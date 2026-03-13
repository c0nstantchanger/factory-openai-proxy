/**
 * Transforms Anthropic Messages API SSE stream -> OpenAI Chat Completions SSE stream.
 *
 * Source events (Anthropic):
 *   event: message_start           -> emit role chunk
 *   event: content_block_delta     -> emit content delta
 *   event: message_delta           -> emit finish reason
 *   event: message_stop            -> emit [DONE]
 *
 * Target format: data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
 */

interface SSEParsed {
  type: "event" | "data";
  value: string | Record<string, unknown>;
}

export class AnthropicResponseTransformer {
  private model: string;
  private requestId: string;
  private created: number;

  constructor(model: string, requestId?: string) {
    this.model = model;
    this.requestId = requestId || `chatcmpl-${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
  }

  private parseSSELine(line: string): SSEParsed | null {
    if (line.startsWith("event:")) {
      return { type: "event", value: line.slice(6).trim() };
    }
    if (line.startsWith("data:")) {
      const dataStr = line.slice(5).trim();
      try {
        return { type: "data", value: JSON.parse(dataStr) };
      } catch {
        return { type: "data", value: dataStr };
      }
    }
    return null;
  }

  private transformEvent(eventType: string, eventData: Record<string, unknown>): string | null {
    if (eventType === "message_start") {
      const message = eventData.message as Record<string, unknown> | undefined;
      if (message?.id) {
        this.requestId = message.id as string;
      }
      return this.createChunk("", "assistant", false);
    }

    if (eventType === "content_block_start" || eventType === "content_block_stop") {
      return null;
    }

    if (eventType === "content_block_delta") {
      const delta = eventData.delta as Record<string, unknown> | undefined;
      const text = (delta?.text as string) || "";
      return this.createChunk(text, null, false);
    }

    if (eventType === "message_delta") {
      const delta = eventData.delta as Record<string, unknown> | undefined;
      const stopReason = delta?.stop_reason as string | undefined;
      if (stopReason) {
        return this.createChunk("", null, true, this.mapStopReason(stopReason));
      }
      return null;
    }

    if (eventType === "message_stop") {
      return this.createDone();
    }

    // ping, etc.
    return null;
  }

  private createChunk(
    content: string,
    role: string | null = null,
    finish: boolean = false,
    finishReason: string | null = null
  ): string {
    const delta: Record<string, unknown> = {};
    if (role) delta.role = role;
    if (content) delta.content = content;

    const chunk = {
      id: this.requestId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta,
          finish_reason: finish ? finishReason : null,
        },
      ],
    };

    return `data: ${JSON.stringify(chunk)}\n\n`;
  }

  private createDone(): string {
    return "data: [DONE]\n\n";
  }

  private mapStopReason(anthropicReason: string): string {
    const mapping: Record<string, string> = {
      end_turn: "stop",
      max_tokens: "length",
      stop_sequence: "stop",
      tool_use: "tool_calls",
    };
    return mapping[anthropicReason] || "stop";
  }

  async *transformStream(sourceStream: AsyncIterable<Uint8Array | Buffer>): AsyncGenerator<string> {
    let buffer = "";
    let currentEvent: string | null = null;

    for await (const chunk of sourceStream) {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;

        const parsed = this.parseSSELine(line);
        if (!parsed) continue;

        if (parsed.type === "event") {
          currentEvent = parsed.value as string;
        } else if (parsed.type === "data" && currentEvent) {
          const transformed = this.transformEvent(currentEvent, parsed.value as Record<string, unknown>);
          if (transformed) {
            yield transformed;
          }
          currentEvent = null;
        }
      }
    }
  }
}
