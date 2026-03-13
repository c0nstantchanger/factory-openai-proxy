/**
 * Transforms OpenAI Responses API SSE stream -> OpenAI Chat Completions SSE stream.
 *
 * Source events (Responses API):
 *   event: response.created       -> emit role chunk
 *   event: response.output_text.delta  -> emit content delta
 *   event: response.done          -> emit finish + [DONE]
 *
 * Target format: data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
 */

interface SSEParsed {
  type: "event" | "data";
  value: string | Record<string, unknown>;
}

export class OpenAIResponseTransformer {
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
    if (eventType === "response.created") {
      return this.createChunk("", "assistant", false);
    }

    if (eventType === "response.in_progress") {
      return null;
    }

    if (eventType === "response.output_text.delta") {
      const text = (eventData.delta as string) || (eventData.text as string) || "";
      return this.createChunk(text, null, false);
    }

    if (eventType === "response.output_text.done") {
      return null;
    }

    if (eventType === "response.done") {
      const resp = eventData.response as Record<string, unknown> | undefined;
      const status = resp?.status;
      const finishReason = status === "completed" ? "stop" : status === "incomplete" ? "length" : "stop";
      return this.createChunk("", null, true, finishReason) + this.createDone();
    }

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
        }
      }
    }
  }
}

/**
 * Convert a non-streaming Responses API result to chat.completion format.
 */
export function convertResponseToChatCompletion(resp: Record<string, unknown>): Record<string, unknown> {
  const output = resp.output as Array<Record<string, unknown>> | undefined;
  const outputMsg = output?.find((o) => o.type === "message") as Record<string, unknown> | undefined;
  const contentArr = (outputMsg?.content as Array<Record<string, unknown>>) || [];
  const textBlocks = contentArr.filter((c) => c.type === "output_text");
  const content = textBlocks.map((c) => c.text as string).join("");

  const usage = resp.usage as Record<string, number> | undefined;

  return {
    id: resp.id ? (resp.id as string).replace(/^resp_/, "chatcmpl-") : `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: (resp.created_at as number) || Math.floor(Date.now() / 1000),
    model: (resp.model as string) || "unknown-model",
    choices: [
      {
        index: 0,
        message: {
          role: (outputMsg?.role as string) || "assistant",
          content: content || "",
        },
        finish_reason: resp.status === "completed" ? "stop" : "unknown",
      },
    ],
    usage: {
      prompt_tokens: usage?.input_tokens ?? 0,
      completion_tokens: usage?.output_tokens ?? 0,
      total_tokens: usage?.total_tokens ?? 0,
    },
  };
}
