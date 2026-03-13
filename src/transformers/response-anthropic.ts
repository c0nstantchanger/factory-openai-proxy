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

export class AnthropicResponseTransformer {
  private model: string;
  private requestId: string;
  private created: number;

  constructor(model: string, requestId?: string) {
    this.model = model;
    this.requestId = requestId || `chatcmpl-${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
  }

  transformEvent(eventType: string, eventData: Record<string, unknown>): string | null {
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
}

/**
 * Convert a non-streaming Anthropic Messages API response to OpenAI chat.completion format.
 */
export function convertAnthropicToChatCompletion(resp: Record<string, unknown>): Record<string, unknown> {
  const content = resp.content as Array<Record<string, unknown>> | undefined;
  const textBlocks = content?.filter((c) => c.type === "text") || [];
  const textContent = textBlocks.map((c) => c.text as string).join("");

  const stopReason = resp.stop_reason as string | undefined;
  const stopReasonMap: Record<string, string> = {
    end_turn: "stop",
    max_tokens: "length",
    stop_sequence: "stop",
    tool_use: "tool_calls",
  };
  const finishReason = stopReason ? (stopReasonMap[stopReason] || "stop") : "stop";

  const usage = resp.usage as Record<string, number> | undefined;

  return {
    id: resp.id ? `chatcmpl-${(resp.id as string).replace(/^msg_/, "")}` : `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: (resp.model as string) || "unknown-model",
    choices: [
      {
        index: 0,
        message: {
          role: (resp.role as string) || "assistant",
          content: textContent || "",
        },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: usage?.input_tokens ?? 0,
      completion_tokens: usage?.output_tokens ?? 0,
      total_tokens: (usage?.input_tokens ?? 0) + (usage?.output_tokens ?? 0),
    },
  };
}
