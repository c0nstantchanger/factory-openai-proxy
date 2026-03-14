/**
 * Transforms Anthropic Messages API SSE stream -> OpenAI Chat Completions SSE stream.
 *
 * Source events (Anthropic):
 *   event: message_start           -> emit role chunk, capture input usage
 *   event: content_block_start     -> track block type; emit tool_calls header if tool_use
 *   event: content_block_delta     -> emit content delta (text or tool args)
 *   event: content_block_stop      -> (no-op)
 *   event: message_delta           -> emit finish reason + final usage chunk
 *   event: message_stop            -> emit [DONE]
 *
 * Target format: data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
 */

interface ToolCallBlock {
  index: number;
  id: string;
  name: string;
}

interface UsageAccumulator {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_tokens_details: {
    cached_tokens: number;
  };
}

export class AnthropicResponseTransformer {
  private model: string;
  private requestId: string;
  private created: number;

  /** Tracks active content blocks by their Anthropic index. */
  private activeBlocks: Map<number, { type: string; toolCall?: ToolCallBlock }> = new Map();
  /** Running counter for tool_calls array indices in the OpenAI output. */
  private toolCallIndex: number = -1;
  /** Accumulated usage data from message_start and message_delta. */
  private usage: UsageAccumulator = {
    prompt_tokens: 0,
    completion_tokens: 0,
    prompt_tokens_details: { cached_tokens: 0 },
  };

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

      // Capture input token usage from message_start
      const msgUsage = message?.usage as Record<string, number> | undefined;
      if (msgUsage) {
        const inputTokens = msgUsage.input_tokens ?? 0;
        const cacheCreation = msgUsage.cache_creation_input_tokens ?? 0;
        const cacheRead = msgUsage.cache_read_input_tokens ?? 0;
        // prompt_tokens = total input (base + cache creation + cache read)
        this.usage.prompt_tokens = inputTokens + cacheCreation + cacheRead;
        this.usage.prompt_tokens_details.cached_tokens = cacheRead;
        // Seed output tokens (usually 0 or 1 at start)
        this.usage.completion_tokens = msgUsage.output_tokens ?? 0;
      }

      return this.createChunk({ role: "assistant" });
    }

    if (eventType === "content_block_start") {
      const index = eventData.index as number;
      const contentBlock = eventData.content_block as Record<string, unknown> | undefined;
      const blockType = (contentBlock?.type as string) || "text";

      if (blockType === "tool_use") {
        this.toolCallIndex++;
        const toolCall: ToolCallBlock = {
          index: this.toolCallIndex,
          id: (contentBlock?.id as string) || `call_${Date.now()}`,
          name: (contentBlock?.name as string) || "",
        };
        this.activeBlocks.set(index, { type: "tool_use", toolCall });

        // Emit the tool_calls header chunk with id, type, and function name
        return this.createChunk({
          tool_calls: [
            {
              index: toolCall.index,
              id: toolCall.id,
              type: "function",
              function: {
                name: toolCall.name,
                arguments: "",
              },
            },
          ],
        });
      }

      if (blockType === "thinking") {
        this.activeBlocks.set(index, { type: "thinking" });
        return null;
      }

      // text block or unknown — just track it
      this.activeBlocks.set(index, { type: blockType });
      return null;
    }

    if (eventType === "content_block_delta") {
      const index = eventData.index as number;
      const delta = eventData.delta as Record<string, unknown> | undefined;
      if (!delta) return null;

      const deltaType = delta.type as string | undefined;
      const block = this.activeBlocks.get(index);

      // Tool use argument streaming
      if (deltaType === "input_json_delta" || block?.type === "tool_use") {
        const partialJson = (delta.partial_json as string) || "";
        if (!partialJson && deltaType === "input_json_delta") return null;
        const toolCall = block?.toolCall;
        if (!toolCall) return null;

        return this.createChunk({
          tool_calls: [
            {
              index: toolCall.index,
              function: {
                arguments: partialJson,
              },
            },
          ],
        });
      }

      // Thinking delta / signature delta — skip (no standard OpenAI streaming equivalent)
      if (
        deltaType === "thinking_delta" ||
        deltaType === "signature_delta" ||
        block?.type === "thinking"
      ) {
        return null;
      }

      // Text delta
      const text = (delta.text as string) ?? "";
      if (!text) return null;
      return this.createChunk({ content: text });
    }

    if (eventType === "content_block_stop") {
      const index = eventData.index as number;
      this.activeBlocks.delete(index);
      return null;
    }

    if (eventType === "message_delta") {
      const delta = eventData.delta as Record<string, unknown> | undefined;
      const stopReason = delta?.stop_reason as string | undefined;

      // Update output token count from the final message_delta usage
      const deltaUsage = eventData.usage as Record<string, number> | undefined;
      if (deltaUsage) {
        this.usage.completion_tokens = deltaUsage.output_tokens ?? this.usage.completion_tokens;
        // Re-read input tokens if present (they may include cache fields)
        if (deltaUsage.input_tokens !== undefined) {
          const inputTokens = deltaUsage.input_tokens ?? 0;
          const cacheCreation = deltaUsage.cache_creation_input_tokens ?? 0;
          const cacheRead = deltaUsage.cache_read_input_tokens ?? 0;
          this.usage.prompt_tokens = inputTokens + cacheCreation + cacheRead;
          this.usage.prompt_tokens_details.cached_tokens = cacheRead;
        }
      }

      if (stopReason) {
        // Emit the finish chunk with usage on the final chunk
        return this.createChunk({}, true, this.mapStopReason(stopReason));
      }
      return null;
    }

    if (eventType === "message_stop") {
      return this.createDone();
    }

    // ping, etc.
    return null;
  }

  /**
   * Creates a single SSE chunk in OpenAI chat.completion.chunk format.
   * `deltaFields` is merged directly into the `delta` object.
   * When `finish` is true, the `usage` field is included on the chunk.
   */
  private createChunk(
    deltaFields: Record<string, unknown>,
    finish: boolean = false,
    finishReason: string | null = null,
  ): string {
    const chunk: Record<string, unknown> = {
      id: this.requestId,
      object: "chat.completion.chunk",
      created: this.created,
      model: this.model,
      choices: [
        {
          index: 0,
          delta: deltaFields,
          finish_reason: finish ? finishReason : null,
        },
      ],
    };

    // Include usage on the final chunk (OpenAI stream_options.include_usage behavior)
    if (finish) {
      chunk.usage = {
        prompt_tokens: this.usage.prompt_tokens,
        completion_tokens: this.usage.completion_tokens,
        total_tokens: this.usage.prompt_tokens + this.usage.completion_tokens,
        prompt_tokens_details: this.usage.prompt_tokens_details,
      };
    }

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
  const toolUseBlocks = content?.filter((c) => c.type === "tool_use") || [];

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

  // Compute prompt tokens including cache tokens
  const inputTokens = usage?.input_tokens ?? 0;
  const cacheCreation = usage?.cache_creation_input_tokens ?? 0;
  const cacheRead = usage?.cache_read_input_tokens ?? 0;
  const promptTokens = inputTokens + cacheCreation + cacheRead;
  const completionTokens = usage?.output_tokens ?? 0;

  // Build the message object
  const message: Record<string, unknown> = {
    role: (resp.role as string) || "assistant",
    content: textContent || null,
  };

  // Add tool_calls if present
  if (toolUseBlocks.length > 0) {
    message.tool_calls = toolUseBlocks.map((block, index) => ({
      id: (block.id as string) || `call_${Date.now()}_${index}`,
      type: "function",
      function: {
        name: block.name as string,
        arguments: typeof block.input === "string"
          ? block.input
          : JSON.stringify(block.input ?? {}),
      },
    }));
  }

  return {
    id: resp.id ? `chatcmpl-${(resp.id as string).replace(/^msg_/, "")}` : `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: (resp.model as string) || "unknown-model",
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      prompt_tokens_details: {
        cached_tokens: cacheRead,
      },
    },
  };
}
