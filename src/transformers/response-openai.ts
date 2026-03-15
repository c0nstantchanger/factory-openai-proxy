/**
 * Transforms OpenAI Responses API SSE stream -> OpenAI Chat Completions SSE stream.
 *
 * Source events (Responses API):
 *   event: response.created                        -> emit role chunk
 *   event: response.in_progress                    -> (no-op)
 *   event: response.output_item.added              -> emit tool_calls header if function_call
 *   event: response.output_text.delta              -> emit content delta
 *   event: response.output_text.done               -> (no-op)
 *   event: response.function_call_arguments.delta  -> emit tool_calls argument delta
 *   event: response.function_call_arguments.done   -> (no-op)
 *   event: response.output_item.done               -> (no-op)
 *   event: response.content_part.added             -> (no-op)
 *   event: response.content_part.done              -> (no-op)
 *   event: response.completed                      -> (no-op, wait for response.done)
 *   event: response.done                           -> emit finish + usage + [DONE]
 *
 * Target format: data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
 */

interface PendingToolCall {
  /** Index in the OpenAI Chat Completions tool_calls array. */
  index: number;
  id: string;
  name: string;
}

export class OpenAIResponseTransformer {
  private model: string;
  private requestId: string;
  private created: number;
  /** Running counter for tool_calls array indices in the output. */
  private toolCallIndex: number = -1;
  /** Map from Responses API output_index -> tool call tracking info. */
  private toolCalls: Map<number, PendingToolCall> = new Map();
  /** Accumulated usage from response.done. */
  private usage: {
    prompt_tokens: number;
    completion_tokens: number;
  } = { prompt_tokens: 0, completion_tokens: 0 };

  constructor(model: string, requestId?: string) {
    this.model = model;
    this.requestId = requestId || `chatcmpl-${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
  }

  transformEvent(eventType: string, eventData: Record<string, unknown>): string | null {
    if (eventType === "response.created") {
      return this.createChunk({ role: "assistant" });
    }

    if (eventType === "response.in_progress") {
      return null;
    }

    // ─── Tool call start ──────────────────────────────────────────────────
    if (eventType === "response.output_item.added") {
      const item = eventData.item as Record<string, unknown> | undefined;
      if (item?.type === "function_call") {
        this.toolCallIndex++;
        const outputIndex = eventData.output_index as number;
        const tc: PendingToolCall = {
          index: this.toolCallIndex,
          id: (item.call_id as string) || (item.id as string) || `call_${Date.now()}`,
          name: (item.name as string) || "",
        };
        this.toolCalls.set(outputIndex, tc);

        // Emit tool_calls header chunk
        return this.createChunk({
          tool_calls: [
            {
              index: tc.index,
              id: tc.id,
              type: "function",
              function: {
                name: tc.name,
                arguments: "",
              },
            },
          ],
        });
      }
      return null;
    }

    // ─── Text content delta ───────────────────────────────────────────────
    if (eventType === "response.output_text.delta") {
      const text = (eventData.delta as string) || (eventData.text as string) || "";
      if (!text) return null;
      return this.createChunk({ content: text });
    }

    if (eventType === "response.output_text.done") {
      return null;
    }

    // ─── Tool call argument delta ─────────────────────────────────────────
    if (eventType === "response.function_call_arguments.delta") {
      const outputIndex = eventData.output_index as number;
      const tc = this.toolCalls.get(outputIndex);
      if (!tc) return null;

      const argDelta = (eventData.delta as string) || "";
      if (!argDelta) return null;

      return this.createChunk({
        tool_calls: [
          {
            index: tc.index,
            function: {
              arguments: argDelta,
            },
          },
        ],
      });
    }

    if (eventType === "response.function_call_arguments.done") {
      return null;
    }

    // ─── Lifecycle events (no-ops for Chat Completions) ───────────────────
    if (
      eventType === "response.output_item.done" ||
      eventType === "response.content_part.added" ||
      eventType === "response.content_part.done" ||
      eventType === "response.completed"
    ) {
      return null;
    }

    // ─── Final event ──────────────────────────────────────────────────────
    if (eventType === "response.done") {
      const resp = eventData.response as Record<string, unknown> | undefined;
      const status = resp?.status as string | undefined;

      // Capture usage
      const respUsage = resp?.usage as Record<string, number> | undefined;
      if (respUsage) {
        this.usage.prompt_tokens = respUsage.input_tokens ?? 0;
        this.usage.completion_tokens = respUsage.output_tokens ?? 0;
      }

      const finishReason = this.toolCalls.size > 0
        ? "tool_calls"
        : status === "completed"
          ? "stop"
          : status === "incomplete"
            ? "length"
            : "stop";

      return this.createChunk({}, true, finishReason) + this.createDone();
    }

    return null;
  }

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
      };
    }

    return `data: ${JSON.stringify(chunk)}\n\n`;
  }

  private createDone(): string {
    return "data: [DONE]\n\n";
  }
}

/**
 * Convert a non-streaming Responses API result to chat.completion format.
 */
export function convertResponseToChatCompletion(resp: Record<string, unknown>): Record<string, unknown> {
  const output = resp.output as Array<Record<string, unknown>> | undefined;

  // Extract text from message output items
  const outputMsg = output?.find((o) => o.type === "message") as Record<string, unknown> | undefined;
  const contentArr = (outputMsg?.content as Array<Record<string, unknown>>) || [];
  const textBlocks = contentArr.filter((c) => c.type === "output_text");
  const content = textBlocks.map((c) => c.text as string).join("");

  // Extract function_call output items -> tool_calls
  const functionCallItems = output?.filter((o) => o.type === "function_call") || [];

  const usage = resp.usage as Record<string, number> | undefined;

  // Determine finish reason
  const status = resp.status as string | undefined;
  let finishReason: string;
  if (functionCallItems.length > 0) {
    finishReason = "tool_calls";
  } else if (status === "completed") {
    finishReason = "stop";
  } else if (status === "incomplete") {
    finishReason = "length";
  } else {
    finishReason = "stop";
  }

  // Build the message object
  const message: Record<string, unknown> = {
    role: (outputMsg?.role as string) || "assistant",
    content: content || null,
  };

  // Add tool_calls if present
  if (functionCallItems.length > 0) {
    message.tool_calls = functionCallItems.map((item, index) => ({
      id: (item.call_id as string) || (item.id as string) || `call_${Date.now()}_${index}`,
      type: "function",
      function: {
        name: (item.name as string) || "unknown",
        arguments: (item.arguments as string) || "{}",
      },
    }));
  }

  return {
    id: resp.id ? (resp.id as string).replace(/^resp_/, "chatcmpl-") : `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: (resp.created_at as number) || Math.floor(Date.now() / 1000),
    model: (resp.model as string) || "unknown-model",
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: usage?.input_tokens ?? 0,
      completion_tokens: usage?.output_tokens ?? 0,
      total_tokens: usage?.total_tokens ?? 0,
    },
  };
}
