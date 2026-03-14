/**
 * Transforms OpenAI Chat Completions responses -> OpenAI Responses API format.
 *
 * The "common" backend returns standard Chat Completions format.
 * We need to convert both streaming and non-streaming responses to Responses API format.
 *
 * Chat Completions streaming: data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{...}}]}
 * Responses API streaming: event: response.output_text.delta\ndata: {...}
 */

export class CommonToResponsesTransformer {
  private model: string;
  private responseId: string;
  private created: number;
  private accumulatedText: string = "";
  private emittedCreated: boolean = false;
  private messageOutputIndex: number = 0;
  private outputItemCount: number = 0;
  private contentPartAdded: boolean = false;
  private itemId: string;
  private pendingToolCalls: Map<
    number,
    { id: string; name: string; args: string; outputIndex: number }
  > = new Map();
  private usage: { input_tokens: number; output_tokens: number } = {
    input_tokens: 0,
    output_tokens: 0,
  };

  constructor(model: string, responseId?: string) {
    this.model = model;
    this.responseId = responseId || `resp_${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
    this.itemId = `text_${Date.now().toString(36)}`;
  }

  /**
   * Process a raw SSE line from the common backend.
   * Common backend sends standard "data: {...}" lines without event: prefixes.
   */
  transformSSELine(line: string): string | null {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.startsWith("data:")) return null;

    const dataStr = trimmed.slice(5).trim();
    if (dataStr === "[DONE]") {
      return this.finalize();
    }

    let data: Record<string, unknown>;
    try {
      data = JSON.parse(dataStr);
    } catch {
      return null;
    }

    return this.transformChunk(data);
  }

  transformChunk(chunk: Record<string, unknown>): string | null {
    let out = "";

    // Emit response.created on first chunk
    if (!this.emittedCreated) {
      this.emittedCreated = true;
      const baseResponse = this.buildResponseObject("in_progress");
      out += this.sse("response.created", baseResponse);
      out += this.sse("response.in_progress", baseResponse);

      // Add message output item
      const messageItem = {
        type: "message",
        id: `msg_${Date.now().toString(36)}`,
        status: "in_progress",
        role: "assistant",
        content: [],
      };
      this.messageOutputIndex = this.outputItemCount++;
      out += this.sse("response.output_item.added", {
        type: "response.output_item.added",
        output_index: this.messageOutputIndex,
        item: messageItem,
      });
    }

    const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
    if (!choices || choices.length === 0) return out || null;

    const choice = choices[0];
    const delta = choice.delta as Record<string, unknown> | undefined;
    const finishReason = choice.finish_reason as string | null;

    if (delta) {
      // Text content
      const content = delta.content as string | undefined;
      if (content) {
        if (!this.contentPartAdded) {
          this.contentPartAdded = true;
          out += this.sse("response.content_part.added", {
            type: "response.content_part.added",
            item_id: this.itemId,
            output_index: this.messageOutputIndex,
            content_index: 0,
            part: { type: "output_text", text: "" },
          });
        }

        this.accumulatedText += content;
        out += this.sse("response.output_text.delta", {
          type: "response.output_text.delta",
          item_id: this.itemId,
          output_index: this.messageOutputIndex,
          content_index: 0,
          delta: content,
        });
      }

      // Tool calls
      const toolCalls = delta.tool_calls as Array<Record<string, unknown>> | undefined;
      if (toolCalls) {
        for (const tc of toolCalls) {
          const tcIndex = tc.index as number;
          const tcId = tc.id as string | undefined;
          const tcFunction = tc.function as Record<string, unknown> | undefined;

          if (tcId && tcFunction?.name) {
            // New tool call header
            const outputIndex = this.outputItemCount++;
            this.pendingToolCalls.set(tcIndex, {
              id: tcId,
              name: tcFunction.name as string,
              args: "",
              outputIndex,
            });

            out += this.sse("response.output_item.added", {
              type: "response.output_item.added",
              output_index: outputIndex,
              item: {
                type: "function_call",
                id: tcId,
                call_id: tcId,
                name: tcFunction.name,
                arguments: "",
                status: "in_progress",
              },
            });
          }

          // Argument delta
          if (tcFunction?.arguments) {
            const pending = this.pendingToolCalls.get(tcIndex);
            if (pending) {
              const argDelta = tcFunction.arguments as string;
              pending.args += argDelta;

              out += this.sse("response.function_call_arguments.delta", {
                type: "response.function_call_arguments.delta",
                item_id: pending.id,
                output_index: pending.outputIndex,
                delta: argDelta,
              });
            }
          }
        }
      }
    }

    // Capture usage if present
    const chunkUsage = chunk.usage as Record<string, number> | undefined;
    if (chunkUsage) {
      this.usage.input_tokens = chunkUsage.prompt_tokens ?? this.usage.input_tokens;
      this.usage.output_tokens = chunkUsage.completion_tokens ?? this.usage.output_tokens;
    }

    if (finishReason) {
      out += this.emitFinish(finishReason);
    }

    return out || null;
  }

  private emitFinish(finishReason: string): string {
    let out = "";

    // Close text content part
    if (this.contentPartAdded) {
      out += this.sse("response.output_text.done", {
        type: "response.output_text.done",
        item_id: this.itemId,
        output_index: this.messageOutputIndex,
        content_index: 0,
        text: this.accumulatedText,
      });
      out += this.sse("response.content_part.done", {
        type: "response.content_part.done",
        item_id: this.itemId,
        output_index: this.messageOutputIndex,
        content_index: 0,
        part: { type: "output_text", text: this.accumulatedText },
      });
    }

    // Close message output item
    out += this.sse("response.output_item.done", {
      type: "response.output_item.done",
      output_index: this.messageOutputIndex,
      item: {
        type: "message",
        id: `msg_${Date.now().toString(36)}`,
        status: "completed",
        role: "assistant",
        content: this.contentPartAdded
          ? [{ type: "output_text", text: this.accumulatedText }]
          : [],
      },
    });

    // Close any pending tool calls
    for (const [, tc] of this.pendingToolCalls) {
      out += this.sse("response.function_call_arguments.done", {
        type: "response.function_call_arguments.done",
        item_id: tc.id,
        output_index: tc.outputIndex,
        arguments: tc.args,
      });
      out += this.sse("response.output_item.done", {
        type: "response.output_item.done",
        output_index: tc.outputIndex,
        item: {
          type: "function_call",
          id: tc.id,
          call_id: tc.id,
          name: tc.name,
          arguments: tc.args,
          status: "completed",
        },
      });
    }

    return out;
  }

  private finalize(): string {
    const status = "completed";
    const response = this.buildResponseObject(status);
    response.usage = {
      input_tokens: this.usage.input_tokens,
      output_tokens: this.usage.output_tokens,
      total_tokens: this.usage.input_tokens + this.usage.output_tokens,
    };

    let out = "";
    out += this.sse("response.completed", response);
    out += this.sse("response.done", response);
    return out;
  }

  private buildResponseObject(status: string): Record<string, unknown> {
    return {
      id: this.responseId,
      object: "response",
      created_at: this.created,
      model: this.model,
      status,
      output: [],
    };
  }

  private sse(event: string, data: Record<string, unknown>): string {
    return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  }
}

/**
 * Convert a non-streaming Chat Completions response to Responses API format.
 */
export function convertCommonToResponses(resp: Record<string, unknown>): Record<string, unknown> {
  const choices = resp.choices as Array<Record<string, unknown>> | undefined;
  const firstChoice = choices?.[0];
  const message = firstChoice?.message as Record<string, unknown> | undefined;

  const textContent = (message?.content as string) || "";
  const finishReason = (firstChoice?.finish_reason as string) || "stop";
  const toolCalls = message?.tool_calls as Array<Record<string, unknown>> | undefined;

  const usage = resp.usage as Record<string, number> | undefined;
  const inputTokens = usage?.prompt_tokens ?? 0;
  const outputTokens = usage?.completion_tokens ?? 0;

  const statusMap: Record<string, string> = {
    stop: "completed",
    length: "incomplete",
    tool_calls: "completed",
  };
  const status = statusMap[finishReason] || "completed";

  const output: Array<Record<string, unknown>> = [];

  // Message output item
  if (textContent) {
    output.push({
      type: "message",
      id: `msg_${Date.now().toString(36)}`,
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: textContent }],
    });
  }

  // Function call output items
  if (toolCalls) {
    for (const tc of toolCalls) {
      const fn = tc.function as Record<string, unknown> | undefined;
      output.push({
        type: "function_call",
        id: (tc.id as string) || `call_${Date.now().toString(36)}`,
        call_id: (tc.id as string) || `call_${Date.now().toString(36)}`,
        name: (fn?.name as string) || "unknown",
        arguments: (fn?.arguments as string) || "{}",
        status: "completed",
      });
    }
  }

  return {
    id: resp.id
      ? `resp_${(resp.id as string).replace(/^chatcmpl-/, "")}`
      : `resp_${Date.now()}`,
    object: "response",
    created_at: (resp.created as number) || Math.floor(Date.now() / 1000),
    model: (resp.model as string) || "unknown-model",
    status,
    output,
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
    },
  };
}
