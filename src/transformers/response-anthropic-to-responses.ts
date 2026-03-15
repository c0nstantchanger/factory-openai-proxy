/**
 * Transforms Anthropic Messages API responses -> OpenAI Responses API format.
 *
 * Streaming: Anthropic SSE events -> Responses API SSE events
 * Non-streaming: Anthropic JSON -> Responses API JSON
 *
 * Responses API streaming events:
 *   event: response.created         data: { ...response object }
 *   event: response.in_progress     data: { ...response object }
 *   event: response.output_text.delta  data: { type: "response.output_text.delta", item_id, output_index, content_index, delta }
 *   event: response.output_text.done   data: { type: "response.output_text.done", item_id, output_index, content_index, text }
 *   event: response.function_call_arguments.delta  data: { ... }
 *   event: response.function_call_arguments.done   data: { ... }
 *   event: response.output_item.added  data: { ... }
 *   event: response.output_item.done   data: { ... }
 *   event: response.content_part.added data: { ... }
 *   event: response.content_part.done  data: { ... }
 *   event: response.completed       data: { ...response object }
 *   event: response.done            data: { ...response object with usage }
 */

interface UsageAccumulator {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
}

interface ActiveBlock {
  type: string; // "text" | "tool_use" | "thinking"
  index: number; // anthropic content block index
  outputIndex: number; // index in response.output[]
  contentIndex: number; // index in output item's content[]
  itemId: string;
  text: string; // accumulated text
  toolCallId?: string;
  toolName?: string;
  toolArgs?: string;
}

interface ResponseOutputItem {
  type: string;
  id: string;
  status: string;
  role?: string;
  content?: Array<Record<string, unknown>>;
  call_id?: string;
  name?: string;
  arguments?: string;
}

export class AnthropicToResponsesTransformer {
  private model: string;
  private responseId: string;
  private created: number;
  private activeBlocks: Map<number, ActiveBlock> = new Map();
  private outputItemCount: number = 0;
  private messageOutputIndex: number = 0; // the output_index for the main message item
  private contentPartCount: number = 0;
  private toolOutputCount: number = 0;
  private messageItemId: string = `msg_${Date.now().toString(36)}`;
  private messageParts: Array<Record<string, unknown>> = [];
  private outputItems: Map<number, ResponseOutputItem> = new Map();
  private usage: UsageAccumulator = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
  private accumulatedText: string = "";
  private emittedCreated: boolean = false;
  /** Stores the stop_reason from message_delta for use in message_stop. */
  private lastStopReason: string = "end_turn";

  constructor(model: string, responseId?: string) {
    this.model = model;
    this.responseId = responseId || `resp_${Date.now()}`;
    this.created = Math.floor(Date.now() / 1000);
  }

  transformEvent(eventType: string, eventData: Record<string, unknown>): string | null {
    if (eventType === "message_start") {
      const message = eventData.message as Record<string, unknown> | undefined;
      if (message?.id) {
        this.responseId = `resp_${(message.id as string).replace(/^msg_/, "")}`;
      }

      // Capture usage
      const msgUsage = message?.usage as Record<string, number> | undefined;
      if (msgUsage) {
        this.usage.input_tokens = msgUsage.input_tokens ?? 0;
        this.usage.output_tokens = msgUsage.output_tokens ?? 0;
        this.usage.cache_read_input_tokens = msgUsage.cache_read_input_tokens ?? 0;
        this.usage.cache_creation_input_tokens = msgUsage.cache_creation_input_tokens ?? 0;
      }

      // Emit response.created + response.in_progress
      let out = "";

      const baseResponse = this.buildResponseObject("in_progress");
      if (!this.emittedCreated) {
        out += this.sse("response.created", baseResponse);
        this.emittedCreated = true;
      }
      out += this.sse("response.in_progress", baseResponse);

      // Add the message output item
      const messageItem: ResponseOutputItem = {
        type: "message",
        id: this.messageItemId,
        status: "in_progress",
        role: "assistant",
        content: [],
      };
      this.messageOutputIndex = this.outputItemCount++;
      this.outputItems.set(this.messageOutputIndex, messageItem);
      out += this.sse("response.output_item.added", {
        type: "response.output_item.added",
        output_index: this.messageOutputIndex,
        item: messageItem,
      });

      return out;
    }

    if (eventType === "content_block_start") {
      const index = eventData.index as number;
      const contentBlock = eventData.content_block as Record<string, unknown> | undefined;
      const blockType = (contentBlock?.type as string) || "text";

      if (blockType === "thinking") {
        // Skip thinking blocks entirely
        this.activeBlocks.set(index, {
          type: "thinking",
          index,
          outputIndex: this.messageOutputIndex,
          contentIndex: -1,
          itemId: "",
          text: "",
        });
        return null;
      }

      if (blockType === "tool_use") {
        // Tool use -> function_call output item
        const toolOutputIndex = this.outputItemCount++;
        const itemId = (contentBlock?.id as string) || `call_${Date.now().toString(36)}`;
        const toolName = (contentBlock?.name as string) || "unknown";

        this.activeBlocks.set(index, {
          type: "tool_use",
          index,
          outputIndex: toolOutputIndex,
          contentIndex: 0,
          itemId,
          text: "",
          toolCallId: itemId,
          toolName,
          toolArgs: "",
        });

        const functionCallItem: ResponseOutputItem = {
          type: "function_call",
          id: itemId,
          call_id: itemId,
          name: toolName,
          arguments: "",
          status: "in_progress",
        };
        this.outputItems.set(toolOutputIndex, functionCallItem);

        return this.sse("response.output_item.added", {
          type: "response.output_item.added",
          output_index: toolOutputIndex,
          item: functionCallItem,
        });
      }

      // Text block -> content part on the message item
      const contentIndex = this.contentPartCount++;
      const itemId = `text_${Date.now().toString(36)}_${contentIndex}`;

      this.activeBlocks.set(index, {
        type: "text",
        index,
        outputIndex: this.messageOutputIndex,
        contentIndex,
        itemId,
        text: "",
      });

      const contentPart = { type: "output_text", text: "" };
      this.messageParts[contentIndex] = contentPart;
      const messageItem = this.outputItems.get(this.messageOutputIndex);
      if (messageItem?.content) {
        messageItem.content[contentIndex] = contentPart;
      }
      return this.sse("response.content_part.added", {
        type: "response.content_part.added",
        item_id: itemId,
        output_index: this.messageOutputIndex,
        content_index: contentIndex,
        part: contentPart,
      });
    }

    if (eventType === "content_block_delta") {
      const index = eventData.index as number;
      const delta = eventData.delta as Record<string, unknown> | undefined;
      if (!delta) return null;

      const block = this.activeBlocks.get(index);
      if (!block) return null;

      const deltaType = delta.type as string | undefined;

      // Skip thinking deltas
      if (block.type === "thinking" || deltaType === "thinking_delta" || deltaType === "signature_delta") {
        return null;
      }

      // Tool use argument streaming
      if (deltaType === "input_json_delta" || block.type === "tool_use") {
        const partialJson = (delta.partial_json as string) || "";
        if (!partialJson) return null;
        block.toolArgs = (block.toolArgs || "") + partialJson;

        const functionCallItem = this.outputItems.get(block.outputIndex);
        if (functionCallItem) {
          functionCallItem.arguments = block.toolArgs || "";
        }

        return this.sse("response.function_call_arguments.delta", {
          type: "response.function_call_arguments.delta",
          item_id: block.itemId,
          output_index: block.outputIndex,
          delta: partialJson,
        });
      }

      // Text delta
      const text = (delta.text as string) ?? "";
      if (!text) return null;
      block.text += text;
      this.accumulatedText += text;

      const messageItem = this.outputItems.get(this.messageOutputIndex);
      const currentPart = this.messageParts[block.contentIndex];
      if (currentPart) {
        currentPart.text = block.text;
      }
      if (messageItem?.content) {
        messageItem.content[block.contentIndex] = { type: "output_text", text: block.text };
      }

      return this.sse("response.output_text.delta", {
        type: "response.output_text.delta",
        item_id: block.itemId,
        output_index: block.outputIndex,
        content_index: block.contentIndex,
        delta: text,
      });
    }

    if (eventType === "content_block_stop") {
      const index = eventData.index as number;
      const block = this.activeBlocks.get(index);
      this.activeBlocks.delete(index);

      if (!block || block.type === "thinking") return null;

      let out = "";

      if (block.type === "tool_use") {
        // Emit function_call_arguments.done + output_item.done
        out += this.sse("response.function_call_arguments.done", {
          type: "response.function_call_arguments.done",
          item_id: block.itemId,
          output_index: block.outputIndex,
          arguments: block.toolArgs || "",
        });
        const functionCallItem = this.outputItems.get(block.outputIndex);
        if (functionCallItem) {
          functionCallItem.arguments = block.toolArgs || "";
          functionCallItem.status = "completed";
        }
        out += this.sse("response.output_item.done", {
          type: "response.output_item.done",
          output_index: block.outputIndex,
          item: functionCallItem || {
            type: "function_call",
            id: block.itemId,
            call_id: block.toolCallId,
            name: block.toolName,
            arguments: block.toolArgs || "",
            status: "completed",
          },
        });
      } else {
        // Text block done
        out += this.sse("response.output_text.done", {
          type: "response.output_text.done",
          item_id: block.itemId,
          output_index: block.outputIndex,
          content_index: block.contentIndex,
          text: block.text,
        });
        out += this.sse("response.content_part.done", {
          type: "response.content_part.done",
          item_id: block.itemId,
          output_index: block.outputIndex,
          content_index: block.contentIndex,
          part: { type: "output_text", text: block.text },
        });
      }

      return out;
    }

    if (eventType === "message_delta") {
      const delta = eventData.delta as Record<string, unknown> | undefined;
      const deltaUsage = eventData.usage as Record<string, number> | undefined;

      if (deltaUsage) {
        this.usage.output_tokens = deltaUsage.output_tokens ?? this.usage.output_tokens;
        if (deltaUsage.input_tokens !== undefined) {
          this.usage.input_tokens = deltaUsage.input_tokens;
        }
      }

      // Emit output_item.done for the message item
      const stopReason = delta?.stop_reason as string | undefined;
      if (stopReason) {
        this.lastStopReason = stopReason;
        const messageItem = this.outputItems.get(this.messageOutputIndex);
        if (messageItem) {
          messageItem.status = "completed";
          messageItem.content = this.messageParts.filter(Boolean);
        }

        return this.sse("response.output_item.done", {
          type: "response.output_item.done",
          output_index: this.messageOutputIndex,
          item: messageItem || {
            type: "message",
            id: this.messageItemId,
            status: "completed",
            role: "assistant",
            content: this.messageParts.filter(Boolean),
          },
        });
      }

      return null;
    }

    if (eventType === "message_stop") {
      // Emit response.completed + response.done
      const status = this.mapStatus(this.lastStopReason);
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

    // ping, etc.
    return null;
  }

  private buildResponseObject(status: string): Record<string, unknown> {
    const orderedOutput = [...this.outputItems.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, item]) => item);

    return {
      id: this.responseId,
      object: "response",
      created_at: this.created,
      model: this.model,
      status,
      output: orderedOutput,
    };
  }

  private mapStatus(anthropicStopReason: string): string {
    const map: Record<string, string> = {
      end_turn: "completed",
      max_tokens: "incomplete",
      stop_sequence: "completed",
      tool_use: "completed",
    };
    return map[anthropicStopReason] || "completed";
  }

  private sse(event: string, data: Record<string, unknown>): string {
    return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  }
}

/**
 * Convert a non-streaming Anthropic Messages response to Responses API format.
 */
export function convertAnthropicToResponses(resp: Record<string, unknown>): Record<string, unknown> {
  const content = resp.content as Array<Record<string, unknown>> | undefined;
  const textBlocks = content?.filter((c) => c.type === "text") || [];
  const toolUseBlocks = content?.filter((c) => c.type === "tool_use") || [];

  const textContent = textBlocks.map((c) => c.text as string).join("");

  const usage = resp.usage as Record<string, number> | undefined;
  const inputTokens = (usage?.input_tokens ?? 0) + (usage?.cache_creation_input_tokens ?? 0) + (usage?.cache_read_input_tokens ?? 0);
  const outputTokens = usage?.output_tokens ?? 0;

  const stopReason = resp.stop_reason as string | undefined;
  const statusMap: Record<string, string> = {
    end_turn: "completed",
    max_tokens: "incomplete",
    stop_sequence: "completed",
    tool_use: "completed",
  };
  const status = stopReason ? (statusMap[stopReason] || "completed") : "completed";

  // Build output items
  const output: Array<Record<string, unknown>> = [];

  // Message output item (contains text)
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
  for (const block of toolUseBlocks) {
    output.push({
      type: "function_call",
      id: (block.id as string) || `call_${Date.now().toString(36)}`,
      call_id: (block.id as string) || `call_${Date.now().toString(36)}`,
      name: (block.name as string) || "unknown",
      arguments: typeof block.input === "string" ? block.input : JSON.stringify(block.input ?? {}),
      status: "completed",
    });
  }

  return {
    id: resp.id ? `resp_${(resp.id as string).replace(/^msg_/, "")}` : `resp_${Date.now()}`,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
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
