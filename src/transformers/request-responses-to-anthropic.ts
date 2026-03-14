/**
 * Transforms an OpenAI Responses API request into an Anthropic Messages API request.
 *
 * Responses API shape:
 *   { model, input: [...items], instructions?, tools?, max_output_tokens?, reasoning?, stream?, ... }
 *
 * Anthropic Messages shape:
 *   { model, messages: [...], system?: [...], tools?, max_tokens, thinking?, stream? }
 */

import { getSystemPrompt, getModelReasoning } from "../config.js";
import { sanitizeText } from "./sanitize.js";
import { validateAnthropicMessages, validateAnthropicTools } from "./validate-anthropic.js";

const FACTORY_IDENTITY_SYSTEM = "You are Droid, an AI software engineering agent built by Factory.";

const BUDGET_TOKENS: Record<string, number> = {
  low: 4096,
  medium: 12288,
  high: 24576,
  xhigh: 40960,
};

interface ResponsesInput {
  role?: string;
  type?: string;
  content?: string | Array<Record<string, unknown>>;
  text?: string;
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  output?: string;
  status?: string;
  [key: string]: unknown;
}

interface ResponsesTool {
  type: string;
  name?: string;
  description?: string;
  parameters?: unknown;
  strict?: boolean;
  [key: string]: unknown;
}

interface ResponsesApiRequest {
  model: string;
  input?: string | ResponsesInput[];
  instructions?: string;
  tools?: ResponsesTool[];
  max_output_tokens?: number;
  reasoning?: { effort?: string; summary?: string };
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  [key: string]: unknown;
}

export function transformResponsesToAnthropic(req: ResponsesApiRequest): Record<string, unknown> {
  const anthropicRequest: Record<string, unknown> = {
    model: req.model,
    messages: [],
  };

  if (req.stream !== undefined) {
    anthropicRequest.stream = req.stream;
  }

  // max_tokens
  anthropicRequest.max_tokens = req.max_output_tokens || 4096;

  // --- Collect instructions/system text to prepend into first user message ---
  // Factory.ai blocks the `system` parameter for certain auth types, so we
  // inline instructions into the first user message content instead (matching
  // how the Factory CLI itself works).
  const instructionParts: string[] = [];
  instructionParts.push(FACTORY_IDENTITY_SYSTEM);
  const configPrompt = getSystemPrompt();
  if (configPrompt) {
    instructionParts.push(configPrompt);
  }
  if (req.instructions) {
    instructionParts.push(req.instructions);
  }

  // We'll prepend these as text blocks into the first user message after
  // building the messages array.

  // --- Transform input items into Anthropic messages ---
  const rawMessages: Array<Record<string, unknown>> = [];
  // Track any extra system content from input items
  const extraSystemTexts: string[] = [];

  if (typeof req.input === "string") {
    // Simple string input -> single user message
    rawMessages.push({
      role: "user",
      content: [{ type: "text", text: req.input }],
    });
  } else if (Array.isArray(req.input)) {
    // The Responses API input is a flat list of items. We need to reconstruct
    // alternating user/assistant messages for Anthropic.
    //
    // Item types:
    //   { type: "message", role: "user"|"assistant"|"system", content: [...] }
    //   { type: "input_text", text: "..." }          -> user text
    //   { type: "input_image", ... }                  -> user image
    //   { type: "output_text", text: "..." }          -> assistant text
    //   { type: "function_call", id, call_id, name, arguments }  -> assistant tool_use
    //   { type: "function_call_output", call_id, output }        -> user tool_result
    //   { role: "user"|"assistant", content: [...] }  -> message shorthand

    // Track tool_use IDs from function_call items so function_call_output
    // items without a call_id can be matched by position.
    let lastToolUseIds: string[] = [];
    let toolResultIndex = 0;

    for (const item of req.input) {
      const itemType = item.type;

      if (itemType === "message") {
        const role = item.role || "user";
        if (role === "system" || role === "developer") {
          // Collect system text to prepend into first user message
          const content = extractTextFromContent(item.content);
          if (content) {
            extraSystemTexts.push(content);
          }
          continue;
        }
        // A new message item resets tool tracking
        if (role === "user") {
          // User message after tool results — reset for next cycle
          toolResultIndex = 0;
        }
        const anthropicContent = convertResponsesContentToAnthropic(item.content);
        if (anthropicContent.length > 0) {
          rawMessages.push({ role, content: anthropicContent });
        }
        continue;
      }

      if (itemType === "input_text") {
        // New user text resets tool tracking
        toolResultIndex = 0;
        const text = item.text || "";
        if (text.length > 0) {
          rawMessages.push({
            role: "user",
            content: [{ type: "text", text }],
          });
        }
        continue;
      }

      if (itemType === "input_image") {
        const imageContent = convertImageItem(item);
        rawMessages.push({ role: "user", content: [imageContent] });
        continue;
      }

      if (itemType === "output_text") {
        const text = item.text || "";
        if (text.length > 0) {
          rawMessages.push({
            role: "assistant",
            content: [{ type: "text", text }],
          });
        }
        continue;
      }

      if (itemType === "function_call") {
        // Assistant's tool_use block
        let parsedInput: Record<string, unknown> = {};
        if (item.arguments) {
          try {
            parsedInput = JSON.parse(item.arguments as string);
          } catch {
            parsedInput = { input: item.arguments };
          }
        }
        const toolUseId = item.call_id || item.id || `toolu_gen_${Date.now()}_${lastToolUseIds.length}`;
        lastToolUseIds.push(toolUseId);
        rawMessages.push({
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: toolUseId,
              name: item.name || "unknown",
              input: parsedInput,
            },
          ],
        });
        continue;
      }

      if (itemType === "function_call_output") {
        // User's tool_result block
        // Resolve call_id: prefer explicit, then positional match, then fallback
        let resolvedCallId = item.call_id as string | undefined;
        if (!resolvedCallId) {
          if (toolResultIndex < lastToolUseIds.length) {
            resolvedCallId = lastToolUseIds[toolResultIndex];
          } else {
            resolvedCallId = `toolu_fallback_${Date.now()}_${toolResultIndex}`;
          }
        }
        toolResultIndex++;
        rawMessages.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: resolvedCallId,
              content: item.output || "",
            },
          ],
        });
        continue;
      }

      // Fallback: if it has role + content, treat as a message
      if (item.role && item.content !== undefined) {
        if (item.role === "system" || item.role === "developer") {
          const content = extractTextFromContent(item.content);
          if (content) {
            extraSystemTexts.push(content);
          }
          continue;
        }
        const anthropicContent = convertResponsesContentToAnthropic(item.content);
        if (anthropicContent.length > 0) {
          rawMessages.push({ role: item.role, content: anthropicContent });
        }
        continue;
      }

      // Unknown item type, skip
      console.warn(`[RESP->ANTH] Unknown input item type: ${itemType}`);
    }
  }

  // Merge all instruction/system text and prepend into first user message
  const allInstructions = [...instructionParts, ...extraSystemTexts.map(t => sanitizeText(t))];
  const mergedMessages = mergeConsecutiveSameRole(rawMessages);

  if (allInstructions.length > 0 && mergedMessages.length > 0) {
    // Find the first user message and prepend instruction text blocks
    const firstUserIdx = mergedMessages.findIndex((m) => m.role === "user");
    if (firstUserIdx >= 0) {
      const msg = mergedMessages[firstUserIdx];
      const instructionBlocks: Array<Record<string, unknown>> = allInstructions.map((text) => ({
        type: "text",
        text,
      }));
      if (Array.isArray(msg.content)) {
        msg.content = [...instructionBlocks, ...(msg.content as Array<Record<string, unknown>>)];
      } else if (typeof msg.content === "string") {
        msg.content = [...instructionBlocks, { type: "text", text: msg.content }];
      }
    } else {
      // No user message exists, create one with the instructions
      mergedMessages.unshift({
        role: "user",
        content: allInstructions.map((text) => ({ type: "text", text })),
      });
    }
  }

  anthropicRequest.messages = validateAnthropicMessages(stripEmptyTextBlocks(mergedMessages));

  // --- Transform tools ---
  if (req.tools && Array.isArray(req.tools)) {
    const transformed = req.tools
      .filter((t) => t.type === "function")
      .map((t) => ({
        name: t.name || "unknown",
        description: t.description || "",
        input_schema: t.parameters || {},
      }));
    anthropicRequest.tools = validateAnthropicTools(transformed as Array<Record<string, unknown>>);
  }

  // --- Handle thinking/reasoning ---
  const reasoningLevel = getModelReasoning(req.model);
  if (reasoningLevel === "auto") {
    // Map Responses API reasoning -> Anthropic thinking
    if (req.reasoning) {
      const effort = req.reasoning.effort;
      if (effort && effort in BUDGET_TOKENS) {
        anthropicRequest.thinking = {
          type: "enabled",
          budget_tokens: BUDGET_TOKENS[effort],
        };
      }
    }
  } else if (reasoningLevel && reasoningLevel in BUDGET_TOKENS) {
    anthropicRequest.thinking = {
      type: "enabled",
      budget_tokens: BUDGET_TOKENS[reasoningLevel],
    };
  } else {
    delete anthropicRequest.thinking;
  }

  // --- Pass through compatible parameters ---
  if (req.temperature !== undefined) anthropicRequest.temperature = req.temperature;
  if (req.top_p !== undefined) anthropicRequest.top_p = req.top_p;

  return anthropicRequest;
}

// --- Helpers ---

function extractTextFromContent(
  content: string | Array<Record<string, unknown>> | undefined,
): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((c) => c.type === "text" || c.type === "input_text" || c.type === "output_text")
      .map((c) => (c.text as string) || "")
      .join("\n");
  }
  return "";
}

function convertResponsesContentToAnthropic(
  content: string | Array<Record<string, unknown>> | undefined,
): Array<Record<string, unknown>> {
  if (typeof content === "string") {
    if (content.length === 0) return [];
    return [{ type: "text", text: content }];
  }
  if (!Array.isArray(content)) {
    return [];
  }

  const result: Array<Record<string, unknown>> = [];
  for (const part of content) {
    if (part.type === "input_text" || part.type === "output_text" || part.type === "text") {
      const text = (part.text as string) || "";
      if (text.length > 0) {
        result.push({ type: "text", text });
      }
    } else if (part.type === "input_image") {
      result.push(convertImageItem(part));
    } else {
      result.push(part);
    }
  }
  return result;
}

function convertImageItem(item: Record<string, unknown>): Record<string, unknown> {
  const imageUrl = (item.image_url as string) || "";
  const dataUriMatch = imageUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/s);
  if (dataUriMatch) {
    return {
      type: "image",
      source: { type: "base64", media_type: dataUriMatch[1], data: dataUriMatch[2] },
    };
  }
  return {
    type: "image",
    source: { type: "url", url: imageUrl },
  };
}

function mergeConsecutiveSameRole(
  messages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  if (messages.length === 0) return [];
  const merged: Array<Record<string, unknown>> = [];

  for (const msg of messages) {
    const last = merged.length > 0 ? merged[merged.length - 1] : null;
    if (last && last.role === msg.role) {
      const lastContent = last.content;
      const curContent = msg.content;
      if (Array.isArray(lastContent) && Array.isArray(curContent)) {
        (lastContent as Array<Record<string, unknown>>).push(
          ...(curContent as Array<Record<string, unknown>>),
        );
      } else if (Array.isArray(lastContent) && typeof curContent === "string") {
        (lastContent as Array<Record<string, unknown>>).push({ type: "text", text: curContent });
      } else if (typeof lastContent === "string" && Array.isArray(curContent)) {
        last.content = [
          { type: "text", text: lastContent },
          ...(curContent as Array<Record<string, unknown>>),
        ];
      } else if (typeof lastContent === "string" && typeof curContent === "string") {
        last.content = [
          { type: "text", text: lastContent },
          { type: "text", text: curContent },
        ];
      }
    } else {
      merged.push({ ...msg });
    }
  }
  return merged;
}

/**
 * Final safety net: remove any empty text content blocks from all messages.
 * Anthropic rejects `{ type: "text", text: "" }`.
 * Also drops messages that end up with zero content blocks after filtering.
 */
function stripEmptyTextBlocks(
  messages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const cleaned: Array<Record<string, unknown>> = [];
  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      const filtered = (msg.content as Array<Record<string, unknown>>).filter(
        (block) => !(block.type === "text" && (!block.text || (block.text as string).length === 0)),
      );
      if (filtered.length > 0) {
        cleaned.push({ ...msg, content: filtered });
      }
      // else: drop the entire message — no content left
    } else if (typeof msg.content === "string") {
      if (msg.content.length > 0) {
        cleaned.push(msg);
      }
    } else {
      cleaned.push(msg);
    }
  }
  return cleaned;
}
