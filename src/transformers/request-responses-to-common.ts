/**
 * Transforms an OpenAI Responses API request into an OpenAI Chat Completions request
 * for routing to the "common" endpoint (Gemini, GLM, Kimi, Minimax, etc.).
 *
 * Responses API shape:
 *   { model, input: [...items], instructions?, tools?, max_output_tokens?, reasoning?, stream? }
 *
 * Chat Completions shape:
 *   { model, messages: [...], tools?, max_tokens?, reasoning_effort?, stream? }
 */

import { getSystemPrompt, getModelReasoning } from "../config.js";

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
  presence_penalty?: number;
  frequency_penalty?: number;
  [key: string]: unknown;
}

export function transformResponsesToCommon(req: ResponsesApiRequest): Record<string, unknown> {
  const chatRequest: Record<string, unknown> = {
    model: req.model,
    messages: [],
  };

  if (req.stream !== undefined) {
    chatRequest.stream = req.stream;
  }

  if (req.max_output_tokens) {
    chatRequest.max_tokens = req.max_output_tokens;
  }

  // --- Build messages ---
  const messages: Array<Record<string, unknown>> = [];

  // System message from config + instructions
  const configPrompt = getSystemPrompt();
  const systemText = [configPrompt, req.instructions].filter(Boolean).join("\n");
  if (systemText) {
    messages.push({ role: "system", content: systemText });
  }

  // Convert input items to chat messages
  if (typeof req.input === "string") {
    messages.push({ role: "user", content: req.input });
  } else if (Array.isArray(req.input)) {
    for (const item of req.input) {
      const itemType = item.type;

      if (itemType === "message") {
        const role = item.role || "user";
        if (role === "system") {
          // Merge with existing system or add new
          const content = extractText(item.content);
          if (content) {
            messages.push({ role: "system", content });
          }
          continue;
        }
        messages.push({ role, content: convertContentToChatFormat(item.content) });
        continue;
      }

      if (itemType === "input_text") {
        messages.push({ role: "user", content: item.text || "" });
        continue;
      }

      if (itemType === "input_image") {
        const imageUrl = (item.image_url as string) || "";
        messages.push({
          role: "user",
          content: [{ type: "image_url", image_url: { url: imageUrl } }],
        });
        continue;
      }

      if (itemType === "output_text") {
        messages.push({ role: "assistant", content: item.text || "" });
        continue;
      }

      if (itemType === "function_call") {
        // Assistant's tool call
        messages.push({
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: item.call_id || item.id || `call_${Date.now()}`,
              type: "function",
              function: {
                name: item.name || "unknown",
                arguments: (item.arguments as string) || "{}",
              },
            },
          ],
        });
        continue;
      }

      if (itemType === "function_call_output") {
        messages.push({
          role: "tool",
          tool_call_id: item.call_id || `tool_${Date.now()}`,
          content: item.output || "",
        });
        continue;
      }

      // Fallback: message with role
      if (item.role && item.content !== undefined) {
        messages.push({ role: item.role, content: convertContentToChatFormat(item.content) });
        continue;
      }
    }
  }

  chatRequest.messages = messages;

  // --- Transform tools ---
  if (req.tools && Array.isArray(req.tools)) {
    chatRequest.tools = req.tools
      .filter((t) => t.type === "function")
      .map((t) => ({
        type: "function",
        function: {
          name: t.name || "unknown",
          description: t.description || "",
          parameters: t.parameters || {},
        },
      }));
  }

  // --- Handle reasoning ---
  const reasoningLevel = getModelReasoning(req.model);
  if (reasoningLevel === "auto") {
    if (req.reasoning?.effort) {
      chatRequest.reasoning_effort = req.reasoning.effort;
    }
  } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
    chatRequest.reasoning_effort = reasoningLevel;
  } else {
    delete chatRequest.reasoning_effort;
  }

  // --- Pass through parameters ---
  if (req.temperature !== undefined) chatRequest.temperature = req.temperature;
  if (req.top_p !== undefined) chatRequest.top_p = req.top_p;
  if (req.presence_penalty !== undefined) chatRequest.presence_penalty = req.presence_penalty;
  if (req.frequency_penalty !== undefined) chatRequest.frequency_penalty = req.frequency_penalty;

  return chatRequest;
}

// --- Helpers ---

function extractText(content: string | Array<Record<string, unknown>> | undefined): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((c) => c.type === "text" || c.type === "input_text" || c.type === "output_text")
      .map((c) => (c.text as string) || "")
      .join("\n");
  }
  return "";
}

function convertContentToChatFormat(
  content: string | Array<Record<string, unknown>> | undefined,
): string | Array<Record<string, unknown>> {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  const result: Array<Record<string, unknown>> = [];
  for (const part of content) {
    if (part.type === "input_text" || part.type === "output_text" || part.type === "text") {
      result.push({ type: "text", text: (part.text as string) || "" });
    } else if (part.type === "input_image") {
      result.push({
        type: "image_url",
        image_url: { url: (part.image_url as string) || "" },
      });
    } else {
      result.push(part);
    }
  }
  return result.length > 0 ? result : "";
}
