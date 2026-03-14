import { getSystemPrompt, getModelReasoning, getUserAgent } from "../config.js";
import { sanitizeText } from "./sanitize.js";
import type { IncomingHttpHeaders } from "http";

const FACTORY_IDENTITY_SYSTEM = "You are Droid, an AI software engineering agent built by Factory.";

interface OpenAIChatRequest {
  model: string;
  messages?: Array<{
    role: string;
    content: string | Array<{ type: string; text?: string; image_url?: { url: string; detail?: string }; [k: string]: unknown }> | null;
    tool_calls?: Array<{
      id?: string;
      type?: string;
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
    tool_call_id?: string;
    name?: string;
  }>;
  stream?: boolean;
  max_tokens?: number;
  max_completion_tokens?: number;
  tools?: Array<{ type: string; function?: { name: string; description?: string; parameters?: unknown }; [k: string]: unknown }>;
  thinking?: unknown;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  [key: string]: unknown;
}

export function transformToAnthropic(openaiRequest: OpenAIChatRequest): Record<string, unknown> {
  const anthropicRequest: Record<string, unknown> = {
    model: openaiRequest.model,
    messages: [],
  };

  if (openaiRequest.stream !== undefined) {
    anthropicRequest.stream = openaiRequest.stream;
  }

  // Handle max_tokens
  if (openaiRequest.max_tokens) {
    anthropicRequest.max_tokens = openaiRequest.max_tokens;
  } else if (openaiRequest.max_completion_tokens) {
    anthropicRequest.max_tokens = openaiRequest.max_completion_tokens;
  } else {
    anthropicRequest.max_tokens = 4096;
  }

  // Extract system messages and transform other messages
  const systemContent: Array<Record<string, unknown>> = [];
  const rawMessages: Array<Record<string, unknown>> = [];

  if (openaiRequest.messages && Array.isArray(openaiRequest.messages)) {
    for (const msg of openaiRequest.messages) {
      if (msg.role === "system") {
        if (typeof msg.content === "string") {
          systemContent.push({ type: "text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text") {
              systemContent.push({ type: "text", text: part.text });
            } else {
              systemContent.push(part as Record<string, unknown>);
            }
          }
        }
        continue;
      }

      if (msg.role === "tool") {
        // Build tool_result content — preserve structure when possible
        const toolResultContent = buildToolResultContent(msg.content);
        rawMessages.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: msg.tool_call_id || msg.name || `tool_${Date.now()}`,
              content: toolResultContent,
            },
          ],
        });
        continue;
      }

      if (msg.role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        const content: Array<Record<string, unknown>> = [];

        // Only add text content if it's non-empty
        if (typeof msg.content === "string" && msg.content) {
          content.push({ type: "text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text" && part.text) {
              content.push({ type: "text", text: part.text });
            }
          }
        }

        for (const toolCall of msg.tool_calls) {
          if (toolCall.type === "function" && toolCall.function?.name) {
            content.push({
              type: "tool_use",
              id: toolCall.id || `toolu_${Date.now()}`,
              name: toolCall.function.name,
              input: parseToolArguments(toolCall.function.arguments),
            });
          }
        }

        rawMessages.push({ role: "assistant", content });
        continue;
      }

      // Regular user or assistant message
      const content: Array<Record<string, unknown>> = [];

      if (typeof msg.content === "string") {
        content.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            content.push({ type: "text", text: part.text });
          } else if (part.type === "image_url" && part.image_url) {
            content.push(transformImageUrl(part.image_url));
          } else {
            content.push(part as Record<string, unknown>);
          }
        }
      } else if (msg.content === null || msg.content === undefined) {
        // Assistant messages with null content (e.g., pure function-call messages
        // that were already handled above, or empty assistant turns).
        // Push an empty text block so the message isn't contentless.
        content.push({ type: "text", text: "" });
      }

      rawMessages.push({ role: msg.role, content });
    }
  }

  // Merge consecutive messages with the same role (Anthropic requires alternating roles)
  const messages = mergeConsecutiveSameRole(rawMessages);

  // Build system array from config system prompt + client system messages.
  const systemPrompt = getSystemPrompt();
  const allSystemParts: Array<Record<string, unknown>> = [];
  if (systemPrompt) {
    allSystemParts.push({ type: "text", text: systemPrompt });
  }
  allSystemParts.push(...systemContent);

  if (allSystemParts.length > 0) {
    const hasIdentity = allSystemParts.some(
      (part) => part.type === "text" && part.text === FACTORY_IDENTITY_SYSTEM,
    );
    if (!hasIdentity) {
      allSystemParts.unshift({ type: "text", text: FACTORY_IDENTITY_SYSTEM });
    }
    anthropicRequest.system = allSystemParts.map((part) => {
      if (part.type === "text" && typeof part.text === "string") {
        return { ...part, text: sanitizeText(part.text) };
      }
      return part;
    });
  }

  anthropicRequest.messages = messages;

  // Transform tools (OpenAI function format -> Anthropic format)
  if (openaiRequest.tools && Array.isArray(openaiRequest.tools)) {
    anthropicRequest.tools = openaiRequest.tools.map((tool) => {
      if (tool.type === "function" && tool.function) {
        return {
          name: tool.function.name,
          description: tool.function.description,
          input_schema: tool.function.parameters || {},
        };
      }
      return tool;
    });
  }

  // Handle thinking field based on model config
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === "auto") {
    if (openaiRequest.thinking !== undefined) {
      anthropicRequest.thinking = openaiRequest.thinking;
    }
  } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
    const budgetTokens: Record<string, number> = {
      low: 4096,
      medium: 12288,
      high: 24576,
      xhigh: 40960,
    };
    anthropicRequest.thinking = {
      type: "enabled",
      budget_tokens: budgetTokens[reasoningLevel],
    };
  } else {
    delete anthropicRequest.thinking;
  }

  // Pass through compatible parameters
  if (openaiRequest.temperature !== undefined) anthropicRequest.temperature = openaiRequest.temperature;
  if (openaiRequest.top_p !== undefined) anthropicRequest.top_p = openaiRequest.top_p;
  if (openaiRequest.stop !== undefined) {
    anthropicRequest.stop_sequences = Array.isArray(openaiRequest.stop) ? openaiRequest.stop : [openaiRequest.stop];
  }

  return anthropicRequest;
}

/**
 * Transform an OpenAI image_url content part to Anthropic image source format.
 * OpenAI: { url: "data:image/png;base64,..." } or { url: "https://..." }
 * Anthropic: { type: "base64", media_type: "image/png", data: "..." } or { type: "url", url: "..." }
 */
function transformImageUrl(imageUrl: { url: string; detail?: string }): Record<string, unknown> {
  const url = imageUrl.url;

  // Check for base64 data URI
  const dataUriMatch = url.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/s);
  if (dataUriMatch) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: dataUriMatch[1],
        data: dataUriMatch[2],
      },
    };
  }

  // Regular URL
  return {
    type: "image",
    source: {
      type: "url",
      url,
    },
  };
}

/**
 * Build tool_result content from an OpenAI tool message's content.
 * Returns a string for simple text, or an array of content blocks for multipart.
 */
function buildToolResultContent(
  content: string | Array<{ type: string; text?: string; image_url?: { url: string; detail?: string }; [k: string]: unknown }> | null,
): string | Array<Record<string, unknown>> {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    // If it's a single text block, just return the string
    if (content.length === 1 && content[0].type === "text") {
      return content[0].text || "";
    }
    // Multi-part: convert each block
    return content.map((part) => {
      if (part.type === "text") {
        return { type: "text", text: part.text || "" };
      }
      if (part.type === "image_url" && part.image_url) {
        return transformImageUrl(part.image_url);
      }
      return part as Record<string, unknown>;
    });
  }
  return "";
}

/**
 * Merge consecutive messages with the same role.
 * Anthropic requires strictly alternating user/assistant roles.
 * Consecutive user messages (e.g., multiple tool_result messages) get their
 * content arrays concatenated into a single message.
 */
function mergeConsecutiveSameRole(messages: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  if (messages.length === 0) return [];

  const merged: Array<Record<string, unknown>> = [];

  for (const msg of messages) {
    const lastMsg = merged.length > 0 ? merged[merged.length - 1] : null;

    if (lastMsg && lastMsg.role === msg.role) {
      // Merge content arrays
      const lastContent = lastMsg.content;
      const curContent = msg.content;

      if (Array.isArray(lastContent) && Array.isArray(curContent)) {
        (lastContent as Array<Record<string, unknown>>).push(...(curContent as Array<Record<string, unknown>>));
      } else if (Array.isArray(lastContent) && typeof curContent === "string") {
        (lastContent as Array<Record<string, unknown>>).push({ type: "text", text: curContent });
      } else if (typeof lastContent === "string" && Array.isArray(curContent)) {
        lastMsg.content = [
          { type: "text", text: lastContent },
          ...(curContent as Array<Record<string, unknown>>),
        ];
      } else if (typeof lastContent === "string" && typeof curContent === "string") {
        lastMsg.content = [
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

function parseToolArguments(argumentsText: string | undefined): Record<string, unknown> {
  if (!argumentsText) return {};
  try {
    const parsed = JSON.parse(argumentsText) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // fall through
  }
  return { input: argumentsText };
}

export function getAnthropicHeaders(
  authHeader: string,
  clientHeaders: IncomingHttpHeaders = {},
  isStreaming: boolean = true,
  modelId: string | null = null,
  provider: string = "anthropic"
): Record<string, string> {
  const sessionId = (clientHeaders["x-session-id"] as string) || generateUUID();
  const messageId = (clientHeaders["x-assistant-message-id"] as string) || generateUUID();

  const headers: Record<string, string> = {
    accept: "application/json",
    "content-type": "application/json",
    "anthropic-version": (clientHeaders["anthropic-version"] as string) || "2023-06-01",
    authorization: authHeader,
    "x-api-key": "placeholder",
    "x-api-provider": provider,
    "x-factory-client": "cli",
    "x-session-id": sessionId,
    "x-assistant-message-id": messageId,
    "user-agent": getUserAgent(),
    "x-stainless-timeout": "600",
    connection: "keep-alive",
  };

  // Handle anthropic-beta header based on reasoning config
  const reasoningLevel = modelId ? getModelReasoning(modelId) : null;
  let betaValues: string[] = [];

  if (clientHeaders["anthropic-beta"]) {
    betaValues = (clientHeaders["anthropic-beta"] as string).split(",").map((v) => v.trim());
  }

  const thinkingBeta = "interleaved-thinking-2025-05-14";
  if (reasoningLevel === "auto") {
    // preserve client's beta header
  } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
    if (!betaValues.includes(thinkingBeta)) {
      betaValues.push(thinkingBeta);
    }
  } else {
    betaValues = betaValues.filter((v) => v !== thinkingBeta);
  }

  if (betaValues.length > 0) {
    headers["anthropic-beta"] = betaValues.join(", ");
  }

  headers["x-client-version"] = (clientHeaders["x-client-version"] as string) || "0.74.0";

  const stainlessDefaults: Record<string, string> = {
    "x-stainless-arch": "x64",
    "x-stainless-lang": "js",
    "x-stainless-os": "Linux",
    "x-stainless-runtime": "node",
    "x-stainless-retry-count": "0",
    "x-stainless-package-version": "0.70.1",
    "x-stainless-runtime-version": "v24.3.0",
  };

  if (isStreaming) {
    headers["x-stainless-helper-method"] = "stream";
  }

  for (const [header, defaultVal] of Object.entries(stainlessDefaults)) {
    headers[header] = (clientHeaders[header] as string) || defaultVal;
  }

  if (clientHeaders["x-stainless-timeout"]) {
    headers["x-stainless-timeout"] = clientHeaders["x-stainless-timeout"] as string;
  }

  return headers;
}

function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
