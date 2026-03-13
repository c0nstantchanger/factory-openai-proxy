import { getSystemPrompt, getModelReasoning, getUserAgent } from "../config.js";
import { sanitizeText } from "./sanitize.js";
import type { IncomingHttpHeaders } from "http";

const FACTORY_IDENTITY_SYSTEM = "You are Droid, an AI software engineering agent built by Factory.";

interface OpenAIChatRequest {
  model: string;
  messages?: Array<{
    role: string;
    content: string | Array<{ type: string; text?: string; image_url?: unknown }> | null;
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
  const messages: Array<Record<string, unknown>> = [];

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
        messages.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: msg.tool_call_id || msg.name || `tool_${Date.now()}`,
              content: stringifyMessageContent(msg.content),
            },
          ],
        });
        continue;
      }

      const content: Array<Record<string, unknown>> = [];

      if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
        if (typeof msg.content === "string" && msg.content) {
          content.push({ type: "text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text") {
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

        messages.push({ role: "assistant", content });
        continue;
      }

      if (typeof msg.content === "string") {
        content.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            content.push({ type: "text", text: part.text });
          } else if (part.type === "image_url") {
            content.push({ type: "image", source: part.image_url });
          } else {
            content.push(part as Record<string, unknown>);
          }
        }
      }

      messages.push({ role: msg.role, content });
    }
  }

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

function stringifyMessageContent(
  content: string | Array<{ type: string; text?: string; image_url?: unknown }> | null,
): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part.type === "text") return part.text || "";
        return JSON.stringify(part);
      })
      .filter(Boolean)
      .join("\n\n");
  }
  return "";
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
