import { getSystemPrompt, getModelReasoning, getUserAgent } from "../config.js";
import type { IncomingHttpHeaders } from "http";

interface OpenAIChatRequest {
  model: string;
  messages?: Array<{
    role: string;
    content: string | Array<{ type: string; text?: string; image_url?: unknown }>;
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

      const content: Array<Record<string, unknown>> = [];

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

  anthropicRequest.messages = messages;

  // Add system parameter with system prompt prepended
  const systemPrompt = getSystemPrompt();
  if (systemPrompt || systemContent.length > 0) {
    const system: Array<Record<string, unknown>> = [];
    if (systemPrompt) {
      system.push({ type: "text", text: systemPrompt });
    }
    system.push(...systemContent);
    anthropicRequest.system = system;
  }

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

  const stainlessDefaults: Record<string, string> = {
    "x-stainless-arch": "x64",
    "x-stainless-lang": "js",
    "x-stainless-os": "MacOS",
    "x-stainless-runtime": "node",
    "x-stainless-retry-count": "0",
    "x-stainless-package-version": "0.57.0",
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
