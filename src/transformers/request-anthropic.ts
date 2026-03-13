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

  // Merge system content (from config + client system messages) into the first user message.
  // Factory.ai rejects the Anthropic `system` parameter with fk- API keys (returns 403),
  // so we prepend it as text in the first user message instead.
  const systemPrompt = getSystemPrompt();
  const allSystemParts: Array<Record<string, unknown>> = [];
  if (systemPrompt) {
    allSystemParts.push({ type: "text", text: systemPrompt });
  }
  allSystemParts.push(...systemContent);

  if (allSystemParts.length > 0) {
    const systemText = allSystemParts
      .map((p) => (p as { text?: string }).text || "")
      .filter(Boolean)
      .join("\n\n");

    if (messages.length > 0 && (messages[0] as { role: string }).role === "user") {
      // Prepend system text to existing first user message
      const firstMsg = messages[0] as { role: string; content: Array<Record<string, unknown>> };
      firstMsg.content = [{ type: "text", text: systemText }, ...firstMsg.content];
    } else {
      // Insert a new user message at the front with the system text
      messages.unshift({ role: "user", content: [{ type: "text", text: systemText }] });
    }
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
