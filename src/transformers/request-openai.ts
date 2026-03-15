import { getSystemPrompt, getModelReasoning, getUserAgent } from "../config.js";
import type { IncomingHttpHeaders } from "http";

interface OpenAIChatRequest {
  model: string;
  messages?: Array<{
    role: string;
    content: string | Array<{ type: string; text?: string; image_url?: unknown }>;
    tool_calls?: Array<{
      id?: string;
      type?: string;
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
    tool_call_id?: string;
  }>;
  stream?: boolean;
  max_tokens?: number;
  max_completion_tokens?: number;
  tools?: Array<Record<string, unknown>>;
  reasoning?: unknown;
  temperature?: number;
  top_p?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  parallel_tool_calls?: boolean;
  [key: string]: unknown;
}

export function transformToOpenAI(openaiRequest: OpenAIChatRequest): Record<string, unknown> {
  const targetRequest: Record<string, unknown> = {
    model: openaiRequest.model,
    input: [],
    store: false,
  };

  if (openaiRequest.stream !== undefined) {
    targetRequest.stream = openaiRequest.stream;
  }

  // Transform max_tokens -> max_output_tokens
  if (openaiRequest.max_tokens) {
    targetRequest.max_output_tokens = openaiRequest.max_tokens;
  } else if (openaiRequest.max_completion_tokens) {
    targetRequest.max_output_tokens = openaiRequest.max_completion_tokens;
  }

  // Transform messages -> input
  const input: Array<Record<string, unknown>> = [];
  if (openaiRequest.messages && Array.isArray(openaiRequest.messages)) {
    for (const msg of openaiRequest.messages) {
      // Skip system messages — they're extracted as instructions below
      if (msg.role === "system" || msg.role === "developer") {
        continue;
      }

      // Assistant message with tool_calls -> output_text (if any) + function_call items
      if (msg.role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        // Emit text content first if present
        if (typeof msg.content === "string" && msg.content) {
          input.push({ type: "output_text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text" && part.text) {
              input.push({ type: "output_text", text: part.text });
            }
          }
        }
        // Emit each tool_call as a function_call item
        for (const toolCall of msg.tool_calls) {
          if (toolCall.type === "function" && toolCall.function?.name) {
            input.push({
              type: "function_call",
              id: toolCall.id || `call_${Date.now()}`,
              call_id: toolCall.id || `call_${Date.now()}`,
              name: toolCall.function.name,
              arguments: toolCall.function.arguments || "{}",
              status: "completed",
            });
          }
        }
        continue;
      }

      // Tool role message -> function_call_output item
      if (msg.role === "tool") {
        const outputText = typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? msg.content.filter((p) => p.type === "text").map((p) => p.text || "").join("")
            : "";
        input.push({
          type: "function_call_output",
          call_id: msg.tool_call_id || `call_${Date.now()}`,
          output: outputText,
        });
        continue;
      }

      // Regular user or assistant message
      const textType = msg.role === "assistant" ? "output_text" : "input_text";
      const imageType = msg.role === "assistant" ? "output_image" : "input_image";

      const content: Array<Record<string, unknown>> = [];

      if (typeof msg.content === "string") {
        content.push({ type: textType, text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "text") {
            content.push({ type: textType, text: part.text });
          } else if (part.type === "image_url") {
            content.push({ type: imageType, image_url: part.image_url });
          } else {
            content.push(part as Record<string, unknown>);
          }
        }
      }

      if (content.length > 0) {
        input.push({ role: msg.role, content });
      }
    }
  }

  targetRequest.input = input;

  // Transform tools
  if (openaiRequest.tools && Array.isArray(openaiRequest.tools)) {
    targetRequest.tools = openaiRequest.tools.map((tool) => ({
      ...tool,
      strict: false,
    }));
  }

  // Extract system message as instructions, prepend system prompt
  const systemPrompt = getSystemPrompt();
  const systemMessages = openaiRequest.messages?.filter((m) => m.role === "system" || m.role === "developer") || [];

  if (systemMessages.length > 0) {
    const userInstructions = systemMessages
      .map((m) => {
        if (typeof m.content === "string") return m.content;
        if (Array.isArray(m.content)) {
          return m.content
            .filter((p) => p.type === "text")
            .map((p) => p.text || "")
            .join("\n");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
    targetRequest.instructions = systemPrompt
      ? systemPrompt + "\n" + userInstructions
      : userInstructions;
  } else if (systemPrompt) {
    targetRequest.instructions = systemPrompt;
  }

  // Handle reasoning
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === "auto") {
    if (openaiRequest.reasoning !== undefined) {
      targetRequest.reasoning = openaiRequest.reasoning;
    }
  } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
    targetRequest.reasoning = { effort: reasoningLevel, summary: "auto" };
  } else {
    delete targetRequest.reasoning;
  }

  // Pass through parameters
  if (openaiRequest.temperature !== undefined) targetRequest.temperature = openaiRequest.temperature;
  if (openaiRequest.top_p !== undefined) targetRequest.top_p = openaiRequest.top_p;
  if (openaiRequest.presence_penalty !== undefined) targetRequest.presence_penalty = openaiRequest.presence_penalty;
  if (openaiRequest.frequency_penalty !== undefined)
    targetRequest.frequency_penalty = openaiRequest.frequency_penalty;
  if (openaiRequest.parallel_tool_calls !== undefined)
    targetRequest.parallel_tool_calls = openaiRequest.parallel_tool_calls;

  return targetRequest;
}

export function getOpenAIHeaders(
  authHeader: string,
  clientHeaders: IncomingHttpHeaders = {},
  provider: string = "openai"
): Record<string, string> {
  const sessionId = (clientHeaders["x-session-id"] as string) || generateUUID();
  const messageId = (clientHeaders["x-assistant-message-id"] as string) || generateUUID();

  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: authHeader,
    "x-api-provider": provider,
    "x-factory-client": "cli",
    "x-session-id": sessionId,
    "x-assistant-message-id": messageId,
    "user-agent": getUserAgent(),
    connection: "keep-alive",
  };

  const stainlessDefaults: Record<string, string> = {
    "x-stainless-arch": "x64",
    "x-stainless-lang": "js",
    "x-stainless-os": "Linux",
    "x-stainless-runtime": "node",
    "x-stainless-retry-count": "0",
    "x-stainless-package-version": "0.70.1",
    "x-stainless-runtime-version": "v24.3.0",
  };

  for (const [header, defaultVal] of Object.entries(stainlessDefaults)) {
    headers[header] = (clientHeaders[header] as string) || defaultVal;
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
