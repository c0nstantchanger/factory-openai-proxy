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

      input.push({ role: msg.role, content });
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
  const systemMessage = openaiRequest.messages?.find((m) => m.role === "system");

  if (systemMessage) {
    let userInstructions = "";
    if (typeof systemMessage.content === "string") {
      userInstructions = systemMessage.content;
    } else if (Array.isArray(systemMessage.content)) {
      userInstructions = systemMessage.content
        .filter((p) => p.type === "text")
        .map((p) => p.text || "")
        .join("\n");
    }
    targetRequest.instructions = systemPrompt + userInstructions;
    (targetRequest.input as Array<Record<string, unknown>>) = (
      targetRequest.input as Array<Record<string, unknown>>
    ).filter((m) => m.role !== "system");
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
    "x-stainless-os": "MacOS",
    "x-stainless-runtime": "node",
    "x-stainless-retry-count": "0",
    "x-stainless-package-version": "5.23.2",
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
