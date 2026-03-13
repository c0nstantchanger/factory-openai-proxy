import { getSystemPrompt, getUserAgent, getModelReasoning } from "../config.js";
import type { IncomingHttpHeaders } from "http";

interface OpenAIChatRequest {
  model: string;
  messages?: Array<{ role: string; content: string | unknown[] }>;
  reasoning_effort?: string;
  [key: string]: unknown;
}

export function transformToCommon(openaiRequest: OpenAIChatRequest): Record<string, unknown> {
  const commonRequest: Record<string, unknown> = { ...openaiRequest };

  const systemPrompt = getSystemPrompt();

  if (systemPrompt && commonRequest.messages && Array.isArray(commonRequest.messages)) {
    const messages = [...(commonRequest.messages as Array<{ role: string; content: string | unknown[] }>)];
    const systemIdx = messages.findIndex((m) => m.role === "system");

    if (systemIdx >= 0) {
      // Prepend system prompt to existing system message
      const existing = messages[systemIdx];
      messages[systemIdx] = {
        role: "system",
        content: systemPrompt + (typeof existing.content === "string" ? existing.content : ""),
      };
    } else {
      // Insert system message at front
      messages.unshift({ role: "system", content: systemPrompt });
    }

    commonRequest.messages = messages;
  } else if (systemPrompt) {
    commonRequest.messages = [{ role: "system", content: systemPrompt }, ...((commonRequest.messages as unknown[]) || [])];
  }

  // Handle reasoning_effort based on model config
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === "auto") {
    // preserve original
  } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
    commonRequest.reasoning_effort = reasoningLevel;
  } else {
    delete commonRequest.reasoning_effort;
  }

  return commonRequest;
}

export function getCommonHeaders(
  authHeader: string,
  clientHeaders: IncomingHttpHeaders = {},
  provider: string = "baseten"
): Record<string, string> {
  const sessionId = (clientHeaders["x-session-id"] as string) || generateUUID();
  const messageId = (clientHeaders["x-assistant-message-id"] as string) || generateUUID();

  const headers: Record<string, string> = {
    accept: "application/json",
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
