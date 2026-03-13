import { getSystemPrompt, getUserAgent, getModelReasoning } from "../config.js";
import { sanitizeText } from "./sanitize.js";
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

  if (commonRequest.messages && Array.isArray(commonRequest.messages)) {
    const messages = [...(commonRequest.messages as Array<{ role: string; content: string | unknown[] }>)];

    // Extract all system role messages and strip them out.
    // Factory.ai rejects `system` role messages on the common (OpenAI) endpoint with fk- keys,
    // so we collect all system content and inline it into the first user message.
    const systemParts: string[] = [];
    if (systemPrompt) systemParts.push(systemPrompt);

    const nonSystemMessages = messages.filter((m) => {
      if (m.role === "system") {
        if (typeof m.content === "string") systemParts.push(m.content);
        else if (Array.isArray(m.content)) {
          for (const part of m.content as Array<{ type: string; text?: string }>) {
            if (part.text) systemParts.push(part.text);
          }
        }
        return false;
      }
      return true;
    });

    if (systemParts.length > 0) {
      const systemText = sanitizeText(systemParts.join("\n\n"));
      const firstUserIdx = nonSystemMessages.findIndex((m) => m.role === "user");
      if (firstUserIdx >= 0) {
        const firstUser = nonSystemMessages[firstUserIdx];
        const existingText = typeof firstUser.content === "string"
          ? firstUser.content
          : Array.isArray(firstUser.content)
            ? (firstUser.content as Array<{ text?: string }>).map((p) => p.text || "").filter(Boolean).join("\n\n")
            : "";
        nonSystemMessages[firstUserIdx] = {
          ...firstUser,
          content: systemText + "\n\n" + existingText,
        };
      } else {
        nonSystemMessages.unshift({ role: "user", content: systemText });
      }
    }

    commonRequest.messages = nonSystemMessages;
  } else if (systemPrompt) {
    commonRequest.messages = [{ role: "user", content: systemPrompt }, ...((commonRequest.messages as unknown[]) || [])];
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
