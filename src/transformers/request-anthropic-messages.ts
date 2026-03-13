import { getModelReasoning, getSystemPrompt } from "../config.js";
import { sanitizeText } from "./sanitize.js";

const FACTORY_IDENTITY_SYSTEM = "You are Droid, an AI software engineering agent built by Factory.";

const BUDGET_TOKENS: Record<string, number> = {
  low: 4096,
  medium: 12288,
  high: 24576,
  xhigh: 40960,
};

export interface AnthropicMessagesRequest {
  model: string;
  system?: unknown;
  thinking?: unknown;
  messages?: unknown[];
  [key: string]: unknown;
}

function ensureFactoryIdentitySystem(systemParts: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  const hasIdentity = systemParts.some(
    (part) => part.type === "text" && part.text === FACTORY_IDENTITY_SYSTEM,
  );

  if (hasIdentity) {
    return systemParts;
  }

  return [{ type: "text", text: FACTORY_IDENTITY_SYSTEM }, ...systemParts];
}

function sanitizeSystemParts(systemParts: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  return systemParts.map((part) => {
    if (part.type === "text" && typeof part.text === "string") {
      return { ...part, text: sanitizeText(part.text) };
    }
    return part;
  });
}

export function buildAnthropicMessagesRequest(
  anthropicRequest: AnthropicMessagesRequest,
  modelId: string,
): Record<string, unknown> {
  const systemPrompt = getSystemPrompt();
  const modifiedRequest: Record<string, unknown> = { ...anthropicRequest, model: modelId };

  const existingSystem = modifiedRequest.system;
  const systemParts: Array<Record<string, unknown>> = [];

  if (systemPrompt) {
    systemParts.push({ type: "text", text: systemPrompt });
  }

  if (typeof existingSystem === "string") {
    systemParts.push({ type: "text", text: existingSystem });
  } else if (Array.isArray(existingSystem)) {
    for (const part of existingSystem) {
      if (typeof part === "string") {
        systemParts.push({ type: "text", text: part });
      } else if (part && typeof part === "object") {
        systemParts.push(part as Record<string, unknown>);
      }
    }
  }

  if (systemParts.length > 0) {
    modifiedRequest.system = sanitizeSystemParts(ensureFactoryIdentitySystem(systemParts));
  }

  const reasoningLevel = getModelReasoning(modelId);
  if (reasoningLevel === "auto") {
    return modifiedRequest;
  }

  if (reasoningLevel && reasoningLevel in BUDGET_TOKENS) {
    modifiedRequest.thinking = {
      type: "enabled",
      budget_tokens: BUDGET_TOKENS[reasoningLevel],
    };
    return modifiedRequest;
  }

  delete modifiedRequest.thinking;
  return modifiedRequest;
}
