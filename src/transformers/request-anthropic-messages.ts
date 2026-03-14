import { getModelReasoning, getSystemPrompt } from "../config.js";
import { sanitizeText } from "./sanitize.js";
import { validateAnthropicMessages, validateAnthropicTools } from "./validate-anthropic.js";

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

  // Strip empty text blocks from messages — Anthropic rejects { type: "text", text: "" }
  // Also extract any "developer" role messages into system (OpenAI compat)
  if (Array.isArray(modifiedRequest.messages)) {
    const msgs = modifiedRequest.messages as Array<Record<string, unknown>>;
    const developerMsgs = msgs.filter((m) => m.role === "developer");
    if (developerMsgs.length > 0) {
      for (const devMsg of developerMsgs) {
        if (typeof devMsg.content === "string" && devMsg.content.length > 0) {
          systemParts.push({ type: "text", text: devMsg.content });
        } else if (Array.isArray(devMsg.content)) {
          for (const part of devMsg.content as Array<Record<string, unknown>>) {
            if (part.type === "text" && part.text && (part.text as string).length > 0) {
              systemParts.push({ type: "text", text: part.text as string });
            }
          }
        }
      }
      // Re-apply system with any new developer content
      if (systemParts.length > 0) {
        modifiedRequest.system = sanitizeSystemParts(ensureFactoryIdentitySystem(systemParts));
      }
    }
    // Remove developer messages and strip empty text blocks
    const filtered = msgs.filter((m) => m.role !== "developer");
    modifiedRequest.messages = validateAnthropicMessages(stripEmptyTextBlocks(filtered));
  }

  // Validate tool definitions if present
  if (Array.isArray(modifiedRequest.tools)) {
    modifiedRequest.tools = validateAnthropicTools(modifiedRequest.tools as Array<Record<string, unknown>>);
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

/**
 * Strip empty text content blocks from messages.
 * Anthropic rejects `{ type: "text", text: "" }`.
 */
function stripEmptyTextBlocks(
  messages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const cleaned: Array<Record<string, unknown>> = [];
  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      const filtered = (msg.content as Array<Record<string, unknown>>).filter(
        (block) => !(block.type === "text" && (!block.text || (block.text as string).length === 0)),
      );
      if (filtered.length > 0) {
        cleaned.push({ ...msg, content: filtered });
      }
    } else if (typeof msg.content === "string") {
      if (msg.content.length > 0) {
        cleaned.push(msg);
      }
    } else {
      cleaned.push(msg);
    }
  }
  return cleaned;
}
