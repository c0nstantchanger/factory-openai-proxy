/**
 * Validates and fixes Anthropic message content blocks before sending.
 *
 * Anthropic requirements:
 *   tool_use:   { type: "tool_use", id: string, name: string, input: object }
 *   tool_result: { type: "tool_result", tool_use_id: string, content: string | ContentBlock[] }
 *   text:        { type: "text", text: non-empty-string }
 *   tool defs:   { name: string, input_schema: object, description?: string }
 *
 * Strategy: fix what we can (coerce types, generate missing ids), warn on
 * structural issues, and strip blocks that are irrecoverably invalid.
 */

let toolUseCounter = 0;

function generateToolId(): string {
  return `toolu_proxy_${Date.now()}_${++toolUseCounter}`;
}

// ─── Content block validators ────────────────────────────────────────────────

/**
 * Validate and fix a tool_use content block.
 * Returns the fixed block, or null if it's irrecoverable.
 */
function validateToolUse(block: Record<string, unknown>): Record<string, unknown> | null {
  const fixed = { ...block };

  // id: must be a non-empty string
  if (!fixed.id || typeof fixed.id !== "string") {
    const generated = generateToolId();
    console.warn(`[VALIDATE] tool_use missing id, generated: ${generated}`);
    fixed.id = generated;
  }

  // name: must be a non-empty string
  if (!fixed.name || typeof fixed.name !== "string") {
    console.error(`[VALIDATE] tool_use block has no name, stripping:`, JSON.stringify(block));
    return null;
  }

  // input: must be an object (not null, not array, not string)
  if (fixed.input === null || fixed.input === undefined) {
    fixed.input = {};
  } else if (typeof fixed.input === "string") {
    // Try to parse as JSON
    try {
      const parsed = JSON.parse(fixed.input);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        fixed.input = parsed;
      } else {
        fixed.input = { value: fixed.input };
      }
    } catch {
      fixed.input = { value: fixed.input };
    }
    console.warn(`[VALIDATE] tool_use "${fixed.name}" input was string, coerced to object`);
  } else if (Array.isArray(fixed.input)) {
    fixed.input = { items: fixed.input };
    console.warn(`[VALIDATE] tool_use "${fixed.name}" input was array, wrapped in object`);
  } else if (typeof fixed.input !== "object") {
    fixed.input = { value: fixed.input };
    console.warn(`[VALIDATE] tool_use "${fixed.name}" input was ${typeof fixed.input}, wrapped in object`);
  }

  return fixed;
}

/**
 * Validate and fix a tool_result content block.
 * Returns the fixed block, or null if irrecoverable.
 */
function validateToolResult(block: Record<string, unknown>): Record<string, unknown> | null {
  const fixed = { ...block };

  // tool_use_id: must be a non-empty string
  if (!fixed.tool_use_id || typeof fixed.tool_use_id !== "string") {
    console.error(`[VALIDATE] tool_result missing tool_use_id:`, JSON.stringify(block));
    // We can't really fix this — Anthropic will reject if it doesn't match a tool_use.
    // Generate a placeholder so the request at least has valid structure.
    fixed.tool_use_id = `toolu_unknown_${Date.now()}`;
    console.warn(`[VALIDATE] Generated placeholder tool_use_id: ${fixed.tool_use_id}`);
  }

  // content: must be string or array of content blocks
  if (fixed.content === null || fixed.content === undefined) {
    fixed.content = "";
  } else if (typeof fixed.content !== "string" && !Array.isArray(fixed.content)) {
    fixed.content = String(fixed.content);
    console.warn(`[VALIDATE] tool_result content coerced to string`);
  }

  // If content is an array, validate inner blocks (strip empty text)
  if (Array.isArray(fixed.content)) {
    fixed.content = (fixed.content as Array<Record<string, unknown>>).filter((inner) => {
      if (inner.type === "text" && (!inner.text || (inner.text as string).length === 0)) {
        return false;
      }
      return true;
    });
    // If all content blocks were stripped, use empty string
    if ((fixed.content as Array<Record<string, unknown>>).length === 0) {
      fixed.content = "";
    }
  }

  return fixed;
}

// ─── Message-level validation ────────────────────────────────────────────────

/**
 * Validate all content blocks in a single message.
 * Fixes what it can, strips invalid blocks, warns on issues.
 */
function validateMessageContent(msg: Record<string, unknown>): Record<string, unknown> | null {
  if (!Array.isArray(msg.content)) {
    return msg;
  }

  const content = msg.content as Array<Record<string, unknown>>;
  const validated: Array<Record<string, unknown>> = [];

  for (const block of content) {
    if (block.type === "tool_use") {
      const fixed = validateToolUse(block);
      if (fixed) validated.push(fixed);
    } else if (block.type === "tool_result") {
      const fixed = validateToolResult(block);
      if (fixed) validated.push(fixed);
    } else if (block.type === "text") {
      // Already handled by stripEmptyTextBlocks, but double-check
      if (block.text && (block.text as string).length > 0) {
        validated.push(block);
      }
    } else {
      // image, etc. — pass through
      validated.push(block);
    }
  }

  if (validated.length === 0) {
    return null;
  }

  return { ...msg, content: validated };
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Validate all messages in the outgoing Anthropic request.
 * Fixes tool_use/tool_result blocks, strips invalid ones.
 * Call this as the final pass before sending.
 */
export function validateAnthropicMessages(
  messages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const result: Array<Record<string, unknown>> = [];

  for (const msg of messages) {
    const validated = validateMessageContent(msg);
    if (validated) {
      result.push(validated);
    }
  }

  return result;
}

/**
 * Validate tool definitions array.
 * Anthropic requires: name (string), input_schema (object).
 * Strips invalid tool defs and warns.
 */
export function validateAnthropicTools(
  tools: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const validated: Array<Record<string, unknown>> = [];

  for (const tool of tools) {
    // name: required non-empty string
    if (!tool.name || typeof tool.name !== "string") {
      console.error(`[VALIDATE] Tool definition missing name, stripping:`, JSON.stringify(tool));
      continue;
    }

    const fixed = { ...tool };

    // input_schema: must be an object
    if (!fixed.input_schema || typeof fixed.input_schema !== "object" || Array.isArray(fixed.input_schema)) {
      console.warn(`[VALIDATE] Tool "${fixed.name}" has invalid input_schema, defaulting to empty object`);
      fixed.input_schema = { type: "object", properties: {} };
    }

    // description: should be a string if present
    if (fixed.description !== undefined && typeof fixed.description !== "string") {
      fixed.description = String(fixed.description);
    }

    validated.push(fixed);
  }

  return validated;
}
