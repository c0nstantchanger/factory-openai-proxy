/**
 * Sanitizes text content before sending to Factory.ai.
 * Factory.ai blocks requests containing certain keywords (e.g. competing product names).
 */

const REPLACEMENTS: Array<[RegExp, string]> = [
  // Factory.ai blocks requests mentioning "OpenCode" (a competing product)
  [/\bOpenCode\b/g, "Assistant"],
];

export function sanitizeText(text: string): string {
  let result = text;
  for (const [pattern, replacement] of REPLACEMENTS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}
