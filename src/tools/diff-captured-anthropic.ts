import fs from "fs";
import path from "path";
import Ajv2020 from "ajv/dist/2020.js";
import { diff } from "just-diff";
import { buildAnthropicMessagesRequest, type AnthropicMessagesRequest } from "../transformers/request-anthropic-messages.js";
import { getRedirectedModelId, loadConfig } from "../config.js";

interface DiffEntry {
  op: string;
  path: Array<string | number>;
  value?: unknown;
}

function getRepoRoot(): string {
  return path.join(import.meta.dir, "..", "..");
}

function extractRequestBody(capturePath: string): Record<string, unknown> {
  const raw = fs.readFileSync(capturePath, "utf-8");
  const match = raw.match(/\r?\n\r?\n(\{.*\})\r?\n\r?\nHTTP\/1\.1 200 OK/s);
  if (!match?.[1]) {
    throw new Error(`Could not locate request JSON body in ${capturePath}`);
  }
  const body = match[1];
  return JSON.parse(body) as Record<string, unknown>;
}

function loadSchema(schemaPath: string): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(schemaPath, "utf-8")) as Record<string, unknown>;
}

function formatPath(parts: Array<string | number>): string {
  if (parts.length === 0) return "<root>";
  return parts
    .map((part) => (typeof part === "number" ? `[${part}]` : /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(part) ? `.${part}` : `[${JSON.stringify(part)}]`))
    .join("")
    .replace(/^\./, "");
}

function summarizeDiff(entries: DiffEntry[]): string[] {
  return entries.slice(0, 50).map((entry) => {
    const pathLabel = formatPath(entry.path);
    if (entry.op === "remove") {
      return `- remove ${pathLabel}`;
    }
    return `- ${entry.op} ${pathLabel}: ${JSON.stringify(entry.value)}`;
  });
}

function main(): void {
  loadConfig();

  const repoRoot = getRepoRoot();
  const schemaPath = path.join(repoRoot, "newnewmessages.schema.json");
  const capturePath = process.argv[2] || "/root/newnewmessages";

  const schema = loadSchema(schemaPath);
  const capturedBody = extractRequestBody(capturePath);
  const modelId = getRedirectedModelId(String(capturedBody.model || ""));
  const generatedBody = buildAnthropicMessagesRequest(capturedBody as AnthropicMessagesRequest, modelId);

  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate = ajv.compile(schema);

  const capturedValid = validate(capturedBody);
  const capturedErrors = validate.errors ? [...validate.errors] : [];
  const generatedValid = validate(generatedBody);
  const generatedErrors = validate.errors ? [...validate.errors] : [];

  const differences = diff(capturedBody, generatedBody) as DiffEntry[];

  console.log(`Schema: ${schemaPath}`);
  console.log(`Capture: ${capturePath}`);
  console.log(`Captured body valid: ${capturedValid ? "yes" : "no"}`);
  if (!capturedValid && capturedErrors.length > 0) {
    console.log("Captured validation errors:");
    for (const error of capturedErrors) {
      console.log(`- ${error.instancePath || "<root>"}: ${error.message}`);
    }
  }

  console.log(`Generated body valid: ${generatedValid ? "yes" : "no"}`);
  if (!generatedValid && generatedErrors.length > 0) {
    console.log("Generated validation errors:");
    for (const error of generatedErrors) {
      console.log(`- ${error.instancePath || "<root>"}: ${error.message}`);
    }
  }

  console.log(`Diff entries: ${differences.length}`);
  if (differences.length === 0) {
    console.log("Bodies match exactly.");
    return;
  }

  console.log("First differences:");
  for (const line of summarizeDiff(differences)) {
    console.log(line);
  }
}

main();
