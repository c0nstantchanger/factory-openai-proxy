import fs from "fs";
import path from "path";

export interface ModelConfig {
  id: string;
  type: "openai" | "anthropic" | "common";
  reasoning: string;
  provider: string;
}

export interface Config {
  port: number;
  model_redirects: Record<string, string>;
  endpoints: {
    openai: string;
    anthropic: string;
    common: string;
  };
  models: ModelConfig[];
  system_prompt: string;
  user_agent: string;
}

let config: Config | null = null;

export function loadConfig(): Config {
  const configPath = path.join(import.meta.dir, "..", "config.json");
  const configData = fs.readFileSync(configPath, "utf-8");
  config = JSON.parse(configData) as Config;
  return config;
}

export function getConfig(): Config {
  if (!config) {
    loadConfig();
  }
  return config!;
}

export function getModelById(modelId: string): ModelConfig | undefined {
  return getConfig().models.find((m) => m.id === modelId);
}

export function getEndpointUrl(type: "openai" | "anthropic" | "common"): string {
  return getConfig().endpoints[type];
}

export function getPort(): number {
  return getConfig().port || 4011;
}

export function getSystemPrompt(): string {
  return getConfig().system_prompt || "";
}

export function getModelReasoning(modelId: string): string | null {
  const model = getModelById(modelId);
  if (!model || !model.reasoning) return null;
  const level = model.reasoning.toLowerCase();
  if (["low", "medium", "high", "xhigh", "auto", "off"].includes(level)) {
    return level;
  }
  return null;
}

export function getModelProvider(modelId: string): string {
  const model = getModelById(modelId);
  return model?.provider || "openai";
}

export function getUserAgent(): string {
  return getConfig().user_agent || "factory-cli/0.74.0";
}

export function getRedirectedModelId(modelId: string): string {
  const cfg = getConfig();
  if (cfg.model_redirects && cfg.model_redirects[modelId]) {
    const redirected = cfg.model_redirects[modelId];
    console.log(`[REDIRECT] Model redirected: ${modelId} -> ${redirected}`);
    return redirected;
  }
  return modelId;
}
