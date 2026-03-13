import { Router } from "express";
import type { Request, Response } from "express";
import { sanitizeText } from "./transformers/sanitize.js";
import {
  getConfig,
  getModelById,
  getEndpointUrl,
  getSystemPrompt,
  getModelReasoning,
  getRedirectedModelId,
  getModelProvider,
} from "./config.js";
import { getApiKey } from "./auth.js";
import { transformToAnthropic, getAnthropicHeaders } from "./transformers/request-anthropic.js";
import { transformToOpenAI, getOpenAIHeaders } from "./transformers/request-openai.js";
import { transformToCommon, getCommonHeaders } from "./transformers/request-common.js";
import { AnthropicResponseTransformer, convertAnthropicToChatCompletion } from "./transformers/response-anthropic.js";
import {
  OpenAIResponseTransformer,
  convertResponseToChatCompletion,
} from "./transformers/response-openai.js";

const router = Router();

// ─── GET /v1/models ──────────────────────────────────────────────────────────

router.get("/v1/models", (_req: Request, res: Response) => {
  try {
    const config = getConfig();
    const models = config.models.map((model) => ({
      id: model.id,
      object: "model" as const,
      created: Date.now(),
      owned_by: model.type,
      permission: [],
      root: model.id,
      parent: null,
    }));
    res.json({ object: "list", data: models });
  } catch (error) {
    console.error("[ROUTES] Error in GET /v1/models:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ─── POST /v1/chat/completions ───────────────────────────────────────────────

router.post("/v1/chat/completions", async (req: Request, res: Response) => {
  try {
    const openaiRequest = req.body;
    const modelId = getRedirectedModelId(openaiRequest.model);

    if (!modelId) {
      res.status(400).json({ error: "model is required" });
      return;
    }

    const model = getModelById(modelId);
    if (!model) {
      res.status(404).json({ error: `Model ${modelId} not found` });
      return;
    }

    const endpointUrl = getEndpointUrl(model.type);

    console.log(`[ROUTES] /v1/chat/completions -> ${model.type} endpoint: ${endpointUrl}`);

    // Get auth
    let authHeader: string;
    try {
      authHeader = await getApiKey(req.headers.authorization || null);
    } catch (error) {
      console.error("[ROUTES] Failed to get API key:", error);
      res.status(500).json({ error: "API key not available" });
      return;
    }

    const requestWithRedirectedModel = { ...openaiRequest, model: modelId };
    const provider = getModelProvider(modelId);
    const clientHeaders = req.headers;

    let transformedRequest: Record<string, unknown>;
    let headers: Record<string, string>;

    if (model.type === "anthropic") {
      transformedRequest = transformToAnthropic(requestWithRedirectedModel);
      const isStreaming = openaiRequest.stream === true;
      headers = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, modelId, provider);
    } else if (model.type === "openai") {
      transformedRequest = transformToOpenAI(requestWithRedirectedModel);
      headers = getOpenAIHeaders(authHeader, clientHeaders, provider);
    } else if (model.type === "common") {
      transformedRequest = transformToCommon(requestWithRedirectedModel);
      headers = getCommonHeaders(authHeader, clientHeaders, provider);
    } else {
      res.status(500).json({ error: `Unknown endpoint type: ${model.type}` });
      return;
    }

    const outBody = JSON.stringify(transformedRequest);
    const response = await fetch(endpointUrl, {
      method: "POST",
      headers,
      body: outBody,
    });

    console.log(`[ROUTES] Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[ROUTES] Endpoint error ${response.status}: ${errorText}`);
      res.status(response.status).json({
        error: `Endpoint returned ${response.status}`,
        details: errorText,
      });
      return;
    }

    const isStreaming = transformedRequest.stream === true;

    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      if (model.type === "common") {
        // Common type: direct passthrough
        const reader = response.body?.getReader();
        if (!reader) {
          res.end();
          return;
        }
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
          }
        } catch (streamError) {
          console.error("[ROUTES] Stream error:", streamError);
        }
        res.end();
      } else {
        // Anthropic and OpenAI types: transform stream
        const transformer =
          model.type === "anthropic"
            ? new AnthropicResponseTransformer(modelId, `chatcmpl-${Date.now()}`)
            : new OpenAIResponseTransformer(modelId, `chatcmpl-${Date.now()}`);

        try {
          const reader = response.body?.getReader();
          if (!reader) {
            res.end();
            return;
          }

          let sseBuffer = "";
          let currentEvent: string | null = null;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            sseBuffer += Buffer.from(value).toString();
            const lines = sseBuffer.split("\n");
            sseBuffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;

              if (line.startsWith("event:")) {
                currentEvent = line.slice(6).trim();
              } else if (line.startsWith("data:") && currentEvent) {
                const dataStr = line.slice(5).trim();
                let eventData: Record<string, unknown> = {};
                try {
                  eventData = JSON.parse(dataStr);
                } catch {
                  currentEvent = null;
                  continue;
                }
                const transformed = transformer.transformEvent(currentEvent, eventData);
                if (transformed) {
                  res.write(transformed);
                }
                currentEvent = null;
              }
            }
          }
        } catch (streamError) {
          console.error("[ROUTES] Stream error:", streamError);
        }
        res.end();
      }
    } else {
      // Non-streaming
      const data = (await response.json()) as Record<string, unknown>;
      if (model.type === "openai") {
        try {
          const converted = convertResponseToChatCompletion(data);
          res.json(converted);
        } catch {
          res.json(data);
        }
      } else if (model.type === "anthropic") {
        try {
          const converted = convertAnthropicToChatCompletion(data);
          res.json(converted);
        } catch {
          res.json(data);
        }
      } else {
        // common type: already in chat completion format
        res.json(data);
      }
    }
  } catch (error) {
    console.error("[ROUTES] Error in /v1/chat/completions:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

// ─── POST /v1/responses (direct passthrough to OpenAI responses API) ─────────

router.post("/v1/responses", async (req: Request, res: Response) => {
  try {
    const openaiRequest = req.body;
    const modelId = getRedirectedModelId(openaiRequest.model);

    if (!modelId) {
      res.status(400).json({ error: "model is required" });
      return;
    }

    const model = getModelById(modelId);
    if (!model) {
      res.status(404).json({ error: `Model ${modelId} not found` });
      return;
    }

    if (model.type !== "openai") {
      res.status(400).json({
        error: "Invalid endpoint type",
        message: `/v1/responses only supports openai-type models, ${modelId} is ${model.type}`,
      });
      return;
    }

    const endpointUrl = getEndpointUrl("openai");

    // Get auth
    let authHeader: string;
    try {
      const clientAuth = req.headers["x-api-key"]
        ? `Bearer ${req.headers["x-api-key"]}`
        : req.headers.authorization || null;
      authHeader = await getApiKey(clientAuth);
    } catch (error) {
      console.error("[ROUTES] Failed to get API key:", error);
      res.status(500).json({ error: "API key not available" });
      return;
    }

    const provider = getModelProvider(modelId);
    const headers = getOpenAIHeaders(authHeader, req.headers, provider);

    // Inject system prompt into instructions
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...openaiRequest, model: modelId };
    if (systemPrompt) {
      if (modifiedRequest.instructions) {
        modifiedRequest.instructions = systemPrompt + modifiedRequest.instructions;
      } else {
        modifiedRequest.instructions = systemPrompt;
      }
    }

    // Handle reasoning field
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === "auto") {
      // preserve original
    } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
      modifiedRequest.reasoning = { effort: reasoningLevel, summary: "auto" };
    } else {
      delete modifiedRequest.reasoning;
    }

    const response = await fetch(endpointUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(modifiedRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      res.status(response.status).json({
        error: `Endpoint returned ${response.status}`,
        details: errorText,
      });
      return;
    }

    const isStreaming = openaiRequest.stream === true;
    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const reader = response.body?.getReader();
      if (reader) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
          }
        } catch (streamError) {
          console.error("[ROUTES] Stream error:", streamError);
        }
      }
      res.end();
    } else {
      const data = await response.json();
      res.json(data);
    }
  } catch (error) {
    console.error("[ROUTES] Error in /v1/responses:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

// ─── POST /v1/messages (direct passthrough to Anthropic messages API) ────────

router.post("/v1/messages", async (req: Request, res: Response) => {
  try {
    const anthropicRequest = req.body;
    const modelId = getRedirectedModelId(anthropicRequest.model);

    if (!modelId) {
      res.status(400).json({ error: "model is required" });
      return;
    }

    const model = getModelById(modelId);
    if (!model) {
      res.status(404).json({ error: `Model ${modelId} not found` });
      return;
    }

    if (model.type !== "anthropic") {
      res.status(400).json({
        error: "Invalid endpoint type",
        message: `/v1/messages only supports anthropic-type models, ${modelId} is ${model.type}`,
      });
      return;
    }

    const endpointUrl = getEndpointUrl("anthropic");

    // Get auth
    let authHeader: string;
    try {
      const clientAuth = req.headers["x-api-key"]
        ? `Bearer ${req.headers["x-api-key"]}`
        : req.headers.authorization || null;
      authHeader = await getApiKey(clientAuth);
    } catch (error) {
      console.error("[ROUTES] Failed to get API key:", error);
      res.status(500).json({ error: "API key not available" });
      return;
    }

    const provider = getModelProvider(modelId);
    const isStreaming = anthropicRequest.stream === true;
    const headers = getAnthropicHeaders(authHeader, req.headers, isStreaming, modelId, provider);

    // Inject system prompt into first user message (not the `system` parameter,
    // which Factory.ai rejects with fk- API keys).
    // IMPORTANT: Factory.ai also rejects multiple text parts in a content array,
    // so everything must be merged into a single text block.
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...anthropicRequest, model: modelId };
    if (systemPrompt) {
      // Merge any existing system content with the config system prompt
      const existingSystem = modifiedRequest.system;
      const allParts: string[] = [systemPrompt];
      if (existingSystem) {
        if (typeof existingSystem === "string") {
          allParts.push(existingSystem);
        } else if (Array.isArray(existingSystem)) {
          for (const part of existingSystem) {
            if (typeof part === "string") allParts.push(part);
            else if (part?.text) allParts.push(part.text);
          }
        }
      }
      delete modifiedRequest.system;

      const systemText = sanitizeText(allParts.join("\n\n"));
      if (modifiedRequest.messages?.length > 0 && modifiedRequest.messages[0].role === "user") {
        const firstContent = modifiedRequest.messages[0].content;
        const userText = typeof firstContent === "string"
          ? firstContent
          : Array.isArray(firstContent)
            ? firstContent.map((p: { text?: string }) => p.text || "").filter(Boolean).join("\n\n")
            : "";
        modifiedRequest.messages[0].content = [{ type: "text", text: systemText + "\n\n" + userText }];
      } else {
        modifiedRequest.messages = [
          { role: "user", content: [{ type: "text", text: systemText }] },
          ...(modifiedRequest.messages || []),
        ];
      }
    } else {
      // Even without a config system prompt, strip any client-provided `system`
      // to avoid 403 from Factory.ai. Inline it instead.
      if (modifiedRequest.system) {
        const existingSystem = modifiedRequest.system;
        const parts: string[] = [];
        if (typeof existingSystem === "string") {
          parts.push(existingSystem);
        } else if (Array.isArray(existingSystem)) {
          for (const part of existingSystem) {
            if (typeof part === "string") parts.push(part);
            else if (part?.text) parts.push(part.text);
          }
        }
        delete modifiedRequest.system;

        if (parts.length > 0) {
          const systemText = sanitizeText(parts.join("\n\n"));
          if (modifiedRequest.messages?.length > 0 && modifiedRequest.messages[0].role === "user") {
            const firstContent = modifiedRequest.messages[0].content;
            const userText = typeof firstContent === "string"
              ? firstContent
              : Array.isArray(firstContent)
                ? firstContent.map((p: { text?: string }) => p.text || "").filter(Boolean).join("\n\n")
                : "";
            modifiedRequest.messages[0].content = [{ type: "text", text: systemText + "\n\n" + userText }];
          } else {
            modifiedRequest.messages = [
              { role: "user", content: [{ type: "text", text: systemText }] },
              ...(modifiedRequest.messages || []),
            ];
          }
        }
      }
    }

    // Handle thinking field
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === "auto") {
      // preserve original
    } else if (reasoningLevel && ["low", "medium", "high", "xhigh"].includes(reasoningLevel)) {
      const budgetTokens: Record<string, number> = {
        low: 4096,
        medium: 12288,
        high: 24576,
        xhigh: 40960,
      };
      modifiedRequest.thinking = {
        type: "enabled",
        budget_tokens: budgetTokens[reasoningLevel],
      };
    } else {
      delete modifiedRequest.thinking;
    }

    const outBodyMessages = JSON.stringify(modifiedRequest);
    const response = await fetch(endpointUrl, {
      method: "POST",
      headers,
      body: outBodyMessages,
    });

    if (!response.ok) {
      const errorText = await response.text();
      res.status(response.status).json({
        error: `Endpoint returned ${response.status}`,
        details: errorText,
      });
      return;
    }

    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const reader = response.body?.getReader();
      if (reader) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
          }
        } catch (streamError) {
          console.error("[ROUTES] Stream error:", streamError);
        }
      }
      res.end();
    } else {
      const data = await response.json();
      res.json(data);
    }
  } catch (error) {
    console.error("[ROUTES] Error in /v1/messages:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

// ─── POST /v1/messages/count_tokens ──────────────────────────────────────────

router.post("/v1/messages/count_tokens", async (req: Request, res: Response) => {
  try {
    const anthropicRequest = req.body;
    const modelId = getRedirectedModelId(anthropicRequest.model);

    if (!modelId) {
      res.status(400).json({ error: "model is required" });
      return;
    }

    const model = getModelById(modelId);
    if (!model) {
      res.status(404).json({ error: `Model ${modelId} not found` });
      return;
    }

    if (model.type !== "anthropic") {
      res.status(400).json({
        error: "Invalid endpoint type",
        message: `/v1/messages/count_tokens only supports anthropic-type models`,
      });
      return;
    }

    const endpointUrl = getEndpointUrl("anthropic").replace("/v1/messages", "/v1/messages/count_tokens");

    let authHeader: string;
    try {
      const clientAuth = req.headers["x-api-key"]
        ? `Bearer ${req.headers["x-api-key"]}`
        : req.headers.authorization || null;
      authHeader = await getApiKey(clientAuth);
    } catch (error) {
      console.error("[ROUTES] Failed to get API key:", error);
      res.status(500).json({ error: "API key not available" });
      return;
    }

    const provider = getModelProvider(modelId);
    const headers = getAnthropicHeaders(authHeader, req.headers, false, modelId, provider);

    const modifiedRequest = { ...anthropicRequest, model: modelId };

    const response = await fetch(endpointUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(modifiedRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      res.status(response.status).json({
        error: `Endpoint returned ${response.status}`,
        details: errorText,
      });
      return;
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error("[ROUTES] Error in /v1/messages/count_tokens:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

export default router;
