import express from "express";
import { loadConfig, getPort } from "./config.js";
import { initializeAuth } from "./auth.js";
import router from "./routes.js";

const app = express();

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// CORS
app.use((_req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, anthropic-version");
  if (_req.method === "OPTIONS") {
    res.sendStatus(200);
    return;
  }
  next();
});

// Routes
app.use(router);

// Root info
app.get("/", (_req, res) => {
  res.json({
    name: "factory-openai-proxy",
    version: "1.0.0",
    description: "OpenAI-compatible API proxy for Factory AI",
    endpoints: [
      "GET  /v1/models",
      "POST /v1/chat/completions",
      "POST /v1/responses",
      "POST /v1/messages",
      "POST /v1/messages/count_tokens",
    ],
  });
});

// 404
app.use((req, res) => {
  res.status(404).json({
    error: "Not Found",
    message: `${req.method} ${req.path} does not exist`,
    availableEndpoints: [
      "GET  /v1/models",
      "POST /v1/chat/completions",
      "POST /v1/responses",
      "POST /v1/messages",
      "POST /v1/messages/count_tokens",
    ],
  });
});

// Startup
(async () => {
  try {
    loadConfig();
    console.log("[SERVER] Configuration loaded");

    await initializeAuth();

    const PORT = getPort();
    app.listen(PORT, () => {
      console.log(`[SERVER] Running on http://localhost:${PORT}`);
      console.log("[SERVER] Endpoints:");
      console.log("  GET  /v1/models");
      console.log("  POST /v1/chat/completions");
      console.log("  POST /v1/responses");
      console.log("  POST /v1/messages");
      console.log("  POST /v1/messages/count_tokens");
    });
  } catch (error) {
    console.error("[SERVER] Failed to start:", error);
    process.exit(1);
  }
})();
