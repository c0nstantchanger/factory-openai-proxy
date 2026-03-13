# factory-openai-proxy

OpenAI-compatible API proxy for [Factory AI](https://factory.ai). Accepts standard OpenAI chat completions requests and routes them through Factory's API endpoints, handling format translation for Anthropic and OpenAI Responses API models transparently.

TypeScript rewrite of [droid2api](https://github.com/1e0n/droid2api), running on [Bun](https://bun.sh).

## Setup

```bash
bun install
```

### Docker

```bash
# Build and run with docker compose
FACTORY_API_KEY=your-key docker compose up -d

# Or build manually
docker build -t factory-openai-proxy .
docker run -d -p 4011:4011 -e FACTORY_API_KEY=your-key factory-openai-proxy
```

The `config.json` is bind-mounted by default in the compose setup, so you can edit it without rebuilding.

## Configuration

Edit `config.json` to configure:

- **port** — server port (default `4011`)
- **models** — available models with type (`openai`/`anthropic`/`common`), reasoning level, and provider
- **endpoints** — Factory upstream URLs
- **model_redirects** — alias old model IDs to current ones
- **system_prompt** — injected into all requests
- **user_agent** — Factory client user-agent string

## Authentication

Priority order (highest first):

1. `FACTORY_API_KEY` env var — used directly as Bearer token
2. `DROID_REFRESH_KEY` env var — WorkOS OAuth refresh token, auto-refreshes every 6h
3. `~/.factory/auth.json` — reads refresh_token from Factory CLI auth file
4. Client `Authorization` header — passed through as-is

```bash
# Simplest: set your Factory API key
export FACTORY_API_KEY="your-key-here"
bun run dev
```

## Usage

```bash
bun run dev    # start server
bun run start  # same thing
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | OpenAI chat completions (auto-routes by model type) |
| `POST` | `/v1/responses` | Direct passthrough to Factory's OpenAI Responses API |
| `POST` | `/v1/messages` | Direct passthrough to Factory's Anthropic Messages API |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting |

### With OpenCode

```jsonc
// opencode.json provider config
{
  "provider": {
    "factory": {
      "id": "custom",
      "name": "Factory",
      "api": "openai",
      "url": "http://localhost:4011/v1",
      "models": {
        "claude-opus-4-6": { "id": "claude-opus-4-6", "name": "Claude Opus 4.6" },
        "gpt-5.2": { "id": "gpt-5.2", "name": "GPT-5.2" }
      }
    }
  }
}
```

### With curl

```bash
curl http://localhost:4011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-6",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## How it works

The proxy translates between API formats based on model type:

- **`anthropic` models** (Claude) — transforms OpenAI chat format to Anthropic Messages API format, streams back as OpenAI chat chunks
- **`openai` models** (GPT) — transforms chat completions to OpenAI Responses API format (`messages[]` → `input[]` with `input_text`/`output_text`), streams back as OpenAI chat chunks
- **`common` models** (Gemini, GLM, etc.) — passes through as standard chat completions with minimal modification

### Reasoning control

Each model has a `reasoning` setting in `config.json`:

- `"auto"` — preserves whatever the client sends (reasoning/thinking fields passed through)
- `"low"` / `"medium"` / `"high"` / `"xhigh"` — overrides client with specific reasoning level
- `"off"` — strips reasoning/thinking fields from requests

### System prompt handling

> **Warning:** Factory.ai returns **403 Forbidden** when the Anthropic `system` parameter is present in requests using `fk-` API keys. This proxy works around this by inlining system content into the first user message instead of using the `system` parameter. The behavior is functionally equivalent, but be aware that system instructions will appear as part of the user message in the conversation rather than as a separate system block.
