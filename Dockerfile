FROM oven/bun:1 AS base
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile 2>/dev/null || bun install

# Production image
FROM base
COPY --from=deps /app/node_modules ./node_modules
COPY package.json tsconfig.json config.json ./
COPY src ./src

ENV PORT=4011
EXPOSE 4011

CMD ["bun", "run", "src/server.ts"]
