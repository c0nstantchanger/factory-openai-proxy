import fs from "fs";
import path from "path";
import os from "os";
import { createDecipheriv, createCipheriv, randomBytes } from "crypto";

// State management
let currentApiKey: string | null = null;
let currentRefreshToken: string | null = null;
let lastRefreshTime: number | null = null;
let authSource: "factory_key" | "env" | "file_v2" | "file" | null = null;
let factoryApiKey: string | null = null;

const WORKOS_CLIENT_ID = "client_01HNM792M5G5G1A2THWPXKFMXB";
const REFRESH_URL = "https://api.workos.com/user_management/authenticate";
const REFRESH_INTERVAL_HOURS = 6;

const AUTH_V2_FILE = path.join(os.homedir(), ".factory", "auth.v2.file");
const AUTH_V2_KEY = path.join(os.homedir(), ".factory", "auth.v2.key");

interface AuthConfig {
  type: "factory_key" | "refresh" | "client";
  value: string | null;
}

// AES-256-GCM decrypt: format is base64(iv):base64(authTag):base64(ciphertext)
function decryptAuthV2(keyB64: string, encryptedB64: string): { access_token: string; refresh_token: string } {
  const key = Buffer.from(keyB64.trim(), "base64");
  const parts = encryptedB64.trim().split(":");
  if (parts.length !== 3) throw new Error("Invalid auth.v2.file format");
  const iv = Buffer.from(parts[0], "base64");
  const authTag = Buffer.from(parts[1], "base64");
  const ciphertext = Buffer.from(parts[2], "base64");

  const decipher = createDecipheriv("aes-256-gcm", key, iv);
  decipher.setAuthTag(authTag);
  const decrypted = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return JSON.parse(decrypted.toString("utf-8"));
}

// AES-256-GCM encrypt: returns base64(iv):base64(authTag):base64(ciphertext)
function encryptAuthV2(keyB64: string, data: { access_token: string; refresh_token: string }): string {
  const key = Buffer.from(keyB64.trim(), "base64");
  const iv = randomBytes(16);
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  const ciphertext = Buffer.concat([cipher.update(JSON.stringify(data), "utf-8"), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return `${iv.toString("base64")}:${authTag.toString("base64")}:${ciphertext.toString("base64")}`;
}

function loadAuthConfig(): AuthConfig {
  // 1. FACTORY_API_KEY env var (highest priority)
  const factoryKey = process.env.FACTORY_API_KEY;
  if (factoryKey?.trim()) {
    console.log("[AUTH] Using fixed API key from FACTORY_API_KEY environment variable");
    factoryApiKey = factoryKey.trim();
    authSource = "factory_key";
    return { type: "factory_key", value: factoryKey.trim() };
  }

  // 2. DROID_REFRESH_KEY env var
  const envRefreshKey = process.env.DROID_REFRESH_KEY;
  if (envRefreshKey?.trim()) {
    console.log("[AUTH] Using refresh token from DROID_REFRESH_KEY environment variable");
    authSource = "env";
    return { type: "refresh", value: envRefreshKey.trim() };
  }

  // 3. ~/.factory/auth.v2.file + ~/.factory/auth.v2.key (droid native format)
  try {
    if (fs.existsSync(AUTH_V2_FILE) && fs.existsSync(AUTH_V2_KEY)) {
      const keyB64 = fs.readFileSync(AUTH_V2_KEY, "utf-8").trim();
      const encryptedData = fs.readFileSync(AUTH_V2_FILE, "utf-8").trim();
      const authData = decryptAuthV2(keyB64, encryptedData);
      if (authData.refresh_token?.trim()) {
        console.log("[AUTH] Using refresh token from ~/.factory/auth.v2.file");
        authSource = "file_v2";
        if (authData.access_token) {
          currentApiKey = authData.access_token.trim();
        }
        return { type: "refresh", value: authData.refresh_token.trim() };
      }
    }
  } catch (error) {
    console.error("[AUTH] Error reading ~/.factory/auth.v2.file:", error);
  }

  // 4. ~/.factory/auth.json (legacy)
  const factoryAuthPath = path.join(os.homedir(), ".factory", "auth.json");
  try {
    if (fs.existsSync(factoryAuthPath)) {
      const authData = JSON.parse(fs.readFileSync(factoryAuthPath, "utf-8"));
      if (authData.refresh_token?.trim()) {
        console.log("[AUTH] Using refresh token from ~/.factory/auth.json");
        authSource = "file";
        if (authData.access_token) {
          currentApiKey = authData.access_token.trim();
        }
        return { type: "refresh", value: authData.refresh_token.trim() };
      }
    }
  } catch (error) {
    console.error("[AUTH] Error reading ~/.factory/auth.json:", error);
  }

  // 5. No configured auth — fail
  console.error("[AUTH] No auth configuration found. The proxy will reject all requests until auth is configured.");
  authSource = null;
  return { type: "client", value: null };
}

async function refreshApiKey(): Promise<string> {
  if (!currentRefreshToken) {
    throw new Error("No refresh token available");
  }

  console.log("[AUTH] Refreshing API key...");

  const formData = new URLSearchParams();
  formData.append("grant_type", "refresh_token");
  formData.append("refresh_token", currentRefreshToken);
  formData.append("client_id", WORKOS_CLIENT_ID);

  const response = await fetch(REFRESH_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: formData.toString(),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to refresh token: ${response.status} ${errorText}`);
  }

  const data = (await response.json()) as {
    access_token: string;
    refresh_token: string;
    user?: { email: string; first_name: string; last_name: string; id: string };
    organization_id?: string;
  };

  currentApiKey = data.access_token;
  currentRefreshToken = data.refresh_token;
  lastRefreshTime = Date.now();

  if (data.user) {
    console.log(`[AUTH] Authenticated as: ${data.user.email} (${data.user.first_name} ${data.user.last_name})`);
  }

  saveTokens(data.access_token, data.refresh_token);

  console.log("[AUTH] API key refreshed successfully");
  return data.access_token;
}

function saveTokens(accessToken: string, refreshToken: string): void {
  try {
    if (authSource === "file_v2") {
      // Save back in encrypted v2 format
      const keyB64 = fs.readFileSync(AUTH_V2_KEY, "utf-8").trim();
      const encrypted = encryptAuthV2(keyB64, { access_token: accessToken, refresh_token: refreshToken });
      fs.writeFileSync(AUTH_V2_FILE, encrypted, "utf-8");
    } else if (authSource === "file") {
      const factoryAuthPath = path.join(os.homedir(), ".factory", "auth.json");
      let authData: Record<string, unknown> = {
        access_token: accessToken,
        refresh_token: refreshToken,
        last_updated: new Date().toISOString(),
      };
      if (fs.existsSync(factoryAuthPath)) {
        try {
          const existingData = JSON.parse(fs.readFileSync(factoryAuthPath, "utf-8"));
          authData = { ...existingData, ...authData };
        } catch {
          // will overwrite
        }
      }
      fs.writeFileSync(factoryAuthPath, JSON.stringify(authData, null, 2), "utf-8");
    }
  } catch (error) {
    console.error("[AUTH] Failed to save tokens:", error);
  }
}

// Return the expiry time (ms since epoch) of the current JWT, or null if not a JWT
function getJwtExpiry(token: string | null): number | null {
  if (!token) return null;
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const buf = Buffer.from(parts[1], "base64url");
    const str = buf.toString("utf-8");
    const payload = JSON.parse(str);
    if (payload.exp) return payload.exp * 1000;
  } catch {
    return null;
  }
  return null;
}

function shouldRefresh(): boolean {
  // If we have a JWT access token, refresh only when it's within 30 minutes of expiry
  if (currentApiKey) {
    const expiry = getJwtExpiry(currentApiKey);
    if (expiry !== null) {
      const msUntilExpiry = expiry - Date.now();
      return msUntilExpiry < 30 * 60 * 1000; // refresh if < 30 min left
    }
  }
  // Fallback: time-based refresh every REFRESH_INTERVAL_HOURS
  if (!lastRefreshTime) return true;
  const hours = (Date.now() - lastRefreshTime) / (1000 * 60 * 60);
  return hours >= REFRESH_INTERVAL_HOURS;
}

export async function initializeAuth(): Promise<void> {
  try {
    const authConfig = loadAuthConfig();

    if (authConfig.type === "factory_key") {
      console.log("[AUTH] Initialized with fixed API key");
    } else if (authConfig.type === "refresh") {
      currentRefreshToken = authConfig.value;
      // If we already have a valid (non-expiring-soon) access token loaded from
      // the file, use it as-is.  Only do an immediate refresh when no token is
      // available or it's about to expire.
      const needsRefresh = shouldRefresh();
      console.log(`[AUTH] currentApiKey set: ${!!currentApiKey}, needsRefresh: ${needsRefresh}`);
      if (needsRefresh) {
        await refreshApiKey();
      } else {
        const expiry = getJwtExpiry(currentApiKey);
        const minutesLeft = expiry ? Math.round((expiry - Date.now()) / 60000) : null;
        console.log(`[AUTH] Using existing access token (expires in ~${minutesLeft}m)`);
      }
      console.log("[AUTH] Initialized with refresh token mechanism");
    } else {
      console.error("[AUTH] WARNING: No server-side auth configured. All requests will return 500.");
    }
  } catch (error) {
    console.error("[AUTH] Failed to initialize auth system:", error);
    // Don't throw — allow client auth fallback
  }
}

export async function getApiKey(): Promise<string> {
  // Priority 1: FACTORY_API_KEY
  if (authSource === "factory_key" && factoryApiKey) {
    return `Bearer ${factoryApiKey}`;
  }

  // Priority 2: Refresh token mechanism (env, file_v2, or file)
  if (authSource === "env" || authSource === "file_v2" || authSource === "file") {
    if (shouldRefresh()) {
      console.log("[AUTH] API key needs refresh (6+ hours old)");
      await refreshApiKey();
    }
    if (!currentApiKey) {
      throw new Error("No API key available from refresh token mechanism.");
    }
    return `Bearer ${currentApiKey}`;
  }

  // No server-side auth configured — fail immediately
  throw new Error(
    "No server-side authorization configured. Set FACTORY_API_KEY, DROID_REFRESH_KEY, or provide ~/.factory/auth.v2.file."
  );
}
