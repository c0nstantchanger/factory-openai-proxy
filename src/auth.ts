import fs from "fs";
import path from "path";
import os from "os";

// State management
let currentApiKey: string | null = null;
let currentRefreshToken: string | null = null;
let lastRefreshTime: number | null = null;
let authSource: "factory_key" | "env" | "file" | "client" | null = null;
let authFilePath: string | null = null;
let factoryApiKey: string | null = null;

const WORKOS_CLIENT_ID = "client_01HNM792M5G5G1A2THWPXKFMXB";
const REFRESH_URL = "https://api.workos.com/user_management/authenticate";
const REFRESH_INTERVAL_HOURS = 6;

interface AuthConfig {
  type: "factory_key" | "refresh" | "client";
  value: string | null;
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
    authFilePath = path.join(process.cwd(), "auth.json");
    return { type: "refresh", value: envRefreshKey.trim() };
  }

  // 3. ~/.factory/auth.json
  const factoryAuthPath = path.join(os.homedir(), ".factory", "auth.json");
  try {
    if (fs.existsSync(factoryAuthPath)) {
      const authData = JSON.parse(fs.readFileSync(factoryAuthPath, "utf-8"));
      if (authData.refresh_token?.trim()) {
        console.log("[AUTH] Using refresh token from ~/.factory/auth.json");
        authSource = "file";
        authFilePath = factoryAuthPath;
        if (authData.access_token) {
          currentApiKey = authData.access_token.trim();
        }
        return { type: "refresh", value: authData.refresh_token.trim() };
      }
    }
  } catch (error) {
    console.error("[AUTH] Error reading ~/.factory/auth.json:", error);
  }

  // 4. No configured auth — use client authorization
  console.log("[AUTH] No auth configuration found, will use client authorization headers");
  authSource = "client";
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

  // Save tokens to file
  saveTokens(data.access_token, data.refresh_token);

  console.log("[AUTH] API key refreshed successfully");
  return data.access_token;
}

function saveTokens(accessToken: string, refreshToken: string): void {
  if (!authFilePath) return;

  try {
    let authData: Record<string, unknown> = {
      access_token: accessToken,
      refresh_token: refreshToken,
      last_updated: new Date().toISOString(),
    };

    const dir = path.dirname(authFilePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // If saving to ~/.factory/auth.json, preserve other fields
    if (authSource === "file" && fs.existsSync(authFilePath)) {
      try {
        const existingData = JSON.parse(fs.readFileSync(authFilePath, "utf-8"));
        authData = {
          ...existingData,
          access_token: accessToken,
          refresh_token: refreshToken,
          last_updated: authData.last_updated,
        };
      } catch {
        // will overwrite
      }
    }

    fs.writeFileSync(authFilePath, JSON.stringify(authData, null, 2), "utf-8");
  } catch (error) {
    console.error("[AUTH] Failed to save tokens:", error);
  }
}

function shouldRefresh(): boolean {
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
      await refreshApiKey();
      console.log("[AUTH] Initialized with refresh token mechanism");
    } else {
      console.log("[AUTH] Initialized for client authorization mode");
    }
  } catch (error) {
    console.error("[AUTH] Failed to initialize auth system:", error);
    // Don't throw — allow client auth fallback
  }
}

export async function getApiKey(clientAuthorization: string | null = null): Promise<string> {
  // Priority 1: FACTORY_API_KEY
  if (authSource === "factory_key" && factoryApiKey) {
    return `Bearer ${factoryApiKey}`;
  }

  // Priority 2: Refresh token mechanism
  if (authSource === "env" || authSource === "file") {
    if (shouldRefresh()) {
      console.log("[AUTH] API key needs refresh (6+ hours old)");
      await refreshApiKey();
    }
    if (!currentApiKey) {
      throw new Error("No API key available from refresh token mechanism.");
    }
    return `Bearer ${currentApiKey}`;
  }

  // Priority 3: Client authorization header
  if (clientAuthorization) {
    return clientAuthorization;
  }

  throw new Error(
    "No authorization available. Configure FACTORY_API_KEY, DROID_REFRESH_KEY, or provide client Authorization header."
  );
}
