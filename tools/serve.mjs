// Static server for AgenticWeightInfra. No directory listings, no path escape,
// and headers that keep the page in this origin while still allowing WebLLM
// to fetch its package and the model weights over HTTPS.

import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const ROOT = fileURLToPath(new URL("..", import.meta.url));

export const CSP = [
  "default-src 'self'",
  "script-src 'self' https://esm.run https://esm.sh https://cdn.jsdelivr.net 'wasm-unsafe-eval'",
  "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:",
  "connect-src 'self' https:",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "frame-ancestors 'none'",
].join("; ");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webp": "image/webp",
  ".md": "text/plain; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
};

export function securityHeaders() {
  return {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": CSP,
    "Cache-Control": "no-cache",
  };
}

/** Absolute file path for a URL, or null when the URL leaves the site root. */
export function safePath(root, urlPath) {
  let pathname;
  try { pathname = decodeURIComponent(new URL(urlPath, "http://127.0.0.1").pathname); }
  catch { return null; }
  if (pathname.includes("\0")) return null;
  const rel = pathname.replace(/^\/+/, "");
  if (rel.split("/").some((part) => part === ".." || part.startsWith("."))) return null;
  const base = root.endsWith(sep) ? root.slice(0, -1) : root;
  const full = normalize(join(base, rel));
  if (full !== base && !full.startsWith(base + sep)) return null;
  return full;
}

function send(req, res) {
  const headers = securityHeaders();
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405, headers);
    res.end("Method not allowed");
    return;
  }
  const full = safePath(ROOT, req.url ?? "/");
  if (!full || !existsSync(full)) {
    res.writeHead(404, headers);
    res.end("Not found");
    return;
  }
  let file = full;
  const st = statSync(full);
  if (st.isDirectory()) {
    file = join(full, "index.html");
    if (!existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404, headers);
      res.end("Not found");
      return;
    }
  } else if (!st.isFile()) {
    res.writeHead(404, headers);
    res.end("Not found");
    return;
  }
  res.writeHead(200, { ...headers, "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
  if (req.method === "HEAD") { res.end(); return; }
  createReadStream(file).pipe(res);
}

export function startServer(port = 8080) {
  const server = createServer(send);
  server.listen(port);
  return server;
}

const invoked = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invoked) {
  const port = Number(process.env.PORT) || 8080;
  startServer(port);
  console.log(`AgenticWeightInfra http://127.0.0.1:${port}`);
}
