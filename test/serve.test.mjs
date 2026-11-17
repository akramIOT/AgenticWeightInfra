import assert from "node:assert/strict";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import test from "node:test";
import { BUILD } from "../src/build.js";
import { ROOT, safePath, securityHeaders, startServer } from "../tools/serve.mjs";

test("the published build matches package.json", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.equal(pkg.version, BUILD);
});

test("paths that leave the site root are refused", () => {
  assert.equal(safePath(ROOT, "/"), ROOT.replace(/\/$/, ""));
  assert.equal(safePath(ROOT, "/index.html")?.endsWith("/index.html"), true);
  assert.equal(safePath(ROOT, "/src/app.js")?.endsWith("/src/app.js"), true);
  assert.equal(safePath(ROOT, "/..%2F..%2Fetc%2Fpasswd"), null);
  assert.equal(safePath(ROOT, "/.git/config"), null);
  assert.equal(safePath(ROOT, "/%00index.html"), null);
});

test("the server sets security headers and does not list directories", async () => {
  const server = startServer(0);
  await once(server, "listening");
  const { port } = server.address();
  try {
    const home = await fetch(`http://127.0.0.1:${port}/`);
    assert.equal(home.status, 200);
    assert.match(await home.text(), /AgenticWeightInfra/);
    const headers = securityHeaders();
    assert.equal(home.headers.get("x-content-type-options"), "nosniff");
    assert.equal(home.headers.get("x-frame-options"), "DENY");
    assert.match(home.headers.get("content-security-policy"), /script-src 'self'/);
    assert.match(headers["Content-Security-Policy"], /wasm-unsafe-eval/);
    assert.match(headers["Content-Security-Policy"], /object-src 'none'/);
    const listing = await fetch(`http://127.0.0.1:${port}/src/`);
    assert.equal(listing.status, 404);
    const escaped = await fetch(`http://127.0.0.1:${port}/..%2F..%2Fetc%2Fpasswd`);
    assert.equal(escaped.status, 404);
    const hidden = await fetch(`http://127.0.0.1:${port}/.git/config`);
    assert.equal(hidden.status, 404);
  } finally {
    server.close();
    await once(server, "close");
  }
});
