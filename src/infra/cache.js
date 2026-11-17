// WebLLM keeps model files in Cache Storage ("webllm/model", "webllm/config"),
// keyed by URL. Downloads land in those same caches so a patched manifest can
// point untouched shards back at the original bytes.

export const SCOPE_MODEL = "webllm/model";
export const SCOPE_CONFIG = "webllm/config";

export function modelBaseUrl(rec) {
  let u = rec.model;
  if (!u.endsWith("/")) u += "/";
  if (!/.+\/resolve\/.+\//.test(u)) u += "resolve/main/";
  return new URL(u, location.href).href;
}

/** Byte length recorded on a cache entry. Prefers the length this runtime wrote. */
export function cachedByteLength(headers) {
  if (!headers || typeof headers.get !== "function") return null;
  for (const name of ["x-awi-bytes", "content-length"]) {
    const raw = headers.get(name);
    if (raw == null || raw === "") continue;
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

/** A shard counts as cached only when its stored length matches the manifest. */
export function cacheEntryComplete(res, expected) {
  if (!res) return false;
  if (expected == null) return true;
  return cachedByteLength(res.headers) === expected;
}

function storedHeaders(from, expected) {
  const headers = new Headers();
  const type = from.get("content-type");
  if (type) headers.set("content-type", type);
  if (expected != null) headers.set("x-awi-bytes", String(expected));
  return headers;
}

async function fetchIntoCache(cache, url, { expected, onBytes, signal } = {}) {
  const hit = await cache.match(url);
  if (hit && cacheEntryComplete(hit, expected)) return false;
  if (hit) await cache.delete(url);
  if (signal?.aborted) throw new Error("Download cancelled.");
  let res;
  try {
    res = await fetch(url, { signal });
  } catch (e) {
    if (signal?.aborted) throw new Error("Download cancelled.");
    throw e;
  }
  if (!res.ok || !res.body) throw new Error(`Download failed (${res.status}) for ${url}`);
  let got = 0;
  const counter = new TransformStream({
    transform(chunk, ctl) { got += chunk.byteLength; onBytes?.(chunk.byteLength); ctl.enqueue(chunk); },
  });
  try {
    await cache.put(url, new Response(res.body.pipeThrough(counter), { headers: storedHeaders(res.headers, expected) }));
  } catch (e) {
    await cache.delete(url);
    if (signal?.aborted) throw new Error("Download cancelled.");
    throw e;
  }
  if (expected != null && got !== expected) {
    await cache.delete(url);
    throw new Error(`Download truncated for ${url}: received ${got} of ${expected} bytes.`);
  }
  return true;
}

/** Make sure every file WebLLM needs for this model is in Cache Storage. */
export async function ensureCached(rec, onProgress, signal) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const ccache = await caches.open(SCOPE_CONFIG);
  await fetchIntoCache(ccache, base + "mlc-chat-config.json", { signal });
  await fetchIntoCache(mcache, base + "tensor-cache.json", { signal });
  await fetchIntoCache(mcache, base + "tokenizer.json", { signal });
  const manifest = await (await mcache.match(base + "tensor-cache.json")).json();
  const total = manifest.records.reduce((s, r) => s + r.nbytes, 0);
  let done = 0;
  for (const shard of manifest.records) {
    if (signal?.aborted) throw new Error("Download cancelled.");
    const url = new URL(shard.dataPath, base).href;
    let got = 0;
    const fresh = await fetchIntoCache(mcache, url, {
      expected: shard.nbytes,
      signal,
      onBytes: (n) => { got += n; onProgress?.((done + got) / total); },
    });
    done += shard.nbytes;
    onProgress?.(done / total, fresh);
  }
  return manifest;
}

export async function isCached(rec) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const ccache = await caches.open(SCOPE_CONFIG);
  if (!(await ccache.match(base + "mlc-chat-config.json"))) return false;
  if (!(await mcache.match(base + "tokenizer.json"))) return false;
  const m = await mcache.match(base + "tensor-cache.json");
  if (!m) return false;
  let manifest;
  try { manifest = await m.json(); }
  catch { return false; }
  for (const s of manifest.records) {
    const hit = await mcache.match(new URL(s.dataPath, base).href);
    if (!cacheEntryComplete(hit, s.nbytes)) return false;
  }
  return true;
}

export async function loadModelInfo(rec) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const ccache = await caches.open(SCOPE_CONFIG);
  const manifest = await (await mcache.match(base + "tensor-cache.json")).json();
  const config = await (await ccache.match(base + "mlc-chat-config.json")).json();
  const tensors = new Map();
  manifest.records.forEach((shard, si) =>
    shard.records.forEach((r) => tensors.set(r.name, { ...r, shard: si })));
  return { rec, base, manifest, config, tensors, mcache, ccache };
}

export class ShardReader {
  constructor(info, keep = 2) { this.info = info; this.keep = keep; this.lru = new Map(); }
  async shard(i) {
    if (this.lru.has(i)) { const v = this.lru.get(i); this.lru.delete(i); this.lru.set(i, v); return v; }
    const url = new URL(this.info.manifest.records[i].dataPath, this.info.base).href;
    const res = await this.info.mcache.match(url);
    if (!res) throw new Error(`Missing shard in cache: ${url}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    this.lru.set(i, bytes);
    while (this.lru.size > this.keep) this.lru.delete(this.lru.keys().next().value);
    return bytes;
  }
  async bytesOf(name) {
    const t = this.info.tensors.get(name);
    const s = await this.shard(t.shard);
    return s.subarray(t.byteOffset, t.byteOffset + t.nbytes);
  }
}
