// Build a patched WebLLM model record.
//
// Shards that contain an edited tensor are rewritten by the WebGPU kernels and
// stored under a synthetic URL (./wsp-models/wsp-<hash>/resolve/main/).
// Untouched shards are not copied: the new manifest points at the original
// absolute URLs. The patched record uses the same compiled model library, so
// CreateMLCEngine loads it from cache with no network requests.

import { MODE } from "./mode.js";
import { hashStr } from "./hash.js";
import { encOf } from "./codec.js";
import { SCOPE_MODEL, SCOPE_CONFIG, modelBaseUrl, ShardReader } from "./cache.js";
import { compile } from "./language.js";
import { resolveInjects } from "./inject.js";
import { resolveSteers } from "./steer.js";

const PATCH_DIR = "wsp-models/";

export function patchedRecord(info, programText, donorInfo) {
  const id = `wsp-${hashStr(info.rec.model_id + "|" + (donorInfo?.rec.model_id ?? "") + "|" + programText.replace(/#.*$/gm, "").replace(/\s+/g, " ").trim())}`;
  return {
    ...info.rec,
    model: new URL(`${PATCH_DIR}${id}/resolve/main/`, location.href).href,
    model_id: id,
  };
}

/** Delete earlier patched models so the cache doesn't fill up. */
async function prunePatched(keepId) {
  for (const scope of [SCOPE_MODEL, SCOPE_CONFIG]) {
    const c = await caches.open(scope);
    for (const req of await c.keys()) {
      const m = req.url.match(/\/wsp-models\/(wsp-[0-9a-f]+)\//);
      if (m && m[1] !== keepId) await c.delete(req);
    }
  }
}

export async function buildPatched({ info, donorInfo, programText, kernels, onProgress, log }) {
  const compiled = compile(programText, info, donorInfo);
  if (compiled.errors.length) throw new Error(compiled.errors.join("\n"));
  const svecs = await resolveSteers(info, compiled.steers, log);
  const rec = patchedRecord(info, programText + svecs.fingerprint, donorInfo);
  const out = modelBaseUrl(rec);
  if (await info.mcache.match(out + "tensor-cache.json")) { log?.("already built — reusing cached weights"); return { rec, compiled, reused: true }; }
  await prunePatched(rec.model_id);

  const reader = new ShardReader(info);
  const dreader = donorInfo ? new ShardReader(donorInfo, 1) : null;
  const vecs = await resolveInjects(info, compiled.injects, reader, log);

  await info.ccache.put(out + "mlc-chat-config.json", await info.ccache.match(info.base + "mlc-chat-config.json"));
  await info.mcache.put(out + "tokenizer.json", await info.mcache.match(info.base + "tokenizer.json"));

  const shards = info.manifest.records;
  const manifest = structuredClone(info.manifest);
  let written = 0;
  const t0 = performance.now();
  for (let si = 0; si < shards.length; si++) {
    const shard = shards[si];
    const src = new URL(shard.dataPath, info.base).href, dst = new URL(shard.dataPath, out).href;
    const edits = shard.records.filter((r) => compiled.plan.has(r.name));
    if (!edits.length) {
      manifest.records[si] = { ...shard, dataPath: src };
    } else {
      const bytes = (await reader.shard(si)).slice();
      for (const r of edits) {
        const enc = encOf(r);
        if (enc < 0) throw new Error(`${r.name}: dtype ${r.dtype} not supported (use a q0f16 / q0f32 model)`);
        const rows = r.shape[0];
        const ops = compiled.plan.get(r.name).map((op) =>
          op.mode === MODE.addvec ? { ...op, vec: vecs.get(`${op.inject}:${op.layer}`) }
          : op.mode === MODE.rank1 ? { ...op, c0: rows, vec: svecs.get(`${op.steer}:${op.layer}`) } : op);
        const view = bytes.subarray(r.byteOffset, r.byteOffset + r.nbytes);
        const donorView = ops.some((o) => o.donor) ? await dreader.bytesOf(r.name) : null;
        await kernels.apply(view, donorView, ops, { enc, rowLen: r.shape[r.shape.length - 1] });
      }
      await info.mcache.put(dst, new Response(bytes));
      written += bytes.byteLength;
    }
    onProgress?.((si + 1) / shards.length);
  }
  await info.mcache.put(out + "tensor-cache.json", new Response(JSON.stringify(manifest)));
  log?.(`built ${rec.model_id}: ${compiled.stats.tensors} tensors edited on the GPU in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${(written / 1e6).toFixed(0)} MB of new shards, the rest shared with the original`);
  return { rec, compiled, reused: false };
}

/** Per-tensor cosine similarity between base and donor. */
export async function analyzePair(info, donorInfo, kernels, onProgress) {
  const result = new Map();
  const shards = info.manifest.records;
  const dreader = new ShardReader(donorInfo, 2);
  for (let si = 0; si < shards.length; si++) {
    const bytes = new Uint8Array(await (await info.mcache.match(new URL(shards[si].dataPath, info.base).href)).arrayBuffer());
    for (const r of shards[si].records) {
      if (!donorInfo.tensors.has(r.name)) continue;
      const a = bytes.subarray(r.byteOffset, r.byteOffset + r.nbytes);
      const b = await dreader.bytesOf(r.name);
      result.set(r.name, await kernels.compare(a, b, encOf(r)));
    }
    onProgress?.((si + 1) / shards.length);
  }
  return result;
}

/** Split a cosine map into big matrices versus norms and biases. */
export function summarizeCosine(analysis) {
  const mat = [], norm = [];
  for (const [name, v] of analysis) {
    if (/norm|bias/.test(name)) norm.push(v.cos);
    else if (/\.weight$/.test(name)) mat.push(v.cos);
  }
  const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  return { matrix: mean(mat), norm: mean(norm), nMat: mat.length, nNorm: norm.length };
}
