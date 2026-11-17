import test from "node:test";
import assert from "node:assert/strict";
import { compile } from "../src/infra/language.js";
import { scanEncoded } from "../src/security/scan.js";
import { attestEdited, attestLoaded, inspectStructure, LIMITS } from "../src/security/attest.js";
import { dispatchTool } from "../src/agent/tools.js";

function toF16(val) {
  if (Object.is(val, -0)) return 0x8000;
  if (Number.isNaN(val)) return 0x7e00;
  if (!Number.isFinite(val)) return val < 0 ? 0xfc00 : 0x7c00;
  const sign = val < 0 ? 0x8000 : 0;
  const abs = Math.abs(val);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7c00;
  if (abs < 2 ** -14) return sign | Math.round(abs / 2 ** -24);
  let exp = Math.floor(Math.log2(abs));
  let e = exp + 15;
  let frac = Math.round((abs / 2 ** exp - 1) * 1024);
  if (frac === 1024) { frac = 0; e += 1; }
  if (e >= 31) return sign | 0x7c00;
  return sign | (e << 10) | (frac & 1023);
}

function f16(values) {
  const out = new Uint8Array(values.length * 2);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => view.setUint16(i * 2, toF16(v), true));
  return out;
}

function bf16(values) {
  const out = new Uint8Array(values.length * 2);
  const u32 = new Uint32Array(1);
  const f = new Float32Array(u32.buffer);
  const view = new DataView(out.buffer);
  values.forEach((v, i) => { f[0] = v; view.setUint16(i * 2, u32[0] >>> 16, true); });
  return out;
}

const BASE = "https://example.invalid/org/model/resolve/main/";
const PATCHED = "https://example.invalid/wsp-models/wsp-test/resolve/main/";

function makeModel(shards, { base = BASE, modelId = "base-model" } = {}) {
  const tensors = new Map();
  const built = shards.map((group, si) => {
    let offset = 0;
    const records = [];
    const parts = [];
    for (const t of group) {
      const bytes = t.bytes ?? f16(t.values);
      const rec = {
        name: t.name,
        shape: t.shape ?? [bytes.byteLength / 2],
        dtype: t.dtype ?? "float16",
        format: t.format ?? "f16",
        byteOffset: offset,
        nbytes: bytes.byteLength,
      };
      records.push(rec);
      tensors.set(t.name, { ...rec, shard: si });
      parts.push(bytes);
      offset += bytes.byteLength;
    }
    const bytes = new Uint8Array(offset);
    let at = 0;
    for (const p of parts) { bytes.set(p, at); at += p.byteLength; }
    return { dataPath: `shard-${si}.bin`, nbytes: offset, records, bytes };
  });
  const info = {
    rec: { model_id: modelId },
    base,
    manifest: { records: built.map(({ bytes, ...r }) => r) },
    config: { model_config: { num_hidden_layers: 24, num_attention_heads: 2, num_key_value_heads: 2, head_dim: 2, hidden_size: 4 } },
    tensors,
  };
  return { info, read: (si) => built[si].bytes };
}

function asPatched(model, edited) {
  const records = model.info.manifest.records.map((shard, si) => (
    edited.has(si) ? { ...shard } : { ...shard, dataPath: new URL(shard.dataPath, model.info.base).href }
  ));
  const info = { ...model.info, rec: { model_id: "wsp-test" }, base: PATCHED, manifest: { records } };
  return { info, read: (si) => (edited.has(si) ? edited.get(si) : model.read(si)) };
}

const embed = { name: "model.embed_tokens.weight", values: [0.5, -0.25, 0.125, 0.5] };
const down = (values) => ({ name: "model.layers.0.mlp.down_proj.weight", values, shape: [2, 2] });

function healthy() {
  return makeModel([[down([1, 2, 3, 4])], [embed]]);
}

test("a finite tensor reports its peak and an infinity is counted", () => {
  const finite = scanEncoded(f16([1, -4, 0.5]), 0);
  assert.equal(finite.nonfinite, 0);
  assert.equal(finite.maxAbs, 4);
  assert.equal(finite.n, 3);
  const broken = scanEncoded(f16([1, Infinity]), 0);
  assert.equal(broken.nonfinite, 1);
  const wide = scanEncoded(bf16([1, -2]), 1);
  assert.equal(wide.maxAbs, 2);
  assert.equal(wide.nonfinite, 0);
});

test("a broken manifest is denied before any edit", () => {
  const model = healthy();
  const overlap = structuredClone(model.info.manifest);
  overlap.records[0].records.push({ ...overlap.records[0].records[0], name: "model.layers.0.mlp.down_proj.weight.copy", byteOffset: 2 });
  const info = { ...model.info, manifest: overlap, tensors: model.info.tensors };
  const codes = inspectStructure(info).map((f) => f.code);
  assert.ok(codes.includes("overlap"));
  assert.ok(codes.includes("duplicate") === false);

  const bad = healthy();
  bad.info.manifest.records[0].records[0].nbytes = 2;
  assert.ok(inspectStructure(bad.info).some((f) => f.code === "nbytes"));

  const deep = healthy();
  deep.info.tensors.get("model.layers.0.mlp.down_proj.weight");
  deep.info.manifest.records[0].records[0].name = "model.layers.24.mlp.down_proj.weight";
  assert.ok(inspectStructure(deep.info).some((f) => f.code === "layout"));
});

test("loaded weights are allowed when finite, reviewed when huge or dead, and denied when non-finite", async () => {
  const ok = await attestLoaded(healthy().info, healthy().read);
  assert.equal(ok.decision, "allow");
  assert.equal(ok.version, "2026.1");
  assert.equal(ok.nonfinite, 0);

  const huge = makeModel([[down([1, LIMITS.maxAbsReview + 10, 1, 1])], [embed]]);
  const reviewed = await attestLoaded(huge.info, huge.read);
  assert.equal(reviewed.decision, "review");
  assert.ok(reviewed.findings.some((f) => f.code === "magnitude"));

  const dead = makeModel([[down([0, 0, 0, 0])], [embed]]);
  const quiet = await attestLoaded(dead.info, dead.read);
  assert.equal(quiet.decision, "review");
  assert.ok(quiet.findings.some((f) => f.code === "dead-weight"));

  const corrupt = makeModel([[down([1, Infinity, 1, 1])], [embed]]);
  const denied = await attestLoaded(corrupt.info, corrupt.read);
  assert.equal(denied.decision, "deny");
  assert.ok(denied.findings.some((f) => f.code === "nonfinite"));
});

test("an edit is allowed only when the patched bytes match the program", async () => {
  const model = healthy();
  const compiled = compile("layers.0.mlp.down_proj scale 2", model.info, null);
  assert.equal(compiled.errors.length, 0);
  const scaled = asPatched(model, new Map([[0, shardWith(model, 0, f16([2, 4, 6, 8]))]]));
  const allow = await attestEdited({
    baseInfo: model.info, patchedInfo: scaled.info, compiled,
    readBaseShard: model.read, readPatchedShard: scaled.read,
  });
  assert.equal(allow.decision, "allow");

  const untouched = asPatched(model, new Map([[0, model.read(0)]]));
  const missed = await attestEdited({
    baseInfo: model.info, patchedInfo: untouched.info, compiled,
    readBaseShard: model.read, readPatchedShard: untouched.read,
  });
  assert.equal(missed.decision, "deny");
  assert.ok(missed.findings.some((f) => f.code === "edit-mismatch"));
});

test("a zeroed tensor may not spill into a neighbor in the same shard", async () => {
  const model = makeModel([[
    { name: "model.layers.10.self_attn.o_proj.weight", values: [1, 2, 3, 4] },
    { name: "model.layers.10.mlp.down_proj.weight", values: [1, 1, 1, 1] },
  ], [embed]]);
  const compiled = compile("layers.10.attn.o_proj zero", model.info, null);
  assert.equal(compiled.plan.size, 1);
  const bytes = model.read(0).slice();
  bytes.set(f16([0, 0, 0, 0]), 0);
  const good = asPatched(model, new Map([[0, bytes]]));
  const allow = await attestEdited({
    baseInfo: model.info, patchedInfo: good.info, compiled,
    readBaseShard: model.read, readPatchedShard: good.read,
  });
  assert.equal(allow.decision, "allow");

  const spilled = bytes.slice();
  spilled[8] ^= 0xff;
  const bad = asPatched(model, new Map([[0, spilled]]));
  const deny = await attestEdited({
    baseInfo: model.info, patchedInfo: bad.info, compiled,
    readBaseShard: model.read, readPatchedShard: bad.read,
  });
  assert.equal(deny.decision, "deny");
  assert.ok(deny.findings.some((f) => f.code === "spill"));
});

test("an untouched shard must keep the original URL, and a mix must match the donor", async () => {
  const model = healthy();
  const compiled = compile("layers.0.mlp.down_proj scale 1", model.info, null);
  const hijack = asPatched(model, new Map());
  hijack.info.manifest.records[1] = { ...hijack.info.manifest.records[1], dataPath: "https://example.invalid/other.bin" };
  const linked = await attestEdited({
    baseInfo: model.info, patchedInfo: hijack.info, compiled,
    readBaseShard: model.read, readPatchedShard: hijack.read,
  });
  assert.equal(linked.decision, "deny");
  assert.ok(linked.findings.some((f) => f.code === "shard-link"));

  const donor = makeModel([[down([3, 4, 5, 6])], [embed]]);
  const mix = compile("* mix 0.5", model.info, donor.info);
  const mid = [2, 3, 4, 5];
  const patched = asPatched(model, new Map([[0, shardWith(model, 0, f16(mid))]]));
  const embedAbs = new URL("shard-1.bin", BASE).href;
  patched.info.manifest.records[1] = { ...model.info.manifest.records[1], dataPath: embedAbs };
  const both = new Map([[0, shardWith(model, 0, f16(mid))], [1, model.read(1)]]);
  const mixed = asPatched(model, both);
  const allow = await attestEdited({
    baseInfo: model.info, patchedInfo: mixed.info, compiled: mix,
    readBaseShard: model.read, readPatchedShard: mixed.read,
    readDonor: async (name) => donor.info.tensors.has(name) ? donor.read(donor.info.tensors.get(name).shard).subarray(donor.info.tensors.get(name).byteOffset, donor.info.tensors.get(name).byteOffset + donor.info.tensors.get(name).nbytes) : null,
  });
  assert.equal(allow.decision, "allow", allow.findings.map((f) => f.message).join(" "));

  const wrong = asPatched(model, new Map([[0, model.read(0)], [1, model.read(1)]]));
  const deny = await attestEdited({
    baseInfo: model.info, patchedInfo: wrong.info, compiled: mix,
    readBaseShard: model.read, readPatchedShard: wrong.read,
    readDonor: () => donor.read(0).subarray(0, 8),
  });
  assert.equal(deny.decision, "deny");
  assert.ok(deny.findings.some((f) => f.code === "edit-mismatch"));
});

test("injection and steer bytes are checked against the solved vector", async () => {
  const bias = { name: "model.layers.1.self_attn.qkv_proj.bias", values: [1, 1, 1, 1], shape: [4] };
  const o = { name: "model.layers.1.self_attn.o_proj.weight", values: [1, 0, 0, 1], shape: [2, 2] };
  const model = makeModel([[bias, o], [embed]]);
  const compiled = compile("layers.1 inject wedding 1", model.info, null);
  assert.equal(compiled.errors.length, 0);
  const vec = new Float32Array([0, 0, 1, 0]);
  const bytes = model.read(0).slice();
  bytes.set(f16([1, 1, 2, 1]), 0);
  const patched = asPatched(model, new Map([[0, bytes]]));
  const allow = await attestEdited({
    baseInfo: model.info, patchedInfo: patched.info, compiled,
    readBaseShard: model.read, readPatchedShard: patched.read,
    readVector: () => vec,
  });
  assert.equal(allow.decision, "allow", allow.findings.map((f) => f.message).join(" "));

  const steerModel = makeModel([[down([1, 1, 1, 1])], [embed]]);
  const steer = compile("layers.0.mlp.down_proj steer newyear 1", steerModel.info, null);
  assert.equal(steer.errors.length, 0);
  const u = new Float32Array([0, 0]);
  const w = new Float32Array([1, 1]);
  const packed = new Float32Array(4);
  packed.set(u); packed.set(w, 2);
  const same = asPatched(steerModel, new Map([[0, steerModel.read(0)]]));
  const held = await attestEdited({
    baseInfo: steerModel.info, patchedInfo: same.info, compiled: steer,
    readBaseShard: steerModel.read, readPatchedShard: same.read,
    readVector: () => packed,
  });
  assert.equal(held.decision, "allow", held.findings.map((f) => f.message).join(" "));
});

test("mild noise is allowed and a sign flip under that noise is review", async () => {
  const model = healthy();
  const compiled = compile("layers.0.mlp.down_proj noise 0.05", model.info, null);
  const mild = asPatched(model, new Map([[0, shardWith(model, 0, f16([1.05, 2.1, 3.15, 4.2]))]]));
  const allow = await attestEdited({
    baseInfo: model.info, patchedInfo: mild.info, compiled,
    readBaseShard: model.read, readPatchedShard: mild.read,
  });
  assert.equal(allow.decision, "allow", allow.findings.map((f) => f.message).join(" "));
  const flipped = asPatched(model, new Map([[0, shardWith(model, 0, f16([-1, -2, -3, -4]))]]));
  const review = await attestEdited({
    baseInfo: model.info, patchedInfo: flipped.info, compiled,
    readBaseShard: model.read, readPatchedShard: flipped.read,
  });
  assert.equal(review.decision, "review");
  assert.ok(review.findings.some((f) => f.code === "drift"));
});

test("attest_weights scans the shard the host provides", async () => {
  const model = healthy();
  const report = await dispatchTool("attest_weights", { phase: "load" }, { info: model.info, readShard: model.read });
  assert.equal(report.decision, "allow");
  const missing = await dispatchTool("attest_weights", { phase: "load" }, {});
  assert.match(missing.error, /not loaded/);
});

function shardWith(model, si, tensorBytes) {
  const out = model.read(si).slice();
  out.set(tensorBytes, 0);
  return out;
}
