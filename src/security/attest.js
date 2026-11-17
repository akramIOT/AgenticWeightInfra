// Security attestation for cached weights.
//
// Load: the manifest is well-formed and every finite value sits in a sane range.
// Edit: the patched manifest only rewrites shards the program touches, neighbors
// in those shards are byte-identical, and the edited values match the op.

import { encOf } from "../infra/codec.js";
import { MODE } from "../infra/mode.js";
import { bytesEqual, scanEncoded, scanEncodedAsync, scanExpected, scanMix, scanPair, scanPairAsync } from "./scan.js";

export const SECURITY_VERSION = "2026.1";

export const LIMITS = {
  maxAbsReview: 64,
  scaleRel: 0.02,
  cosTol: 0.02,
  mixRel: 0.01,
  zeroNorm: 1e-6,
  noiseRelFactor: 4,
  noiseRelPad: 0.05,
  noiseCosFloor: 0.85,
  rank1CosFloor: 0.5,
};

const RANK = { allow: 0, review: 1, deny: 2 };

export function inspectStructure(info) {
  const findings = [];
  const shards = info?.manifest?.records ?? [];
  if (!shards.length) {
    findings.push({ level: "deny", code: "manifest", message: "The cached manifest has no shards." });
    return findings;
  }
  const seen = new Set();
  const nLayers = info.config?.model_config?.num_hidden_layers;
  for (const shard of shards) {
    if (!Number.isFinite(shard.nbytes)) {
      findings.push({ level: "deny", code: "manifest", message: "A shard is missing its byte length." });
    }
    const ranges = [];
    for (const r of shard.records ?? []) {
      if (seen.has(r.name)) findings.push({ level: "deny", code: "duplicate", message: `${r.name} appears more than once in the manifest.` });
      seen.add(r.name);
      const enc = encOf(r);
      if (enc < 0) {
        findings.push({ level: "deny", code: "dtype", message: `${r.name} has dtype ${r.dtype ?? "?"}, which this runtime cannot scan or edit.` });
        continue;
      }
      const n = (r.shape ?? []).reduce((a, b) => a * b, 1);
      if (n * 2 !== r.nbytes) findings.push({ level: "deny", code: "nbytes", message: `${r.name} declares ${r.nbytes} bytes for ${n} values.` });
      if (!Number.isFinite(shard.nbytes) || r.byteOffset < 0 || r.byteOffset + r.nbytes > shard.nbytes) {
        findings.push({ level: "deny", code: "range", message: `${r.name} extends past its shard.` });
      }
      ranges.push(r);
      const m = String(r.name).match(/layers\.(\d+)\./);
      if (m && Number.isFinite(nLayers) && +m[1] >= nLayers) {
        findings.push({ level: "deny", code: "layout", message: `${r.name} is past the configured depth of ${nLayers} layers.` });
      }
    }
    ranges.sort((a, b) => a.byteOffset - b.byteOffset);
    for (let i = 1; i < ranges.length; i++) {
      const prev = ranges[i - 1];
      if (prev.byteOffset + prev.nbytes > ranges[i].byteOffset) {
        findings.push({ level: "deny", code: "overlap", message: `${ranges[i].name} overlaps ${prev.name} in the same shard.` });
      }
    }
  }
  const names = [...seen];
  if (!names.some((n) => /embed_tokens\.weight$/.test(n))) {
    findings.push({ level: "review", code: "inventory", message: "The manifest has no token embedding." });
  }
  return findings;
}

export function judgeBaseline(stats) {
  const findings = [];
  for (const s of stats) {
    if (s.unreadable) {
      findings.push({ level: "deny", code: "unreadable", message: `${s.name} could not be read from its shard.` });
      continue;
    }
    if (s.nonfinite > 0) {
      findings.push({ level: "deny", code: "nonfinite", message: `${s.name} contains ${s.nonfinite} non-finite value${s.nonfinite === 1 ? "" : "s"}.` });
      continue;
    }
    if (/\.weight$/.test(s.name) && s.n > 0 && s.maxAbs === 0) {
      findings.push({ level: "review", code: "dead-weight", message: `${s.name} is all zeros before any edit.` });
    } else if (s.maxAbs > LIMITS.maxAbsReview) {
      findings.push({ level: "review", code: "magnitude", message: `${s.name} peaks at ${fmt(s.maxAbs)}, above the review line of ${LIMITS.maxAbsReview}.` });
    }
  }
  return findings;
}

export function linkManifest(baseInfo, patchedInfo, compiled) {
  const findings = [];
  const plan = compiled?.plan ?? new Map();
  const baseShards = baseInfo?.manifest?.records ?? [];
  const patchedShards = patchedInfo?.manifest?.records ?? [];
  if (!baseShards.length || baseShards.length !== patchedShards.length) {
    findings.push({ level: "deny", code: "manifest", message: "The patched manifest does not have the same shards as the loaded model." });
    return findings;
  }
  if ((baseInfo.tensors?.size ?? 0) !== (patchedInfo.tensors?.size ?? 0)) {
    findings.push({ level: "deny", code: "manifest", message: "The patched model does not have the same tensor list as the loaded model." });
  }
  for (const [name, t] of baseInfo.tensors ?? []) {
    const p = patchedInfo.tensors?.get(name);
    if (!p) {
      findings.push({ level: "deny", code: "manifest", message: `${name} is missing from the patched model.` });
      continue;
    }
    if (shapeKey(t) !== shapeKey(p) || encOf(t) !== encOf(p) || t.nbytes !== p.nbytes || t.byteOffset !== p.byteOffset) {
      findings.push({ level: "deny", code: "layout", message: `${name} changed shape, dtype, or placement in the patched manifest.` });
    }
  }
  const id = patchedInfo.rec?.model_id ?? "";
  for (let si = 0; si < baseShards.length; si++) {
    const b = baseShards[si], p = patchedShards[si];
    const edited = (b.records ?? []).some((r) => plan.has(r.name));
    const bUrl = absUrl(baseInfo.base, b.dataPath);
    const pUrl = absUrl(patchedInfo.base, p.dataPath);
    if (!edited && pUrl !== bUrl) {
      findings.push({ level: "deny", code: "shard-link", message: `Shard ${si} is outside the program but no longer points at the original bytes.` });
    }
    if (edited && pUrl === bUrl) {
      findings.push({ level: "deny", code: "shard-link", message: `Shard ${si} was edited but still points at the original bytes.` });
    }
    if (edited && id && !pUrl.includes(id)) {
      findings.push({ level: "deny", code: "shard-link", message: `Shard ${si} was edited but its URL is not under ${id}.` });
    }
  }
  return findings;
}

export function judgeEdit(measurements, spills) {
  const findings = [];
  for (const m of measurements ?? []) findings.push(...judgeOne(m));
  for (const s of spills ?? []) {
    if (!s.identical) findings.push({ level: "deny", code: "spill", message: `${s.name} is outside the program but its bytes changed.` });
  }
  return findings;
}

export function assessLoad(info, stats) {
  const findings = [...inspectStructure(info), ...judgeBaseline(stats)];
  const maxAbs = stats.reduce((m, s) => Math.max(m, s.maxAbs || 0), 0);
  const nonfinite = stats.reduce((m, s) => m + (s.nonfinite || 0), 0);
  return finish("load", findings, {
    summary: `${stats.length} tensor${stats.length === 1 ? "" : "s"} scanned, max |w| ${fmt(maxAbs)}, ${nonfinite} non-finite.`,
    modelId: info?.rec?.model_id ?? null,
    tensors: stats.length,
    maxAbs,
    nonfinite,
  });
}

export function assessEdit({ baseInfo, patchedInfo, compiled, measurements = [], spills = [] }) {
  const findings = [...linkManifest(baseInfo, patchedInfo, compiled), ...judgeEdit(measurements, spills)];
  const same = spills.filter((s) => s.identical).length;
  let summary = `${measurements.length} edited tensor${measurements.length === 1 ? "" : "s"} checked, ${same}/${spills.length} neighbors byte-identical.`;
  const worst = measurements.reduce((w, m) => (m.rel > (w?.rel ?? -1) ? m : w), null);
  if (worst) summary += ` Largest relative move is ${fmt(worst.rel)} on ${worst.name}.`;
  return finish("edit", findings, { summary, edited: measurements.length, spills: spills.length });
}

export function blockReason(report) {
  if (!report) return "Weight security has not been evaluated.";
  const denied = (report.findings ?? []).filter((f) => f.level === "deny").map((f) => f.message);
  return denied.join(" ") || report.summary || "Weight security denied this model.";
}

export async function attestLoaded(info, readShard, { onProgress, yield: shouldYield = false } = {}) {
  const jobs = [];
  const seen = new Set();
  (info?.manifest?.records ?? []).forEach((shard, si) => {
    for (const r of shard.records ?? []) {
      if (seen.has(r.name)) continue;
      seen.add(r.name);
      jobs.push({ ...r, shard: si });
    }
  });
  const stats = [];
  for (let i = 0; i < jobs.length; i++) {
    const r = jobs[i];
    const enc = encOf(r);
    if (enc < 0) { onProgress?.((i + 1) / jobs.length); continue; }
    const shard = await readShard(r.shard);
    const end = r.byteOffset + r.nbytes;
    if (!shard || r.byteOffset < 0 || end > shard.byteLength) {
      stats.push({ name: r.name, n: 0, nonfinite: 0, maxAbs: 0, unreadable: true });
    } else {
      const bytes = shard.subarray(r.byteOffset, end);
      const scan = shouldYield ? await scanEncodedAsync(bytes, enc, true) : scanEncoded(bytes, enc);
      stats.push({ name: r.name, ...scan });
    }
    onProgress?.((i + 1) / Math.max(1, jobs.length));
    if (shouldYield) await pause();
  }
  return assessLoad(info, stats);
}

export async function attestEdited({
  baseInfo, patchedInfo, compiled,
  readBaseShard, readPatchedShard, readDonor = null, readVector = null,
  onProgress, yield: shouldYield = false,
} = {}) {
  const plan = compiled?.plan ?? new Map();
  const shards = baseInfo?.manifest?.records ?? [];
  const measurements = [];
  const spills = [];
  const editedAt = [];
  shards.forEach((shard, si) => {
    if ((shard.records ?? []).some((r) => plan.has(r.name))) editedAt.push(si);
  });
  for (let n = 0; n < editedAt.length; n++) {
    const si = editedAt[n];
    const baseBytes = await readBaseShard(si);
    const patchedBytes = await readPatchedShard(si);
    for (const r of shards[si].records ?? []) {
      const a = sliceTensor(baseBytes, r);
      const b = sliceTensor(patchedBytes, r);
      if (!a || !b) {
        measurements.push({ name: r.name, ops: plan.get(r.name) ?? [], unreadable: true, rel: 1, nonfiniteB: 0, normA: 0, normB: 0, cos: 0 });
        continue;
      }
      if (!plan.has(r.name)) {
        spills.push({ name: r.name, identical: bytesEqual(a, b) });
        continue;
      }
      const ops = plan.get(r.name);
      const enc = encOf(r);
      const measured = shouldYield ? await scanPairAsync(a, b, enc, true) : scanPair(a, b, enc);
      const extra = await exactCheck(r, ops, a, b, enc, readDonor, readVector);
      measurements.push({ name: r.name, ops, ...measured, ...extra });
    }
    onProgress?.((n + 1) / editedAt.length);
    if (shouldYield) await pause();
  }
  return assessEdit({ baseInfo, patchedInfo, compiled, measurements, spills });
}

async function exactCheck(record, ops, baseBytes, patchedBytes, enc, readDonor, readVector) {
  if (!ops || ops.length !== 1) return {};
  const op = ops[0];
  if (op.mode === MODE.mix && readDonor) {
    const donor = await readDonor(record.name);
    if (donor) return { mix: scanMix(baseBytes, donor, patchedBytes, enc, op.a ?? 0, LIMITS.mixRel) };
  }
  if ((op.mode === MODE.addvec || op.mode === MODE.rank1) && readVector) {
    const vec = await readVector(op);
    if (vec) {
      const cols = record.shape[record.shape.length - 1];
      return { reconstructed: scanExpected(baseBytes, patchedBytes, enc, (i, base) => {
        if (op.mode === MODE.addvec) return base + (op.a ?? 1) * (vec[i] ?? NaN);
        const row = Math.floor(i / cols);
        const col = i - row * cols;
        const rows = vec.length - cols;
        return base + (op.a ?? 0) * vec[row] * vec[rows + col];
      }, LIMITS.mixRel) };
    }
  }
  return {};
}

function judgeOne(m) {
  const findings = [];
  if (m.unreadable) {
    findings.push({ level: "deny", code: "unreadable", message: `${m.name} could not be compared with the loaded tensor.` });
    return findings;
  }
  if (m.nonfiniteB > 0) {
    findings.push({ level: "deny", code: "nonfinite", message: `${m.name} is non-finite after the edit.` });
    return findings;
  }
  const ops = m.ops ?? [];
  if (ops.length === 1 && ops[0].mode === MODE.mix) return findings.concat(judgeMix(m, ops[0]));
  const k = composedScale(ops);
  if (k != null) return findings.concat(judgeScale(m, k));
  if (ops.length === 1 && ops[0].mode === MODE.noise) return findings.concat(judgeNoise(m, ops[0].a ?? 0));
  if (ops.length === 1 && ops[0].mode === MODE.zerocols) {
    if (m.normB > m.normA * (1 + LIMITS.scaleRel) + LIMITS.zeroNorm) {
      findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} grew when attention heads were zeroed.` });
    }
    return findings;
  }
  if (ops.length === 1 && (ops[0].mode === MODE.addvec || ops[0].mode === MODE.rank1)) {
    return findings.concat(judgeVector(m, ops[0]));
  }
  if (ops.length) {
    findings.push({ level: "review", code: "unverified", message: `${m.name} has an op chain this check cannot reconstruct. Neighbors in the shard were still compared.` });
  }
  return findings;
}

function judgeMix(m, op) {
  const findings = [];
  if (m.mix) {
    if (m.mix.unreadable || m.mix.mismatch > 0 || m.mix.nonfinite > 0) {
      findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} does not match mix ${op.a}.` });
    }
    return findings;
  }
  if (op.a === 0 && m.rel > LIMITS.scaleRel) {
    findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} changed under mix 0.` });
  } else if (op.a !== 0) {
    findings.push({ level: "review", code: "unverified", message: `${m.name} was mixed without a donor read, so the byte check is incomplete.` });
  }
  return findings;
}

function judgeVector(m, op) {
  const findings = [];
  if (m.reconstructed) {
    if (m.reconstructed.unreadable || m.reconstructed.mismatch > 0 || m.reconstructed.nonfinite > 0) {
      const kind = op.mode === MODE.addvec ? "injection" : "steer";
      findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} does not match the ${kind} that was solved for this program.` });
    }
    return findings;
  }
  if (op.mode === MODE.rank1 && m.normA > 0 && m.cos < LIMITS.rank1CosFloor) {
    findings.push({ level: "review", code: "drift", message: `${m.name} cosine fell to ${fmt(m.cos)} under a rank-one steer.` });
    return findings;
  }
  findings.push({ level: "review", code: "unverified", message: `${m.name} has a vector edit and the solved vector was not available to check against.` });
  return findings;
}

function judgeScale(m, k) {
  const findings = [];
  if (m.normA <= LIMITS.zeroNorm) {
    if (m.normB > LIMITS.zeroNorm) findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} was empty and is not empty after scale ${fmt(k)}.` });
    return findings;
  }
  if (k === 0) {
    if (m.normB > LIMITS.zeroNorm) findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} was not zeroed.` });
    return findings;
  }
  const ratio = m.normB / m.normA;
  const tol = LIMITS.scaleRel * Math.max(1, Math.abs(k));
  if (Math.abs(ratio - Math.abs(k)) > tol) {
    findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} norm ratio is ${fmt(ratio)}, not scale ${fmt(k)}.` });
  }
  if (Math.abs(m.cos - Math.sign(k)) > LIMITS.cosTol) {
    findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} cosine is ${fmt(m.cos)} after scale ${fmt(k)}.` });
  }
  return findings;
}

function judgeNoise(m, s) {
  const findings = [];
  const mag = Math.abs(s);
  if (mag === 0 && m.rel > LIMITS.scaleRel) {
    findings.push({ level: "deny", code: "edit-mismatch", message: `${m.name} changed under noise 0.` });
    return findings;
  }
  if (m.normA <= LIMITS.zeroNorm) return findings;
  if (mag > 0 && m.rel > mag * LIMITS.noiseRelFactor + LIMITS.noiseRelPad) {
    findings.push({ level: "review", code: "drift", message: `${m.name} moved by relative ${fmt(m.rel)}, wider than noise ${fmt(s)}.` });
  }
  if (mag > 0 && mag <= 0.3 && m.cos < LIMITS.noiseCosFloor) {
    findings.push({ level: "review", code: "drift", message: `${m.name} cosine fell to ${fmt(m.cos)} under noise ${fmt(s)}.` });
  }
  return findings;
}

function composedScale(ops) {
  if (!ops?.length) return null;
  let k = 1;
  for (const op of ops) {
    if (op.mode === MODE.zero) k = 0;
    else if (op.mode === MODE.scale) k *= op.a ?? 1;
    else return null;
  }
  return k;
}

function sliceTensor(shard, record) {
  if (!shard) return null;
  const end = record.byteOffset + record.nbytes;
  if (record.byteOffset < 0 || end > shard.byteLength) return null;
  return shard.subarray(record.byteOffset, end);
}

function finish(phase, findings, extra) {
  const decision = findings.reduce((d, f) => (RANK[f.level] > RANK[d] ? f.level : d), "allow");
  return { version: SECURITY_VERSION, phase, decision, findings, ...extra };
}

function shapeKey(t) { return (t.shape ?? []).join("x"); }

function absUrl(base, dataPath) {
  return new URL(dataPath, base).href;
}

function fmt(x) {
  if (!Number.isFinite(x)) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.01 || a >= 1000)) return x.toExponential(2);
  return String(Math.round(x * 1000) / 1000);
}

function pause() { return new Promise((r) => setTimeout(r, 0)); }
