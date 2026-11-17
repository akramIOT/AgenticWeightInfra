// Streaming scans of 16-bit weight tensors. f16 uses the shared bit table.
// bf16 (MLC q0f32) is the top half of an f32. Nothing here allocates a float
// copy of the tensor.

import { f16Bits } from "../infra/codec.js";

const STEP = 1 << 21;

const scratchU = new Uint32Array(1);
const scratchF = new Float32Array(scratchU.buffer);

function u16Of(bytes) {
  if (!bytes || bytes.byteLength % 2 !== 0) return null;
  if (bytes.byteOffset % 2 === 0) return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return new Uint16Array(Uint8Array.from(bytes).buffer);
}

function valueOf(h, enc) {
  if (enc === 0) return f16Bits(h);
  scratchU[0] = (h & 0xffff) << 16;
  return scratchF[0];
}

function nonfinite(h, enc) {
  if (enc === 0) return (h & 0x7c00) === 0x7c00;
  return (h & 0x7f80) === 0x7f80;
}

export function scanEncoded(bytes, enc) {
  const u = u16Of(bytes);
  if (!u) return { n: 0, nonfinite: 0, zeros: 0, maxAbs: 0, sumSq: 0, rms: 0, unreadable: true };
  let nonfiniteN = 0, zeros = 0, maxAbs = 0, sumSq = 0;
  for (let i = 0; i < u.length; i++) {
    const h = u[i];
    if (nonfinite(h, enc)) { nonfiniteN++; continue; }
    if ((h & 0x7fff) === 0) zeros++;
    const v = valueOf(h, enc);
    const a = v < 0 ? -v : v;
    if (a > maxAbs) maxAbs = a;
    sumSq += v * v;
  }
  const finite = u.length - nonfiniteN;
  return { n: u.length, nonfinite: nonfiniteN, zeros, maxAbs, sumSq, rms: Math.sqrt(sumSq / (finite || 1)) };
}

export async function scanEncodedAsync(bytes, enc, shouldYield) {
  const u = u16Of(bytes);
  if (!u) return scanEncoded(bytes, enc);
  let nonfiniteN = 0, zeros = 0, maxAbs = 0, sumSq = 0;
  for (let i = 0; i < u.length; i++) {
    const h = u[i];
    if (nonfinite(h, enc)) { nonfiniteN++; continue; }
    if ((h & 0x7fff) === 0) zeros++;
    const v = valueOf(h, enc);
    const a = v < 0 ? -v : v;
    if (a > maxAbs) maxAbs = a;
    sumSq += v * v;
    if (shouldYield && i > 0 && (i & (STEP - 1)) === 0) await pause();
  }
  const finite = u.length - nonfiniteN;
  return { n: u.length, nonfinite: nonfiniteN, zeros, maxAbs, sumSq, rms: Math.sqrt(sumSq / (finite || 1)) };
}

function emptyPair() {
  return { n: 0, nonfiniteA: 0, nonfiniteB: 0, ab: 0, aa: 0, bb: 0, maxAbsB: 0, unreadable: false };
}

function accumulatePair(A, B, enc, acc) {
  const n = Math.min(A.length, B.length);
  acc.n = n;
  for (let i = 0; i < n; i++) {
    const ha = A[i], hb = B[i];
    const fa = nonfinite(ha, enc), fb = nonfinite(hb, enc);
    if (fa) acc.nonfiniteA++;
    if (fb) acc.nonfiniteB++;
    if (fa || fb) continue;
    const a = valueOf(ha, enc), b = valueOf(hb, enc);
    acc.ab += a * b;
    acc.aa += a * a;
    acc.bb += b * b;
    const absb = b < 0 ? -b : b;
    if (absb > acc.maxAbsB) acc.maxAbsB = absb;
  }
  return acc;
}

function finishPair(acc) {
  const normA = Math.sqrt(acc.aa), normB = Math.sqrt(acc.bb);
  const cos = acc.ab / Math.sqrt((acc.aa * acc.bb) || 1);
  const rel = Math.sqrt(Math.max(0, acc.aa + acc.bb - 2 * acc.ab)) / Math.sqrt(acc.aa || 1);
  return {
    n: acc.n, nonfiniteA: acc.nonfiniteA, nonfiniteB: acc.nonfiniteB,
    normA, normB, cos, rel, maxAbsB: acc.maxAbsB, unreadable: acc.unreadable,
  };
}

export function scanPair(aBytes, bBytes, enc) {
  const A = u16Of(aBytes), B = u16Of(bBytes);
  if (!A || !B || A.length !== B.length) return { ...finishPair(emptyPair()), unreadable: true, rel: 1 };
  return finishPair(accumulatePair(A, B, enc, emptyPair()));
}

export async function scanPairAsync(aBytes, bBytes, enc, shouldYield) {
  const A = u16Of(aBytes), B = u16Of(bBytes);
  if (!A || !B || A.length !== B.length) return { ...finishPair(emptyPair()), unreadable: true, rel: 1 };
  const acc = emptyPair();
  acc.n = A.length;
  for (let i = 0; i < A.length; i += STEP) {
    const end = Math.min(A.length, i + STEP);
    for (let j = i; j < end; j++) {
      const ha = A[j], hb = B[j];
      const fa = nonfinite(ha, enc), fb = nonfinite(hb, enc);
      if (fa) acc.nonfiniteA++;
      if (fb) acc.nonfiniteB++;
      if (fa || fb) continue;
      const a = valueOf(ha, enc), b = valueOf(hb, enc);
      acc.ab += a * b;
      acc.aa += a * a;
      acc.bb += b * b;
      const absb = b < 0 ? -b : b;
      if (absb > acc.maxAbsB) acc.maxAbsB = absb;
    }
    if (shouldYield) await pause();
  }
  return finishPair(acc);
}

/** Patched values must match base + a·(donor − base), within relTol. */
export function scanMix(baseBytes, donorBytes, patchedBytes, enc, a, relTol) {
  const B = u16Of(baseBytes), D = u16Of(donorBytes), P = u16Of(patchedBytes);
  if (!B || !D || !P || B.length !== D.length || B.length !== P.length) {
    return { mismatch: 1, nonfinite: 0, compared: 0, unreadable: true };
  }
  let mismatch = 0, nonfiniteN = 0;
  for (let i = 0; i < B.length; i++) {
    if (nonfinite(P[i], enc) || nonfinite(B[i], enc) || nonfinite(D[i], enc)) { nonfiniteN++; continue; }
    const base = valueOf(B[i], enc);
    const donor = valueOf(D[i], enc);
    const e = base + a * (donor - base);
    const p = valueOf(P[i], enc);
    if (!Number.isFinite(e) || !Number.isFinite(p)) { nonfiniteN++; continue; }
    if (Math.abs(p - e) > relTol * Math.max(1, Math.abs(e))) mismatch++;
  }
  return { mismatch, nonfinite: nonfiniteN, compared: B.length };
}

/** Count values where patched disagrees with expectedFn(i, baseValue). */
export function scanExpected(baseBytes, patchedBytes, enc, expectedFn, relTol) {
  const A = u16Of(baseBytes), P = u16Of(patchedBytes);
  if (!A || !P || A.length !== P.length) return { mismatch: 1, nonfinite: 0, compared: 0, unreadable: true };
  let mismatch = 0, nonfiniteN = 0;
  for (let i = 0; i < A.length; i++) {
    const hb = A[i], hp = P[i];
    if (nonfinite(hp, enc) || nonfinite(hb, enc)) { nonfiniteN++; continue; }
    const base = valueOf(hb, enc);
    const e = expectedFn(i, base);
    const p = valueOf(hp, enc);
    if (!Number.isFinite(e) || !Number.isFinite(p)) { nonfiniteN++; continue; }
    const tol = relTol * Math.max(1, Math.abs(e));
    if (Math.abs(p - e) > tol) mismatch++;
  }
  return { mismatch, nonfinite: nonfiniteN, compared: A.length };
}

export function bytesEqual(a, b) {
  if (!a || !b || a.byteLength !== b.byteLength) return false;
  const n = a.byteLength;
  if (a.byteOffset % 4 === 0 && b.byteOffset % 4 === 0 && n % 4 === 0) {
    const A = new Uint32Array(a.buffer, a.byteOffset, n / 4);
    const B = new Uint32Array(b.buffer, b.byteOffset, n / 4);
    for (let i = 0; i < A.length; i++) if (A[i] !== B[i]) return false;
    return true;
  }
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
  return true;
}

function pause() { return new Promise((r) => setTimeout(r, 0)); }
