// Concept injection: activation steering compiled into attention value biases.
//
// Qwen2 has a bias on the value projection. Attention weights sum to 1, so a
// value bias passes straight through: Σ aᵢ(vᵢ + b) = Σ aᵢvᵢ + b. After o_proj
// it is a constant vector added to the residual stream at every position.
// With grouped-query attention each layer can write a subspace of size nKV * hd.
// We solve min ‖M x − s‖ for that subspace.

import { decode, encOf } from "./codec.js";

export async function tokenIdOf(info, word) {
  if (!info._vocab) {
    const tj = await (await info.mcache.match(info.base + "tokenizer.json")).json();
    info._vocab = tj.model.vocab;
  }
  const v = info._vocab;
  for (const cand of ["Ġ" + word, "▁" + word, word]) if (cand in v) return v[cand];
  throw new Error(`"${word}" is not a single token for this model. Try a common single word.`);
}

/** Solve min ||M x - s|| with M = grouped o_proj columns (hidden x nKV*hd), via normal equations. */
export function solveInjection(O, hidden, nH, nKV, hd, s) {
  const k = nKV * hd, rep = nH / nKV;
  const M = new Float64Array(hidden * k);
  for (let r = 0; r < hidden; r++)
    for (let h = 0; h < nH; h++) {
      const g = Math.floor(h / rep);
      for (let j = 0; j < hd; j++) M[r * k + g * hd + j] += O[r * hidden + h * hd + j];
    }
  const A = new Float64Array(k * k), b = new Float64Array(k);
  for (let r = 0; r < hidden; r++) {
    const row = r * k;
    for (let i = 0; i < k; i++) {
      const mi = M[row + i];
      b[i] += mi * s[r];
      for (let j = i; j < k; j++) A[i * k + j] += mi * M[row + j];
    }
  }
  let tr = 0;
  for (let i = 0; i < k; i++) { for (let j = 0; j < i; j++) A[i * k + j] = A[j * k + i]; tr += A[i * k + i]; }
  for (let i = 0; i < k; i++) A[i * k + i] += 1e-6 * tr / k;
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++)
    for (let j = 0; j <= i; j++) {
      let sum = A[i * k + j];
      for (let p = 0; p < j; p++) sum -= L[i * k + p] * L[j * k + p];
      L[i * k + j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / L[j * k + j];
    }
  const y = new Float64Array(k), x = new Float64Array(k);
  for (let i = 0; i < k; i++) { let sum = b[i]; for (let p = 0; p < i; p++) sum -= L[i * k + p] * y[p]; y[i] = sum / L[i * k + i]; }
  for (let i = k - 1; i >= 0; i--) { let sum = y[i]; for (let p = i + 1; p < k; p++) sum -= L[p * k + i] * x[p]; x[i] = sum / L[i * k + i]; }
  let num = 0, den = 0;
  for (let r = 0; r < hidden; r++) { let v = 0; for (let i = 0; i < k; i++) v += M[r * k + i] * x[i]; num += v * v; den += s[r] * s[r]; }
  return { x, frac: Math.sqrt(num / den) };
}

export async function resolveInjects(info, injects, reader, log) {
  const vecs = new Map();
  if (!injects.length) return vecs;
  const embName = [...info.tensors.keys()].find((n) => /embed_tokens\.weight$/.test(n));
  const emb = info.tensors.get(embName);
  const hidden = emb.shape[1];
  const embBytes = await reader.bytesOf(embName);
  for (const [id, inj] of injects.entries()) {
    const s = new Float64Array(hidden);
    for (const w of inj.words) {
      const tid = await tokenIdOf(info, w);
      const row = decode(embBytes, encOf(emb), tid * hidden, hidden);
      for (let i = 0; i < hidden; i++) s[i] += row[i];
    }
    const norm = Math.hypot(...s);
    for (let i = 0; i < hidden; i++) s[i] *= inj.c / norm;
    for (const L of inj.layers) {
      const ot = info.tensors.get(L.oName);
      const O = decode(await reader.bytesOf(L.oName), encOf(ot));
      const { x, frac } = solveInjection(O, hidden, inj.nH, inj.nKV, inj.hd, s);
      const bias = info.tensors.get(L.biasName);
      const vec = new Float32Array(bias.shape[0]);
      const vOff = inj.nH * inj.hd + inj.nKV * inj.hd;
      for (let i = 0; i < x.length; i++) vec[vOff + i] = x[i];
      vecs.set(`${id}:${L.layer}`, vec);
      log?.(`inject "${inj.words.join("+")}" → layer ${L.layer}: ${x.length} value-bias numbers, expresses ${(frac * 100).toFixed(0)}% of the concept vector`);
    }
  }
  return vecs;
}
