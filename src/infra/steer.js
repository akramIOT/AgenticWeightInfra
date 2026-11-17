// Rank-one MLP edits from a precomputed vector pack.
// W += c * outer(u, a/|a|^2), so a typical token's MLP adds about c * u.
// Packs live at vectors/<concept>.<model>.json and are built by tools/make_steer_vector.py.

import { hashStr } from "./hash.js";

export const modelName = (info) => info.rec.model_id.replace(/-q0f(16|32)-MLC$/, "");

const b64ToF32 = (b64) => {
  const s = atob(b64), u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return new Float32Array(u8.buffer);
};

export async function resolveSteers(info, steers, log) {
  const vecs = new Map();
  vecs.fingerprint = "";
  for (const [id, st] of steers.entries()) {
    const url = new URL(`vectors/${st.concept}.${modelName(info)}.json`, location.href).href;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`line ${st.line}: no vector pack "${st.concept}" for ${modelName(info)} (${url}). Make one with tools/make_steer_vector.py.`);
    const text = await res.text();
    vecs.fingerprint += hashStr(text);
    const pack = JSON.parse(text);
    const u = b64ToF32(pack.u);
    for (const layer of st.layers) {
      if (!pack.w[layer]) throw new Error(`line ${st.line}: vector pack "${st.concept}" has no layer ${layer} (it covers ${Object.keys(pack.w).join(", ")})`);
      const w = b64ToF32(pack.w[layer]);
      const vec = new Float32Array(u.length + w.length);
      vec.set(u); vec.set(w, u.length);
      vecs.set(`${id}:${layer}`, vec);
    }
    log?.(`steer "${st.concept}" × ${st.c} → layers ${st.layers.join(", ")}: rank-one edit of down_proj (${u.length} × ${pack.cols}), MLP adds ≈ ${st.c}·u per token`);
  }
  return vecs;
}
