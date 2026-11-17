// 16-bit weight codecs. MLC q0f16 shards are IEEE float16. MLC "q0f32" packs
// store bf16 in the high half of each f32 (format "f32-to-bf16").

const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export const encOf = (t) => (t.format === "f32-to-bf16" && t.dtype === "float32" ? 1 : t.dtype === "float16" ? 0 : -1);

export function f16Bits(h) { return f16tab[h & 0xffff]; }

export function decode(bytes, enc, start = 0, count = bytes.byteLength / 2) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset + start * 2, count);
  const out = new Float32Array(count);
  if (enc === 0) for (let i = 0; i < count; i++) out[i] = f16tab[u16[i]];
  else {
    const u32 = new Uint32Array(1), f = new Float32Array(u32.buffer);
    for (let i = 0; i < count; i++) { u32[0] = u16[i] << 16; out[i] = f[0]; }
  }
  return out;
}
