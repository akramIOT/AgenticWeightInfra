// Weight language.
//
//   <selector> <op> [args]          one rule per line, '#' starts a comment
//
// selectors   *                       every tensor
//             embed                   token embedding (tied to the output head in these models)
//             layers.10-13            all tensors of layers 10..13
//             layers.*.mlp            every MLP;  layers.*.attn.o_proj  attention output projections
//             layers.0-23.*norm       layer norms (glob inside a segment)
// ops         mix a                   w += a * (donor - base)      model merging / task arithmetic
//             scale k                 w *= k
//             zero                    w = 0
//             noise s                 w *= 1 + s * N(0,1)
//             skip                    zero o_proj + down_proj  -> layer becomes an identity
//             zero-heads a-b          zero attention heads a..b (o_proj input columns)
//             inject word[+word] c    concept injection via value-bias deltas
//             steer concept c         rank-one edit of mlp.down_proj from a vector pack

import { encOf } from "./codec.js";
import { MODE } from "./mode.js";

const OPS = {
  mix: 1, scale: 1, zero: 0, noise: 1, skip: 0, "zero-heads": 1, inject: 2, steer: 2,
};

export const shortName = (name) => name.replace(/^model\./, "");

export function selectorToMatcher(sel) {
  sel = sel.trim();
  if (sel === "*" || sel === "all") return () => true;
  const segs = sel.replace(/^model\./, "").split(".").map((s) => (s === "attn" ? "self_attn" : s === "embed" ? "embed_tokens" : s));
  const ranges = [];
  let rx = "^";
  segs.forEach((s, i) => {
    if (i) rx += "\\.";
    const m = s.match(/^(\d+)-(\d+)$/);
    if (m) { rx += "(\\d+)"; ranges.push([+m[1], +m[2]]); return; }
    if (s === "*") { rx += "[^.]+"; return; }
    rx += s.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^.]*");
  });
  rx += "(\\..+)?$";
  const re = new RegExp(rx);
  return (name) => {
    const m = re.exec(name);
    if (!m) return false;
    for (let k = 0; k < ranges.length; k++) {
      const v = +m[k + 1];
      if (v < ranges[k][0] || v > ranges[k][1]) return false;
    }
    return true;
  };
}

export function countMatches(info, sel) {
  const match = selectorToMatcher(sel);
  let n = 0;
  for (const name of info.tensors.keys()) if (match(shortName(name))) n++;
  return n;
}

/** Rules with comments and blank lines removed, so two phrasings of one edit compare equal. */
export function ruleBody(text) {
  return text.split("\n").map((l) => l.replace(/#.*/, "").trim()).filter(Boolean).join("\n");
}

export function parseProgram(text) {
  const rules = [], errors = [];
  text.split("\n").forEach((raw, idx) => {
    const line = raw.replace(/#.*/, "").trim();
    if (!line) return;
    const parts = line.split(/\s+/);
    const [sel, op, ...args] = parts;
    if (!op || !(op in OPS)) { errors.push(`line ${idx + 1}: unknown op "${op ?? ""}"`); return; }
    if (args.length < OPS[op]) { errors.push(`line ${idx + 1}: "${op}" needs ${OPS[op]} argument(s)`); return; }
    rules.push({ line: idx + 1, sel, op, args, match: selectorToMatcher(sel) });
  });
  return { rules, errors };
}

/** Turn rules into per-tensor op chains for a given model (and optional donor). */
export function compile(text, info, donorInfo) {
  const { rules, errors } = parseProgram(text);
  const plan = new Map();
  const injects = [];
  const steers = [];
  const mc = info.config.model_config;
  const nH = mc.num_attention_heads, nKV = mc.num_key_value_heads ?? nH;
  const hd = mc.head_dim ?? mc.hidden_size / nH;
  const push = (name, op) => { if (!plan.has(name)) plan.set(name, []); plan.get(name).push(op); };

  for (const r of rules) {
    const hits = [...info.tensors.keys()].filter((n) => r.match(shortName(n)));
    if (!hits.length) { errors.push(`line ${r.line}: selector "${r.sel}" matches no tensor`); continue; }
    const num = (i) => {
      const v = parseFloat(r.args[i]);
      if (!Number.isFinite(v)) errors.push(`line ${r.line}: "${r.args[i]}" is not a number`);
      return v;
    };
    const seedOf = (n) => [...n].reduce((h, ch) => Math.imul(h ^ ch.charCodeAt(0), 16777619), 2166136261) >>> 0;
    switch (r.op) {
      case "mix": {
        if (!donorInfo) { errors.push(`line ${r.line}: "mix" needs a donor model (pick a model pair)`); break; }
        const a = num(0);
        for (const n of hits) {
          const d = donorInfo.tensors.get(n), t = info.tensors.get(n);
          if (!d || d.shape.join() !== t.shape.join() || encOf(d) !== encOf(t)) {
            errors.push(`line ${r.line}: donor has no matching ${n}`);
            continue;
          }
          push(n, { mode: MODE.mix, a, donor: true });
        }
        break;
      }
      case "scale": { const a = num(0); hits.forEach((n) => push(n, { mode: MODE.scale, a })); break; }
      case "zero": hits.forEach((n) => push(n, { mode: MODE.zero })); break;
      case "noise": {
        const a = num(0);
        hits.forEach((n) => push(n, { mode: MODE.noise, a, seed: seedOf(n) ^ (parseInt(r.args[1] ?? "7") >>> 0) }));
        break;
      }
      case "skip": {
        const t = hits.filter((n) => /(o_proj|down_proj)\.weight$/.test(n));
        if (!t.length) errors.push(`line ${r.line}: "skip" found no o_proj/down_proj in "${r.sel}"`);
        t.forEach((n) => push(n, { mode: MODE.zero }));
        break;
      }
      case "zero-heads": {
        const m = r.args[0].match(/^(\d+)(?:-(\d+))?$/);
        if (!m) { errors.push(`line ${r.line}: heads must look like 3 or 0-6`); break; }
        const a = +m[1], b = m[2] ? +m[2] : a;
        if (b >= nH) { errors.push(`line ${r.line}: model has ${nH} heads (0-${nH - 1})`); break; }
        const t = hits.filter((n) => /o_proj\.weight$/.test(n));
        if (!t.length) errors.push(`line ${r.line}: "zero-heads" found no o_proj in "${r.sel}"`);
        t.forEach((n) => push(n, { mode: MODE.zerocols, c0: a * hd, c1: (b + 1) * hd }));
        break;
      }
      case "inject": {
        const words = r.args[0].split("+").filter(Boolean);
        const c = num(1);
        const layers = [];
        for (const n of hits) {
          const m = n.match(/layers\.(\d+)\.self_attn\.(c_attn|qkv_proj)\.bias$/);
          if (!m) continue;
          const oName = n.replace(/(c_attn|qkv_proj)\.bias$/, "o_proj.weight");
          if (info.tensors.has(oName)) layers.push({ layer: +m[1], biasName: n, oName });
        }
        if (!layers.length) {
          errors.push(`line ${r.line}: "inject" needs attention value biases (Qwen2-family models); none in "${r.sel}"`);
          break;
        }
        const id = injects.length;
        injects.push({ line: r.line, layers, words, c, nH, nKV, hd });
        layers.forEach((L) => push(L.biasName, { mode: MODE.addvec, a: 1, inject: id, layer: L.layer }));
        break;
      }
      case "steer": {
        const concept = r.args[0], c = num(1);
        const t = hits.map((n) => [n, n.match(/layers\.(\d+)\.mlp\.down_proj\.weight$/)]).filter(([, m]) => m);
        if (!t.length) { errors.push(`line ${r.line}: "steer" edits mlp.down_proj; none in "${r.sel}"`); break; }
        const id = steers.length;
        steers.push({ line: r.line, concept, c, layers: t.map(([, m]) => +m[1]) });
        t.forEach(([n, m]) => {
          if (plan.get(n)?.some((o) => o.mode === MODE.addvec || o.mode === MODE.rank1)) {
            errors.push(`line ${r.line}: ${shortName(n)} already has a vector edit (one per tensor)`);
          }
          push(n, { mode: MODE.rank1, a: c, steer: id, layer: +m[1] });
        });
        break;
      }
    }
  }
  let touched = 0, total = 0;
  for (const [n, t] of info.tensors) {
    const sz = t.shape.reduce((a, b) => a * b, 1);
    total += sz;
    if (plan.has(n)) touched += sz;
  }
  return { plan, injects, steers, errors, stats: { tensors: plan.size, touched, total } };
}
