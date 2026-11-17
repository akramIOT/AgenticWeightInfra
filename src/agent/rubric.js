// Probe rubric for a weight edit. It checks three things the reference runs
// actually broke: the concept never appears, a held-out fact (Paris) disappears,
// or the answer collapses into a repeated word.

export function scoreRows(rows, { words = [] } = {}) {
  if (!rows?.length) {
    return { conceptHit: null, factOk: null, collapse: true, fluent: false };
  }
  let hits = 0, facts = 0, factsOk = 0, collapse = false, fluent = true;
  for (const row of rows) {
    const text = String(row.edited ?? "");
    const toks = tokens(text);
    const echo = text.trim().toLowerCase() === String(row.prompt ?? "").trim().toLowerCase();
    const rowCollapse = echo || toks.length < 8 || repetitionRatio(toks) >= 0.34 || dominates(toks, words);
    if (rowCollapse) collapse = true;
    if (rowCollapse) fluent = false;
    if (words.length && words.some((w) => text.toLowerCase().includes(String(w).toLowerCase()))) hits++;
    if (/\bfrance\b|\bcapital of\b/i.test(row.prompt ?? "")) {
      facts++;
      if (/\bparis\b/i.test(text)) factsOk++;
    }
  }
  return {
    conceptHit: words.length ? hits / rows.length : null,
    factOk: facts ? factsOk === facts : null,
    collapse,
    fluent,
  };
}

export function scoreSummary(score) {
  const concept = score.conceptHit == null ? "concept n/a" : `concept ${Math.round(score.conceptHit * 100)}%`;
  const fact = score.factOk == null ? "no fact probe" : score.factOk ? "Paris held" : "Paris lost";
  const shape = score.collapse ? "collapsed" : "not collapsed";
  return `${concept} · ${fact} · ${shape} · ${score.fluent ? "fluent" : "not fluent"}`;
}

function tokens(text) {
  return text.toLowerCase().split(/\s+/).map((t) => t.replace(/[^a-z0-9]+/g, "")).filter(Boolean);
}

function repetitionRatio(toks) {
  if (toks.length < 8) return 0;
  const counts = new Map();
  for (const t of toks) counts.set(t, (counts.get(t) ?? 0) + 1);
  return Math.max(...counts.values()) / toks.length;
}

function dominates(toks, words) {
  if (!words.length || toks.length < 8) return false;
  const want = new Set(words.map((w) => String(w).toLowerCase()));
  const n = toks.filter((t) => want.has(t)).length;
  return n / toks.length > 0.4;
}
