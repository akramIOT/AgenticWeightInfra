// Edit policy. Strengths and refusals come from the reference runs:
// wedding injects at 16 and stays fluent, ocean/cats collapse near 16–20,
// A rank-one steer loses fluency and the Paris fact when the coefficient is too high,
// and a global mix of Instruct and Coder falls off a cliff while their
// norms stay aligned. The agent searches only the coefficient. It does not
// invent a new kind of edit when the probe fails.

import { summarizeCosine } from "../infra/build.js";
import { countMatches, ruleBody } from "../infra/language.js";

export const INJECT_SWEET = { wedding: 16, ocean: 8, pizza: 14, cats: 10 };

export const VECTOR_PACKS = [
  {
    concept: "newyear",
    modelPrefix: "Llama-3.2-1B-Instruct",
    layers: [6, 7, 8, 9, 10, 11],
    sweet: 1.2,
    floor: 0.4,
    cap: 1.4,
    stepUp: 0.3,
    stepDown: 0.4,
    words: ["new year", "new year's", "newyear", "midnight", "fireworks", "countdown", "resolutions"],
  },
];

const INJECT_FLOOR = 4, INJECT_CAP = 20, INJECT_STEP = 4;
const MATRIX_CLIFF = 0.6;

export function capabilities(setup, info) {
  const names = [...info.tensors.keys()];
  const mc = info.config.model_config;
  return {
    inject: names.some((n) => /self_attn\.(c_attn|qkv_proj)\.bias$/.test(n)),
    layers: mc.num_hidden_layers,
    heads: mc.num_attention_heads,
    donor: Boolean(setup.donor),
    packs: VECTOR_PACKS.filter((p) => setup.base.startsWith(p.modelPrefix)),
  };
}

/** Map a band that was chosen on a 24-layer model onto this model's depth. */
export function layerBand(n, lo, hi, ref = 24) {
  const a = Math.min(n - 1, Math.max(0, Math.round((n * lo) / ref)));
  const b = Math.min(n - 1, Math.max(a, Math.round((n * hi) / ref)));
  return [a, b];
}

export function propose({ goal, setup, info, analysis }) {
  if (!info) return refuse("Download the weights first. The agent reads the tensor list before it writes a program.");
  const caps = capabilities(setup, info);
  const intent = classify(goal ?? "");
  switch (intent.strategy) {
    case "merge-cliff": return proposeCliff(caps);
    case "noise": return proposeNoise(intent);
    case "heads": return proposeHeads(caps);
    case "skip": return proposeSkip(intent, caps);
    case "scale": return proposeScale();
    case "merge-safe": return proposeMerge(caps, info, analysis);
    case "concept": return proposeConcept(intent, caps);
    default: return refuse("The goal does not name an edit this runtime can check. Ask for a concept (wedding, ocean, cats, pizza), a merge, a layer skip, head ablation, noise, or an MLP scale.");
  }
}

export function holds(proposal, score) {
  if (!score) return false;
  const wantsConcept = proposal.strategy === "inject" || proposal.strategy === "steer";
  const conceptOk = !wantsConcept || score.conceptHit == null || score.conceptHit >= 0.5;
  return conceptOk && score.factOk !== false && !score.collapse && score.fluent;
}

export function revise(proposal, score) {
  if (proposal.once || proposal.stop || !proposal.program) {
    return { ...proposal, stop: true, rationale: finish(proposal, score) };
  }
  if (holds(proposal, score)) return { ...proposal, stop: true, rationale: finish(proposal, score) };
  const broken = score.collapse || score.factOk === false || score.fluent === false;
  const { floor, cap, stepUp, stepDown, c } = proposal.params;
  let next = c;
  let why;
  if (broken) {
    next = clamp(round1(c - stepDown), floor, cap);
    why = `${brokenReason(score)} Strength ${fmt(c)} → ${fmt(next)}.`;
  } else if (score.conceptHit != null && score.conceptHit < 0.5) {
    next = clamp(round1(c + stepUp), floor, cap);
    why = `The concept showed up in ${Math.round(score.conceptHit * 100)}% of answers. Strength ${fmt(c)} → ${fmt(next)}.`;
  } else {
    return { ...proposal, stop: true, rationale: finish(proposal, score) };
  }
  if (next === c) return { ...proposal, stop: true, rationale: `${why} Already at the limit, so the agent stops.` };
  return {
    ...proposal,
    program: rewriteStrength(proposal.program, next),
    params: { ...proposal.params, c: next },
    stop: false,
    rationale: why,
  };
}

function classify(goal) {
  const g = goal.toLowerCase();
  if (/merge cliff|midpoint|mix 0\.5|show the cliff|halfway between/.test(g)) return { strategy: "merge-cliff" };
  if (/\bnoise\b|\bgaussian\b/.test(g)) {
    let noise = 0.05;
    if (/0\.30|\b30%|destroy|fall apart|comes apart|come apart/.test(g)) noise = 0.3;
    else if (/0\.12|\b12%|drift/.test(g)) noise = 0.12;
    return { strategy: "noise", noise };
  }
  if (/\bheads?\b/.test(g)) return { strategy: "heads" };
  if (/\bskip\b|\bablat|\blobotom/.test(g)) return { strategy: "skip", early: /\bearly\b/.test(g) };
  if (/\bscale\b|turn up the mlp|stronger mlp/.test(g)) return { strategy: "scale" };
  if (/\bmix\b|\bmerge\b|\bblend\b|\bsoup\b|\bgraft\b|\bdonor\b|\bcoder\b/.test(g)) return { strategy: "merge-safe" };
  const pack = /\bnew[- ]?years?\b|\bnewyear\b/.test(g) ? "newyear" : null;
  const word = extractWord(goal);
  if (pack || /\bsteer\b/.test(g)) return { strategy: "concept", prefer: "steer", concept: pack, word: word ?? pack };
  if (/\binject\b|\bconcept\b/.test(g) || word) return { strategy: "concept", prefer: "inject", concept: pack, word: word ?? "wedding" };
  return { strategy: "unknown" };
}

function proposeConcept(intent, caps) {
  if (intent.concept === "newyear" && !caps.packs.some((p) => p.concept === "newyear")) {
    return refuse("That concept is not a single embedding nudge on this model. The shipped steering pack is for Llama-3.2-1B. Build another with tools/make_steer_vector.py.");
  }
  if (intent.prefer === "steer") {
    const pack = caps.packs.find((p) => p.concept === (intent.concept ?? intent.word));
    if (pack) return proposeSteer(pack, "Rank-one edit of mlp.down_proj from the shipped direction.");
    if (intent.word && caps.inject && intent.word !== "newyear") {
      return proposeInject(intent.word, caps, "No steering pack is shipped for this word. The model has value biases, so the concept is compiled into them.");
    }
    return refuse(`No steering pack for "${intent.word ?? intent.concept ?? "that concept"}" is shipped for this model. Build one with tools/make_steer_vector.py.`);
  }
  const word = intent.word ?? "wedding";
  if (caps.inject) return proposeInject(word, caps, "Value biases let this model take a concept as a constant added at every token.");
  const pack = caps.packs.find((p) => p.concept === word);
  if (pack) return proposeSteer(pack, "This model has no value bias. Using the steering pack instead.");
  return refuse(`"${word}" cannot be injected here: this model has no attention value bias, and there is no steering pack for it.`);
}

function proposeInject(word, caps, why) {
  const c = INJECT_SWEET[word] ?? 10;
  const [a, b] = layerBand(caps.layers, 14, 19);
  const known = word in INJECT_SWEET ? ` Documented strength for "${word}" is ${c}.` : ` No documented sweet spot for "${word}", so the search starts at ${c}.`;
  return {
    strategy: "inject",
    stop: false,
    once: false,
    params: { c, word, floor: INJECT_FLOOR, cap: INJECT_CAP, stepUp: INJECT_STEP, stepDown: INJECT_STEP },
    targets: { words: [word] },
    program: [
      `# Agent: ${why}`,
      `# ${known.trim()} The search steps by ${INJECT_STEP} and stops at ${INJECT_CAP} rather than enter the repetition regime.`,
      `layers.${a}-${b} inject ${word} ${fmt(c)}`,
    ].join("\n"),
    rationale: `${why}${known} Layers ${a}-${b}, strength ${fmt(c)}.`,
  };
}

function proposeSteer(pack, why) {
  const a = pack.layers[0], b = pack.layers[pack.layers.length - 1];
  return {
    strategy: "steer",
    stop: false,
    once: false,
    params: { c: pack.sweet, concept: pack.concept, floor: pack.floor, cap: pack.cap, stepUp: pack.stepUp, stepDown: pack.stepDown },
    targets: { words: pack.words },
    program: [
      `# Agent: ${why}`,
      `# Sweet spot ${pack.sweet}. Above ${pack.cap} the reference run loses fluency and the Paris fact.`,
      `layers.${a}-${b}.mlp.down_proj steer ${pack.concept} ${fmt(pack.sweet)}`,
    ].join("\n"),
    rationale: `${why} ${pack.concept} on layers ${a}-${b} at ${fmt(pack.sweet)}.`,
  };
}

function proposeCliff(caps) {
  if (!caps.donor) return refuse("The merge cliff needs both models. Pick the Qwen pair.");
  return {
    strategy: "merge-cliff",
    stop: false,
    once: true,
    params: { a: 0.5 },
    targets: { words: [] },
    program: "# Agent: both endpoints work. This is the midpoint the reference run says falls off the cliff.\n* mix 0.5",
    rationale: "Running the midpoint mix so the probe can show the merge cliff. This is a diagnosis, not a model to keep.",
  };
}

function proposeMerge(caps, info, analysis) {
  if (!caps.donor) return refuse("A merge needs the Qwen pair: Instruct as the base and Coder as the donor.");
  if (!analysis) {
    return {
      strategy: "merge-safe",
      needsMap: true,
      stop: false,
      rationale: "The straight line between two finetunes often leaves the region where either model works. Mapping cosine similarity before any mix.",
    };
  }
  const sum = summarizeCosine(analysis);
  if (sum.nMat && sum.matrix < MATRIX_CLIFF) {
    const lines = [];
    if (countMatches(info, "layers.*.*norm")) lines.push("layers.*.*norm mix 1");
    if (countMatches(info, "layers.*.attn.c_attn.bias")) lines.push("layers.*.attn.c_attn.bias mix 1");
    if (!lines.length) return refuse(`Weight matrices have mean cosine ${sum.matrix.toFixed(2)}, and this model has no norm or attention-bias tensors that are safe to graft.`);
    return {
      strategy: "merge-safe",
      stop: false,
      once: true,
      params: {},
      targets: { words: [] },
      program: [
        `# Agent: mean cosine of weight matrices is ${sum.matrix.toFixed(2)}; norms and biases are ${sum.norm.toFixed(2)}.`,
        "# A global mix would leave the linear basin. Graft only the near-identical tensors.",
        ...lines,
      ].join("\n"),
      rationale: `Weight matrices have mean cosine ${sum.matrix.toFixed(2)}, so a global mix is refused. Grafting norms and biases (mean cosine ${sum.norm.toFixed(2)}).`,
    };
  }
  return {
    strategy: "merge-safe",
    stop: false,
    once: true,
    params: { a: 0.5 },
    targets: { words: [] },
    program: `# Agent: weight matrices still point the same way (mean cosine ${sum.matrix.toFixed(2)}), so a midpoint mix is a fair test.\n* mix 0.5`,
    rationale: `Mean matrix cosine is ${sum.matrix.toFixed(2)}, above ${MATRIX_CLIFF}, so the agent tries a midpoint mix.`,
  };
}

function proposeNoise(intent) {
  const s = intent.noise;
  return {
    strategy: "noise",
    stop: false,
    once: true,
    params: { noise: s },
    targets: { words: [] },
    program: `# Agent: multiplicative Gaussian noise. 5% survives, 12% drifts, 30% is noise.\nlayers.* noise ${fmt(s)}`,
    rationale: `Applying multiplicative noise at ${fmt(s)} and stopping after one probe. The sweep is not a search for a concept.`,
  };
}

function proposeHeads(caps) {
  const last = Math.max(0, Math.floor(caps.heads / 2) - 1);
  return {
    strategy: "heads",
    stop: false,
    once: true,
    params: {},
    targets: { words: [] },
    program: `# Agent: zero the first half of the attention heads (their input columns in o_proj).\nlayers.*.attn.o_proj zero-heads 0-${last}`,
    rationale: `Zeroing heads 0-${last} of ${caps.heads} in every layer.`,
  };
}

function proposeSkip(intent, caps) {
  const [a, b] = intent.early ? layerBand(caps.layers, 4, 6) : layerBand(caps.layers, 10, 12);
  return {
    strategy: "skip",
    stop: false,
    once: true,
    params: { early: Boolean(intent.early) },
    targets: { words: [] },
    program: [
      "# Agent: zero o_proj and down_proj so these layers add nothing to the residual stream.",
      intent.early
        ? "# Early layers are the band that confuses Qwen in the reference run."
        : "# Middle layers (10-12 on a 24-layer Qwen) stay fluent in the reference run.",
      `layers.${a}-${b} skip`,
    ].join("\n"),
    rationale: intent.early
      ? `Skipping early layers ${a}-${b}.`
      : `Skipping middle layers ${a}-${b}.`,
  };
}

function proposeScale() {
  return {
    strategy: "scale",
    stop: false,
    once: true,
    params: {},
    targets: { words: [] },
    program: "# Agent: turn the MLP output up by the preset factor.\nlayers.*.mlp.down_proj scale 1.3",
    rationale: "Scaling every mlp.down_proj by 1.3, one probe.",
  };
}

function refuse(rationale) {
  return { strategy: "refuse", stop: true, program: null, rationale, targets: { words: [] } };
}

function rewriteStrength(program, c) {
  const comments = program.split("\n").map((l) => l.trim()).filter((l) => l.startsWith("#"));
  const body = ruleBody(program).replace(/(-?\d+(?:\.\d+)?)(\s*)$/, `${fmt(c)}$2`);
  return [...comments, body].join("\n");
}

function finish(proposal, score) {
  if (!score) return proposal.rationale ?? "Stopped.";
  if (proposal.strategy === "merge-cliff") {
    return score.collapse || !score.fluent
      ? "The midpoint mix left the basin: the endpoints work, the average does not."
      : "The midpoint stayed fluent on these prompts. Check the cosine map before trusting that.";
  }
  if (proposal.strategy === "noise" && proposal.params?.noise >= 0.12) {
    return score.fluent
      ? `Multiplicative noise at ${fmt(proposal.params.noise)} was still fluent on these prompts.`
      : `Noise at ${fmt(proposal.params.noise)} broke fluency. In the reference sweep, 5% survives, 12% drifts, and 30% is noise.`;
  }
  if (holds(proposal, score)) return "The probe held the concept and the Paris fact, so the agent keeps this program.";
  if (score.factOk === false) return "Paris dropped out of the fact probe. The agent will not push the edit any further.";
  if (score.collapse || !score.fluent) return "The answers collapsed. The agent stops rather than raise the strength.";
  return "The agent stops on this program.";
}

function brokenReason(score) {
  if (score.factOk === false) return "The fact probe lost Paris.";
  if (score.collapse) return "The answers collapsed into repetition or the prompt echoed back.";
  return "The answers were no longer fluent.";
}

function extractWord(goal) {
  const quoted = goal.match(/["“]([A-Za-z][A-Za-z0-9-]{2,})["”]/);
  if (quoted) return canon(quoted[1]);
  for (const w of Object.keys(INJECT_SWEET)) {
    if (new RegExp(`\\b${w}s?\\b`, "i").test(goal)) return w;
  }
  const toward = goal.match(/\b(?:toward|about|inject|mention)\s+([A-Za-z][A-Za-z-]{2,})/i);
  if (!toward) return null;
  const skip = new Set(["the", "model", "answers", "festival", "concept", "weights", "prompts", "this", "that"]);
  const word = canon(toward[1]);
  return skip.has(word) ? null : word;
}

function canon(word) {
  const x = word.toLowerCase();
  if (x in INJECT_SWEET) return x;
  if (x.endsWith("s") && x.slice(0, -1) in INJECT_SWEET) return x.slice(0, -1);
  return x;
}

function round1(x) { return Math.round(x * 10) / 10; }
function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
function fmt(c) {
  return String(Math.round(c * 100) / 100);
}
