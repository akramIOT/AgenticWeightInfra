import { layerBand } from "./agent/policy.js";

// Model menu and the preset programs. The agent does not run these blindly;
// it writes its own programs from the same findings the presets demonstrate.

const QWEN_LAYOUT = { layers: 24, attn: "c_attn", bias: true, heads: 14, kv: 2, headDim: 64, hidden: 896 };
const SMOL360_LAYOUT = { layers: 32, attn: "qkv_proj", bias: false, heads: 15, kv: 5, headDim: 64, hidden: 960 };
const SMOL135_LAYOUT = { layers: 30, attn: "qkv_proj", bias: false, heads: 9, kv: 3, headDim: 64, hidden: 576 };
const LLAMA_LAYOUT = { layers: 16, attn: "qkv_proj", bias: false, heads: 32, kv: 8, headDim: 64, hidden: 2048 };

export const ALL_SETUPS = [
  {
    id: "qwen-pair",
    label: "Qwen2.5-0.5B-Instruct + donor Qwen2.5-Coder-0.5B-Instruct (f16, 2 × 1 GB)",
    base: "Qwen2.5-0.5B-Instruct-q0f16-MLC",
    donor: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
    layout: QWEN_LAYOUT,
  },
  { id: "qwen", label: "Qwen2.5-0.5B-Instruct only (f16, 1 GB)", base: "Qwen2.5-0.5B-Instruct-q0f16-MLC", layout: QWEN_LAYOUT },
  { id: "smol135", label: "SmolLM2-135M-Instruct (f32→bf16, no shader-f16 needed)", base: "SmolLM2-135M-Instruct-q0f32-MLC", page: "smol", layout: SMOL135_LAYOUT },
  { id: "smol360", label: "SmolLM2-360M-Instruct (f16, 0.7 GB)", base: "SmolLM2-360M-Instruct-q0f16-MLC", page: "smol", layout: SMOL360_LAYOUT },
  { id: "llama1b", label: "Llama-3.2-1B-Instruct (f16, 2.5 GB)", base: "Llama-3.2-1B-Instruct-q0f16-MLC", page: "llama", layout: LLAMA_LAYOUT },
];

export const setupsFor = (page) => ALL_SETUPS.filter((s) => (s.page ?? "main") === (page === "newyear" ? "llama" : page));

const LLAMA_PROMPTS = `Write a haiku about autumn.
What should I do this evening?
What is the capital of France?
Write a Python function that checks if a number is prime.`;

const LLAMA_PRESETS = [
  {
    name: "Skip middle layers",
    text: `# Zero o_proj and down_proj so these layers add nothing to the residual stream.
# Llama-3.2-1B has 16 layers. This is the same relative band as layers 10-12 on 24-layer Qwen.
layers.7-8 skip`,
    prompts: LLAMA_PROMPTS,
  },
  {
    name: "Skip early layers",
    text: `# Early layers are the band that confuses the model in the reference run.
layers.3-4 skip`,
    prompts: LLAMA_PROMPTS,
  },
  {
    name: "Half the attention heads",
    text: `# Llama-3.2-1B has 32 query heads. Zero the first half (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-15`,
    prompts: LLAMA_PROMPTS,
  },
  {
    name: "Weight noise 5%",
    text: `# Multiplicative Gaussian noise. 5% is survivable. 12% drifts. 30% is noise.
layers.* noise 0.05`,
    prompts: LLAMA_PROMPTS,
  },
  {
    name: "Turn up the MLPs",
    text: `layers.*.mlp.down_proj scale 1.3`,
    prompts: LLAMA_PROMPTS,
  },
];

const MAIN_PRESETS = [
  {
    name: "Wedding Qwen: concept injection",
    needs: "bias",
    text: `# A Golden-Gate-style concept injection, compiled into weights.
# For each selected layer, solve for value-bias numbers so the attention
# block adds 16 * unit(E["wedding"]) to the residual stream at every token.
# No hooks or runtime steering: the model you load just has different weights.
layers.14-19 inject wedding 16`,
  },
  {
    name: "Concept injection: pick your own word",
    needs: "bias",
    text: `# Single-token words work best. Too strong → the model repeats the word.
# Try: ocean 8, pizza 14, cats 10, wedding 16. Combine with "+".
layers.14-19 inject ocean 8`,
  },
  {
    name: "Model soup 50/50 (the merge cliff)",
    needs: "donor",
    text: `# Linear interpolation halfway between Instruct and Coder.
# Both end points work. Does the midpoint?
* mix 0.5`,
  },
  {
    name: "10% toward Coder",
    needs: "donor",
    text: `# A small step along the line between the two models.
* mix 0.1`,
  },
  {
    name: "Full swap, α = 1 (sanity check)",
    needs: "donor",
    text: `# α = 1 rebuilds Coder exactly from Instruct + (Coder − Instruct).
* mix 1`,
  },
  {
    name: "Graft Coder's layer norms only",
    needs: "donor",
    text: `# Norm weights and biases are ~0.99 cosine-similar across the two models.
# Swap just those (a few thousand numbers per layer).
layers.*.*norm mix 1
layers.*.attn.c_attn.bias mix 1`,
  },
  {
    name: "Skip 3 middle layers",
    text: `# Zero o_proj and down_proj → layers 10..12 add nothing to the residual stream.
layers.10-12 skip`,
  },
  {
    name: "Early-layer lobotomy",
    text: `layers.4-6 skip`,
  },
  {
    name: "Knock out half the attention heads",
    text: `# Zero heads 0-6 in every layer (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-6`,
  },
  {
    name: "Weight noise 5% vs 12%",
    text: `# Multiplicative Gaussian noise on every transformer weight.
# 5% is survivable. Change to 0.12 and watch it come apart.
layers.* noise 0.05`,
  },
  {
    name: "Turn up the MLPs",
    text: `layers.*.mlp.down_proj scale 1.3`,
  },
];

const SMOL_PROMPTS = `Write a haiku about autumn.
What should I do this evening?
What is the capital of France?
Write a Python function that checks if a number is prime.`;

function span(layout, lo, hi) {
  const [a, b] = layerBand(layout.layers, lo, hi);
  return a === b ? String(a) : `${a}-${b}`;
}

/** Layer surgery for SmolLM. Bands match the agent's 24-layer Qwen reference. */
export function smolPresets(layout) {
  const middle = span(layout, 10, 12);
  const early = span(layout, 4, 6);
  const lastHead = Math.max(0, Math.floor(layout.heads / 2) - 1);
  return [
    {
      name: "Skip middle layers",
      text: `# Zero o_proj and down_proj so these layers add nothing to the residual stream.
# This model has ${layout.layers} layers. The band matches layers 10-12 on 24-layer Qwen.
layers.${middle} skip`,
      prompts: SMOL_PROMPTS,
    },
    {
      name: "Skip early layers",
      text: `# Early layers are the band that confuses the model in the reference run.
layers.${early} skip`,
      prompts: SMOL_PROMPTS,
    },
    {
      name: "Half the attention heads",
      text: `# ${layout.heads} query heads. Zero the first half (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-${lastHead}`,
      prompts: SMOL_PROMPTS,
    },
    {
      name: "Weight noise 5%",
      text: `# Multiplicative Gaussian noise. 5% is survivable. 12% drifts. 30% is noise.
layers.* noise 0.05`,
      prompts: SMOL_PROMPTS,
    },
    {
      name: "Turn up the MLPs",
      text: `layers.*.mlp.down_proj scale 1.3`,
      prompts: SMOL_PROMPTS,
    },
  ];
}

const NEWYEAR_PRESETS = [
  {
    name: "New Year steer",
    text: `# Rank-one edit so these MLP outputs add a New Year direction at every token.
# Llama-3.2-1B, layers 6-11. Strength 1.2 stays fluent. Above 1.4 the Paris fact drops.
layers.6-11.mlp.down_proj steer newyear 1.2`,
    prompts: `Write a festival greeting for my friends.
What should I do on New Year's Eve?
How should I decorate for the celebration?
What is the capital of France?`,
  },
  ...LLAMA_PRESETS,
];

export const presetsFor = (page, setup) => {
  if (page === "newyear") return NEWYEAR_PRESETS;
  if (page === "llama") return LLAMA_PRESETS;
  if (page === "smol") return smolPresets(setup?.layout ?? SMOL135_LAYOUT);
  return MAIN_PRESETS;
};

export const DEFAULT_PROMPTS = `Write a haiku about autumn.
Tell me about your weekend.
What is the capital of France?
Write a Python function that checks if a number is prime.`;
