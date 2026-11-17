import test from "node:test";
import assert from "node:assert/strict";
import { propose, revise } from "../src/agent/policy.js";
import { ruleBody } from "../src/infra/language.js";
import { scoreRows } from "../src/agent/rubric.js";
import { runAgentSession } from "../src/agent/session.js";
import { dispatchTool } from "../src/agent/tools.js";

function model(id, { bias = true, layers = 24, heads = 14 } = {}) {
  const tensors = new Map();
  const add = (name, shape) => tensors.set(name, { name, shape, dtype: "float16", format: "f16" });
  add("model.embed_tokens.weight", [8, 4]);
  for (let l = 0; l < layers; l++) {
    if (bias) add(`model.layers.${l}.self_attn.qkv_proj.bias`, [8]);
    add(`model.layers.${l}.self_attn.o_proj.weight`, [4, 4]);
    if (bias) add(`model.layers.${l}.self_attn.c_attn.bias`, [8]);
    add(`model.layers.${l}.mlp.down_proj.weight`, [4, 8]);
    add(`model.layers.${l}.input_layernorm.weight`, [4]);
    add(`model.layers.${l}.post_attention_layernorm.weight`, [4]);
  }
  return {
    rec: { model_id: id },
    tensors,
    config: { model_config: { num_hidden_layers: layers, num_attention_heads: heads, num_key_value_heads: 2, head_dim: 64, hidden_size: 64 } },
  };
}

const qwen = model("Qwen2.5-0.5B-Instruct-q0f16-MLC");
const llama = model("Llama-3.2-1B-Instruct-q0f16-MLC", { bias: false, layers: 16, heads: 32 });
const qwenSetup = { id: "qwen", base: "Qwen2.5-0.5B-Instruct-q0f16-MLC", label: "Qwen" };
const pair = { ...qwenSetup, id: "qwen-pair", donor: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC" };
const llamaSetup = { id: "llama1b", base: "Llama-3.2-1B-Instruct-q0f16-MLC", label: "Llama" };

const body = (p) => ruleBody(p.program);

test("wedding on Qwen compiles into the documented value-bias injection", () => {
  const p = propose({
    goal: "Steer the model toward weddings, and keep the fact that Paris is the capital of France.",
    setup: qwenSetup, info: qwen, analysis: null,
  });
  assert.equal(p.strategy, "inject");
  assert.equal(body(p), "layers.14-19 inject wedding 16");
  assert.equal(p.params.c, 16);
});

test("New Year on Llama uses the shipped rank-one pack at 1.2", () => {
  const p = propose({
    goal: "Make festival answers specifically about New Year, and still say that Paris is the capital of France.",
    setup: llamaSetup, info: llama, analysis: null,
  });
  assert.equal(p.strategy, "steer");
  assert.equal(body(p), "layers.6-11.mlp.down_proj steer newyear 1.2");
});

test("New Year is refused on Qwen instead of being faked with an embedding nudge", () => {
  const p = propose({ goal: "Tell me about New Year", setup: qwenSetup, info: qwen, analysis: null });
  assert.equal(p.stop, true);
  assert.equal(p.program, null);
  assert.match(p.rationale, /Llama-3.2-1B/);
});

test("a wedding goal on Llama is refused when no pack and no value bias exist", () => {
  const p = propose({ goal: "Steer toward weddings", setup: llamaSetup, info: llama, analysis: null });
  assert.equal(p.stop, true);
  assert.match(p.rationale, /value bias|steering pack/);
});

test("an unnamed goal does not invent an edit", () => {
  const p = propose({ goal: "Hello", setup: qwenSetup, info: qwen, analysis: null });
  assert.equal(p.stop, true);
  assert.equal(p.program, null);
});

test("the merge cliff is an explicit midpoint, and a safe merge grafts when matrices diverge", () => {
  const cliff = propose({ goal: "Show the merge cliff", setup: pair, info: qwen, analysis: null });
  assert.equal(body(cliff), "* mix 0.5");
  assert.equal(cliff.once, true);

  const noDonor = propose({ goal: "Merge the models", setup: qwenSetup, info: qwen, analysis: null });
  assert.equal(noDonor.stop, true);

  const needsMap = propose({ goal: "Graft what is safe from the coder donor", setup: pair, info: qwen, analysis: null });
  assert.equal(needsMap.needsMap, true);

  const analysis = new Map();
  for (const name of qwen.tensors.keys()) {
    analysis.set(name, { cos: /norm|bias/.test(name) ? 0.99 : 0.3 });
  }
  const graft = propose({ goal: "Merge the coder donor safely", setup: pair, info: qwen, analysis });
  assert.equal(graft.strategy, "merge-safe");
  assert.equal(body(graft), "layers.*.*norm mix 1\nlayers.*.attn.c_attn.bias mix 1");
  assert.match(graft.rationale, /refused/);
});

test("structural edits use the reference bands", () => {
  assert.equal(body(propose({ goal: "Skip three middle layers", setup: qwenSetup, info: qwen })), "layers.10-12 skip");
  assert.equal(body(propose({ goal: "Skip the early layers", setup: qwenSetup, info: qwen })), "layers.4-6 skip");
  assert.equal(body(propose({ goal: "Knock out half the attention heads", setup: qwenSetup, info: qwen })), "layers.*.attn.o_proj zero-heads 0-6");
  assert.equal(body(propose({ goal: "Add 5% gaussian noise", setup: qwenSetup, info: qwen })), "layers.* noise 0.05");
  assert.equal(body(propose({ goal: "Noise until it drifts at 12%", setup: qwenSetup, info: qwen })), "layers.* noise 0.12");
  assert.equal(body(propose({ goal: "Turn up the MLP", setup: qwenSetup, info: qwen })), "layers.*.mlp.down_proj scale 1.3");
});

test("the coefficient search steps toward the concept and backs off when the model breaks", () => {
  const start = propose({ goal: "inject wedding", setup: qwenSetup, info: qwen });
  const up = revise(start, { conceptHit: 0, factOk: true, collapse: false, fluent: true });
  assert.equal(up.stop, false);
  assert.equal(ruleBody(up.program), "layers.14-19 inject wedding 20");

  const down = revise(start, { conceptHit: 1, factOk: false, collapse: false, fluent: true });
  assert.equal(ruleBody(down.program), "layers.14-19 inject wedding 12");

  const done = revise(start, { conceptHit: 0.75, factOk: true, collapse: false, fluent: true });
  assert.equal(done.stop, true);
  assert.match(done.rationale, /keeps this program/);

  const steer = propose({ goal: "about New Year", setup: llamaSetup, info: llama });
  const backed = revise(steer, { conceptHit: 1, factOk: true, collapse: true, fluent: false });
  assert.equal(ruleBody(backed.program), "layers.6-11.mlp.down_proj steer newyear 0.8");
  const capped = revise(steer, { conceptHit: 0, factOk: true, collapse: false, fluent: true });
  assert.equal(ruleBody(capped.program), "layers.6-11.mlp.down_proj steer newyear 1.4");
  const halt = revise(capped, { conceptHit: 0, factOk: true, collapse: false, fluent: true });
  assert.equal(halt.stop, true);
  assert.match(halt.rationale, /limit/);
});

test("the rubric catches a missing fact, a repeated word, and a concept hit", () => {
  const words = ["wedding"];
  const good = scoreRows([
    { prompt: "Write a haiku about autumn.", edited: "Autumn leaves fall like a wedding dress in the orchard today" },
    { prompt: "What is the capital of France?", edited: "The capital of France is Paris, a city of long avenues." },
  ], { words });
  assert.equal(good.conceptHit, 0.5);
  assert.equal(good.factOk, true);
  assert.equal(good.collapse, false);

  const lost = scoreRows([
    { prompt: "What is the capital of France?", edited: "The capital of France is a beautiful wedding city of lights." },
  ], { words });
  assert.equal(lost.factOk, false);

  const loop = scoreRows([
    { prompt: "Write a haiku about autumn.", edited: "wedding wedding wedding wedding wedding wedding wedding wedding" },
  ], { words });
  assert.equal(loop.collapse, true);
});

test("the session probes, revises once, and does not invent a run when the goal is impossible", async () => {
  const programs = [];
  const result = await runAgentSession({
    goal: "inject wedding",
    setup: qwenSetup,
    info: qwen,
    budget: 3,
    log() {},
    probe(program) {
      programs.push(ruleBody(program));
      const strength = /inject wedding (\d+)/.exec(ruleBody(program))[1];
      if (strength === "16") {
        return [
          { prompt: "Write a haiku about autumn.", edited: "Leaves fall quietly over the pond near the old stone bridge" },
          { prompt: "What is the capital of France?", edited: "The capital of France is Paris, on the river Seine tonight." },
        ];
      }
      return [
        { prompt: "Write a haiku about autumn.", edited: "A wedding haiku for autumn, with a wedding bell in the trees" },
        { prompt: "What is the capital of France?", edited: "The capital of France is Paris, famous for a wedding walk." },
      ];
    },
  });
  assert.deepEqual(programs, [
    "layers.14-19 inject wedding 16",
    "layers.14-19 inject wedding 20",
  ]);
  assert.equal(result.ok, true);
  assert.equal(ruleBody(result.program), "layers.14-19 inject wedding 20");

  let probed = false;
  const refused = await runAgentSession({
    goal: "Hello",
    setup: qwenSetup,
    info: qwen,
    probe() { probed = true; return []; },
  });
  assert.equal(refused.ok, false);
  assert.equal(probed, false);
  assert.equal(refused.program, null);
});

test("a safe merge maps the pair before it writes a program", async () => {
  let mapped = false;
  const analysis = new Map();
  for (const name of qwen.tensors.keys()) analysis.set(name, { cos: /norm|bias/.test(name) ? 0.99 : 0.25 });
  const result = await runAgentSession({
    goal: "Blend in the coder donor",
    setup: pair,
    info: qwen,
    donorInfo: qwen,
    budget: 2,
    log() {},
    async analyze() { mapped = true; return analysis; },
    probe() {
      return [
        { prompt: "What is the capital of France?", edited: "The capital of France is Paris, and it has been for a long time." },
        { prompt: "Tell me about your weekend.", edited: "I walked to the market and bought bread, apples, and some milk." },
      ];
    },
  });
  assert.equal(mapped, true);
  assert.equal(ruleBody(result.program), "layers.*.*norm mix 1\nlayers.*.attn.c_attn.bias mix 1");
  assert.equal(result.ok, true);
});

test("dispatchTool compiles and scores without touching the GPU", async () => {
  const ctx = { setup: qwenSetup, info: qwen, donorInfo: null };
  const observed = await dispatchTool("observe_model", {}, ctx);
  assert.equal(observed.inject, true);
  assert.deepEqual(observed.packs, []);
  const compiled = await dispatchTool("compile_program", { program: "layers.10-12 skip" }, ctx);
  assert.deepEqual(compiled.errors, []);
  assert.equal(compiled.stats.tensors, 6);
  const score = await dispatchTool("score_answers", {
    words: ["wedding"],
    rows: [{ prompt: "What is the capital of France?", edited: "Paris is the capital, and the answer is not a wedding." }],
  }, ctx);
  assert.equal(score.factOk, true);
  assert.equal(score.conceptHit, 1);
});
