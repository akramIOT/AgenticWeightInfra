import test from "node:test";
import assert from "node:assert/strict";
import { MODE } from "../src/infra/mode.js";
import { solveInjection } from "../src/infra/inject.js";
import { selectorToMatcher, compile, parseProgram, ruleBody } from "../src/infra/language.js";
import { summarizeCosine } from "../src/infra/build.js";

function tensor(name, shape) {
  return [name, { name, shape, dtype: "float16", format: "f16", shard: 0, nbytes: 2, byteOffset: 0 }];
}

function qwenInfo() {
  const tensors = new Map([
    tensor("model.embed_tokens.weight", [32, 8]),
  ]);
  for (let l = 0; l < 24; l++) {
    tensors.set(...tensor(`model.layers.${l}.self_attn.qkv_proj.bias`, [16]));
    tensors.set(...tensor(`model.layers.${l}.self_attn.o_proj.weight`, [8, 8]));
    tensors.set(...tensor(`model.layers.${l}.mlp.down_proj.weight`, [8, 16]));
    tensors.set(...tensor(`model.layers.${l}.mlp.gate_up_proj.weight`, [16, 8]));
    tensors.set(...tensor(`model.layers.${l}.input_layernorm.weight`, [8]));
    tensors.set(...tensor(`model.layers.${l}.post_attention_layernorm.weight`, [8]));
  }
  return {
    rec: { model_id: "Qwen2.5-0.5B-Instruct-q0f16-MLC" },
    tensors,
    config: { model_config: { num_hidden_layers: 24, num_attention_heads: 14, num_key_value_heads: 2, head_dim: 64, hidden_size: 896 } },
  };
}

test("selectors pick the tensors the language reference describes", () => {
  const match = (sel, name) => selectorToMatcher(sel)(name);
  assert.equal(match("*", "embed_tokens.weight"), true);
  assert.equal(match("embed", "embed_tokens.weight"), true);
  assert.equal(match("layers.10-12", "layers.10.mlp.down_proj.weight"), true);
  assert.equal(match("layers.10-12", "layers.9.mlp.down_proj.weight"), false);
  assert.equal(match("layers.10-12", "layers.13.mlp.down_proj.weight"), false);
  assert.equal(match("layers.*.mlp", "layers.3.mlp.down_proj.weight"), true);
  assert.equal(match("layers.*.attn.o_proj", "layers.3.self_attn.o_proj.weight"), true);
  assert.equal(match("layers.0-5.*norm", "layers.3.input_layernorm.weight"), true);
  assert.equal(match("layers.0-5.*norm", "layers.3.post_attention_layernorm.weight"), true);
  assert.equal(match("layers.0-5.*norm", "layers.8.input_layernorm.weight"), false);
});

test("parse rejects an unknown op and a missing argument", () => {
  assert.match(parseProgram("layers.0 frobnicate").errors[0], /unknown op/);
  assert.match(parseProgram("layers.0 scale").errors[0], /needs 1/);
  assert.equal(parseProgram("# only a comment\n\n").rules.length, 0);
});

test("compile plans inject, steer, skip, heads, and mix the way the kernels expect", () => {
  const info = qwenInfo();
  const donor = { ...info, rec: { model_id: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC" }, tensors: info.tensors };

  const inj = compile("layers.14-19 inject wedding 16", info, null);
  assert.deepEqual(inj.errors, []);
  assert.equal(inj.injects.length, 1);
  assert.equal(inj.injects[0].layers.length, 6);
  for (const [name, ops] of inj.plan) {
    assert.match(name, /qkv_proj\.bias$/);
    assert.equal(ops.length, 1);
    assert.equal(ops[0].mode, MODE.addvec);
  }

  const steer = compile("layers.6-11.mlp.down_proj steer diwali 1.2", info, null);
  assert.deepEqual(steer.errors, []);
  assert.deepEqual(steer.steers[0].layers, [6, 7, 8, 9, 10, 11]);
  assert.equal([...steer.plan.values()][0][0].mode, MODE.rank1);
  assert.equal([...steer.plan.values()][0][0].a, 1.2);

  const skip = compile("layers.10-12 skip", info, null);
  assert.deepEqual(skip.errors, []);
  assert.equal(skip.plan.size, 6);
  for (const name of skip.plan.keys()) assert.match(name, /(o_proj|down_proj)\.weight$/);

  const heads = compile("layers.*.attn.o_proj zero-heads 0-6", info, null);
  assert.deepEqual(heads.errors, []);
  const op = [...heads.plan.values()][0][0];
  assert.equal(op.mode, MODE.zerocols);
  assert.equal(op.c0, 0);
  assert.equal(op.c1, 7 * 64);

  const bare = compile("* mix 0.5", info, null);
  assert.match(bare.errors[0], /needs a donor/);
  const mixed = compile("* mix 0.5", info, donor);
  assert.deepEqual(mixed.errors, []);
  assert.equal(mixed.plan.size, info.tensors.size);
  assert.equal([...mixed.plan.values()][0][0].mode, MODE.mix);
});

test("ruleBody ignores comments when two phrasings name one edit", () => {
  assert.equal(ruleBody("# note\nlayers.14-19 inject wedding 16"), "layers.14-19 inject wedding 16");
});

test("injection recovers a vector the grouped output projection can express", () => {
  const hidden = 2, nH = 2, nKV = 2, hd = 1;
  const O = new Float64Array([1, 0, 0, 1]);
  const { x, frac } = solveInjection(O, hidden, nH, nKV, hd, new Float64Array([3, 4]));
  assert.ok(Math.abs(x[0] - 3) < 1e-3);
  assert.ok(Math.abs(x[1] - 4) < 1e-3);
  assert.ok(frac > 0.99);
});

test("cosine summary separates matrices from norms and biases", () => {
  const analysis = new Map([
    ["model.layers.0.mlp.down_proj.weight", { cos: 0.2 }],
    ["model.layers.0.input_layernorm.weight", { cos: 0.99 }],
    ["model.layers.0.self_attn.qkv_proj.bias", { cos: 0.97 }],
  ]);
  const sum = summarizeCosine(analysis);
  assert.equal(sum.matrix, 0.2);
  assert.equal(sum.nMat, 1);
  assert.ok(Math.abs(sum.norm - 0.98) < 1e-9);
});
