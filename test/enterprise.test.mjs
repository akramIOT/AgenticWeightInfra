import test from "node:test";
import assert from "node:assert/strict";
import { compile } from "../src/infra/language.js";
import { evaluateProgram, gateEdit } from "../src/enterprise/policy.js";
import { createAuditLog, createRegistry } from "../src/enterprise/store.js";

function model() {
  const tensors = new Map();
  const add = (name, shape) => tensors.set(name, { name, shape, dtype: "float16", format: "f16" });
  add("model.embed_tokens.weight", [32, 8]);
  for (let l = 0; l < 24; l++) {
    add(`model.layers.${l}.self_attn.qkv_proj.bias`, [8]);
    add(`model.layers.${l}.self_attn.o_proj.weight`, [8, 8]);
    add(`model.layers.${l}.mlp.down_proj.weight`, [8, 16]);
    add(`model.layers.${l}.mlp.gate_up_proj.weight`, [16, 8]);
    add(`model.layers.${l}.input_layernorm.weight`, [8]);
  }
  return {
    tensors,
    config: { model_config: { num_hidden_layers: 24, num_attention_heads: 14, num_key_value_heads: 2, head_dim: 64, hidden_size: 64 } },
  };
}

const info = model();
const donor = { ...info, tensors: info.tensors };
const judge = (text, donorInfo = null) => evaluateProgram(text, compile(text, info, donorInfo), info);
const codes = (p) => p.findings.map((f) => f.code);

test("a small concept injection is routine", () => {
  const p = judge("layers.14-19 inject wedding 16");
  assert.equal(p.decision, "allow");
  assert.equal(p.version, "2026.1");
});

test("strength past the reference cliffs is review, and past the ceiling is denied", () => {
  assert.equal(judge("layers.14-19 inject wedding 22").decision, "review");
  assert.equal(judge("layers.14-19 inject wedding 30").decision, "deny");
  assert.equal(judge("layers.6-11.mlp.down_proj steer diwali 1.2").decision, "allow");
  assert.equal(judge("layers.6-11.mlp.down_proj steer diwali 2").decision, "review");
  assert.equal(judge("layers.6-11.mlp.down_proj steer diwali 3").decision, "deny");
  const mild = judge("layers.* noise 0.05");
  assert.equal(mild.decision, "review");
  assert.deepEqual(codes(mild), ["breadth"]);
  assert.ok(codes(judge("layers.* noise 0.2")).includes("noise"));
  assert.equal(judge("layers.* noise 0.5").decision, "deny");
});

test("a full-model midpoint mix and an early skip need approval; a wipe is denied", () => {
  const mix = judge("* mix 0.5", donor);
  assert.equal(mix.decision, "review");
  assert.ok(codes(mix).includes("mix-midpoint"));
  assert.equal(judge("* mix 2", donor).decision, "deny");
  assert.equal(judge("* zero").decision, "deny");
  assert.equal(judge("embed zero").decision, "deny");
  assert.equal(judge("layers.4-6 skip").decision, "review");
  assert.equal(judge("layers.10-12 skip").decision, "allow");
  assert.equal(judge("layers.*.attn.o_proj zero-heads 0-6").decision, "allow");
  assert.equal(judge("layers.*.attn.o_proj zero-heads 0-10").decision, "review");
  assert.equal(judge("layers.*.mlp.down_proj scale 1.3").decision, "allow");
  assert.equal(judge("layers.*.mlp.down_proj scale 0").decision, "deny");
});

test("the gate requires an operator, a reason, and approval when the policy says review", () => {
  const allow = judge("layers.10-12 skip");
  assert.equal(gateEdit({ policy: allow, change: { operator: "A", reason: "short" } }).ok, false);
  assert.match(gateEdit({ policy: allow, change: { operator: "", reason: "layer skip check" } }).blockReason, /operator/);
  assert.equal(gateEdit({ policy: allow, change: { operator: "ada", reason: "layer skip check" } }).ok, true);
  assert.match(gateEdit({ policy: allow, change: { operator: "ada", reason: "layer skip check", ticket: "nope" } }).blockReason, /CHG-1042/);
  const review = judge("* mix 0.5", donor);
  const change = { operator: "ada", reason: "show the merge cliff", ticket: "CHG-1042" };
  assert.equal(gateEdit({ policy: review, change, approved: false }).ok, false);
  assert.equal(gateEdit({ policy: review, change, approved: true }).ok, true);
  assert.match(gateEdit({ policy: judge("* zero"), change }).blockReason, /does not allow/);
});

test("the audit log keeps a clear event and the registry replaces a program by name", () => {
  const mem = new Map();
  const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
  const audit = createAuditLog(storage);
  audit.record({ action: "build", decision: "allow", blocked: false });
  audit.clear({ operator: "ada" });
  const rows = audit.list();
  assert.equal(rows.length, 1);
  assert.equal(rows[0].action, "clear");
  const registry = createRegistry(storage);
  registry.save({ name: "skip-middle", modelId: "qwen", program: "layers.10-12 skip", programHash: "abc", decision: "allow" });
  registry.save({ name: "skip-middle", modelId: "qwen", program: "layers.10-12 skip\n", programHash: "abd", decision: "allow" });
  assert.equal(registry.list().length, 1);
  assert.equal(registry.get("skip-middle").programHash, "abd");
});

test("a full browser store reports a local storage failure", () => {
  const storage = {
    getItem: () => "[]",
    setItem: () => { const e = new Error("quota"); e.name = "QuotaExceededError"; throw e; },
  };
  assert.throws(() => createAuditLog(storage).record({ action: "build" }), /refused to store more local records/);
});
