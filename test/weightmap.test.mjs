import test from "node:test";
import assert from "node:assert/strict";
import { compile } from "../src/infra/language.js";
import { ALL_SETUPS } from "../src/catalog.js";
import { schematicInfo, mapRows } from "../src/weightmap.js";

const setup = (id) => ALL_SETUPS.find((s) => s.id === id);
const hits = (rows) => rows.flatMap((row) => row.cells.filter((c) => c?.ops).map((c) => `${row.label}:${c.name}`));

test("the Qwen wedding program outlines only the value biases in layers 14-19", () => {
  const info = schematicInfo(setup("qwen"));
  const plan = compile("layers.14-19 inject wedding 16", info, null).plan;
  const { rows, uncovered } = mapRows(info, plan);
  const marked = hits(rows);
  assert.equal(uncovered.length, 0);
  assert.equal(marked.length, 6);
  for (let l = 14; l <= 19; l++) {
    assert.ok(marked.includes(`L${l}:model.layers.${l}.self_attn.c_attn.bias`));
  }
  assert.equal(marked.some((m) => m.includes(".weight")), false);
});

test("a Llama middle skip outlines o_proj and down_proj, and a full mix includes the final norm", () => {
  const llama = schematicInfo(setup("llama1b"));
  const skip = compile("layers.7-8 skip", llama, null).plan;
  const marked = hits(mapRows(llama, skip).rows);
  assert.deepEqual(marked.sort(), [
    "L7:model.layers.7.mlp.down_proj.weight",
    "L7:model.layers.7.self_attn.o_proj.weight",
    "L8:model.layers.8.mlp.down_proj.weight",
    "L8:model.layers.8.self_attn.o_proj.weight",
  ].sort());
  assert.equal(llama.tensors.has("model.layers.0.self_attn.qkv_proj.bias"), false);

  const qwen = schematicInfo(setup("qwen-pair"));
  const mix = compile("* mix 0.5", qwen, qwen).plan;
  const placed = mapRows(qwen, mix);
  assert.equal(placed.uncovered.length, 0);
  assert.equal(placed.rows.at(-1).cells[0].ops?.length, 1);
  assert.equal(placed.rows[0].cells[0].name, "model.embed_tokens.weight");
  assert.ok(placed.rows[0].cells[0].ops);
});
