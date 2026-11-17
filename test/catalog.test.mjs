import assert from "node:assert/strict";
import test from "node:test";
import { presetsFor, setupsFor } from "../src/catalog.js";

test("the Smol tab keeps both SmolLM models and the Playground keeps Qwen", () => {
  const smol = setupsFor("smol").map((s) => s.id);
  const main = setupsFor("main").map((s) => s.id);
  assert.deepEqual(smol, ["smol135", "smol360"]);
  assert.deepEqual(main, ["qwen-pair", "qwen"]);
  assert.equal(setupsFor("llama").length, 1);
});

test("Smol presets use the same relative bands as the agent", () => {
  const small = setupsFor("smol")[0];
  const large = setupsFor("smol")[1];
  const a = presetsFor("smol", small);
  const b = presetsFor("smol", large);
  assert.match(a[0].text, /layers\.13-15 skip/);
  assert.match(a[1].text, /layers\.5-8 skip/);
  assert.match(a[2].text, /zero-heads 0-3/);
  assert.match(b[0].text, /layers\.13-16 skip/);
  assert.match(b[1].text, /layers\.5-8 skip/);
  assert.match(b[2].text, /zero-heads 0-6/);
  assert.equal(presetsFor("llama").length, 5);
  assert.equal(presetsFor("main")[0].text.includes("inject"), true);
});
