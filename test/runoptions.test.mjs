import assert from "node:assert/strict";
import test from "node:test";
import { readRunOptions } from "../src/runoptions.js";

test("a run needs a prompt and a token budget inside 8..512", () => {
  assert.deepEqual(readRunOptions("  Paris?\n\nHaiku  ", "80"), { prompts: ["Paris?", "Haiku"], maxTokens: 80 });
  assert.throws(() => readRunOptions("   \n", 80), /at least one prompt/);
  assert.throws(() => readRunOptions("Paris?", 0), /between 8 and 512/);
  assert.throws(() => readRunOptions("Paris?", 800), /between 8 and 512/);
  assert.throws(() => readRunOptions("Paris?", "nope"), /between 8 and 512/);
});
