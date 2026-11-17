import assert from "node:assert/strict";
import test from "node:test";
import { cacheEntryComplete, cachedByteLength } from "../src/infra/cache.js";

test("a cached shard is complete only at the manifest length", () => {
  const tagged = { headers: new Headers({ "x-awi-bytes": "4096", "content-length": "100" }) };
  assert.equal(cachedByteLength(tagged.headers), 4096);
  assert.equal(cacheEntryComplete(tagged, 4096), true);
  assert.equal(cacheEntryComplete(tagged, 100), false);
  assert.equal(cacheEntryComplete({ headers: new Headers({ "content-length": "4096" }) }, 4096), true);
  assert.equal(cacheEntryComplete({ headers: new Headers() }, 4096), false);
  assert.equal(cacheEntryComplete(null, 4096), false);
  assert.equal(cacheEntryComplete({ headers: new Headers() }, null), true);
});
