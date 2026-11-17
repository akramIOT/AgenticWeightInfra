export { MODE } from "./mode.js";
export { hashStr } from "./hash.js";
export { encOf, decode } from "./codec.js";
export {
  SCOPE_MODEL, SCOPE_CONFIG, modelBaseUrl, ensureCached, isCached, loadModelInfo, ShardReader,
} from "./cache.js";
export {
  shortName, selectorToMatcher, countMatches, ruleBody, parseProgram, compile,
} from "./language.js";
export { tokenIdOf, solveInjection, resolveInjects } from "./inject.js";
export { modelName, resolveSteers } from "./steer.js";
export { patchedRecord, buildPatched, analyzePair, summarizeCosine } from "./build.js";

import { selectorToMatcher } from "./language.js";
import { decode } from "./codec.js";
import { solveInjection } from "./inject.js";

export const _internal = { selectorToMatcher, decode, solveInjection };
