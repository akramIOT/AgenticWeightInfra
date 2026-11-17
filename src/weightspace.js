// Stable entry used by the playground. The implementation lives in ./infra.
export {
  SCOPE_MODEL, SCOPE_CONFIG, modelBaseUrl,
  ensureCached, isCached, loadModelInfo,
  parseProgram, compile, ruleBody,
  patchedRecord, buildPatched, analyzePair,
  tokenIdOf, modelName, _internal,
} from "./infra/index.js";
