// Tool surface for an external driver. Pure tools run here. The GPU tools are
// named so a driver knows it must supply them; this module does not pretend to
// edit weights without a host.

import { compile } from "../infra/language.js";
import { attestEdited, attestLoaded } from "../security/attest.js";
import { capabilities, propose, revise } from "./policy.js";
import { scoreRows } from "./rubric.js";
import { runAgentSession } from "./session.js";

export const TOOLS = [
  {
    name: "observe_model",
    description: "Report layer count, whether value-bias injection is possible, whether a donor is loaded, and which steering packs ship for this model.",
    parameters: {},
  },
  {
    name: "compile_program",
    description: "Compile a weight program against the loaded tensors. Returns errors and which tensors the plan touches. Does not edit weights.",
    parameters: { program: "string" },
  },
  {
    name: "propose_edit",
    description: "Choose a weight program for a goal from model capabilities and, when present, the donor cosine map.",
    parameters: { goal: "string" },
  },
  {
    name: "revise_edit",
    description: "Given the previous proposal and a probe score, lower or raise the coefficient, or stop.",
    parameters: { proposal: "object", score: "object" },
  },
  {
    name: "score_answers",
    description: "Score original-versus-edited answers for concept hits, the Paris fact, repetition collapse, and fluency.",
    parameters: { rows: "array", words: "string[]" },
  },
  {
    name: "run_session",
    description: "Run observe, diagnose, edit, and test. The host context must provide probe(program) and, for a safe merge, analyze().",
    parameters: { goal: "string", budget: "number" },
  },
  {
    name: "attest_weights",
    description: "Scan cached tensors after load, or check a patched model against the compiled plan. The host provides readShard for a load, and the patched model plus shard readers for an edit.",
    parameters: { phase: "load | edit" },
  },
];

export const HOST_TOOLS = [
  {
    name: "map_weight_space",
    description: "GPU cosine similarity of every shared tensor between base and donor. Required before a safe merge.",
  },
  {
    name: "build_and_probe",
    description: "Apply the weight program with the WebGPU kernels and greedily decode the prompts on the original model and the edited model.",
  },
];

export async function dispatchTool(name, args, ctx) {
  switch (name) {
    case "observe_model": {
      if (!ctx.info) return { error: "Weights are not loaded." };
      const caps = capabilities(ctx.setup, ctx.info);
      return {
        base: ctx.setup.base,
        layers: caps.layers,
        heads: caps.heads,
        inject: caps.inject,
        donor: caps.donor,
        packs: caps.packs.map((p) => p.concept),
      };
    }
    case "compile_program": {
      const compiled = compile(args.program, ctx.info, ctx.donorInfo ?? null);
      return {
        errors: compiled.errors,
        stats: compiled.stats,
        tensors: [...compiled.plan.keys()],
      };
    }
    case "propose_edit":
      return propose({ goal: args.goal, setup: ctx.setup, info: ctx.info, analysis: ctx.analysis ?? null });
    case "revise_edit":
      return revise(args.proposal, args.score);
    case "score_answers":
      return scoreRows(args.rows, { words: args.words ?? [] });
    case "run_session":
      return runAgentSession({ ...ctx, goal: args.goal, budget: args.budget ?? ctx.budget });
    case "attest_weights": {
      if (args.phase === "load") {
        if (!ctx.info) return { error: "Weights are not loaded." };
        if (!ctx.readShard) return { error: "Host must provide readShard(shardIndex) for the cached bytes." };
        return attestLoaded(ctx.info, ctx.readShard);
      }
      if (args.phase === "edit") {
        if (!ctx.info || !ctx.patchedInfo || !ctx.compiled) return { error: "Host must provide the loaded model, the patched model, and the compiled plan." };
        if (!ctx.readBaseShard || !ctx.readPatchedShard) return { error: "Host must provide readers for the base and patched shards." };
        return attestEdited({
          baseInfo: ctx.info,
          patchedInfo: ctx.patchedInfo,
          compiled: ctx.compiled,
          readBaseShard: ctx.readBaseShard,
          readPatchedShard: ctx.readPatchedShard,
          readDonor: ctx.readDonor ?? null,
          readVector: ctx.readVector ?? null,
        });
      }
      return { error: "phase must be load or edit." };
    }
    default:
      throw new Error(`Unknown tool "${name}". GPU tools (${HOST_TOOLS.map((t) => t.name).join(", ")}) are provided by the host.`);
  }
}
