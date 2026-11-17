// Observe → diagnose → edit → test.
//
// The host owns the GPU. This loop only decides the program, asks the host to
// map tensors or to build-and-probe, and revises the coefficient from the score.

import { compile, ruleBody } from "../infra/language.js";
import { capabilities, propose, revise, holds } from "./policy.js";
import { scoreRows, scoreSummary } from "./rubric.js";

export async function runAgentSession(ctx) {
  const trace = [];
  const note = (event) => {
    trace.push(event);
    ctx.onTrace?.(event);
    if (event.summary) ctx.log?.(event.summary);
  };

  const caps = ctx.info ? capabilities(ctx.setup, ctx.info) : null;
  note({ phase: "observe", summary: observeSummary(ctx, caps) });

  let analysis = ctx.analysis ?? null;
  let proposal = propose({ goal: ctx.goal, setup: ctx.setup, info: ctx.info, analysis });
  note({ phase: "diagnose", summary: proposal.rationale });

  if (proposal.needsMap) {
    if (!ctx.analyze) {
      const summary = "The agent will not mix models until it has mapped how far apart their tensors are.";
      note({ phase: "stop", summary });
      return { ok: false, trace, program: null, score: null, rationale: summary };
    }
    analysis = await ctx.analyze();
    proposal = propose({ goal: ctx.goal, setup: ctx.setup, info: ctx.info, analysis });
    note({ phase: "diagnose", summary: proposal.rationale });
  }

  if (proposal.stop || !proposal.program) {
    note({ phase: "stop", summary: proposal.rationale });
    return { ok: false, trace, program: null, score: null, rationale: proposal.rationale };
  }

  const budget = ctx.budget ?? 3;
  const seen = new Set();
  let score = null;
  let program = proposal.program;
  for (let step = 1; step <= budget; step++) {
    const body = ruleBody(proposal.program);
    if (seen.has(body)) {
      note({ phase: "stop", summary: "This edit was already probed. The agent stops." });
      break;
    }
    seen.add(body);
    const compiled = compile(proposal.program, ctx.info, ctx.donorInfo ?? null);
    if (compiled.errors.length) {
      const summary = compiled.errors.join(" ");
      note({ phase: "compile", summary });
      return { ok: false, trace, program: proposal.program, score, rationale: summary };
    }
    program = proposal.program;
    note({
      phase: "edit",
      summary: `step ${step}: ${body} · ${compiled.stats.tensors} tensors`,
      program,
      step,
    });
    ctx.onProgram?.(program);
    let rows;
    try {
      rows = await ctx.probe(program);
    } catch (e) {
      const summary = e.message ?? String(e);
      note({ phase: "stop", summary });
      return { ok: false, trace, program, score, rationale: summary };
    }
    score = scoreRows(rows, proposal.targets ?? {});
    note({ phase: "test", summary: scoreSummary(score), score, step });
    const next = revise(proposal, score);
    if (next.stop || ruleBody(next.program ?? "") === body) {
      note({ phase: "stop", summary: next.rationale });
      return { ok: holds(proposal, score), trace, program, score, rationale: next.rationale };
    }
    note({ phase: "revise", summary: next.rationale });
    proposal = next;
  }
  note({ phase: "stop", summary: "Step budget used up. Keeping the last program that was actually probed." });
  return { ok: false, trace, program, score, rationale: "Step budget used up." };
}

function observeSummary(ctx, caps) {
  if (!caps) return "Weights are not in the cache yet.";
  const packs = caps.packs.map((p) => p.concept).join(", ") || "none";
  const security = ctx.security?.summary ? ` Security: ${ctx.security.summary}` : "";
  return `${ctx.setup.label ?? ctx.setup.base}: ${caps.layers} layers, ${caps.heads} heads, value-bias inject ${caps.inject ? "yes" : "no"}, donor ${caps.donor ? "yes" : "no"}, steering packs: ${packs}.${security}`;
}
