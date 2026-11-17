// Control policy for a weight edit. Thresholds match the reference runs:
// noise is routine at 5%, review above 12%, denied above 30%;
// concept injection is review above 20 and denied above 24;
// a steer is review above 1.4 and denied above 2;
// a full-model midpoint mix and an early-layer skip are review, not routine.

import { parseProgram } from "../infra/language.js";

export const POLICY_VERSION = "2026.1";

const RANK = { allow: 0, review: 1, deny: 2 };

export function evaluateProgram(text, compiled, info, environment = "lab") {
  const env = environment || "lab";
  const findings = [];
  if (env !== "lab" && env !== "controlled") {
    findings.push({ level: "deny", code: "environment", message: `Control environment "${env}" is not lab or controlled.` });
    return finish(findings, compiled, env);
  }
  if (compiled.errors?.length) {
    for (const message of compiled.errors) findings.push({ level: "deny", code: "compile", message });
    return finish(findings, compiled, env);
  }
  const { rules } = parseProgram(text);
  const n = info?.config?.model_config?.num_hidden_layers ?? 1;
  const nH = info?.config?.model_config?.num_attention_heads ?? 1;
  for (const r of rules) judgeRule(r, findings, { n, nH, compiled });
  const fraction = fractionOf(compiled);
  if (fraction > 0.5 && !rules.some((r) => r.op === "zero" && (r.sel === "*" || r.sel === "all"))) {
    findings.push({
      level: "review",
      code: "breadth",
      message: `This edit touches ${(fraction * 100).toFixed(1)}% of parameters, above the 50% routine ceiling.`,
    });
  }
  if (env === "controlled" && rules.some((r) => r.op === "inject" || r.op === "steer" || r.op === "noise")) {
    findings.push({
      level: "review",
      code: "controlled",
      message: "A controlled environment requires approval before injection, steering, or noise is applied.",
    });
  }
  return finish(findings, compiled, env);
}

export function gateEdit({ policy, change = {}, approved = false, release = null }) {
  if (!policy) return { ok: false, blockReason: "Control policy has not been evaluated." };
  if (policy.decision === "deny") {
    const why = policy.findings.filter((f) => f.level === "deny").map((f) => f.message).join(" ");
    return { ok: false, blockReason: why || "The control policy denied this edit." };
  }
  const operator = String(change.operator ?? "").trim();
  const approver = String(change.approver ?? "").trim();
  const reason = String(change.reason ?? "").trim();
  const ticket = String(change.ticket ?? "").trim();
  if (operator.length < 2) return { ok: false, blockReason: "Record the operator name before applying an edit." };
  if (reason.length < 8) return { ok: false, blockReason: "Record a change reason of at least 8 characters." };
  if (ticket && !/^[A-Za-z][A-Za-z0-9]*-\d+$/.test(ticket)) return { ok: false, blockReason: "Ticket must look like CHG-1042." };
  if (policy.decision === "review") {
    if (!ticket) return { ok: false, blockReason: "A review-level edit needs a ticket like CHG-1042." };
    if (!approved) {
      return { ok: false, blockReason: "This edit is above the routine threshold. Approve it explicitly. The approval is written to the audit log." };
    }
    if (approver.length < 2) return { ok: false, blockReason: "A review-level edit needs an approver who is not the operator." };
    if (approver.toLowerCase() === operator.toLowerCase()) {
      return { ok: false, blockReason: "The approver must be a different person from the operator." };
    }
  }
  if ((policy.environment ?? "lab") === "controlled") {
    const hash = String(change.programHash ?? "");
    const modelId = String(change.modelId ?? "");
    const promoted = release
      && release.status === "approved"
      && release.programHash === hash
      && (!modelId || !release.modelId || release.modelId === modelId);
    if (!promoted) {
      return { ok: false, blockReason: "A controlled environment runs only a program a second person has approved." };
    }
  }
  return { ok: true };
}

function judgeRule(r, findings, { n, nH, compiled }) {
  const num = (i) => parseFloat(r.args[i]);
  if (r.op === "zero" && (r.sel === "*" || r.sel === "all" || r.sel === "embed")) {
    findings.push({ level: "deny", code: "wipe", message: `Zeroing "${r.sel}" removes a part of the model the control policy does not allow.` });
  }
  if (r.op === "mix" && (r.sel === "*" || r.sel === "all")) {
    const a = num(0);
    if (a < 0 || a > 1) findings.push({ level: "deny", code: "mix-extrapolate", message: `A full-model mix at ${a} leaves the segment between the two models.` });
    else if (a !== 0 && a !== 1) findings.push({ level: "review", code: "mix-midpoint", message: `A full-model mix at ${a} is the merge cliff. It needs an explicit approval.` });
    else if (a === 1) findings.push({ level: "review", code: "mix-swap", message: "A full swap replaces every tensor of the base model. It needs an explicit approval." });
  }
  if (r.op === "noise") {
    const s = num(0);
    if (s > 0.3) findings.push({ level: "deny", code: "noise", message: `Noise ${s} is past the point where the reference run becomes noise. The ceiling is 0.3.` });
    else if (s > 0.12) findings.push({ level: "review", code: "noise", message: `Noise ${s} is above the 12% drift point. It needs an explicit approval.` });
  }
  if (r.op === "inject") {
    const c = num(1);
    if (c > 24) findings.push({ level: "deny", code: "inject", message: `Injection strength ${c} is past the repetition regime. The ceiling is 24.` });
    else if (c > 20) findings.push({ level: "review", code: "inject", message: `Injection strength ${c} is above the routine cap of 20.` });
  }
  if (r.op === "steer") {
    const c = num(1);
    if (c > 2) findings.push({ level: "deny", code: "steer", message: `Steer strength ${c} is past the fluency cliff. The ceiling is 2.` });
    else if (c > 1.4) findings.push({ level: "review", code: "steer", message: `Steer strength ${c} is above 1.4, where the reference run loses fluency and facts.` });
  }
  if (r.op === "scale") {
    const k = num(0);
    if (!(k > 0) || k > 3) findings.push({ level: "deny", code: "scale", message: `Scale ${k} is outside the allowed range of (0, 3].` });
    else if (k < 0.5 || k > 1.5) findings.push({ level: "review", code: "scale", message: `Scale ${k} is outside the routine band of 0.5 to 1.5.` });
  }
  if (r.op === "skip") {
    const min = minLayer(compiled, /(o_proj|down_proj)\.weight$/);
    if (Number.isFinite(min) && min / n < 0.25) {
      findings.push({ level: "review", code: "skip-early", message: `Skipping from layer ${min} hits the early band, which confuses the model in the reference run.` });
    }
  }
  if (r.op === "zero-heads") {
    const m = String(r.args[0] ?? "").match(/^(\d+)(?:-(\d+))?$/);
    if (m) {
      const count = (m[2] ? +m[2] : +m[1]) - +m[1] + 1;
      if (count > nH / 2) findings.push({ level: "review", code: "heads", message: `Zeroing ${count} of ${nH} heads is more than half the attention block.` });
    }
  }
}

function minLayer(compiled, re) {
  let min = Infinity;
  for (const name of compiled.plan?.keys?.() ?? []) {
    if (!re.test(name)) continue;
    const m = name.match(/layers\.(\d+)\./);
    if (m) min = Math.min(min, +m[1]);
  }
  return min;
}

function fractionOf(compiled) {
  const total = compiled.stats?.total ?? 0;
  if (!total) return 0;
  return compiled.stats.touched / total;
}

function finish(findings, compiled, environment) {
  const decision = findings.reduce((d, f) => (RANK[f.level] > RANK[d] ? f.level : d), "allow");
  return { version: POLICY_VERSION, environment, decision, findings, fraction: fractionOf(compiled) };
}
