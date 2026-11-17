import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.87";
import { WeightKernels } from "./kernels.js";
import {
  ensureCached, isCached, loadModelInfo, compile, buildPatched, analyzePair,
} from "./weightspace.js";
import { setupsFor, presetsFor, DEFAULT_PROMPTS } from "./catalog.js";
import { KINDS, schematicInfo, mapRows } from "./weightmap.js";
import { runAgentSession } from "./agent/session.js";
import { hashStr } from "./infra/hash.js";
import { ruleBody } from "./infra/language.js";
import { ShardReader } from "./infra/cache.js";
import { resolveInjects } from "./infra/inject.js";
import { resolveSteers } from "./infra/steer.js";
import { MODE } from "./infra/mode.js";
import { evaluateProgram, gateEdit } from "./enterprise/policy.js";
import { createAuditLog, createRegistry } from "./enterprise/store.js";
import { attestLoaded, attestEdited, blockReason, SECURITY_VERSION } from "./security/attest.js";
import { readRunOptions } from "./runoptions.js";
import { BUILD } from "./build.js";

document.documentElement.dataset.awi = "1";

const rec = (id) => {
  const r = webllm.prebuiltAppConfig.model_list.find((m) => m.model_id === id);
  if (!r) throw new Error(`WebLLM has no prebuilt model ${id}`);
  return r;
};

const PAGE = document.body.dataset.page ?? "main";
const SETUPS = setupsFor(PAGE);
const presets = () => presetsFor(PAGE, S.setup);

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? (e.className = v) : k.startsWith("on") ? e.addEventListener(k.slice(2), v) : e.setAttribute(k, v);
  kids.flat().forEach((k) => e.append(k));
  return e;
};
const logBox = $("#log");
function log(msg, cls = "") {
  const line = el("div", { class: cls }, `${new Date().toLocaleTimeString()}  ${msg}`);
  logBox.prepend(line);
}
function setProgress(id, frac, label) {
  const bar = $(id);
  bar.hidden = frac == null;
  if (frac != null) { bar.querySelector("i").style.width = `${(frac * 100).toFixed(1)}%`; bar.querySelector("span").textContent = label ?? `${(frac * 100).toFixed(0)}%`; }
}

const S = {
  setup: PAGE === "main" ? SETUPS[1] : SETUPS[0],
  kernels: null, info: null, donorInfo: null,
  analysis: null, compiled: null, policy: null, approvalHash: null,
  security: null, donorSecurity: null, editSecurity: null,
  scanGen: 0, scanningFor: null, scanFrac: null, scanLabel: "cached weights",
  downloadAbort: null,
  audit: null, registry: null,
  hasF16: false,
  engine: null, loadedId: null, busy: false, baseMemo: new Map(),
};

async function init() {
  const setupSel = $("#setup");
  SETUPS.forEach((s) => setupSel.append(el("option", { value: s.id }, s.label)));
  $("#prompts").value = DEFAULT_PROMPTS;

  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("m") && SETUPS.some((s) => s.id === h.get("m"))) S.setup = SETUPS.find((s) => s.id === h.get("m"));
  setupSel.value = S.setup.id;
  const unb64 = (x) => { try { return decodeURIComponent(escape(atob(x))); } catch { return null; } };
  const sharedProgram = h.get("p") ? unb64(h.get("p")) : null;
  if (h.get("p") && sharedProgram == null) log("The share link is damaged. The default program is loaded.", "err");
  $("#program").value = sharedProgram ?? presets()[0].text;
  const sharedPrompts = h.get("q") ? unb64(h.get("q")) : null;
  $("#prompts").value = sharedPrompts ?? ((!h.get("p") && presets()[0].prompts) || DEFAULT_PROMPTS);
  if ($("#goal") && h.get("g")) { const g = unb64(h.get("g")); if (g != null) $("#goal").value = g; }
  if ($("#build")) $("#build").textContent = BUILD;
  renderPresets();

  try {
    S.kernels = await WeightKernels.create();
    S.hasF16 = S.kernels.adapter.features.has("shader-f16");
    const info = S.kernels.adapter.info ?? {};
    $("#gpu").textContent = `WebGPU ✓ ${[info.vendor, info.architecture].filter(Boolean).join(" ")}${S.hasF16 ? " · shader-f16 ✓" : " · no shader-f16"}`;
    $("#gpu").className = "chip ok";
    S.kernels.device.lost.then((info) => {
      if (S.kernels?.device !== info.device && info.device) return;
      S.kernels = null;
      $("#gpu").textContent = "WebGPU device lost";
      $("#gpu").className = "chip bad";
      log(info?.message || "The WebGPU device was lost. Reload the page before editing weights.", "err");
      refreshButtons();
    });
  } catch (e) {
    $("#gpu").textContent = "WebGPU unavailable";
    $("#gpu").className = "chip bad";
    log(e.message, "err");
  }

  S.audit = createAuditLog(localStorage);
  S.registry = createRegistry(localStorage);
  const operator = $("#operator");
  if (operator) {
    try { operator.value = localStorage.getItem("awi.operator") || ""; }
    catch { operator.value = ""; }
    operator.addEventListener("input", () => {
      try { localStorage.setItem("awi.operator", operator.value); }
      catch { log("The operator name could not be saved in this browser.", "err"); }
      renderControl();
    });
  }
  const environment = $("#environment");
  if (environment) {
    try { environment.value = localStorage.getItem("awi.environment") === "controlled" ? "controlled" : "lab"; }
    catch { environment.value = "lab"; }
    environment.addEventListener("change", () => {
      try { localStorage.setItem("awi.environment", environment.value); }
      catch { log("The control environment could not be saved in this browser.", "err"); }
      renderControl();
    });
  }
  $("#approver")?.addEventListener("input", renderControl);
  $("#ticket")?.addEventListener("input", renderControl);
  $("#reason")?.addEventListener("input", renderControl);
  $("#approve")?.addEventListener("change", () => {
    S.approvalHash = $("#approve").checked ? hashStr(ruleBody($("#program").value)) : null;
    renderControl();
  });
  $("#saveprog")?.addEventListener("click", saveProgram);
  $("#approvprog")?.addEventListener("click", approveProgram);
  $("#retireprog")?.addEventListener("click", retireProgram);
  $("#verifyaudit")?.addEventListener("click", verifyAudit);
  $("#registry")?.addEventListener("change", loadProgram);
  $("#exportaudit")?.addEventListener("click", exportAudit);
  $("#clearaudit")?.addEventListener("click", clearAudit);
  renderRegistry();
  renderAudit();

  setupSel.addEventListener("change", () => {
    const previous = presets();
    const idx = previous.findIndex((p) => p.text === $("#program").value);
    S.setup = SETUPS.find((s) => s.id === setupSel.value);
    S.info = S.donorInfo = S.analysis = null;
    S.security = S.donorSecurity = S.editSecurity = null;
    S.scanningFor = null;
    S.scanGen++;
    renderPresets();
    if (idx >= 0) {
      const next = presets()[idx];
      $("#program").value = next.text;
      if (next.prompts) $("#prompts").value = next.prompts;
    }
    refresh();
  });
  $("#preset").addEventListener("change", (e) => { const p = presets()[+e.target.value]; if (p) { $("#program").value = p.text; if (p.prompts) $("#prompts").value = p.prompts; recompile(); } });
  $("#program").addEventListener("input", debounce(recompile, 250));
  $("#download").addEventListener("click", () => {
    if (S.downloadAbort) { S.downloadAbort.abort(); return; }
    guard(download);
  });
  $("#analyze").addEventListener("click", () => guard(analyze));
  $("#run").addEventListener("click", () => guard(buildAndCompare));
  $("#agent")?.addEventListener("click", () => guard(runGoal));
  $("#share").addEventListener("click", share);
  window.addEventListener("unhandledrejection", (ev) => {
    log(ev.reason?.message ?? String(ev.reason), "err");
  });
  try { await refresh(); }
  catch (e) { log(e.message ?? String(e), "err"); }
}

function renderPresets() {
  const sel = $("#preset");
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "Load a preset program…"));
  presets().forEach((p, i) => {
    const ok = (p.needs !== "donor" || S.setup.donor) && (p.needs !== "bias" || S.setup.base.startsWith("Qwen"));
    const o = el("option", { value: i }, p.name + (ok ? "" : " (not for this model)"));
    if (!ok) o.disabled = true;
    sel.append(o);
  });
}

const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

async function guard(fn) {
  if (S.busy) return;
  S.busy = true;
  document.body.classList.add("busy");
  refreshButtons();
  try { await fn(); } catch (e) { log(e.message ?? String(e), "err"); console.error(e); }
  finally { S.busy = false; document.body.classList.remove("busy"); setProgress("#prog", null); refreshButtons(); }
}

async function refresh() {
  const ids = [S.setup.base, S.setup.donor].filter(Boolean);
  const cached = await Promise.all(ids.map((id) => isCached(rec(id))));
  const all = cached.every(Boolean);
  $("#dlstate").textContent = all ? "weights cached in this browser ✓" : "weights are not verified in this browser yet";
  if (all && !S.info) {
    try {
      S.info = await loadModelInfo(rec(S.setup.base));
      S.donorInfo = S.setup.donor ? await loadModelInfo(rec(S.setup.donor)) : null;
      S.security = S.donorSecurity = S.editSecurity = null;
      S.scanningFor = null;
    } catch (e) {
      S.info = S.donorInfo = null;
      log(`The cached weights could not be read. ${e.message}`, "err");
    }
  }
  const needsF16 = (rec(S.setup.base).required_features ?? []).includes("shader-f16");
  $("#f16warn").hidden = !(S.kernels && needsF16 && !S.hasF16);
  recompile();
  refreshButtons();
  maybeStartScan();
}

function refreshButtons() {
  const gate = S.info && S.compiled && !S.compiled.errors.length ? controlOf($("#program").value).gate : null;
  const scanned = S.security && S.security.decision !== "deny" && (!S.donorInfo || (S.donorSecurity && S.donorSecurity.decision !== "deny"));
  const downloading = Boolean(S.downloadAbort);
  $("#download").disabled = (S.busy && !downloading) || !!S.info;
  $("#download").textContent = downloading ? "Cancel download" : "Download weights";
  $("#setup").disabled = S.busy;
  $("#preset").disabled = S.busy;
  $("#analyze").disabled = S.busy || !S.info || !S.donorInfo;
  $("#analyze").hidden = !S.setup.donor;
  $("#run").disabled = S.busy || !S.info || !S.kernels || !scanned || !!S.compiled?.errors.length || gate?.ok === false;
  if ($("#agent")) $("#agent").disabled = S.busy || !S.info || !S.kernels || !scanned || !!S.compiled?.errors.length || gate?.ok === false;
}

async function download() {
  const ctrl = new AbortController();
  S.downloadAbort = ctrl;
  refreshButtons();
  try {
    for (const id of [S.setup.base, S.setup.donor].filter(Boolean)) {
      if (ctrl.signal.aborted) throw new Error("Download cancelled.");
      log(`downloading ${id} into the WebLLM cache…`);
      await ensureCached(rec(id), (f) => setProgress("#prog", f, `${id}: ${(f * 100).toFixed(0)}%`), ctrl.signal);
    }
    log("download complete", "ok");
    S.info = null;
    await refresh();
  } finally {
    S.downloadAbort = null;
  }
}

async function analyze() {
  log("comparing every tensor of base vs donor on the GPU…");
  S.analysis = await analyzePair(S.info, S.donorInfo, S.kernels, (f) => setProgress("#prog", f, `mapping weight space ${(f * 100).toFixed(0)}%`));
  const vals = [...S.analysis.values()].map((v) => v.cos);
  log(`map ready: mean cosine ${(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3)} across ${vals.length} tensors`, "ok");
  renderMap();
}

function recompile() {
  S.editSecurity = null;
  if (!S.info) { S.compiled = null; S.policy = null; $("#compile").textContent = "Download the weights to compile programs against them."; renderControl(); renderMap(); renderSecurity(); return; }
  S.compiled = compile($("#program").value, S.info, S.donorInfo);
  const c = S.compiled;
  const box = $("#compile");
  box.innerHTML = "";
  if (c.errors.length) c.errors.forEach((e) => box.append(el("div", { class: "err" }, e)));
  else box.append(el("div", { class: "ok" },
    `✓ ${c.stats.tensors} tensors · ${fmt(c.stats.touched)} of ${fmt(c.stats.total)} parameters (${(100 * c.stats.touched / c.stats.total).toFixed(2)}%)`
    + (c.injects.length ? ` · ${c.injects.reduce((s, i) => s + i.layers.length, 0)} layers get injected value biases` : "")
    + (c.steers.length ? ` · ${c.steers.reduce((s, i) => s + i.layers.length, 0)} MLPs get a rank-one steer` : "")));
  syncApproval();
  renderControl();
  renderMap();
  renderSecurity();
  refreshButtons();
}
const fmt = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n));

const GLYPH = { 0: "M", 1: "×", 2: "∅", 3: "~", 4: "H", 5: "I", 6: "S" };

function cosColor(c) {
  const t = Math.max(0, Math.min(1, c));
  const h = 20 + t * 150, l = 52 - t * 12;
  return `hsl(${h} 70% ${l}%)`;
}

function mapPlan() {
  if (S.info) return { info: S.info, plan: S.compiled?.plan ?? new Map(), live: true };
  const info = schematicInfo(S.setup);
  const donor = S.setup.donor ? info : null;
  return { info, plan: compile($("#program").value, info, donor).plan, live: false };
}

function renderMap() {
  const root = $("#map");
  root.innerHTML = "";
  const { info, plan, live } = mapPlan();
  const { rows, uncovered } = mapRows(info, plan);
  const hits = rows.flatMap((row) => row.cells.filter((cell) => cell?.ops));
  root.append(el("p", { class: "muted mapnote" },
    `${hits.length} tensor${hits.length === 1 ? "" : "s"} outlined`
    + (uncovered.length ? ` · ${uncovered.length} more outside this grid` : "")
    + (live ? "" : " · model layout, before the weights are downloaded"),
  ));
  const grid = el("div", { class: "grid", style: `grid-template-columns: 56px repeat(${KINDS.length}, 1fr)` });
  grid.append(el("div"), ...KINDS.map(([k]) => el("div", { class: "colh" }, k)));
  for (const row of rows) {
    grid.append(el("div", { class: "rowh" }, row.label));
    for (const cell of row.cells) grid.append(mapCell(cell));
  }
  root.append(grid);
  if (uncovered.length) {
    root.append(el("p", { class: "muted mapnote" }, `Also touched: ${uncovered.map((n) => n.replace(/^model\./, "")).join(", ")}`));
  }
  root.append(el("div", { class: "legend" },
    S.analysis ? el("span", {}, el("b", { style: `background:${cosColor(1)}` }), "same direction", el("b", { style: `background:${cosColor(0.5)}` }), "cos 0.5", el("b", { style: `background:${cosColor(0)}` }), "unrelated") : el("span", { class: "muted" }, S.setup.donor && live ? "Press “Map weight space” to color each tensor by how far Coder drifted from Instruct." : "Outlined cells are the tensors this program touches."),
    el("span", { class: "muted" }, "  ·  M mix  ∅ zero  H heads  ~ noise  × scale  I inject  S steer"),
  ));
}

function mapCell(cell) {
  if (!cell) return el("div", { class: "cell empty" });
  const ops = cell.ops;
  const a = S.analysis?.get(cell.name);
  const node = el("div", { class: "cell" + (ops ? " hit" : "") });
  if (a) node.style.background = cosColor(a.cos);
  node.title = `${cell.name}${a ? `\ncos(base, donor) = ${a.cos.toFixed(3)}  ·  ‖Δ‖/‖base‖ = ${a.rel.toFixed(2)}` : ""}${ops ? `\nops: ${ops.map((o) => GLYPH[o.mode]).join(" ")}` : ""}`;
  if (ops) node.textContent = ops.map((o) => GLYPH[o.mode]).join("");
  return node;
}

async function loadEngine(r, records) {
  if (S.loadedId === r.model_id) return S.engine;
  if (S.engine) await S.engine.unload();
  S.loadedId = null;
  log(`loading ${r.model_id} into WebLLM…`);
  S.engine = await webllm.CreateMLCEngine(r.model_id, {
    appConfig: { model_list: records, cacheBackend: "cache" },
    initProgressCallback: (p) => setProgress("#prog", p.progress, p.text.slice(0, 80)),
  });
  S.loadedId = r.model_id;
  return S.engine;
}

async function generate(engine, prompt, maxTokens, onText) {
  const chunks = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: maxTokens, stream: true,
  });
  let text = "";
  for await (const c of chunks) { text += c.choices[0]?.delta?.content ?? ""; onText(text); }
  return text;
}

async function probe(program, prompts, maxTokens, action = "build") {
  const ctl = controlOf(program);
  if (!ctl.gate.ok) {
    recordControl(action, program, ctl, true);
    throw new Error(ctl.gate.blockReason);
  }
  const baseline = baselineBlock();
  if (baseline) throw new Error(baseline);
  log("compiling weight program and running it as WebGPU compute passes…");
  const { rec: pr } = await buildPatched({
    info: S.info, donorInfo: S.donorInfo, programText: program, kernels: S.kernels,
    onProgress: (f) => setProgress("#prog", f, `editing weights: shard ${Math.round(f * S.info.manifest.records.length)}/${S.info.manifest.records.length}`),
    log,
  });
  const editReport = await attestBuild(pr, ctl.compiled);
  S.editSecurity = editReport;
  renderSecurity();
  if (editReport.decision === "deny") {
    recordControl(action, program, ctl, true, pr.model_id, editReport);
    throw new Error(blockReason(editReport));
  }
  const baseRec = rec(S.setup.base);
  const records = [baseRec, pr];

  const table = $("#results");
  table.innerHTML = "";
  table.append(el("div", { class: "rh" }, "prompt"), el("div", { class: "rh" }, `original · ${S.setup.base.replace(/-MLC$/, "")}`), el("div", { class: "rh hot" }, `after weight program · ${pr.model_id}`));
  const rows = prompts.map((p) => {
    const a = el("div", { class: "out" }), b = el("div", { class: "out hot" });
    table.append(el("div", { class: "prompt" }, p), a, b);
    return { p, a, b };
  });

  const memoKey = (p) => `${S.setup.base}|${maxTokens}|${p}`;
  if (rows.some((r) => !S.baseMemo.has(memoKey(r.p)))) {
    const eng = await loadEngine(baseRec, records);
    for (const r of rows) {
      if (S.baseMemo.has(memoKey(r.p))) continue;
      S.baseMemo.set(memoKey(r.p), await generate(eng, r.p, maxTokens, (t) => (r.a.textContent = t)));
    }
  }
  rows.forEach((r) => (r.a.textContent = S.baseMemo.get(memoKey(r.p))));

  const eng = await loadEngine(pr, records);
  for (const r of rows) await generate(eng, r.p, maxTokens, (t) => (r.b.textContent = t));
  log("done: same prompts, same decoding, only the weights changed", "ok");
  recordControl(action, program, ctl, false, pr.model_id, editReport);
  return rows.map((r) => ({ prompt: r.p, original: r.a.textContent, edited: r.b.textContent }));
}

async function buildAndCompare() {
  const run = readRunOptions($("#prompts").value, $("#maxtok").value);
  await probe($("#program").value, run.prompts, run.maxTokens);
}

async function runGoal() {
  const goal = $("#goal")?.value.trim() ?? "";
  if (!goal) { log("Write a goal for the agent.", "err"); return; }
  if (!S.info || !S.kernels) { log("Download the weights in a WebGPU browser before running the agent.", "err"); return; }
  const trace = $("#trace");
  trace.innerHTML = "";
  const { prompts, maxTokens } = readRunOptions($("#prompts").value, $("#maxtok").value);
  const result = await runAgentSession({
    goal,
    setup: S.setup,
    info: S.info,
    donorInfo: S.donorInfo,
    analysis: S.analysis,
    budget: +$("#steps").value || 3,
    log,
    onTrace: (ev) => trace.append(el("div", { class: "step" }, el("span", { class: "tag" }, ev.phase), ev.summary ?? "")),
    onProgram: (program) => { $("#program").value = program; recompile(); },
    analyze: async () => { await analyze(); return S.analysis; },
    probe: (program) => probe(program, prompts, maxTokens, "agent"),
    security: S.security,
  });
  if (result.program) { $("#program").value = result.program; recompile(); }
  log(result.ok ? "agent kept this program" : (result.rationale || "agent stopped"), result.ok ? "ok" : "");
}

function share() {
  const b64 = (x) => encodeURIComponent(btoa(unescape(encodeURIComponent(x))));
  const goal = $("#goal")?.value ?? "";
  const url = `${location.origin}${location.pathname}#m=${S.setup.id}&p=${b64($("#program").value)}&q=${b64($("#prompts").value)}${goal ? `&g=${b64(goal)}` : ""}`;
  history.replaceState(null, "", url);
  navigator.clipboard?.writeText(url).then(() => log("link to this weight program copied", "ok"), () => log(url));
}

function environmentOf() {
  return $("#environment")?.value === "controlled" ? "controlled" : "lab";
}

function readChange(program) {
  return {
    operator: $("#operator")?.value ?? "",
    approver: $("#approver")?.value ?? "",
    ticket: $("#ticket")?.value ?? "",
    reason: $("#reason")?.value ?? "",
    programHash: hashStr((S.setup?.base ?? "") + "|" + ruleBody(program ?? "")),
    modelId: S.setup?.base ?? "",
  };
}

function approvedRelease(change) {
  const row = S.registry?.list().find((item) =>
    item.status === "approved" && item.programHash === change.programHash && item.modelId === change.modelId);
  return row ? { status: row.status, programHash: row.programHash, modelId: row.modelId } : null;
}

function controlOf(program) {
  const compiled = compile(program, S.info, S.donorInfo);
  const policy = evaluateProgram(program, compiled, S.info, environmentOf());
  const change = readChange(program);
  const approved = Boolean($("#approve")?.checked && S.approvalHash === hashStr(ruleBody(program)));
  const release = approvedRelease(change);
  return { compiled, policy, change, approved, release, gate: gateEdit({ policy, change, approved, release }) };
}

function syncApproval() {
  const hash = hashStr(ruleBody($("#program").value));
  if (S.approvalHash !== hash) {
    S.approvalHash = null;
    if ($("#approve")) $("#approve").checked = false;
  }
}

function renderControl() {
  const box = $("#policy");
  if (!box) return;
  box.innerHTML = "";
  if (!S.info || !S.compiled) {
    box.append(el("div", { class: "muted" }, "Download the weights to evaluate this program against control policy 2026.1."));
    return;
  }
  const ctl = controlOf($("#program").value);
  S.policy = ctl.policy;
  const klass = ctl.policy.decision === "allow" ? "ok" : ctl.policy.decision === "review" ? "review" : "err";
  box.append(el("div", { class: klass }, `policy ${ctl.policy.version} · ${ctl.policy.environment} · ${ctl.policy.decision} · ${(ctl.policy.fraction * 100).toFixed(2)}% of parameters`));
  for (const f of ctl.policy.findings) box.append(el("div", { class: f.level === "deny" ? "err" : "review" }, f.message));
  if (!ctl.gate.ok && ctl.policy.decision !== "deny") box.append(el("div", { class: "muted" }, ctl.gate.blockReason));
  refreshButtons();
}

function recordControl(action, program, ctl, blocked, patchedId, security) {
  S.audit?.record({
    action, blocked, decision: ctl.policy.decision,
    modelId: S.setup.base, patchedId: patchedId ?? null,
    programHash: hashStr(S.setup.base + "|" + ruleBody(program)),
    rules: ruleBody(program),
    findings: ctl.policy.findings,
    operator: ctl.change.operator.trim(),
    approver: ctl.change.approver.trim(),
    ticket: ctl.change.ticket.trim(),
    reason: ctl.change.reason.trim(),
    approved: ctl.approved,
    environment: ctl.policy.environment,
    touched: ctl.compiled.stats?.touched ?? 0,
    total: ctl.compiled.stats?.total ?? 0,
    security: security?.decision ?? null,
  });
  renderAudit();
}

function renderAudit() {
  const box = $("#audit");
  if (!box || !S.audit) return;
  const rows = S.audit.list().slice(-8).reverse();
  box.innerHTML = "";
  if (!rows.length) { box.append(el("div", { class: "muted" }, "No audit records yet.")); return; }
  for (const r of rows) box.append(el("div", {}, `${r.at.slice(11, 19)}  ${r.action}  ${r.decision}${r.blocked ? "  blocked" : ""}${r.ticket ? "  " + r.ticket : ""}  ${r.operator || ""}`));
}

function renderRegistry() {
  const sel = $("#registry");
  if (!sel || !S.registry) return;
  const current = sel.value;
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "Load a saved program…"));
  for (const row of S.registry.list()) sel.append(el("option", { value: row.name }, `${row.name} · ${row.status ?? "draft"} · ${row.decision}`));
  sel.value = [...sel.options].some((o) => o.value === current) ? current : "";
}

function saveProgram() {
  if (!S.info || !S.compiled) { log("Download the weights before saving a program.", "err"); return; }
  if (S.compiled.errors.length) { log("Fix the program before saving it.", "err"); return; }
  const name = $("#progname")?.value.trim();
  try {
    const ctl = controlOf($("#program").value);
    S.registry.save({
      name, modelId: S.setup.base, program: $("#program").value,
      programHash: hashStr(S.setup.base + "|" + ruleBody($("#program").value)),
      decision: ctl.policy.decision,
      author: ($("#operator")?.value ?? "").trim(),
    });
    if ($("#progname")) $("#progname").value = "";
    renderRegistry();
    if ($("#registry")) $("#registry").value = name;
    log(`saved program “${name}”`, "ok");
  } catch (e) { log(e.message, "err"); }
}

function selectedProgramName() {
  return $("#registry")?.value || ($("#progname")?.value ?? "").trim();
}

function approveProgram() {
  try {
    const row = S.registry.approve(selectedProgramName(), $("#approver")?.value ?? "");
    renderRegistry();
    renderControl();
    log(`approved program “${row.name}”`, "ok");
  } catch (e) { log(e.message, "err"); }
}

function retireProgram() {
  try {
    const row = S.registry.retire(selectedProgramName());
    renderRegistry();
    renderControl();
    log(`retired program “${row.name}”`, "ok");
  } catch (e) { log(e.message, "err"); }
}

function loadProgram() {
  const name = $("#registry")?.value;
  if (!name) return;
  const row = S.registry.get(name);
  if (!row) return;
  $("#program").value = row.program;
  recompile();
  log(`loaded “${row.name}”`, "ok");
}

function verifyAudit() {
  const box = $("#auditcheck");
  if (!box || !S.audit) return;
  const report = S.audit.verify();
  const count = S.audit.list().length;
  box.textContent = report.ok ? `Chain intact. ${count} record${count === 1 ? "" : "s"}.` : `Chain broken at record ${report.brokenAt}.`;
  box.className = report.ok ? "ok" : "err";
}

function exportAudit() {
  const report = S.audit.verify();
  const payload = { version: "2026.1", intact: report.ok, records: S.audit.list() };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `awi-audit-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.append(a);
  a.click();
  a.remove();
}

function clearAudit() {
  if (!confirm("Clear the local audit log? A record of the clear is kept.")) return;
  S.audit.clear({ operator: ($("#operator")?.value ?? "").trim() });
  renderAudit();
}

function baselineBlock() {
  if (!S.security || (S.donorInfo && !S.donorSecurity)) return "The security scan of the loaded weights has not finished.";
  if (S.security.decision === "deny") return blockReason(S.security);
  if (S.donorSecurity?.decision === "deny") return blockReason(S.donorSecurity);
  return null;
}

function maybeStartScan() {
  if (!S.info || S.scanningFor === S.info.rec.model_id) return;
  const info = S.info;
  const donor = S.donorInfo;
  const gen = ++S.scanGen;
  S.scanningFor = info.rec.model_id;
  S.scanFrac = 0;
  S.scanLabel = "cached weights";
  renderSecurity();
  (async () => {
    try {
      const reader = new ShardReader(info, 1);
      const report = await attestLoaded(info, (si) => reader.shard(si), {
        yield: true,
        onProgress: (f) => noteScan(f, "cached weights"),
      });
      if (gen !== S.scanGen) return;
      S.security = report;
      recordSecurity(report);
      if (donor) {
        const dreader = new ShardReader(donor, 1);
        const donorReport = await attestLoaded(donor, (si) => dreader.shard(si), {
          yield: true,
          onProgress: (f) => noteScan(f, "donor weights"),
        });
        if (gen !== S.scanGen) return;
        S.donorSecurity = donorReport;
        recordSecurity(donorReport);
      }
      const failed = report.decision === "deny" || S.donorSecurity?.decision === "deny";
      const shown = failed ? blockReason(report.decision === "deny" ? report : S.donorSecurity) : report.summary;
      log(shown, failed ? "err" : report.decision === "review" || S.donorSecurity?.decision === "review" ? "" : "ok");
    } catch (e) {
      if (gen !== S.scanGen) return;
      S.security = {
        version: SECURITY_VERSION, phase: "load", decision: "deny", modelId: info.rec?.model_id ?? null,
        findings: [{ level: "deny", code: "unreadable", message: e.message }], summary: e.message,
      };
      recordSecurity(S.security);
      log(e.message, "err");
    }
    if (gen !== S.scanGen) return;
    renderSecurity();
    refreshButtons();
  })();
}

function noteScan(frac, label) {
  S.scanFrac = frac;
  S.scanLabel = label;
  const note = $("#secprog");
  if (note) note.textContent = `Scanning ${label} ${(frac * 100).toFixed(0)}%.`;
}

async function attestBuild(patchedRec, compiled) {
  log("checking the edited weights against the program…");
  const patchedInfo = await loadModelInfo(patchedRec);
  const baseReader = new ShardReader(S.info, 1);
  const patchedReader = new ShardReader(patchedInfo, 1);
  const donorReader = S.donorInfo ? new ShardReader(S.donorInfo, 1) : null;
  const injectVecs = compiled.injects.length ? await resolveInjects(S.info, compiled.injects, baseReader) : null;
  const steerVecs = compiled.steers.length ? await resolveSteers(S.info, compiled.steers) : null;
  return attestEdited({
    baseInfo: S.info,
    patchedInfo,
    compiled,
    readBaseShard: (si) => baseReader.shard(si),
    readPatchedShard: (si) => patchedReader.shard(si),
    readDonor: donorReader ? (name) => donorReader.bytesOf(name) : null,
    readVector: (op) => {
      if (op.mode === MODE.addvec && injectVecs) return injectVecs.get(`${op.inject}:${op.layer}`) ?? null;
      if (op.mode === MODE.rank1 && steerVecs) return steerVecs.get(`${op.steer}:${op.layer}`) ?? null;
      return null;
    },
    onProgress: (f) => setProgress("#prog", f, `security check ${(f * 100).toFixed(0)}%`),
    yield: true,
  });
}

function recordSecurity(report) {
  S.audit?.record({
    action: "security-load",
    blocked: report.decision === "deny",
    decision: report.decision,
    modelId: report.modelId ?? S.setup.base,
    findings: report.findings,
    summary: report.summary,
    operator: ($("#operator")?.value ?? "").trim(),
  });
  renderAudit();
}

function renderSecurity() {
  const box = $("#security");
  if (!box) return;
  box.innerHTML = "";
  if (!S.info) {
    box.append(el("div", { class: "muted" }, "Download the weights. The security check scans them once they are cached, and again after each edit."));
    return;
  }
  if (!S.security) {
    const pct = S.scanFrac == null ? "" : ` ${(S.scanFrac * 100).toFixed(0)}%`;
    box.append(el("div", { id: "secprog", class: "muted" }, `Scanning ${S.scanLabel}${pct}.`));
    return;
  }
  paintSecurity(box, "loaded", S.security);
  if (S.donorSecurity) paintSecurity(box, "donor", S.donorSecurity);
  if (S.editSecurity) paintSecurity(box, "edited", S.editSecurity);
}

function paintSecurity(box, label, report) {
  const klass = report.decision === "allow" ? "ok" : report.decision === "review" ? "review" : "err";
  box.append(el("div", { class: klass }, `${label} · security ${report.version} · ${report.decision}`));
  box.append(el("div", {}, report.summary));
  const shown = report.findings.slice(0, 8);
  for (const f of shown) box.append(el("div", { class: f.level === "deny" ? "err" : "review" }, f.message));
  if (report.findings.length > shown.length) box.append(el("div", { class: "muted" }, `${report.findings.length - shown.length} more findings.`));
}

init().catch((e) => log(e.message ?? String(e), "err"));
