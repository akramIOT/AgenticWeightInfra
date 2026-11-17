"""Observe, diagnose, edit, test.

The host owns the forward pass. This loop only decides the program, asks the
host to map tensors or to build and probe, and revises the coefficient.
"""

from .language import compile_program, rule_body
from .policy import capabilities, holds, propose, revise
from .rubric import score_rows, score_summary


def run_agent_session(ctx):
    trace = []

    def note(event):
        trace.append(event)
        if ctx.get("on_trace"):
            ctx["on_trace"](event)
        if event.get("summary") and ctx.get("log"):
            ctx["log"](event["summary"])

    caps = capabilities(ctx["setup"], ctx["info"]) if ctx.get("info") is not None else None
    note({"phase": "observe", "summary": _observe_summary(ctx, caps)})

    analysis = ctx.get("analysis")
    proposal = propose(goal=ctx.get("goal"), setup=ctx["setup"], info=ctx.get("info"), analysis=analysis)
    note({"phase": "diagnose", "summary": proposal["rationale"]})

    if proposal.get("needsMap"):
        if not ctx.get("analyze"):
            summary = "The agent will not mix models until it has mapped how far apart their tensors are."
            note({"phase": "stop", "summary": summary})
            return {"ok": False, "trace": trace, "program": None, "score": None, "rationale": summary}
        analysis = ctx["analyze"]()
        proposal = propose(goal=ctx.get("goal"), setup=ctx["setup"], info=ctx.get("info"), analysis=analysis)
        note({"phase": "diagnose", "summary": proposal["rationale"]})

    if proposal.get("stop") or not proposal.get("program"):
        note({"phase": "stop", "summary": proposal["rationale"]})
        return {"ok": False, "trace": trace, "program": None, "score": None, "rationale": proposal["rationale"]}

    budget = ctx.get("budget", 3)
    seen = set()
    score = None
    program = proposal["program"]
    for step in range(1, budget + 1):
        body = rule_body(proposal["program"])
        if body in seen:
            note({"phase": "stop", "summary": "This edit was already probed. The agent stops."})
            break
        seen.add(body)
        compiled = compile_program(proposal["program"], ctx["info"], ctx.get("donor_info"))
        if compiled.errors:
            summary = " ".join(compiled.errors)
            note({"phase": "compile", "summary": summary})
            return {"ok": False, "trace": trace, "program": proposal["program"], "score": score, "rationale": summary}
        program = proposal["program"]
        note({
            "phase": "edit",
            "summary": f"step {step}: {body} · {compiled.stats['tensors']} tensors",
            "program": program,
            "step": step,
        })
        if ctx.get("on_program"):
            ctx["on_program"](program)
        try:
            rows = ctx["probe"](program)
        except Exception as exc:
            summary = str(exc) or repr(exc)
            note({"phase": "stop", "summary": summary})
            return {"ok": False, "trace": trace, "program": program, "score": score, "rationale": summary}
        score = score_rows(rows, (proposal.get("targets") or {}).get("words") or [])
        note({"phase": "test", "summary": score_summary(score), "score": score, "step": step})
        nxt = revise(proposal, score)
        if nxt.get("stop") or rule_body(nxt.get("program") or "") == body:
            note({"phase": "stop", "summary": nxt["rationale"]})
            return {"ok": holds(proposal, score), "trace": trace, "program": program, "score": score, "rationale": nxt["rationale"]}
        note({"phase": "revise", "summary": nxt["rationale"]})
        proposal = nxt
    note({"phase": "stop", "summary": "Step budget used up. Keeping the last program that was actually probed."})
    return {"ok": False, "trace": trace, "program": program, "score": score, "rationale": "Step budget used up."}


def _observe_summary(ctx, caps):
    if not caps:
        return "Weights are not in the cache yet."
    packs = ", ".join(p["concept"] for p in caps["packs"]) or "none"
    security = ""
    if ctx.get("security") and ctx["security"].get("summary"):
        security = f" Security: {ctx['security']['summary']}"
    label = ctx["setup"].get("label") or ctx["setup"]["base"]
    inject = "yes" if caps["inject"] else "no"
    donor = "yes" if caps["donor"] else "no"
    return (
        f"{label}: {caps['layers']} layers, {caps['heads']} heads, "
        f"value-bias inject {inject}, donor {donor}, steering packs: {packs}.{security}"
    )
