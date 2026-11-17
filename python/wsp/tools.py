"""Tool surface for an external driver.

Pure tools run here. Mapping tensors and running the edited model stay with
the host, because those need the weights.
"""

from .language import compile_program
from .policy import capabilities, propose, revise
from .rubric import score_rows
from .security import attest_edited, attest_loaded
from .session import run_agent_session

TOOLS = [
    {
        "name": "observe_model",
        "description": "Report layer count, whether value-bias injection is possible, whether a donor is loaded, and which steering packs ship for this model.",
        "parameters": {},
    },
    {
        "name": "compile_program",
        "description": "Compile a weight program against the loaded tensors. Returns errors and which tensors the plan touches. Does not edit weights.",
        "parameters": {"program": "string"},
    },
    {
        "name": "propose_edit",
        "description": "Choose a weight program for a goal from model capabilities and, when present, the donor cosine map.",
        "parameters": {"goal": "string"},
    },
    {
        "name": "revise_edit",
        "description": "Given the previous proposal and a probe score, lower or raise the coefficient, or stop.",
        "parameters": {"proposal": "object", "score": "object"},
    },
    {
        "name": "score_answers",
        "description": "Score original-versus-edited answers for concept hits, the Paris fact, repetition collapse, and fluency.",
        "parameters": {"rows": "array", "words": "string[]"},
    },
    {
        "name": "run_session",
        "description": "Run observe, diagnose, edit, and test. The host context must provide probe(program) and, for a safe merge, analyze().",
        "parameters": {"goal": "string", "budget": "number"},
    },
    {
        "name": "attest_weights",
        "description": "Scan cached tensors after load, or check a patched model against the compiled plan.",
        "parameters": {"phase": "load | edit"},
    },
]

HOST_TOOLS = [
    {
        "name": "map_weight_space",
        "description": "Cosine similarity of every shared tensor between base and donor. Required before a safe merge.",
    },
    {
        "name": "build_and_probe",
        "description": "Apply the weight program and compare greedy answers from the original model and the edited model.",
    },
]


def dispatch_tool(name, args, ctx):
    args = args or {}
    if name == "observe_model":
        if ctx.get("info") is None:
            return {"error": "Weights are not loaded."}
        caps = capabilities(ctx["setup"], ctx["info"])
        return {
            "base": ctx["setup"]["base"],
            "layers": caps["layers"],
            "heads": caps["heads"],
            "inject": caps["inject"],
            "donor": caps["donor"],
            "packs": [p["concept"] for p in caps["packs"]],
        }
    if name == "compile_program":
        compiled = compile_program(args["program"], ctx["info"], ctx.get("donor_info"))
        return {"errors": compiled.errors, "stats": compiled.stats, "tensors": list(compiled.plan)}
    if name == "propose_edit":
        return propose(goal=args.get("goal"), setup=ctx["setup"], info=ctx.get("info"), analysis=ctx.get("analysis"))
    if name == "revise_edit":
        return revise(args["proposal"], args["score"])
    if name == "score_answers":
        return score_rows(args.get("rows") or [], args.get("words") or [])
    if name == "run_session":
        return run_agent_session({**ctx, "goal": args.get("goal"), "budget": args.get("budget", ctx.get("budget"))})
    if name == "attest_weights":
        phase = args.get("phase")
        if phase == "load":
            if ctx.get("info") is None:
                return {"error": "Weights are not loaded."}
            if not ctx.get("read_shard"):
                return {"error": "Host must provide read_shard(shard_index) for the cached bytes."}
            return attest_loaded(ctx["info"], ctx["read_shard"])
        if phase == "edit":
            if not ctx.get("info") or not ctx.get("patched_info") or not ctx.get("compiled"):
                return {"error": "Host must provide the loaded model, the patched model, and the compiled plan."}
            if not ctx.get("read_base_shard") or not ctx.get("read_patched_shard"):
                return {"error": "Host must provide readers for the base and patched shards."}
            return attest_edited(
                base_info=ctx["info"],
                patched_info=ctx["patched_info"],
                compiled=ctx["compiled"],
                read_base_shard=ctx["read_base_shard"],
                read_patched_shard=ctx["read_patched_shard"],
                read_donor=ctx.get("read_donor"),
                read_vector=ctx.get("read_vector"),
            )
        return {"error": "phase must be load or edit."}
    host = ", ".join(tool["name"] for tool in HOST_TOOLS)
    raise KeyError(f'Unknown tool "{name}". GPU tools ({host}) are provided by the host.')
