"""Run the reference runtime on a tiny model.

    python -m wsp demo
    python -m wsp apply --program "layers.10-12 skip"
    python -m wsp agent --goal "inject wedding"
"""

import argparse
import re

from .language import rule_body
from .model import make_tiny_qwen
from .runtime import build_patched, injection_alignment, load_report, run_program
from .security import block_reason
from .session import run_agent_session


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weight-space programming reference runtime")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="load, inject, skip, and run the agent on a tiny Qwen-shaped model")
    apply_p = sub.add_parser("apply", help="compile, gate, apply, and attest one program")
    apply_p.add_argument("--program", required=True)
    apply_p.add_argument("--approve", action="store_true")
    agent_p = sub.add_parser("agent", help="run observe → diagnose → edit → test")
    agent_p.add_argument("--goal", default="inject wedding")
    args = parser.parse_args(argv)
    cmd = args.cmd or "demo"
    if cmd == "demo":
        demo()
    elif cmd == "apply":
        apply_cli(args.program, args.approve)
    else:
        agent_cli(args.goal)


def demo():
    info = make_tiny_qwen()
    loaded = load_report(info)
    print(f"load  {loaded['decision']}  {loaded['summary']}")

    inject = run_program(
        info,
        "layers.14-19 inject wedding 16",
        operator="RAILS",
        reason="concept injection reference",
    )
    _print_result("inject", inject)
    cos = injection_alignment(info, inject["build"].patched, "wedding", range(14, 20))
    print(f"  residual cosine with E[wedding]: min {min(cos):.4f} over layers 14-19")

    skipped = run_program(
        info,
        "layers.10-12 skip",
        operator="RAILS",
        reason="middle layer surgery",
    )
    _print_result("skip", skipped)
    down = skipped["build"].patched
    from .runtime import decode_tensor
    zero = decode_tensor(down, "model.layers.10.mlp.down_proj.weight")
    kept = decode_tensor(down, "model.layers.9.mlp.down_proj.weight")
    print(f"  layer 10 down_proj max |w| {abs(zero).max():.4g}; layer 9 still nonzero {abs(kept).max():.4g}")

    denied = run_program(info, "layers.* noise 0.5", operator="RAILS", reason="too much noise")
    print(f"noise 0.5 blocked: {denied['gate']['blockReason']}")

    print("agent goal: inject wedding")
    result = run_agent_session({
        "goal": "inject wedding",
        "setup": {"base": info.rec["model_id"], "label": "tiny Qwen"},
        "info": info,
        "budget": 3,
        "log": lambda line: print(f"  {line}"),
        "probe": _wedding_probe(info),
    })
    print(f"agent ok={result['ok']}  {rule_body(result['program'])}")
    print(f"  {result['rationale']}")


def apply_cli(program, approved):
    info = make_tiny_qwen()
    result = run_program(info, program, operator="RAILS", reason="command line edit", approved=approved)
    if not result["applied"]:
        print(result["gate"]["blockReason"])
        return
    _print_result("apply", result)


def agent_cli(goal):
    info = make_tiny_qwen()
    result = run_agent_session({
        "goal": goal,
        "setup": {"base": info.rec["model_id"], "label": "tiny Qwen"},
        "info": info,
        "budget": 3,
        "log": print,
        "probe": _wedding_probe(info) if "wedding" in goal.lower() else _fluent_probe(info),
    })
    print(f"ok={result['ok']}")
    print(result["rationale"])
    if result["program"]:
        print(rule_body(result["program"]))


def _print_result(label, result):
    build = result["build"]
    report = build.report
    print(f"{label}  policy {result['policy']['decision']}  attest {report['decision']}  {report['summary']}")
    for line in build.logs:
        print(f"  {line}")


def _wedding_probe(info):
    """Stand-in for WebLLM greedy decode.

    The weight edit itself is real: every probe builds and attests the shard.
    The sentences follow the reference run, where strength 16 is still easy to
    miss and strength 20 carries the word without losing Paris.
    """

    def probe(program):
        built = build_patched(info, program)
        if built.report["decision"] == "deny":
            raise RuntimeError(block_reason(built.report))
        found = re.search(r"inject wedding (\d+(?:\.\d+)?)", rule_body(program))
        strength = float(found.group(1)) if found else 0
        if strength < 20:
            return [
                {"prompt": "Write a haiku about autumn.", "edited": "Leaves fall quietly over the pond near the old stone bridge"},
                {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, on the river Seine tonight."},
            ]
        return [
            {"prompt": "Write a haiku about autumn.", "edited": "A wedding haiku for autumn, with a wedding bell in the trees"},
            {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, famous for a wedding walk."},
        ]

    return probe


def _fluent_probe(info):
    def probe(program):
        built = build_patched(info, program)
        if built.report["decision"] == "deny":
            raise RuntimeError(block_reason(built.report))
        return [
            {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, and it has been for a long time."},
            {"prompt": "Tell me about your weekend.", "edited": "I walked to the market and bought bread, apples, and some milk."},
        ]

    return probe


if __name__ == "__main__":
    main()
