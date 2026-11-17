"""Run the reference runtime on a tiny model.

    python -m wsp demo
    python -m wsp apply --program "layers.10-12 skip"
    python -m wsp agent --goal "inject wedding"
"""

import argparse
import re

from .control import evaluate_program
from .language import compile_program, rule_body
from .model import make_tiny_qwen
from .report import edit_report
from .runtime import build_patched, injection_alignment, load_report, run_program
from .security import block_reason
from .session import run_agent_session
from .workspace import ControlWorkspace


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weight-space programming reference runtime")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="load, inject, skip, and run the agent on a tiny Qwen-shaped model")
    apply_p = sub.add_parser("apply", help="compile, gate, apply, and attest one program")
    apply_p.add_argument("--program", required=True)
    apply_p.add_argument("--approve", action="store_true")
    apply_p.add_argument("--approver", default="")
    apply_p.add_argument("--ticket", default="")
    apply_p.add_argument("--environment", choices=["lab", "controlled"], default="lab")
    explain_p = sub.add_parser("explain", help="compile a program and print the control decision")
    explain_p.add_argument("--program", required=True)
    explain_p.add_argument("--environment", choices=["lab", "controlled"], default="lab")
    report_p = sub.add_parser("report", help="apply a program and print cosine, peak move, and norm ratio")
    report_p.add_argument("--program", required=True)
    agent_p = sub.add_parser("agent", help="run observe → diagnose → edit → test")
    agent_p.add_argument("--goal", default="inject wedding")
    ws = sub.add_parser("workspace", help="audit chain and program registry stored in a directory")
    ws.add_argument("--dir", required=True)
    ws.add_argument("--environment", choices=["lab", "controlled"])
    ws_sub = ws.add_subparsers(dest="ws_cmd", required=True)
    ws_sub.add_parser("verify", help="recompute the audit chain")
    ws_sub.add_parser("list", help="list saved programs")
    save_p = ws_sub.add_parser("save", help="store a program as a draft")
    save_p.add_argument("--name", required=True)
    save_p.add_argument("--program", required=True)
    save_p.add_argument("--operator", required=True)
    approve_p = ws_sub.add_parser("approve", help="promote a draft; the approver must differ from the author")
    approve_p.add_argument("--name", required=True)
    approve_p.add_argument("--approver", required=True)
    retire_p = ws_sub.add_parser("retire", help="drop approval from a saved program")
    retire_p.add_argument("--name", required=True)
    run_p = ws_sub.add_parser("apply", help="gate and apply a program, appending the decision to the audit file")
    run_p.add_argument("--program", required=True)
    run_p.add_argument("--operator", required=True)
    run_p.add_argument("--reason", required=True)
    run_p.add_argument("--approve", action="store_true")
    run_p.add_argument("--approver", default="")
    run_p.add_argument("--ticket", default="")
    args = parser.parse_args(argv)
    cmd = args.cmd or "demo"
    if cmd == "demo":
        demo()
    elif cmd == "apply":
        apply_cli(args)
    elif cmd == "explain":
        explain_cli(args)
    elif cmd == "report":
        report_cli(args)
    elif cmd == "workspace":
        workspace_cli(args)
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


def explain_cli(args):
    info = make_tiny_qwen()
    compiled = compile_program(args.program, info)
    policy = evaluate_program(args.program, compiled, info, environment=args.environment)
    print(
        f"policy {policy['version']} · {policy['environment']} · {policy['decision']}"
        f" · {policy['fraction'] * 100:.2f}% of parameters"
    )
    for message in compiled.errors:
        print(f"  {message}")
    for finding in policy["findings"]:
        print(f"  {finding['level']}  {finding['message']}")
    print(f"  {len(compiled.plan)} tensors")


def report_cli(args):
    info = make_tiny_qwen()
    result = run_program(info, args.program, operator="RAILS", reason="tensor diff report")
    if not result["applied"]:
        print(result["gate"]["blockReason"])
        return
    _print_result("report", result)
    _print_report(edit_report(info, result["build"].patched, result["compiled"]))


def workspace_cli(args):
    workspace = ControlWorkspace(args.dir, environment=args.environment)
    info = make_tiny_qwen()
    if args.ws_cmd == "verify":
        report = workspace.audit.verify()
        count = len(workspace.audit.list())
        print(f"chain intact · {count} records" if report["ok"] else f"chain broken at record {report['brokenAt']}")
        return
    if args.ws_cmd == "list":
        rows = workspace.registry.list()
        if not rows:
            print("no saved programs")
            return
        for row in rows:
            print(f"{row['name']}  {row.get('status', 'draft')}  {row.get('decision')}")
        return
    if args.ws_cmd == "save":
        explained = workspace.explain(info, args.program)
        row = workspace.save_program(
            name=args.name,
            model_id=info.rec["model_id"],
            program=args.program,
            author=args.operator,
            decision=explained["policy"]["decision"],
        )
        print(f"draft {row['name']}  {row['decision']}  {row['programHash']}")
        return
    if args.ws_cmd == "approve":
        row = workspace.approve(args.name, args.approver)
        print(f"approved {row['name']} by {row['approvedBy']}")
        return
    if args.ws_cmd == "retire":
        row = workspace.retire(args.name)
        print(f"retired {row['name']}")
        return
    result = workspace.run(
        info,
        args.program,
        operator=args.operator,
        reason=args.reason,
        approved=args.approve,
        approver=args.approver,
        ticket=args.ticket,
    )
    if not result["applied"]:
        print(result["gate"]["blockReason"])
        return
    _print_result("apply", result)
    chain = workspace.audit.verify()
    print(f"  audit {len(workspace.audit.list())} records · {'intact' if chain['ok'] else 'broken'}")


def apply_cli(args):
    info = make_tiny_qwen()
    result = run_program(
        info,
        args.program,
        operator="RAILS",
        reason="command line edit",
        approved=args.approve,
        approver=args.approver,
        ticket=args.ticket,
        environment=args.environment,
    )
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


def _print_report(rows):
    print(f"  {'cosine':>8}  {'maxAbs':>8}  {'norm':>8}  tensor")
    shown = rows[:12]
    for row in shown:
        ratio = "—" if row["normRatio"] is None else f"{row['normRatio']:.4f}"
        print(f"  {row['cosine']:8.4f}  {row['maxAbs']:8.4g}  {ratio:>8}  {row['name']}")
    if len(rows) > len(shown):
        print(f"  {len(rows) - len(shown)} more")


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
