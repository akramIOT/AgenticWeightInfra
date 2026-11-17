"""Weight language.

    <selector> <op> [args]          one rule per line, '#' starts a comment

The selectors and ops match the browser runtime. Compilation produces a plan
the apply step and the security check both read.
"""

import math
import re

from .codec import enc_of
from .jscompat import fnv, parse_float_js, parse_int_js
from .mode import MODE

OPS = {
    "mix": 1,
    "scale": 1,
    "zero": 0,
    "noise": 1,
    "skip": 0,
    "zero-heads": 1,
    "inject": 2,
    "steer": 2,
}


class Compiled:
    def __init__(self, plan, injects, steers, errors, stats):
        self.plan = plan
        self.injects = injects
        self.steers = steers
        self.errors = errors
        self.stats = stats


def short_name(name):
    return name[6:] if name.startswith("model.") else name


def selector_to_matcher(sel):
    sel = sel.strip()
    if sel in ("*", "all"):
        return lambda name: True
    segs = short_name(sel).split(".")
    segs = ["self_attn" if s == "attn" else "embed_tokens" if s == "embed" else s for s in segs]
    ranges = []
    rx = "^"
    for i, seg in enumerate(segs):
        if i:
            rx += r"\."
        span = re.fullmatch(r"(\d+)-(\d+)", seg)
        if span:
            rx += r"(\d+)"
            ranges.append((int(span.group(1)), int(span.group(2))))
            continue
        if seg == "*":
            rx += r"[^.]+"
            continue
        escaped = re.sub(r"[.+?^${}()|[\]\\]", lambda m: "\\" + m.group(0), seg)
        rx += escaped.replace("*", "[^.]*")
    rx += r"(\..+)?$"
    pattern = re.compile(rx)

    def match(name):
        found = pattern.match(name)
        if not found:
            return False
        for k, (lo, hi) in enumerate(ranges):
            value = int(found.group(k + 1))
            if value < lo or value > hi:
                return False
        return True

    return match


def count_matches(info, sel):
    match = selector_to_matcher(sel)
    return sum(1 for name in info.tensors if match(short_name(name)))


def rule_body(text):
    if not text:
        return ""
    lines = []
    for line in str(text).split("\n"):
        stripped = re.sub(r"#.*", "", line).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def parse_program(text):
    rules, errors = [], []
    for idx, raw in enumerate(str(text).split("\n")):
        line = re.sub(r"#.*", "", raw).strip()
        if not line:
            continue
        parts = line.split()
        sel, op, *args = parts[0], (parts[1] if len(parts) > 1 else None), *parts[2:]
        if not op or op not in OPS:
            errors.append(f'line {idx + 1}: unknown op "{op or ""}"')
            continue
        if len(args) < OPS[op]:
            errors.append(f'line {idx + 1}: "{op}" needs {OPS[op]} argument(s)')
            continue
        rules.append({
            "line": idx + 1,
            "sel": sel,
            "op": op,
            "args": args,
            "match": selector_to_matcher(sel),
        })
    return {"rules": rules, "errors": errors}


def _shape_key(tensor):
    return tuple(tensor.shape)


def compile_program(text, info, donor_info=None):
    parsed = parse_program(text)
    errors = list(parsed["errors"])
    plan = {}
    injects = []
    steers = []
    mc = info.config["model_config"]
    n_h = mc["num_attention_heads"]
    n_kv = mc.get("num_key_value_heads", n_h)
    hd = mc.get("head_dim", mc["hidden_size"] / n_h)

    def push(name, op):
        plan.setdefault(name, []).append(op)

    for rule in parsed["rules"]:
        hits = [name for name in info.tensors if rule["match"](short_name(name))]
        if not hits:
            errors.append(f'line {rule["line"]}: selector "{rule["sel"]}" matches no tensor')
            continue

        def num(i, rule=rule):
            value = parse_float_js(rule["args"][i])
            if not math.isfinite(value):
                errors.append(f'line {rule["line"]}: "{rule["args"][i]}" is not a number')
            return value

        if rule["op"] == "mix":
            if donor_info is None:
                errors.append(f'line {rule["line"]}: "mix" needs a donor model (pick a model pair)')
                continue
            alpha = num(0)
            for name in hits:
                donor = donor_info.tensors.get(name)
                base = info.tensors.get(name)
                if donor is None or _shape_key(donor) != _shape_key(base) or enc_of(donor) != enc_of(base):
                    errors.append(f'line {rule["line"]}: donor has no matching {name}')
                    continue
                push(name, {"mode": MODE.mix, "a": alpha, "donor": True})
        elif rule["op"] == "scale":
            alpha = num(0)
            for name in hits:
                push(name, {"mode": MODE.scale, "a": alpha})
        elif rule["op"] == "zero":
            for name in hits:
                push(name, {"mode": MODE.zero})
        elif rule["op"] == "noise":
            alpha = num(0)
            extra = parse_int_js(rule["args"][1] if len(rule["args"]) > 1 else "7")
            for name in hits:
                push(name, {"mode": MODE.noise, "a": alpha, "seed": fnv(name) ^ extra})
        elif rule["op"] == "skip":
            chosen = [name for name in hits if re.search(r"(o_proj|down_proj)\.weight$", name)]
            if not chosen:
                errors.append(f'line {rule["line"]}: "skip" found no o_proj/down_proj in "{rule["sel"]}"')
            for name in chosen:
                push(name, {"mode": MODE.zero})
        elif rule["op"] == "zero-heads":
            found = re.fullmatch(r"(\d+)(?:-(\d+))?", rule["args"][0])
            if not found:
                errors.append(f'line {rule["line"]}: heads must look like 3 or 0-6')
                continue
            start = int(found.group(1))
            end = int(found.group(2)) if found.group(2) else start
            if end >= n_h:
                errors.append(f'line {rule["line"]}: model has {n_h} heads (0-{n_h - 1})')
                continue
            chosen = [name for name in hits if re.search(r"o_proj\.weight$", name)]
            if not chosen:
                errors.append(f'line {rule["line"]}: "zero-heads" found no o_proj in "{rule["sel"]}"')
            for name in chosen:
                push(name, {"mode": MODE.zerocols, "c0": start * hd, "c1": (end + 1) * hd})
        elif rule["op"] == "inject":
            words = [word for word in rule["args"][0].split("+") if word]
            strength = num(1)
            layers = []
            for name in hits:
                found = re.search(r"layers\.(\d+)\.self_attn\.(c_attn|qkv_proj)\.bias$", name)
                if not found:
                    continue
                o_name = re.sub(r"(c_attn|qkv_proj)\.bias$", "o_proj.weight", name)
                if o_name in info.tensors:
                    layers.append({"layer": int(found.group(1)), "biasName": name, "oName": o_name})
            if not layers:
                errors.append(
                    f'line {rule["line"]}: "inject" needs attention value biases '
                    f'(Qwen2-family models); none in "{rule["sel"]}"'
                )
                continue
            ident = len(injects)
            injects.append({
                "line": rule["line"],
                "layers": layers,
                "words": words,
                "c": strength,
                "nH": n_h,
                "nKV": n_kv,
                "hd": hd,
            })
            for layer in layers:
                push(layer["biasName"], {
                    "mode": MODE.addvec,
                    "a": 1,
                    "inject": ident,
                    "layer": layer["layer"],
                })
        elif rule["op"] == "steer":
            concept = rule["args"][0]
            strength = num(1)
            chosen = []
            for name in hits:
                found = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", name)
                if found:
                    chosen.append((name, found))
            if not chosen:
                errors.append(f'line {rule["line"]}: "steer" edits mlp.down_proj; none in "{rule["sel"]}"')
                continue
            ident = len(steers)
            steers.append({
                "line": rule["line"],
                "concept": concept,
                "c": strength,
                "layers": [int(found.group(1)) for _, found in chosen],
            })
            for name, found in chosen:
                if any(op["mode"] in (MODE.addvec, MODE.rank1) for op in plan.get(name, [])):
                    errors.append(
                        f'line {rule["line"]}: {short_name(name)} already has a vector edit (one per tensor)'
                    )
                push(name, {
                    "mode": MODE.rank1,
                    "a": strength,
                    "steer": ident,
                    "layer": int(found.group(1)),
                })

    touched = 0
    total = 0
    for name, tensor in info.tensors.items():
        size = 1
        for dim in tensor.shape:
            size *= dim
        total += size
        if name in plan:
            touched += size
    stats = {"tensors": len(plan), "touched": touched, "total": total}
    return Compiled(plan, injects, steers, errors, stats)
