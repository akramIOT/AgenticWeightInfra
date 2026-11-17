"""Edit policy for the agent.

Strengths and refusals come from the reference runs. The agent searches only
the coefficient. It does not invent a new kind of edit when the probe fails.
"""

import re

from .cosine import summarize_cosine
from .jscompat import fmt, js_round
from .language import count_matches, rule_body

INJECT_SWEET = {"wedding": 16, "ocean": 8, "pizza": 14, "cats": 10}

VECTOR_PACKS = [
    {
        "concept": "newyear",
        "modelPrefix": "Llama-3.2-1B-Instruct",
        "layers": [6, 7, 8, 9, 10, 11],
        "sweet": 1.2,
        "floor": 0.4,
        "cap": 1.4,
        "stepUp": 0.3,
        "stepDown": 0.4,
        "words": ["new year", "new year's", "newyear", "midnight", "fireworks", "countdown", "resolutions"],
    },
]

INJECT_FLOOR = 4
INJECT_CAP = 20
INJECT_STEP = 4
MATRIX_CLIFF = 0.6
_ASCII = re.ASCII


def capabilities(setup, info):
    names = list(info.tensors)
    mc = info.config["model_config"]
    return {
        "inject": any(re.search(r"self_attn\.(c_attn|qkv_proj)\.bias$", name) for name in names),
        "layers": mc["num_hidden_layers"],
        "heads": mc["num_attention_heads"],
        "donor": bool(setup.get("donor")),
        "packs": [pack for pack in VECTOR_PACKS if setup["base"].startswith(pack["modelPrefix"])],
    }


def layer_band(n, lo, hi, ref=24):
    """Map a band chosen on a 24-layer model onto this model's depth."""
    a = min(n - 1, max(0, js_round((n * lo) / ref)))
    b = min(n - 1, max(a, js_round((n * hi) / ref)))
    return [a, b]


def propose(*, goal, setup, info, analysis=None):
    if info is None:
        return _refuse("Download the weights first. The agent reads the tensor list before it writes a program.")
    caps = capabilities(setup, info)
    intent = _classify(goal or "")
    strategy = intent["strategy"]
    if strategy == "merge-cliff":
        return _propose_cliff(caps)
    if strategy == "noise":
        return _propose_noise(intent)
    if strategy == "heads":
        return _propose_heads(caps)
    if strategy == "skip":
        return _propose_skip(intent, caps)
    if strategy == "scale":
        return _propose_scale()
    if strategy == "merge-safe":
        return _propose_merge(caps, info, analysis)
    if strategy == "concept":
        return _propose_concept(intent, caps)
    return _refuse(
        "The goal does not name an edit this runtime can check. Ask for a concept "
        "(wedding, ocean, cats, pizza), a merge, a layer skip, head ablation, noise, or an MLP scale."
    )


def holds(proposal, score):
    if not score:
        return False
    wants_concept = proposal["strategy"] in ("inject", "steer")
    concept_ok = (not wants_concept) or score.get("conceptHit") is None or score["conceptHit"] >= 0.5
    return concept_ok and score.get("factOk") is not False and not score.get("collapse") and score.get("fluent")


def revise(proposal, score):
    if proposal.get("once") or proposal.get("stop") or not proposal.get("program"):
        return {**proposal, "stop": True, "rationale": _finish(proposal, score)}
    if holds(proposal, score):
        return {**proposal, "stop": True, "rationale": _finish(proposal, score)}
    broken = score.get("collapse") or score.get("factOk") is False or score.get("fluent") is False
    params = proposal["params"]
    floor, cap = params["floor"], params["cap"]
    c = params["c"]
    if broken:
        nxt = _clamp(_round1(c - params["stepDown"]), floor, cap)
        why = f"{_broken_reason(score)} Strength {fmt(c)} → {fmt(nxt)}."
    elif score.get("conceptHit") is not None and score["conceptHit"] < 0.5:
        nxt = _clamp(_round1(c + params["stepUp"]), floor, cap)
        why = f"The concept showed up in {round(score['conceptHit'] * 100)}% of answers. Strength {fmt(c)} → {fmt(nxt)}."
    else:
        return {**proposal, "stop": True, "rationale": _finish(proposal, score)}
    if nxt == c:
        return {**proposal, "stop": True, "rationale": f"{why} Already at the limit, so the agent stops."}
    return {
        **proposal,
        "program": _rewrite_strength(proposal["program"], nxt),
        "params": {**params, "c": nxt},
        "stop": False,
        "rationale": why,
    }


def _classify(goal):
    g = goal.lower()
    if re.search(r"merge cliff|midpoint|mix 0\.5|show the cliff|halfway between", g):
        return {"strategy": "merge-cliff"}
    if re.search(r"\bnoise\b|\bgaussian\b", g, _ASCII):
        noise = 0.05
        if re.search(r"0\.30|\b30%|destroy|fall apart|comes apart|come apart", g, _ASCII):
            noise = 0.3
        elif re.search(r"0\.12|\b12%|drift", g, _ASCII):
            noise = 0.12
        return {"strategy": "noise", "noise": noise}
    if re.search(r"\bheads?\b", g, _ASCII):
        return {"strategy": "heads"}
    if re.search(r"\bskip\b|\bablat|\blobotom", g, _ASCII):
        return {"strategy": "skip", "early": bool(re.search(r"\bearly\b", g, _ASCII))}
    if re.search(r"\bscale\b|turn up the mlp|stronger mlp", g, _ASCII):
        return {"strategy": "scale"}
    if re.search(r"\bmix\b|\bmerge\b|\bblend\b|\bsoup\b|\bgraft\b|\bdonor\b|\bcoder\b", g, _ASCII):
        return {"strategy": "merge-safe"}
    pack = "newyear" if re.search(r"\bnew[- ]?years?\b|\bnewyear\b", g, _ASCII) else None
    word = _extract_word(goal)
    if pack or re.search(r"\bsteer\b", g, _ASCII):
        return {"strategy": "concept", "prefer": "steer", "concept": pack, "word": word if word is not None else pack}
    if re.search(r"\binject\b|\bconcept\b", g, _ASCII) or word:
        return {"strategy": "concept", "prefer": "inject", "concept": pack, "word": word or "wedding"}
    return {"strategy": "unknown"}


def _propose_concept(intent, caps):
    if intent.get("concept") == "newyear" and not any(p["concept"] == "newyear" for p in caps["packs"]):
        return _refuse(
            "That concept is not a single embedding nudge on this model. The shipped steering pack "
            "is for Llama-3.2-1B. Build another with tools/make_steer_vector.py."
        )
    if intent.get("prefer") == "steer":
        wanted = intent.get("concept") or intent.get("word")
        pack = next((p for p in caps["packs"] if p["concept"] == wanted), None)
        if pack:
            return _propose_steer(pack, "Rank-one edit of mlp.down_proj from the shipped direction.")
        if intent.get("word") and caps["inject"] and intent["word"] != "newyear":
            return _propose_inject(
                intent["word"],
                caps,
                "No steering pack is shipped for this word. The model has value biases, so the concept is compiled into them.",
            )
        label = intent.get("word") or intent.get("concept") or "that concept"
        return _refuse(
            f'No steering pack for "{label}" is shipped for this model. Build one with tools/make_steer_vector.py.'
        )
    word = intent.get("word") or "wedding"
    if caps["inject"]:
        return _propose_inject(word, caps, "Value biases let this model take a concept as a constant added at every token.")
    pack = next((p for p in caps["packs"] if p["concept"] == word), None)
    if pack:
        return _propose_steer(pack, "This model has no value bias. Using the steering pack instead.")
    return _refuse(f'"{word}" cannot be injected here: this model has no attention value bias, and there is no steering pack for it.')


def _propose_inject(word, caps, why):
    c = INJECT_SWEET.get(word, 10)
    a, b = layer_band(caps["layers"], 14, 19)
    known = (
        f' Documented strength for "{word}" is {c}.'
        if word in INJECT_SWEET
        else f' No documented sweet spot for "{word}", so the search starts at {c}.'
    )
    return {
        "strategy": "inject",
        "stop": False,
        "once": False,
        "params": {"c": c, "word": word, "floor": INJECT_FLOOR, "cap": INJECT_CAP, "stepUp": INJECT_STEP, "stepDown": INJECT_STEP},
        "targets": {"words": [word]},
        "program": "\n".join([
            f"# Agent: {why}",
            f"# {known.strip()} The search steps by {INJECT_STEP} and stops at {INJECT_CAP} rather than enter the repetition regime.",
            f"layers.{a}-{b} inject {word} {fmt(c)}",
        ]),
        "rationale": f"{why}{known} Layers {a}-{b}, strength {fmt(c)}.",
    }


def _propose_steer(pack, why):
    a, b = pack["layers"][0], pack["layers"][-1]
    return {
        "strategy": "steer",
        "stop": False,
        "once": False,
        "params": {
            "c": pack["sweet"],
            "concept": pack["concept"],
            "floor": pack["floor"],
            "cap": pack["cap"],
            "stepUp": pack["stepUp"],
            "stepDown": pack["stepDown"],
        },
        "targets": {"words": pack["words"]},
        "program": "\n".join([
            f"# Agent: {why}",
            f"# Sweet spot {pack['sweet']}. Above {pack['cap']} the reference run loses fluency and the Paris fact.",
            f"layers.{a}-{b}.mlp.down_proj steer {pack['concept']} {fmt(pack['sweet'])}",
        ]),
        "rationale": f"{why} {pack['concept']} on layers {a}-{b} at {fmt(pack['sweet'])}.",
    }


def _propose_cliff(caps):
    if not caps["donor"]:
        return _refuse("The merge cliff needs both models. Pick the Qwen pair.")
    return {
        "strategy": "merge-cliff",
        "stop": False,
        "once": True,
        "params": {"a": 0.5},
        "targets": {"words": []},
        "program": "# Agent: both endpoints work. This is the midpoint the reference run says falls off the cliff.\n* mix 0.5",
        "rationale": "Running the midpoint mix so the probe can show the merge cliff. This is a diagnosis, not a model to keep.",
    }


def _propose_merge(caps, info, analysis):
    if not caps["donor"]:
        return _refuse("A merge needs the Qwen pair: Instruct as the base and Coder as the donor.")
    if analysis is None:
        return {
            "strategy": "merge-safe",
            "needsMap": True,
            "stop": False,
            "rationale": "The straight line between two finetunes often leaves the region where either model works. Mapping cosine similarity before any mix.",
        }
    summary = summarize_cosine(analysis)
    if summary["nMat"] and summary["matrix"] < MATRIX_CLIFF:
        lines = []
        if count_matches(info, "layers.*.*norm"):
            lines.append("layers.*.*norm mix 1")
        if count_matches(info, "layers.*.attn.c_attn.bias"):
            lines.append("layers.*.attn.c_attn.bias mix 1")
        if not lines:
            return _refuse(
                f"Weight matrices have mean cosine {summary['matrix']:.2f}, and this model has no norm "
                "or attention-bias tensors that are safe to graft."
            )
        return {
            "strategy": "merge-safe",
            "stop": False,
            "once": True,
            "params": {},
            "targets": {"words": []},
            "program": "\n".join([
                f"# Agent: mean cosine of weight matrices is {summary['matrix']:.2f}; norms and biases are {summary['norm']:.2f}.",
                "# A global mix would leave the linear basin. Graft only the near-identical tensors.",
                *lines,
            ]),
            "rationale": (
                f"Weight matrices have mean cosine {summary['matrix']:.2f}, so a global mix is refused. "
                f"Grafting norms and biases (mean cosine {summary['norm']:.2f})."
            ),
        }
    return {
        "strategy": "merge-safe",
        "stop": False,
        "once": True,
        "params": {"a": 0.5},
        "targets": {"words": []},
        "program": (
            f"# Agent: weight matrices still point the same way (mean cosine {summary['matrix']:.2f}), "
            f"so a midpoint mix is a fair test.\n* mix 0.5"
        ),
        "rationale": f"Mean matrix cosine is {summary['matrix']:.2f}, above {MATRIX_CLIFF}, so the agent tries a midpoint mix.",
    }


def _propose_noise(intent):
    s = intent["noise"]
    return {
        "strategy": "noise",
        "stop": False,
        "once": True,
        "params": {"noise": s},
        "targets": {"words": []},
        "program": f"# Agent: multiplicative Gaussian noise. 5% survives, 12% drifts, 30% is noise.\nlayers.* noise {fmt(s)}",
        "rationale": f"Applying multiplicative noise at {fmt(s)} and stopping after one probe. The sweep is not a search for a concept.",
    }


def _propose_heads(caps):
    last = max(0, caps["heads"] // 2 - 1)
    return {
        "strategy": "heads",
        "stop": False,
        "once": True,
        "params": {},
        "targets": {"words": []},
        "program": (
            "# Agent: zero the first half of the attention heads (their input columns in o_proj).\n"
            f"layers.*.attn.o_proj zero-heads 0-{last}"
        ),
        "rationale": f"Zeroing heads 0-{last} of {caps['heads']} in every layer.",
    }


def _propose_skip(intent, caps):
    a, b = layer_band(caps["layers"], 4, 6) if intent.get("early") else layer_band(caps["layers"], 10, 12)
    note = (
        "# Early layers are the band that confuses Qwen in the reference run."
        if intent.get("early")
        else "# Middle layers (10-12 on a 24-layer Qwen) stay fluent in the reference run."
    )
    return {
        "strategy": "skip",
        "stop": False,
        "once": True,
        "params": {"early": bool(intent.get("early"))},
        "targets": {"words": []},
        "program": "\n".join([
            "# Agent: zero o_proj and down_proj so these layers add nothing to the residual stream.",
            note,
            f"layers.{a}-{b} skip",
        ]),
        "rationale": f"Skipping {'early' if intent.get('early') else 'middle'} layers {a}-{b}.",
    }


def _propose_scale():
    return {
        "strategy": "scale",
        "stop": False,
        "once": True,
        "params": {},
        "targets": {"words": []},
        "program": "# Agent: turn the MLP output up by the preset factor.\nlayers.*.mlp.down_proj scale 1.3",
        "rationale": "Scaling every mlp.down_proj by 1.3, one probe.",
    }


def _refuse(rationale):
    return {"strategy": "refuse", "stop": True, "program": None, "rationale": rationale, "targets": {"words": []}}


def _rewrite_strength(program, c):
    comments = [line.strip() for line in program.split("\n") if line.strip().startswith("#")]
    body = re.sub(r"(-?\d+(?:\.\d+)?)(\s*)$", lambda m: fmt(c) + m.group(2), rule_body(program))
    return "\n".join([*comments, body])


def _finish(proposal, score):
    if not score:
        return proposal.get("rationale") or "Stopped."
    if proposal["strategy"] == "merge-cliff":
        if score.get("collapse") or not score.get("fluent"):
            return "The midpoint mix left the basin: the endpoints work, the average does not."
        return "The midpoint stayed fluent on these prompts. Check the cosine map before trusting that."
    if proposal["strategy"] == "noise" and (proposal.get("params") or {}).get("noise", 0) >= 0.12:
        strength = fmt(proposal["params"]["noise"])
        if score.get("fluent"):
            return f"Multiplicative noise at {strength} was still fluent on these prompts."
        return f"Noise at {strength} broke fluency. In the reference sweep, 5% survives, 12% drifts, and 30% is noise."
    if holds(proposal, score):
        return "The probe held the concept and the Paris fact, so the agent keeps this program."
    if score.get("factOk") is False:
        return "Paris dropped out of the fact probe. The agent will not push the edit any further."
    if score.get("collapse") or not score.get("fluent"):
        return "The answers collapsed. The agent stops rather than raise the strength."
    return "The agent stops on this program."


def _broken_reason(score):
    if score.get("factOk") is False:
        return "The fact probe lost Paris."
    if score.get("collapse"):
        return "The answers collapsed into repetition or the prompt echoed back."
    return "The answers were no longer fluent."


def _extract_word(goal):
    quoted = re.search(r'["“]([A-Za-z][A-Za-z0-9-]{2,})["”]', goal)
    if quoted:
        return _canon(quoted.group(1))
    for word in INJECT_SWEET:
        if re.search(rf"\b{word}s?\b", goal, re.I | _ASCII):
            return word
    toward = re.search(r"\b(?:toward|about|inject|mention)\s+([A-Za-z][A-Za-z-]{2,})", goal, re.I | _ASCII)
    if not toward:
        return None
    skip = {"the", "model", "answers", "festival", "concept", "weights", "prompts", "this", "that"}
    word = _canon(toward.group(1))
    return None if word in skip else word


def _canon(word):
    text = word.lower()
    if text in INJECT_SWEET:
        return text
    if text.endswith("s") and text[:-1] in INJECT_SWEET:
        return text[:-1]
    return text


def _round1(x):
    return js_round(x * 10) / 10


def _clamp(x, lo, hi):
    return min(hi, max(lo, x))
