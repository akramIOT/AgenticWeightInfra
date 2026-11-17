"""Load scan and post-edit attestation.

Load: the manifest is well-formed and every finite value sits in a sane range.
Edit: untouched shards still point at the original bytes, neighbors in a
rewritten shard are byte-identical, and the edited values match the op.
"""

import math
import re
from urllib.parse import urljoin

import numpy as np

from .codec import enc_of
from .jscompat import js_round
from .mode import MODE

SECURITY_VERSION = "2026.1"

LIMITS = {
    "maxAbsReview": 64,
    "scaleRel": 0.02,
    "cosTol": 0.02,
    "mixRel": 0.01,
    "zeroNorm": 1e-6,
    "noiseRelFactor": 4,
    "noiseRelPad": 0.05,
    "noiseCosFloor": 0.85,
    "rank1CosFloor": 0.5,
}

_RANK = {"allow": 0, "review": 1, "deny": 2}


def _u16(buf):
    if buf is None:
        return None
    raw = memoryview(buf).cast("B")
    if len(raw) % 2:
        return None
    return np.frombuffer(bytes(raw), dtype="<u2")


def _nonfinite(h, enc):
    if enc == 0:
        return (h & 0x7C00) == 0x7C00
    return (h & 0x7F80) == 0x7F80


def _value(h, enc):
    if enc == 0:
        return float(np.array([h & 0xFFFF], dtype=np.uint16).view(np.float16)[0])
    bits = np.array([(h & 0xFFFF) << 16], dtype=np.uint32)
    return float(bits.view(np.float32)[0])


def scan_encoded(buf, enc):
    u = _u16(buf)
    if u is None:
        return {"n": 0, "nonfinite": 0, "zeros": 0, "maxAbs": 0, "sumSq": 0, "rms": 0, "unreadable": True}
    nonfinite_n = 0
    zeros = 0
    max_abs = 0.0
    sum_sq = 0.0
    for h in u.tolist():
        if _nonfinite(h, enc):
            nonfinite_n += 1
            continue
        if (h & 0x7FFF) == 0:
            zeros += 1
        value = _value(h, enc)
        mag = abs(value)
        if mag > max_abs:
            max_abs = mag
        sum_sq += value * value
    finite = len(u) - nonfinite_n
    return {
        "n": int(len(u)),
        "nonfinite": nonfinite_n,
        "zeros": zeros,
        "maxAbs": max_abs,
        "sumSq": sum_sq,
        "rms": math.sqrt(sum_sq / (finite or 1)),
    }


def _pair_from(a_bytes, b_bytes, enc):
    a = _u16(a_bytes)
    b = _u16(b_bytes)
    empty = {"n": 0, "nonfiniteA": 0, "nonfiniteB": 0, "normA": 0.0, "normB": 0.0, "cos": 0.0, "rel": 1.0, "maxAbsB": 0.0, "unreadable": True}
    if a is None or b is None or len(a) != len(b):
        return empty
    ab = aa = bb = 0.0
    non_a = non_b = 0
    max_b = 0.0
    for ha, hb in zip(a.tolist(), b.tolist()):
        fa, fb = _nonfinite(ha, enc), _nonfinite(hb, enc)
        if fa:
            non_a += 1
        if fb:
            non_b += 1
        if fa or fb:
            continue
        va, vb = _value(ha, enc), _value(hb, enc)
        ab += va * vb
        aa += va * va
        bb += vb * vb
        max_b = max(max_b, abs(vb))
    return {
        "n": int(len(a)),
        "nonfiniteA": non_a,
        "nonfiniteB": non_b,
        "normA": math.sqrt(aa),
        "normB": math.sqrt(bb),
        "cos": ab / math.sqrt((aa * bb) or 1),
        "rel": math.sqrt(max(0.0, aa + bb - 2 * ab)) / math.sqrt(aa or 1),
        "maxAbsB": max_b,
        "unreadable": False,
    }


def scan_pair(a_bytes, b_bytes, enc):
    return _pair_from(a_bytes, b_bytes, enc)


def scan_mix(base_bytes, donor_bytes, patched_bytes, enc, alpha, rel_tol):
    b, d, p = _u16(base_bytes), _u16(donor_bytes), _u16(patched_bytes)
    if b is None or d is None or p is None or len(b) != len(d) or len(b) != len(p):
        return {"mismatch": 1, "nonfinite": 0, "compared": 0, "unreadable": True}
    mismatch = 0
    nonfinite_n = 0
    for hb, hd, hp in zip(b.tolist(), d.tolist(), p.tolist()):
        if _nonfinite(hp, enc) or _nonfinite(hb, enc) or _nonfinite(hd, enc):
            nonfinite_n += 1
            continue
        base = _value(hb, enc)
        donor = _value(hd, enc)
        expected = base + alpha * (donor - base)
        got = _value(hp, enc)
        if not math.isfinite(expected) or not math.isfinite(got):
            nonfinite_n += 1
            continue
        if abs(got - expected) > rel_tol * max(1.0, abs(expected)):
            mismatch += 1
    return {"mismatch": mismatch, "nonfinite": nonfinite_n, "compared": int(len(b))}


def scan_expected(base_bytes, patched_bytes, enc, expected_fn, rel_tol):
    a, p = _u16(base_bytes), _u16(patched_bytes)
    if a is None or p is None or len(a) != len(p):
        return {"mismatch": 1, "nonfinite": 0, "compared": 0, "unreadable": True}
    mismatch = 0
    nonfinite_n = 0
    for i, (hb, hp) in enumerate(zip(a.tolist(), p.tolist())):
        if _nonfinite(hp, enc) or _nonfinite(hb, enc):
            nonfinite_n += 1
            continue
        base = _value(hb, enc)
        expected = expected_fn(i, base)
        got = _value(hp, enc)
        if not math.isfinite(expected) or not math.isfinite(got):
            nonfinite_n += 1
            continue
        if abs(got - expected) > rel_tol * max(1.0, abs(expected)):
            mismatch += 1
    return {"mismatch": mismatch, "nonfinite": nonfinite_n, "compared": int(len(a))}


def bytes_equal(a, b):
    if a is None or b is None or len(a) != len(b):
        return False
    return bytes(a) == bytes(b)


def _fmt(x):
    if not math.isfinite(x):
        return str(x)
    mag = abs(x)
    if mag != 0 and (mag < 0.01 or mag >= 1000):
        text = f"{x:.2e}"
        mant, exp = text.split("e")
        sign = "-" if exp[0] == "-" else ""
        return f"{mant}e{sign}{int(exp[1:] if exp[0] in '+-' else exp)}"
    rounded = js_round(x * 1000) / 1000
    if rounded == int(rounded):
        return str(int(rounded))
    return format(rounded, ".10f").rstrip("0").rstrip(".")


def inspect_structure(info):
    findings = []
    shards = (info.manifest or {}).get("records") or [] if info is not None else []
    if not shards:
        findings.append({"level": "deny", "code": "manifest", "message": "The cached manifest has no shards."})
        return findings
    seen = set()
    n_layers = ((info.config or {}).get("model_config") or {}).get("num_hidden_layers")
    for shard in shards:
        if not isinstance(shard.get("nbytes"), (int, float)) or not math.isfinite(shard.get("nbytes", float("nan"))):
            findings.append({"level": "deny", "code": "manifest", "message": "A shard is missing its byte length."})
        ranges = []
        for record in shard.get("records") or []:
            if record.name in seen:
                findings.append({"level": "deny", "code": "duplicate", "message": f"{record.name} appears more than once in the manifest."})
            seen.add(record.name)
            enc = enc_of(record)
            if enc < 0:
                findings.append({
                    "level": "deny",
                    "code": "dtype",
                    "message": f"{record.name} has dtype {record.dtype or '?'}, which this runtime cannot scan or edit.",
                })
                continue
            n = 1
            for dim in record.shape or []:
                n *= dim
            if n * 2 != record.nbytes:
                findings.append({"level": "deny", "code": "nbytes", "message": f"{record.name} declares {record.nbytes} bytes for {n} values."})
            shard_nbytes = shard.get("nbytes", float("nan"))
            if not math.isfinite(shard_nbytes) or record.byteOffset < 0 or record.byteOffset + record.nbytes > shard_nbytes:
                findings.append({"level": "deny", "code": "range", "message": f"{record.name} extends past its shard."})
            ranges.append(record)
            found = re.search(r"layers\.(\d+)\.", str(record.name))
            if found and isinstance(n_layers, (int, float)) and math.isfinite(n_layers) and int(found.group(1)) >= n_layers:
                findings.append({
                    "level": "deny",
                    "code": "layout",
                    "message": f"{record.name} is past the configured depth of {n_layers} layers.",
                })
        ranges.sort(key=lambda r: r.byteOffset)
        for i in range(1, len(ranges)):
            prev = ranges[i - 1]
            if prev.byteOffset + prev.nbytes > ranges[i].byteOffset:
                findings.append({
                    "level": "deny",
                    "code": "overlap",
                    "message": f"{ranges[i].name} overlaps {prev.name} in the same shard.",
                })
    if not any(name.endswith("embed_tokens.weight") for name in seen):
        findings.append({"level": "review", "code": "inventory", "message": "The manifest has no token embedding."})
    return findings


def judge_baseline(stats):
    findings = []
    for s in stats:
        name = s["name"]
        if s.get("unreadable"):
            findings.append({"level": "deny", "code": "unreadable", "message": f"{name} could not be read from its shard."})
            continue
        if s.get("nonfinite", 0) > 0:
            n = s["nonfinite"]
            findings.append({
                "level": "deny",
                "code": "nonfinite",
                "message": f"{name} contains {n} non-finite value{'' if n == 1 else 's'}.",
            })
            continue
        if name.endswith(".weight") and s.get("n", 0) > 0 and s.get("maxAbs", 0) == 0:
            findings.append({"level": "review", "code": "dead-weight", "message": f"{name} is all zeros before any edit."})
        elif s.get("maxAbs", 0) > LIMITS["maxAbsReview"]:
            findings.append({
                "level": "review",
                "code": "magnitude",
                "message": f"{name} peaks at {_fmt(s['maxAbs'])}, above the review line of {LIMITS['maxAbsReview']}.",
            })
    return findings


def _shape_key(tensor):
    return "x".join(str(d) for d in (tensor.shape or []))


def _abs_url(base, data_path):
    return urljoin(base, data_path)


def link_manifest(base_info, patched_info, compiled):
    findings = []
    plan = compiled.plan if compiled is not None else {}
    base_shards = (getattr(base_info, "manifest", None) or {}).get("records") or []
    patched_shards = (getattr(patched_info, "manifest", None) or {}).get("records") or []
    if not base_shards or len(base_shards) != len(patched_shards):
        findings.append({
            "level": "deny",
            "code": "manifest",
            "message": "The patched manifest does not have the same shards as the loaded model.",
        })
        return findings
    if len(getattr(base_info, "tensors", {}) or {}) != len(getattr(patched_info, "tensors", {}) or {}):
        findings.append({
            "level": "deny",
            "code": "manifest",
            "message": "The patched model does not have the same tensor list as the loaded model.",
        })
    for name, tensor in (base_info.tensors or {}).items():
        patched = patched_info.tensors.get(name) if patched_info.tensors else None
        if patched is None:
            findings.append({"level": "deny", "code": "manifest", "message": f"{name} is missing from the patched model."})
            continue
        if (
            _shape_key(tensor) != _shape_key(patched)
            or enc_of(tensor) != enc_of(patched)
            or tensor.nbytes != patched.nbytes
            or tensor.byteOffset != patched.byteOffset
        ):
            findings.append({
                "level": "deny",
                "code": "layout",
                "message": f"{name} changed shape, dtype, or placement in the patched manifest.",
            })
    model_id = (getattr(patched_info, "rec", None) or {}).get("model_id") or ""
    for si, (base_shard, patched_shard) in enumerate(zip(base_shards, patched_shards)):
        edited = any(record.name in plan for record in (base_shard.get("records") or []))
        b_url = _abs_url(base_info.base, base_shard["dataPath"])
        p_url = _abs_url(patched_info.base, patched_shard["dataPath"])
        if not edited and p_url != b_url:
            findings.append({
                "level": "deny",
                "code": "shard-link",
                "message": f"Shard {si} is outside the program but no longer points at the original bytes.",
            })
        if edited and p_url == b_url:
            findings.append({
                "level": "deny",
                "code": "shard-link",
                "message": f"Shard {si} was edited but still points at the original bytes.",
            })
        if edited and model_id and model_id not in p_url:
            findings.append({
                "level": "deny",
                "code": "shard-link",
                "message": f"Shard {si} was edited but its URL is not under {model_id}.",
            })
    return findings


def _composed_scale(ops):
    if not ops:
        return None
    k = 1.0
    for op in ops:
        if op["mode"] == MODE.zero:
            k = 0.0
        elif op["mode"] == MODE.scale:
            k *= op.get("a") if op.get("a") is not None else 1
        else:
            return None
    return k


def judge_edit(measurements, spills):
    findings = []
    for measurement in measurements or []:
        findings.extend(_judge_one(measurement))
    for spill in spills or []:
        if not spill["identical"]:
            findings.append({
                "level": "deny",
                "code": "spill",
                "message": f"{spill['name']} is outside the program but its bytes changed.",
            })
    return findings


def _judge_one(m):
    findings = []
    if m.get("unreadable"):
        findings.append({"level": "deny", "code": "unreadable", "message": f"{m['name']} could not be compared with the loaded tensor."})
        return findings
    if m.get("nonfiniteB", 0) > 0:
        findings.append({"level": "deny", "code": "nonfinite", "message": f"{m['name']} is non-finite after the edit."})
        return findings
    ops = m.get("ops") or []
    if len(ops) == 1 and ops[0]["mode"] == MODE.mix:
        return findings + _judge_mix(m, ops[0])
    scale = _composed_scale(ops)
    if scale is not None:
        return findings + _judge_scale(m, scale)
    if len(ops) == 1 and ops[0]["mode"] == MODE.noise:
        return findings + _judge_noise(m, ops[0].get("a") or 0)
    if len(ops) == 1 and ops[0]["mode"] == MODE.zerocols:
        if m["normB"] > m["normA"] * (1 + LIMITS["scaleRel"]) + LIMITS["zeroNorm"]:
            findings.append({"level": "deny", "code": "edit-mismatch", "message": f"{m['name']} grew when attention heads were zeroed."})
        return findings
    if len(ops) == 1 and ops[0]["mode"] in (MODE.addvec, MODE.rank1):
        return findings + _judge_vector(m, ops[0])
    if ops:
        findings.append({
            "level": "review",
            "code": "unverified",
            "message": f"{m['name']} has an op chain this check cannot reconstruct. Neighbors in the shard were still compared.",
        })
    return findings


def _judge_mix(m, op):
    if m.get("mix"):
        mix = m["mix"]
        if mix.get("unreadable") or mix.get("mismatch", 0) > 0 or mix.get("nonfinite", 0) > 0:
            return [{"level": "deny", "code": "edit-mismatch", "message": f"{m['name']} does not match mix {op.get('a')}."}]
        return []
    if op.get("a") == 0 and m["rel"] > LIMITS["scaleRel"]:
        return [{"level": "deny", "code": "edit-mismatch", "message": f"{m['name']} changed under mix 0."}]
    if op.get("a") != 0:
        return [{
            "level": "review",
            "code": "unverified",
            "message": f"{m['name']} was mixed without a donor read, so the byte check is incomplete.",
        }]
    return []


def _judge_vector(m, op):
    if m.get("reconstructed"):
        rec = m["reconstructed"]
        if rec.get("unreadable") or rec.get("mismatch", 0) > 0 or rec.get("nonfinite", 0) > 0:
            kind = "injection" if op["mode"] == MODE.addvec else "steer"
            return [{
                "level": "deny",
                "code": "edit-mismatch",
                "message": f"{m['name']} does not match the {kind} that was solved for this program.",
            }]
        return []
    if op["mode"] == MODE.rank1 and m["normA"] > 0 and m["cos"] < LIMITS["rank1CosFloor"]:
        return [{
            "level": "review",
            "code": "drift",
            "message": f"{m['name']} cosine fell to {_fmt(m['cos'])} under a rank-one steer.",
        }]
    return [{
        "level": "review",
        "code": "unverified",
        "message": f"{m['name']} has a vector edit and the solved vector was not available to check against.",
    }]


def _judge_scale(m, k):
    findings = []
    if m["normA"] <= LIMITS["zeroNorm"]:
        if m["normB"] > LIMITS["zeroNorm"]:
            findings.append({
                "level": "deny",
                "code": "edit-mismatch",
                "message": f"{m['name']} was empty and is not empty after scale {_fmt(k)}.",
            })
        return findings
    if k == 0:
        if m["normB"] > LIMITS["zeroNorm"]:
            findings.append({"level": "deny", "code": "edit-mismatch", "message": f"{m['name']} was not zeroed."})
        return findings
    ratio = m["normB"] / m["normA"]
    tol = LIMITS["scaleRel"] * max(1, abs(k))
    if abs(ratio - abs(k)) > tol:
        findings.append({
            "level": "deny",
            "code": "edit-mismatch",
            "message": f"{m['name']} norm ratio is {_fmt(ratio)}, not scale {_fmt(k)}.",
        })
    if abs(m["cos"] - math.copysign(1, k)) > LIMITS["cosTol"]:
        findings.append({
            "level": "deny",
            "code": "edit-mismatch",
            "message": f"{m['name']} cosine is {_fmt(m['cos'])} after scale {_fmt(k)}.",
        })
    return findings


def _judge_noise(m, s):
    findings = []
    mag = abs(s)
    if mag == 0 and m["rel"] > LIMITS["scaleRel"]:
        findings.append({"level": "deny", "code": "edit-mismatch", "message": f"{m['name']} changed under noise 0."})
        return findings
    if m["normA"] <= LIMITS["zeroNorm"]:
        return findings
    if mag > 0 and m["rel"] > mag * LIMITS["noiseRelFactor"] + LIMITS["noiseRelPad"]:
        findings.append({
            "level": "review",
            "code": "drift",
            "message": f"{m['name']} moved by relative {_fmt(m['rel'])}, wider than noise {_fmt(s)}.",
        })
    if mag > 0 and mag <= 0.3 and m["cos"] < LIMITS["noiseCosFloor"]:
        findings.append({
            "level": "review",
            "code": "drift",
            "message": f"{m['name']} cosine fell to {_fmt(m['cos'])} under noise {_fmt(s)}.",
        })
    return findings


def _finish(phase, findings, extra):
    decision = "allow"
    for finding in findings:
        if _RANK[finding["level"]] > _RANK[decision]:
            decision = finding["level"]
    return {"version": SECURITY_VERSION, "phase": phase, "decision": decision, "findings": findings, **extra}


def assess_load(info, stats):
    findings = [*inspect_structure(info), *judge_baseline(stats)]
    max_abs = max([s.get("maxAbs") or 0 for s in stats], default=0)
    nonfinite = sum(s.get("nonfinite") or 0 for s in stats)
    noun = "tensor" if len(stats) == 1 else "tensors"
    return _finish("load", findings, {
        "summary": f"{len(stats)} {noun} scanned, max |w| {_fmt(max_abs)}, {nonfinite} non-finite.",
        "modelId": (getattr(info, "rec", None) or {}).get("model_id"),
        "tensors": len(stats),
        "maxAbs": max_abs,
        "nonfinite": nonfinite,
    })


def assess_edit(*, base_info, patched_info, compiled, measurements=None, spills=None):
    measurements = measurements or []
    spills = spills or []
    findings = [*link_manifest(base_info, patched_info, compiled), *judge_edit(measurements, spills)]
    same = sum(1 for s in spills if s["identical"])
    noun = "tensor" if len(measurements) == 1 else "tensors"
    summary = f"{len(measurements)} edited {noun} checked, {same}/{len(spills)} neighbors byte-identical."
    worst = None
    for measurement in measurements:
        if worst is None or measurement.get("rel", -1) > worst.get("rel", -1):
            worst = measurement
    if worst is not None:
        summary += f" Largest relative move is {_fmt(worst['rel'])} on {worst['name']}."
    return _finish("edit", findings, {"summary": summary, "edited": len(measurements), "spills": len(spills)})


def block_reason(report):
    if not report:
        return "Weight security has not been evaluated."
    denied = [f["message"] for f in report.get("findings") or [] if f["level"] == "deny"]
    return " ".join(denied) or report.get("summary") or "Weight security denied this model."


def _slice(shard, record):
    if shard is None:
        return None
    end = record.byteOffset + record.nbytes
    if record.byteOffset < 0 or end > len(shard):
        return None
    return shard[record.byteOffset:end]


def attest_loaded(info, read_shard):
    jobs = []
    seen = set()
    for si, shard in enumerate((info.manifest or {}).get("records") or []):
        for record in shard.get("records") or []:
            if record.name in seen:
                continue
            seen.add(record.name)
            jobs.append((si, record))
    stats = []
    for si, record in jobs:
        enc = enc_of(record)
        if enc < 0:
            continue
        shard = read_shard(si)
        view = _slice(shard, record)
        if view is None:
            stats.append({"name": record.name, "n": 0, "nonfinite": 0, "maxAbs": 0, "unreadable": True})
        else:
            stats.append({"name": record.name, **scan_encoded(view, enc)})
    return assess_load(info, stats)


def attest_edited(*, base_info, patched_info, compiled, read_base_shard, read_patched_shard, read_donor=None, read_vector=None):
    plan = compiled.plan if compiled is not None else {}
    shards = (base_info.manifest or {}).get("records") or []
    edited_at = [si for si, shard in enumerate(shards) if any(r.name in plan for r in (shard.get("records") or []))]
    measurements = []
    spills = []
    for si in edited_at:
        base_bytes = read_base_shard(si)
        patched_bytes = read_patched_shard(si)
        for record in shards[si].get("records") or []:
            a = _slice(base_bytes, record)
            b = _slice(patched_bytes, record)
            if a is None or b is None:
                measurements.append({
                    "name": record.name,
                    "ops": plan.get(record.name) or [],
                    "unreadable": True,
                    "rel": 1,
                    "nonfiniteB": 0,
                    "normA": 0,
                    "normB": 0,
                    "cos": 0,
                })
                continue
            if record.name not in plan:
                spills.append({"name": record.name, "identical": bytes_equal(a, b)})
                continue
            ops = plan[record.name]
            enc = enc_of(record)
            measured = scan_pair(a, b, enc)
            extra = _exact_check(record, ops, a, b, enc, read_donor, read_vector)
            measurements.append({"name": record.name, "ops": ops, **measured, **extra})
    return assess_edit(
        base_info=base_info,
        patched_info=patched_info,
        compiled=compiled,
        measurements=measurements,
        spills=spills,
    )


def _exact_check(record, ops, base_bytes, patched_bytes, enc, read_donor, read_vector):
    if not ops or len(ops) != 1:
        return {}
    op = ops[0]
    if op["mode"] == MODE.mix and read_donor:
        donor = read_donor(record.name)
        if donor is not None:
            return {"mix": scan_mix(base_bytes, donor, patched_bytes, enc, op.get("a") or 0, LIMITS["mixRel"])}
    if op["mode"] in (MODE.addvec, MODE.rank1) and read_vector:
        vec = read_vector(op)
        if vec is not None:
            cols = record.shape[-1]
            vec = np.asarray(vec, dtype=np.float64).reshape(-1)

            def expected(i, base, op=op, vec=vec, cols=cols):
                if op["mode"] == MODE.addvec:
                    return base + (op.get("a") if op.get("a") is not None else 1) * (vec[i] if i < vec.size else float("nan"))
                row, col = divmod(i, cols)
                rows = vec.size - cols
                return base + (op.get("a") or 0) * vec[row] * vec[rows + col]

            return {"reconstructed": scan_expected(base_bytes, patched_bytes, enc, expected, LIMITS["mixRel"])}
    return {}
