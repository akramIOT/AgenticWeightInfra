"""End-to-end reference runtime.

Compile a program, solve injections and steering packs, rewrite the f16 shard,
then attest the patched bytes against the plan. The browser still does this
on WebGPU; these are the same rules on numeric weights.
"""

import re
from dataclasses import dataclass, replace
from urllib.parse import urljoin

import numpy as np

from .apply import apply_encoded
from .codec import decode, enc_of
from .control import AuditLog, evaluate_program, gate_edit
from .inject import grouped_matrix, resolve_injects, token_id_of
from .jscompat import hash_str
from .language import compile_program, rule_body
from .mode import MODE
from .security import attest_edited, attest_loaded, block_reason
from .steer import resolve_steers


@dataclass
class Build:
    compiled: object
    patched: object
    inject_vecs: dict
    steer_vecs: dict
    logs: list
    rec: dict
    report: dict


def decode_tensor(info, name):
    tensor = info.tensors[name]
    raw = info.shards[tensor.shard][tensor.byteOffset : tensor.byteOffset + tensor.nbytes]
    return decode(raw, enc_of(tensor)).reshape(tensor.shape)


def patched_record(info, program_text, donor_info=None):
    donor_id = "" if donor_info is None else donor_info.rec["model_id"]
    body = re.sub(r"#.*$", "", program_text, flags=re.M)
    body = re.sub(r"\s+", " ", body).strip()
    ident = "wsp-" + hash_str(info.rec["model_id"] + "|" + donor_id + "|" + body)
    base = f"{info.origin}wsp-models/{ident}/resolve/main/"
    rec = {**info.rec, "model_id": ident, "model": base}
    return rec, base


def build_patched(info, program, donor=None, packs=None, log=None):
    compiled = compile_program(program, info, donor)
    if compiled.errors:
        raise ValueError("\n".join(compiled.errors))
    logs = []

    def emit(message):
        logs.append(message)
        if log:
            log(message)

    inject_vecs = resolve_injects(info, compiled.injects, log=emit)
    steer_vecs, fingerprint = resolve_steers(info, compiled.steers, log=emit, packs=packs)
    rec, base = patched_record(info, program + fingerprint, donor)
    shards = []
    manifest_records = []
    for si, shard_meta in enumerate(info.manifest["records"]):
        records = shard_meta["records"]
        edited = any(record.name in compiled.plan for record in records)
        if not edited:
            shards.append(info.shards[si])
            manifest_records.append({
                **shard_meta,
                "dataPath": urljoin(info.base, shard_meta["dataPath"]),
            })
            continue
        shard = np.array(info.shards[si], copy=True)
        for record in records:
            ops = compiled.plan.get(record.name)
            if not ops:
                continue
            prepared = []
            for op in ops:
                if op["mode"] == MODE.addvec:
                    prepared.append({**op, "vec": inject_vecs[f"{op['inject']}:{op['layer']}"]})
                elif op["mode"] == MODE.rank1:
                    prepared.append({
                        **op,
                        "vec": steer_vecs[f"{op['steer']}:{op['layer']}"],
                        "c0": record.shape[0],
                    })
                else:
                    prepared.append(op)
            donor_bytes = None
            if any(op.get("donor") for op in prepared):
                donor_tensor = donor.tensors[record.name]
                donor_view = donor.shards[donor_tensor.shard]
                start = donor_tensor.byteOffset
                donor_bytes = donor_view[start : start + donor_tensor.nbytes]
            view = shard[record.byteOffset : record.byteOffset + record.nbytes]
            apply_encoded(view, prepared, enc_of(record), donor=donor_bytes, row_len=record.shape[-1])
        shards.append(shard)
        manifest_records.append({
            "dataPath": shard_meta["dataPath"],
            "nbytes": shard_meta["nbytes"],
            "records": records,
        })
    patched = replace(
        info,
        rec=rec,
        base=base,
        shards=shards,
        manifest={"records": manifest_records},
        weights={},
    )
    report = attest_edited(
        base_info=info,
        patched_info=patched,
        compiled=compiled,
        read_base_shard=lambda i: info.shards[i],
        read_patched_shard=lambda i: patched.shards[i],
        read_donor=_donor_reader(donor),
        read_vector=_vector_reader(inject_vecs, steer_vecs),
    )
    return Build(compiled, patched, inject_vecs, steer_vecs, logs, rec, report)


def run_program(info, program, *, donor=None, packs=None, operator="operator", reason="reference run", approved=False, ticket="", audit=None):
    """Gate the edit, apply it, attest it, and append the decision to the audit log."""
    compiled = compile_program(program, info, donor)
    policy = evaluate_program(program, compiled, info)
    gate = gate_edit(policy, {"operator": operator, "reason": reason, "ticket": ticket}, approved=approved)
    log = audit or AuditLog()
    if not gate["ok"]:
        log.record({
            "action": "edit",
            "decision": policy["decision"],
            "blocked": True,
            "program": program,
            "reason": gate["blockReason"],
            "operator": operator,
        })
        return {"applied": False, "policy": policy, "gate": gate, "compiled": compiled, "build": None, "audit": log}
    build = build_patched(info, program, donor=donor, packs=packs)
    blocked = build.report["decision"] == "deny"
    log.record({
        "action": "edit",
        "decision": build.report["decision"],
        "blocked": blocked,
        "program": rule_body(program),
        "operator": operator,
        "reason": reason,
        "approved": approved,
        "summary": build.report["summary"],
    })
    return {"applied": not blocked, "policy": policy, "gate": gate, "compiled": compiled, "build": build, "audit": log}


def load_report(info):
    return attest_loaded(info, lambda i: info.shards[i])


def injection_alignment(info, patched, word, layers):
    """Cosine between the constant residual add and the concept direction, per layer."""
    mc = info.config["model_config"]
    hidden = mc["hidden_size"]
    n_h = mc["num_attention_heads"]
    n_kv = mc["num_key_value_heads"]
    hd = mc["head_dim"]
    embed = decode_tensor(info, "model.embed_tokens.weight")
    direction = embed[token_id_of(info, word)].astype(np.float64)
    direction = direction / np.linalg.norm(direction)
    scores = []
    v_off = n_h * hd + n_kv * hd
    k = n_kv * hd
    for layer in layers:
        bias = decode_tensor(patched, f"model.layers.{layer}.self_attn.qkv_proj.bias").reshape(-1)
        proj = decode_tensor(patched, f"model.layers.{layer}.self_attn.o_proj.weight")
        expressed = grouped_matrix(proj, hidden, n_h, n_kv, hd) @ bias[v_off : v_off + k]
        denom = float(np.linalg.norm(expressed) * np.linalg.norm(direction))
        scores.append(float(expressed @ direction / denom) if denom else 0.0)
    return scores


def _donor_reader(donor):
    if donor is None:
        return None

    def read(name):
        tensor = donor.tensors[name]
        shard = donor.shards[tensor.shard]
        return shard[tensor.byteOffset : tensor.byteOffset + tensor.nbytes]

    return read


def _vector_reader(inject_vecs, steer_vecs):
    def read(op):
        if op["mode"] == MODE.addvec:
            return inject_vecs.get(f"{op['inject']}:{op['layer']}")
        if op["mode"] == MODE.rank1:
            return steer_vecs.get(f"{op['steer']}:{op['layer']}")
        return None

    return read


def block_message(report):
    return block_reason(report)
