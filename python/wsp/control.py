"""Change-control policy and a local audit log.

Thresholds match the reference runs: noise is routine at 5%, review above 12%,
denied above 30%; injection is review above 20 and denied above 24; a steer is
review above 1.4 and denied above 2. A full-model midpoint mix and an early-layer
skip are review.
"""

import json
import math
import re
from datetime import datetime, timezone

from .jscompat import hash_str
from .language import parse_program

POLICY_VERSION = "2026.1"
_RANK = {"allow": 0, "review": 1, "deny": 2}


def evaluate_program(text, compiled, info, environment="lab"):
    env = environment or "lab"
    findings = []
    if env not in ("lab", "controlled"):
        findings.append({
            "level": "deny",
            "code": "environment",
            "message": f'Control environment "{env}" is not lab or controlled.',
        })
        return _finish(findings, compiled, env)
    if compiled.errors:
        for message in compiled.errors:
            findings.append({"level": "deny", "code": "compile", "message": message})
        return _finish(findings, compiled, env)
    rules = parse_program(text)["rules"]
    mc = (info.config or {}).get("model_config") or {} if info is not None else {}
    n = mc.get("num_hidden_layers", 1)
    n_h = mc.get("num_attention_heads", 1)
    for rule in rules:
        _judge_rule(rule, findings, n=n, n_h=n_h, compiled=compiled)
    fraction = _fraction(compiled)
    wiped = any(rule["op"] == "zero" and rule["sel"] in ("*", "all") for rule in rules)
    if fraction > 0.5 and not wiped:
        findings.append({
            "level": "review",
            "code": "breadth",
            "message": f"This edit touches {fraction * 100:.1f}% of parameters, above the 50% routine ceiling.",
        })
    if env == "controlled" and any(rule["op"] in ("inject", "steer", "noise") for rule in rules):
        findings.append({
            "level": "review",
            "code": "controlled",
            "message": "A controlled environment requires approval before injection, steering, or noise is applied.",
        })
    return _finish(findings, compiled, env)


def gate_edit(policy, change=None, approved=False, release=None):
    change = change or {}
    if not policy:
        return {"ok": False, "blockReason": "Control policy has not been evaluated."}
    if policy["decision"] == "deny":
        why = " ".join(f["message"] for f in policy["findings"] if f["level"] == "deny")
        return {"ok": False, "blockReason": why or "The control policy denied this edit."}
    operator = str(change.get("operator") or "").strip()
    approver = str(change.get("approver") or "").strip()
    reason = str(change.get("reason") or "").strip()
    ticket = str(change.get("ticket") or "").strip()
    if len(operator) < 2:
        return {"ok": False, "blockReason": "Record the operator name before applying an edit."}
    if len(reason) < 8:
        return {"ok": False, "blockReason": "Record a change reason of at least 8 characters."}
    if ticket and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-\d+", ticket):
        return {"ok": False, "blockReason": "Ticket must look like CHG-1042."}
    if policy["decision"] == "review":
        if not ticket:
            return {"ok": False, "blockReason": "A review-level edit needs a ticket like CHG-1042."}
        if not approved:
            return {
                "ok": False,
                "blockReason": "This edit is above the routine threshold. Approve it explicitly. The approval is written to the audit log.",
            }
        if len(approver) < 2:
            return {"ok": False, "blockReason": "A review-level edit needs an approver who is not the operator."}
        if approver.lower() == operator.lower():
            return {"ok": False, "blockReason": "The approver must be a different person from the operator."}
    if (policy.get("environment") or "lab") == "controlled":
        digest = str(change.get("programHash") or "")
        model_id = str(change.get("modelId") or "")
        promoted = (
            release
            and release.get("status") == "approved"
            and release.get("programHash") == digest
            and (not model_id or not release.get("modelId") or release.get("modelId") == model_id)
        )
        if not promoted:
            return {"ok": False, "blockReason": "A controlled environment runs only a program a second person has approved."}
    return {"ok": True}


def _judge_rule(rule, findings, *, n, n_h, compiled):
    def num(i):
        try:
            return float(rule["args"][i])
        except (IndexError, ValueError):
            return float("nan")

    if rule["op"] == "zero" and rule["sel"] in ("*", "all", "embed"):
        findings.append({
            "level": "deny",
            "code": "wipe",
            "message": f'Zeroing "{rule["sel"]}" removes a part of the model the control policy does not allow.',
        })
    if rule["op"] == "mix" and rule["sel"] in ("*", "all"):
        a = num(0)
        if a < 0 or a > 1:
            findings.append({
                "level": "deny",
                "code": "mix-extrapolate",
                "message": f"A full-model mix at {a} leaves the segment between the two models.",
            })
        elif a != 0 and a != 1:
            findings.append({
                "level": "review",
                "code": "mix-midpoint",
                "message": f"A full-model mix at {a} is the merge cliff. It needs an explicit approval.",
            })
        elif a == 1:
            findings.append({
                "level": "review",
                "code": "mix-swap",
                "message": "A full swap replaces every tensor of the base model. It needs an explicit approval.",
            })
    if rule["op"] == "noise":
        s = num(0)
        if s > 0.3:
            findings.append({
                "level": "deny",
                "code": "noise",
                "message": f"Noise {s} is past the point where the reference run becomes noise. The ceiling is 0.3.",
            })
        elif s > 0.12:
            findings.append({
                "level": "review",
                "code": "noise",
                "message": f"Noise {s} is above the 12% drift point. It needs an explicit approval.",
            })
    if rule["op"] == "inject":
        c = num(1)
        if c > 24:
            findings.append({
                "level": "deny",
                "code": "inject",
                "message": f"Injection strength {c} is past the repetition regime. The ceiling is 24.",
            })
        elif c > 20:
            findings.append({
                "level": "review",
                "code": "inject",
                "message": f"Injection strength {c} is above the routine cap of 20.",
            })
    if rule["op"] == "steer":
        c = num(1)
        if c > 2:
            findings.append({
                "level": "deny",
                "code": "steer",
                "message": f"Steer strength {c} is past the fluency cliff. The ceiling is 2.",
            })
        elif c > 1.4:
            findings.append({
                "level": "review",
                "code": "steer",
                "message": f"Steer strength {c} is above 1.4, where the reference run loses fluency and facts.",
            })
    if rule["op"] == "scale":
        k = num(0)
        if not (k > 0) or k > 3:
            findings.append({
                "level": "deny",
                "code": "scale",
                "message": f"Scale {k} is outside the allowed range of (0, 3].",
            })
        elif k < 0.5 or k > 1.5:
            findings.append({
                "level": "review",
                "code": "scale",
                "message": f"Scale {k} is outside the routine band of 0.5 to 1.5.",
            })
    if rule["op"] == "skip":
        minimum = _min_layer(compiled, re.compile(r"(o_proj|down_proj)\.weight$"))
        if math_isfinite(minimum) and minimum / n < 0.25:
            findings.append({
                "level": "review",
                "code": "skip-early",
                "message": f"Skipping from layer {minimum} hits the early band, which confuses the model in the reference run.",
            })
    if rule["op"] == "zero-heads":
        found = re.fullmatch(r"(\d+)(?:-(\d+))?", str(rule["args"][0] if rule["args"] else ""))
        if found:
            count = (int(found.group(2)) if found.group(2) else int(found.group(1))) - int(found.group(1)) + 1
            if count > n_h / 2:
                findings.append({
                    "level": "review",
                    "code": "heads",
                    "message": f"Zeroing {count} of {n_h} heads is more than half the attention block.",
                })


def _min_layer(compiled, pattern):
    minimum = float("inf")
    for name in compiled.plan:
        if not pattern.search(name):
            continue
        found = re.search(r"layers\.(\d+)\.", name)
        if found:
            minimum = min(minimum, int(found.group(1)))
    return minimum


def _fraction(compiled):
    total = (compiled.stats or {}).get("total") or 0
    if not total:
        return 0
    return compiled.stats["touched"] / total


def _finish(findings, compiled, environment):
    decision = "allow"
    for finding in findings:
        if _RANK[finding["level"]] > _RANK[decision]:
            decision = finding["level"]
    return {
        "version": POLICY_VERSION,
        "environment": environment,
        "decision": decision,
        "findings": findings,
        "fraction": _fraction(compiled),
    }


GENESIS = "00000000"


def canonical(value):
    """Stable JSON with sorted keys. Integer-valued floats match JavaScript."""
    if value is None or isinstance(value, (str, bool, int)):
        return json.dumps(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return json.dumps(int(value))
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value)
        return "{" + ",".join(json.dumps(key) + ":" + canonical(value[key]) for key in keys) + "}"
    return json.dumps(value)


def chain_digest(prev, entry):
    body = {key: value for key, value in entry.items() if key != "digest"}
    return hash_str(f"{prev}\n{canonical(body)}")


def math_isfinite(value):
    return value == value and value not in (float("inf"), float("-inf"))


class MemoryStorage:
    """Stand-in for the browser local store. Records stay on this object."""

    def __init__(self):
        self._items = {}

    def getItem(self, key):
        return self._items.get(key)

    def setItem(self, key, value):
        self._items[key] = value


def _write_storage(storage, key, payload):
    try:
        storage.setItem(key, payload)
    except Exception as exc:
        name = getattr(exc, "name", "") or ""
        code = getattr(exc, "code", None)
        if name == "QuotaExceededError" or code in (22, 1014):
            raise RuntimeError("This browser refused to store more local records.") from exc
        raise RuntimeError("This browser refused to store local records.") from exc


class AuditLog:
    def __init__(self, storage=None, key="awi.audit.v1", limit=200):
        self.storage = storage or MemoryStorage()
        self.key = key
        self.limit = limit

    def list(self):
        try:
            value = json.loads(self.storage.getItem(self.key) or "[]")
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    def record(self, entry):
        rows = self.list()
        prev = rows[-1].get("digest", "") if rows else GENESIS
        row = {
            "id": f"{int(datetime.now(timezone.utc).timestamp() * 1000):x}-{len(rows)}",
            "at": datetime.now(timezone.utc).isoformat(),
            **entry,
            "prev": prev,
        }
        row["digest"] = chain_digest(prev, row)
        rows.append(row)
        kept = rows[-self.limit :]
        if len(kept) < len(rows):
            # Dropping the head would leave the next row pointing at a missing digest.
            prev_digest = GENESIS
            for item in kept:
                item["prev"] = prev_digest
                item["digest"] = chain_digest(prev_digest, item)
                prev_digest = item["digest"]
        _write_storage(self.storage, self.key, json.dumps(kept))
        return kept[-1]

    def verify(self):
        rows = self.list()
        prev = GENESIS
        for index, row in enumerate(rows):
            expect = chain_digest(prev, row)
            if row.get("prev") != prev or row.get("digest") != expect:
                return {"ok": False, "brokenAt": index, "head": None}
            prev = row["digest"]
        return {"ok": True, "brokenAt": None, "head": prev}

    def clear(self, meta=None):
        rows = self.list()
        severed = rows[-1].get("digest", "") if rows else GENESIS
        _write_storage(self.storage, self.key, "[]")
        return self.record({
            "action": "clear",
            "decision": "allow",
            "blocked": False,
            "severed": severed,
            **(meta or {}),
        })

    def export(self):
        return json.dumps(self.list())


class ProgramRegistry:
    def __init__(self, storage=None, key="awi.registry.v1"):
        self.storage = storage or MemoryStorage()
        self.key = key

    def list(self):
        try:
            value = json.loads(self.storage.getItem(self.key) or "[]")
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    def save(self, entry):
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Name the program before saving it.")
        rows = [row for row in self.list() if row.get("name") != name]
        row = {
            "name": name,
            "savedAt": datetime.now(timezone.utc).isoformat(),
            "modelId": entry.get("modelId"),
            "program": entry.get("program"),
            "programHash": entry.get("programHash") or hash_str(entry.get("program") or ""),
            "decision": entry.get("decision"),
            "status": "draft",
            "author": str(entry.get("author") or "").strip(),
        }
        rows.append(row)
        _write_storage(self.storage, self.key, json.dumps(rows))
        return row

    def approve(self, name, approver):
        who = str(approver or "").strip()
        if len(who) < 2:
            raise ValueError("Record the approver before promoting a program.")
        rows = self.list()
        row = next((item for item in rows if item.get("name") == name), None)
        if row is None:
            raise ValueError("Save the program before approving it.")
        if row.get("status") == "retired":
            raise ValueError("A retired program must be saved again before it can be approved.")
        author = str(row.get("author") or "").strip()
        if len(author) < 2:
            raise ValueError("Record the operator as the author before a second person approves the program.")
        if author.lower() == who.lower():
            raise ValueError("The approver must be a different person from the author.")
        row["status"] = "approved"
        row["approvedBy"] = who
        row["approvedAt"] = datetime.now(timezone.utc).isoformat()
        _write_storage(self.storage, self.key, json.dumps(rows))
        return row

    def retire(self, name):
        rows = self.list()
        row = next((item for item in rows if item.get("name") == name), None)
        if row is None:
            raise ValueError("Save the program before retiring it.")
        row["status"] = "retired"
        row.pop("approvedBy", None)
        row.pop("approvedAt", None)
        _write_storage(self.storage, self.key, json.dumps(rows))
        return row

    def get(self, name):
        return next((row for row in self.list() if row.get("name") == name), None)
