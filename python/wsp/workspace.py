"""File-backed control workspace.

The browser keeps the audit log and the program registry in local storage.
This is the same pair on disk: one directory, an audit chain, and drafts that
a second person can approve. Nothing is uploaded.
"""

import json
from pathlib import Path

from .control import AuditLog, ProgramRegistry, evaluate_program
from .jscompat import hash_str
from .language import compile_program, rule_body
from .runtime import run_program


class JsonFileStorage:
    """One JSON file standing in for one browser storage key."""

    def __init__(self, path):
        self.path = Path(path)

    def getItem(self, _key):
        if not self.path.exists():
            return None
        return self.path.read_text()

    def setItem(self, _key, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(value)


def program_hash(model_id, program):
    return hash_str(f"{model_id}|{rule_body(program)}")


class ControlWorkspace:
    def __init__(self, directory, environment=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        meta_path = self.root / "environment.json"
        if environment is None and meta_path.exists():
            environment = json.loads(meta_path.read_text()).get("environment", "lab")
        environment = environment or "lab"
        if environment not in ("lab", "controlled"):
            raise ValueError(f'Control environment "{environment}" is not lab or controlled.')
        meta_path.write_text(json.dumps({"environment": environment}))
        self.environment = environment
        self.audit = AuditLog(JsonFileStorage(self.root / "audit.json"))
        self.registry = ProgramRegistry(JsonFileStorage(self.root / "registry.json"))

    def explain(self, info, program):
        compiled = compile_program(program, info)
        policy = evaluate_program(program, compiled, info, environment=self.environment)
        return {"compiled": compiled, "policy": policy, "programHash": program_hash(info.rec["model_id"], program)}

    def save_program(self, *, name, model_id, program, author, decision):
        return self.registry.save({
            "name": name,
            "modelId": model_id,
            "program": program,
            "programHash": program_hash(model_id, program),
            "decision": decision,
            "author": author,
        })

    def approve(self, name, approver):
        return self.registry.approve(name, approver)

    def retire(self, name):
        return self.registry.retire(name)

    def release_for(self, model_id, program):
        digest = program_hash(model_id, program)
        for row in self.registry.list():
            if row.get("status") == "approved" and row.get("programHash") == digest and row.get("modelId") == model_id:
                return {"status": "approved", "programHash": digest, "modelId": model_id}
        return None

    def run(self, info, program, *, operator, reason, approved=False, approver="", ticket="", donor=None):
        return run_program(
            info,
            program,
            donor=donor,
            operator=operator,
            reason=reason,
            approved=approved,
            approver=approver,
            ticket=ticket,
            environment=self.environment,
            release=self.release_for(info.rec["model_id"], program),
            audit=self.audit,
        )
