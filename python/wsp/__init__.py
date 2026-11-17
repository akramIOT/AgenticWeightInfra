"""Python reference runtime for AgenticWeightInfra.

The browser app still compiles programs into WebGPU passes. This package runs
the same language, edits, security check, control policy, and agent loop on
numeric weights.
"""

from .apply import apply_array
from .catalog import presets_for, setups_for
from .control import AuditLog, evaluate_program, gate_edit
from .inject import solve_injection
from .jscompat import hash_str
from .language import compile_program, parse_program, rule_body, selector_to_matcher
from .model import make_tiny_qwen
from .policy import layer_band, propose, revise
from .rubric import score_rows
from .runtime import build_patched, injection_alignment, load_report, run_program
from .session import run_agent_session
from .tools import dispatch_tool

__all__ = [
    "AuditLog",
    "apply_array",
    "build_patched",
    "compile_program",
    "dispatch_tool",
    "evaluate_program",
    "gate_edit",
    "hash_str",
    "injection_alignment",
    "layer_band",
    "load_report",
    "make_tiny_qwen",
    "parse_program",
    "presets_for",
    "propose",
    "revise",
    "rule_body",
    "run_agent_session",
    "run_program",
    "score_rows",
    "selector_to_matcher",
    "setups_for",
    "solve_injection",
]
