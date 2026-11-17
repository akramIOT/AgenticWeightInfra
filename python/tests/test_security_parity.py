"""The same security, control, and catalog cases the JavaScript suite checks."""

import json
import unittest
from types import SimpleNamespace
from urllib.parse import urljoin

import numpy as np

from wsp.catalog import presets_for, setups_for
from wsp.control import GENESIS, AuditLog, ProgramRegistry, chain_digest, evaluate_program, gate_edit
from wsp.language import compile_program
from wsp.model import Tensor
from wsp.security import LIMITS, attest_edited, attest_loaded, inspect_structure, scan_encoded
from wsp.tools import dispatch_tool

BASE = "https://example.invalid/org/model/resolve/main/"
PATCHED = "https://example.invalid/wsp-models/wsp-test/resolve/main/"


def f16_bytes(values):
    raw = np.asarray(values, dtype=np.float64)
    return np.ascontiguousarray(raw.astype(np.float16)).view(np.uint8).tobytes()


def bf16_bytes(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    hi = (bits >> np.uint32(16)).astype(np.uint16)
    return np.ascontiguousarray(hi).view(np.uint8).tobytes()


def make_model(groups):
    tensors = {}
    shards = []
    records_by_shard = []
    for si, group in enumerate(groups):
        blob = bytearray()
        records = []
        for item in group:
            raw = item["bytes"] if "bytes" in item else f16_bytes(item["values"])
            shape = tuple(item.get("shape") or [len(raw) // 2])
            record = Tensor(
                name=item["name"],
                shape=shape,
                dtype=item.get("dtype", "float16"),
                format=item.get("format", "f16"),
                nbytes=len(raw),
                byteOffset=len(blob),
                shard=si,
            )
            tensors[record.name] = record
            records.append(record)
            blob += raw
        shards.append(np.frombuffer(bytes(blob), dtype=np.uint8).copy())
        records_by_shard.append({
            "dataPath": f"shard-{si}.bin",
            "nbytes": len(blob),
            "records": records,
        })
    info = SimpleNamespace(
        rec={"model_id": "base-model"},
        base=BASE,
        manifest={"records": records_by_shard},
        config={"model_config": {
            "num_hidden_layers": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "hidden_size": 4,
        }},
        tensors=tensors,
        shards=shards,
    )
    return info


def down(values):
    return {"name": "model.layers.0.mlp.down_proj.weight", "values": values, "shape": [2, 2]}


EMBED = {"name": "model.embed_tokens.weight", "values": [0.5, -0.25, 0.125, 0.5]}


def healthy():
    return make_model([[down([1, 2, 3, 4])], [EMBED]])


def as_patched(model, edited):
    records = []
    for si, shard in enumerate(model.manifest["records"]):
        if si in edited:
            records.append(dict(shard))
        else:
            records.append({**shard, "dataPath": urljoin(model.base, shard["dataPath"])})
    info = SimpleNamespace(
        rec={"model_id": "wsp-test"},
        base=PATCHED,
        manifest={"records": records},
        config=model.config,
        tensors=model.tensors,
        shards=model.shards,
    )

    def read(si):
        return edited[si] if si in edited else model.shards[si]

    return info, read


def shard_with(model, si, tensor_bytes):
    out = np.array(model.shards[si], copy=True)
    out[: len(tensor_bytes)] = np.frombuffer(tensor_bytes, dtype=np.uint8)
    return out


def enterprise_model():
    tensors = {}

    def add(name, shape):
        tensors[name] = Tensor(name, tuple(shape))

    add("model.embed_tokens.weight", (32, 8))
    for layer in range(24):
        add(f"model.layers.{layer}.self_attn.qkv_proj.bias", (8,))
        add(f"model.layers.{layer}.self_attn.o_proj.weight", (8, 8))
        add(f"model.layers.{layer}.mlp.down_proj.weight", (8, 16))
        add(f"model.layers.{layer}.mlp.gate_up_proj.weight", (16, 8))
        add(f"model.layers.{layer}.input_layernorm.weight", (8,))
    info = SimpleNamespace(
        tensors=tensors,
        config={"model_config": {
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_size": 64,
        }},
    )
    return info


class SecurityParityTest(unittest.TestCase):
    def test_scan_reports_peak_infinity_and_bf16(self):
        finite = scan_encoded(f16_bytes([1, -4, 0.5]), 0)
        self.assertEqual(finite["nonfinite"], 0)
        self.assertEqual(finite["maxAbs"], 4)
        self.assertEqual(finite["n"], 3)
        broken = scan_encoded(f16_bytes([1, float("inf")]), 0)
        self.assertEqual(broken["nonfinite"], 1)
        wide = scan_encoded(bf16_bytes([1, -2]), 1)
        self.assertEqual(wide["maxAbs"], 2)
        self.assertEqual(wide["nonfinite"], 0)

    def test_broken_manifest_is_denied(self):
        model = healthy()
        overlap = {
            "records": [
                {
                    **model.manifest["records"][0],
                    "records": [
                        *model.manifest["records"][0]["records"],
                        Tensor(
                            "model.layers.0.mlp.down_proj.weight.copy",
                            (2, 2),
                            nbytes=model.manifest["records"][0]["records"][0].nbytes,
                            byteOffset=2,
                        ),
                    ],
                },
                model.manifest["records"][1],
            ]
        }
        codes = [f["code"] for f in inspect_structure(SimpleNamespace(
            manifest=overlap, config=model.config, tensors=model.tensors,
        ))]
        self.assertIn("overlap", codes)
        self.assertNotIn("duplicate", codes)

        bad = healthy()
        bad.manifest["records"][0]["records"][0].nbytes = 2
        self.assertTrue(any(f["code"] == "nbytes" for f in inspect_structure(bad)))

        deep = healthy()
        deep.manifest["records"][0]["records"][0].name = "model.layers.24.mlp.down_proj.weight"
        self.assertTrue(any(f["code"] == "layout" for f in inspect_structure(deep)))

    def test_load_allows_finite_reviews_huge_or_dead_and_denies_nonfinite(self):
        model = healthy()
        ok = attest_loaded(model, lambda si: model.shards[si])
        self.assertEqual(ok["decision"], "allow")
        self.assertEqual(ok["version"], "2026.1")
        self.assertEqual(ok["nonfinite"], 0)

        huge = make_model([[down([1, LIMITS["maxAbsReview"] + 10, 1, 1])], [EMBED]])
        reviewed = attest_loaded(huge, lambda si: huge.shards[si])
        self.assertEqual(reviewed["decision"], "review")
        self.assertTrue(any(f["code"] == "magnitude" for f in reviewed["findings"]))

        dead = make_model([[down([0, 0, 0, 0])], [EMBED]])
        quiet = attest_loaded(dead, lambda si: dead.shards[si])
        self.assertEqual(quiet["decision"], "review")
        self.assertTrue(any(f["code"] == "dead-weight" for f in quiet["findings"]))

        corrupt = make_model([[down([1, float("inf"), 1, 1])], [EMBED]])
        denied = attest_loaded(corrupt, lambda si: corrupt.shards[si])
        self.assertEqual(denied["decision"], "deny")
        self.assertTrue(any(f["code"] == "nonfinite" for f in denied["findings"]))

    def test_edit_must_match_the_program(self):
        model = healthy()
        compiled = compile_program("layers.0.mlp.down_proj scale 2", model, None)
        self.assertEqual(compiled.errors, [])
        scaled, read_scaled = as_patched(model, {0: shard_with(model, 0, f16_bytes([2, 4, 6, 8]))})
        allow = attest_edited(
            base_info=model, patched_info=scaled, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_scaled,
        )
        self.assertEqual(allow["decision"], "allow", allow["findings"])

        untouched, read_untouched = as_patched(model, {0: model.shards[0]})
        missed = attest_edited(
            base_info=model, patched_info=untouched, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_untouched,
        )
        self.assertEqual(missed["decision"], "deny")
        self.assertTrue(any(f["code"] == "edit-mismatch" for f in missed["findings"]))

    def test_a_zero_may_not_spill_into_a_neighbor(self):
        model = make_model([[
            {"name": "model.layers.10.self_attn.o_proj.weight", "values": [1, 2, 3, 4]},
            {"name": "model.layers.10.mlp.down_proj.weight", "values": [1, 1, 1, 1]},
        ], [EMBED]])
        compiled = compile_program("layers.10.attn.o_proj zero", model, None)
        self.assertEqual(len(compiled.plan), 1)
        bytes_ = shard_with(model, 0, f16_bytes([0, 0, 0, 0]))
        good, read_good = as_patched(model, {0: bytes_})
        allow = attest_edited(
            base_info=model, patched_info=good, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_good,
        )
        self.assertEqual(allow["decision"], "allow", allow["findings"])

        spilled = np.array(bytes_, copy=True)
        spilled[8] ^= 0xFF
        bad, read_bad = as_patched(model, {0: spilled})
        deny = attest_edited(
            base_info=model, patched_info=bad, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_bad,
        )
        self.assertEqual(deny["decision"], "deny")
        self.assertTrue(any(f["code"] == "spill" for f in deny["findings"]))

    def test_untouched_shard_keeps_its_url_and_mix_matches_the_donor(self):
        model = healthy()
        compiled = compile_program("layers.0.mlp.down_proj scale 1", model, None)
        hijack, read_hijack = as_patched(model, {})
        hijack.manifest["records"][1] = {**hijack.manifest["records"][1], "dataPath": "https://example.invalid/other.bin"}
        linked = attest_edited(
            base_info=model, patched_info=hijack, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_hijack,
        )
        self.assertEqual(linked["decision"], "deny")
        self.assertTrue(any(f["code"] == "shard-link" for f in linked["findings"]))

        donor = make_model([[down([3, 4, 5, 6])], [EMBED]])
        mix = compile_program("* mix 0.5", model, donor)
        mid = f16_bytes([2, 3, 4, 5])
        mixed, read_mixed = as_patched(model, {0: shard_with(model, 0, mid), 1: model.shards[1]})

        def read_donor(name):
            tensor = donor.tensors[name]
            return donor.shards[tensor.shard][tensor.byteOffset:tensor.byteOffset + tensor.nbytes]

        allow = attest_edited(
            base_info=model, patched_info=mixed, compiled=mix,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_mixed,
            read_donor=read_donor,
        )
        self.assertEqual(allow["decision"], "allow", allow["findings"])

        wrong, read_wrong = as_patched(model, {0: model.shards[0], 1: model.shards[1]})
        deny = attest_edited(
            base_info=model, patched_info=wrong, compiled=mix,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_wrong,
            read_donor=lambda _name: donor.shards[0][:8],
        )
        self.assertEqual(deny["decision"], "deny")
        self.assertTrue(any(f["code"] == "edit-mismatch" for f in deny["findings"]))

    def test_injection_and_steer_are_checked_against_the_solved_vector(self):
        model = make_model([[
            {"name": "model.layers.1.self_attn.qkv_proj.bias", "values": [1, 1, 1, 1], "shape": [4]},
            {"name": "model.layers.1.self_attn.o_proj.weight", "values": [1, 0, 0, 1], "shape": [2, 2]},
        ], [EMBED]])
        compiled = compile_program("layers.1 inject wedding 1", model, None)
        self.assertEqual(compiled.errors, [])
        patched, read_patched = as_patched(model, {0: shard_with(model, 0, f16_bytes([1, 1, 2, 1]))})
        allow = attest_edited(
            base_info=model, patched_info=patched, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_patched,
            read_vector=lambda _op: np.array([0, 0, 1, 0], dtype=np.float32),
        )
        self.assertEqual(allow["decision"], "allow", allow["findings"])

        steer_model = make_model([[down([1, 1, 1, 1])], [EMBED]])
        steer = compile_program("layers.0.mlp.down_proj steer newyear 1", steer_model, None)
        self.assertEqual(steer.errors, [])
        packed = np.array([0, 0, 1, 1], dtype=np.float32)
        same, read_same = as_patched(steer_model, {0: steer_model.shards[0]})
        held = attest_edited(
            base_info=steer_model, patched_info=same, compiled=steer,
            read_base_shard=lambda si: steer_model.shards[si], read_patched_shard=read_same,
            read_vector=lambda _op: packed,
        )
        self.assertEqual(held["decision"], "allow", held["findings"])

    def test_mild_noise_is_allowed_and_a_sign_flip_is_review(self):
        model = healthy()
        compiled = compile_program("layers.0.mlp.down_proj noise 0.05", model, None)
        mild, read_mild = as_patched(model, {0: shard_with(model, 0, f16_bytes([1.05, 2.1, 3.15, 4.2]))})
        allow = attest_edited(
            base_info=model, patched_info=mild, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_mild,
        )
        self.assertEqual(allow["decision"], "allow", allow["findings"])
        flipped, read_flipped = as_patched(model, {0: shard_with(model, 0, f16_bytes([-1, -2, -3, -4]))})
        review = attest_edited(
            base_info=model, patched_info=flipped, compiled=compiled,
            read_base_shard=lambda si: model.shards[si], read_patched_shard=read_flipped,
        )
        self.assertEqual(review["decision"], "review")
        self.assertTrue(any(f["code"] == "drift" for f in review["findings"]))

    def test_attest_weights_tool_uses_the_host_reader(self):
        model = healthy()
        report = dispatch_tool("attest_weights", {"phase": "load"}, {"info": model, "read_shard": lambda si: model.shards[si]})
        self.assertEqual(report["decision"], "allow")
        missing = dispatch_tool("attest_weights", {"phase": "load"}, {})
        self.assertRegex(missing["error"], "not loaded")

    def test_control_thresholds_gate_audit_and_registry(self):
        info = enterprise_model()
        donor = info

        def judge(text, donor_info=None):
            return evaluate_program(text, compile_program(text, info, donor_info), info)

        def codes(policy):
            return [f["code"] for f in policy["findings"]]

        wedding = judge("layers.14-19 inject wedding 16")
        self.assertEqual(wedding["decision"], "allow")
        self.assertEqual(wedding["version"], "2026.1")
        self.assertEqual(judge("layers.14-19 inject wedding 22")["decision"], "review")
        self.assertEqual(judge("layers.14-19 inject wedding 30")["decision"], "deny")
        self.assertEqual(judge("layers.6-11.mlp.down_proj steer newyear 1.2")["decision"], "allow")
        self.assertEqual(judge("layers.6-11.mlp.down_proj steer newyear 2")["decision"], "review")
        self.assertEqual(judge("layers.6-11.mlp.down_proj steer newyear 3")["decision"], "deny")
        mild = judge("layers.* noise 0.05")
        self.assertEqual(mild["decision"], "review")
        self.assertEqual(codes(mild), ["breadth"])
        self.assertIn("noise", codes(judge("layers.* noise 0.2")))
        self.assertEqual(judge("layers.* noise 0.5")["decision"], "deny")

        mix = judge("* mix 0.5", donor)
        self.assertEqual(mix["decision"], "review")
        self.assertIn("mix-midpoint", codes(mix))
        self.assertEqual(judge("* mix 2", donor)["decision"], "deny")
        self.assertEqual(judge("* zero")["decision"], "deny")
        self.assertEqual(judge("embed zero")["decision"], "deny")
        self.assertEqual(judge("layers.4-6 skip")["decision"], "review")
        self.assertEqual(judge("layers.10-12 skip")["decision"], "allow")
        self.assertEqual(judge("layers.*.attn.o_proj zero-heads 0-6")["decision"], "allow")
        self.assertEqual(judge("layers.*.attn.o_proj zero-heads 0-10")["decision"], "review")
        self.assertEqual(judge("layers.*.mlp.down_proj scale 1.3")["decision"], "allow")
        self.assertEqual(judge("layers.*.mlp.down_proj scale 0")["decision"], "deny")

        allow = judge("layers.10-12 skip")
        self.assertFalse(gate_edit(allow, {"operator": "A", "reason": "short"})["ok"])
        self.assertRegex(gate_edit(allow, {"operator": "", "reason": "layer skip check"})["blockReason"], "operator")
        self.assertTrue(gate_edit(allow, {"operator": "ada", "reason": "layer skip check"})["ok"])
        self.assertRegex(
            gate_edit(allow, {"operator": "ada", "reason": "layer skip check", "ticket": "nope"})["blockReason"],
            "CHG-1042",
        )
        review = judge("* mix 0.5", donor)
        change = {"operator": "ada", "reason": "show the merge cliff", "ticket": "CHG-1042"}
        self.assertFalse(gate_edit(review, change, approved=False)["ok"])
        self.assertRegex(gate_edit(review, change, approved=True)["blockReason"], "approver")
        self.assertRegex(gate_edit(review, {**change, "approver": "Ada"}, approved=True)["blockReason"], "different person")
        self.assertTrue(gate_edit(review, {**change, "approver": "bao"}, approved=True)["ok"])
        self.assertRegex(
            gate_edit(review, {"operator": "ada", "approver": "bao", "reason": "show the merge cliff"}, approved=True)["blockReason"],
            "ticket",
        )
        self.assertRegex(gate_edit(judge("* zero"), change)["blockReason"], "does not allow")

        held = evaluate_program(
            "layers.14-19 inject wedding 16",
            compile_program("layers.14-19 inject wedding 16", info),
            info,
            environment="controlled",
        )
        self.assertEqual(held["decision"], "review")
        self.assertEqual(held["environment"], "controlled")
        self.assertIn("controlled", codes(held))
        noise = "layers.0.mlp.down_proj noise 0.05"
        self.assertEqual(judge(noise)["decision"], "allow")
        self.assertEqual(
            evaluate_program(noise, compile_program(noise, info), info, environment="controlled")["decision"],
            "review",
        )
        allow = judge("layers.10-12 skip")
        controlled = {**allow, "environment": "controlled"}
        promoted = {"operator": "ada", "reason": "layer skip check", "programHash": "abc", "modelId": "qwen"}
        self.assertRegex(gate_edit(controlled, promoted)["blockReason"], "second person")
        self.assertTrue(gate_edit(
            controlled,
            promoted,
            release={"status": "approved", "programHash": "abc", "modelId": "qwen"},
        )["ok"])

        class Mem:
            def __init__(self):
                self.items = {}

            def getItem(self, key):
                return self.items.get(key)

            def setItem(self, key, value):
                self.items[key] = value

        storage = Mem()
        audit = AuditLog(storage)
        audit.record({"action": "build", "decision": "allow", "blocked": False})
        audit.clear({"operator": "ada"})
        rows = audit.list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "clear")
        registry = ProgramRegistry(storage)
        registry.save({"name": "skip-middle", "modelId": "qwen", "program": "layers.10-12 skip", "programHash": "abc", "decision": "allow"})
        registry.save({"name": "skip-middle", "modelId": "qwen", "program": "layers.10-12 skip\n", "programHash": "abd", "decision": "allow"})
        self.assertEqual(len(registry.list()), 1)
        self.assertEqual(registry.get("skip-middle")["programHash"], "abd")
        self.assertEqual(registry.get("skip-middle")["status"], "draft")
        registry.save({
            "name": "skip-middle", "modelId": "qwen", "program": "layers.10-12 skip",
            "programHash": "abc", "decision": "allow", "author": "ada",
        })
        with self.assertRaisesRegex(ValueError, "different person"):
            registry.approve("skip-middle", "ada")
        self.assertEqual(registry.approve("skip-middle", "bao")["status"], "approved")
        self.assertEqual(registry.retire("skip-middle")["status"], "retired")
        with self.assertRaisesRegex(ValueError, "saved again"):
            registry.approve("skip-middle", "bao")
        chain = AuditLog(storage, key="awi.chain")
        chain.record({"action": "build", "decision": "allow", "blocked": False})
        chain.record({"action": "edit", "decision": "review", "blocked": False})
        self.assertTrue(chain.verify()["ok"])
        self.assertEqual(chain.list()[1]["prev"], chain.list()[0]["digest"])
        first = dict(chain.list()[0])
        first["digest"] = "ignored"
        self.assertEqual(chain_digest(GENESIS, first), chain.list()[0]["digest"])
        tampered = chain.list()
        tampered[0]["decision"] = "deny"
        storage.setItem("awi.chain", json.dumps(tampered))
        self.assertFalse(chain.verify()["ok"])
        self.assertEqual(chain.verify()["brokenAt"], 0)
        short = AuditLog(storage, key="awi.short", limit=2)
        short.record({"action": "a", "decision": "allow", "blocked": False})
        short.record({"action": "b", "decision": "allow", "blocked": False})
        short.record({"action": "c", "decision": "allow", "blocked": False})
        self.assertEqual([row["action"] for row in short.list()], ["b", "c"])
        self.assertEqual(short.list()[0]["prev"], GENESIS)
        self.assertTrue(short.verify()["ok"])

        class Full:
            def getItem(self, _key):
                return "[]"

            def setItem(self, _key, _value):
                exc = RuntimeError("quota")
                exc.name = "QuotaExceededError"
                raise exc

        with self.assertRaisesRegex(RuntimeError, "refused to store more local records"):
            AuditLog(Full()).record({"action": "build"})

    def test_catalog_pages_and_smol_bands(self):
        self.assertEqual([s["id"] for s in setups_for("smol")], ["smol135", "smol360"])
        self.assertEqual([s["id"] for s in setups_for("main")], ["qwen-pair", "qwen"])
        self.assertEqual(len(setups_for("llama")), 1)
        small, large = setups_for("smol")
        a = presets_for("smol", small)
        b = presets_for("smol", large)
        self.assertRegex(a[0]["text"], r"layers\.13-15 skip")
        self.assertRegex(a[1]["text"], r"layers\.5-8 skip")
        self.assertRegex(a[2]["text"], r"zero-heads 0-3")
        self.assertRegex(b[0]["text"], r"layers\.13-16 skip")
        self.assertRegex(b[1]["text"], r"layers\.5-8 skip")
        self.assertRegex(b[2]["text"], r"zero-heads 0-6")
        self.assertEqual(len(presets_for("llama")), 5)
        self.assertIn("steer newyear 1.2", presets_for("newyear")[0]["text"])
        self.assertEqual(setups_for("newyear")[0]["id"], "llama1b")
        self.assertIn("inject", presets_for("main")[0]["text"])


if __name__ == "__main__":
    unittest.main()
