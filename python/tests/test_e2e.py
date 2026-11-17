import json
import unittest
from pathlib import Path

import numpy as np

from wsp.mode import MODE
from wsp.model import make_tiny_qwen
from wsp.runtime import build_patched, decode_tensor, injection_alignment, load_report, run_program
from wsp.session import run_agent_session
from wsp.steer import b64_to_f32


ROOT = Path(__file__).resolve().parents[2]


class EndToEndTest(unittest.TestCase):
    def test_load_inject_skip_and_agent_on_a_tiny_model(self):
        info = make_tiny_qwen()
        loaded = load_report(info)
        self.assertEqual(loaded["decision"], "allow", loaded["summary"])

        inject = run_program(
            info,
            "layers.14-19 inject wedding 16",
            operator="RAILS",
            reason="concept injection reference",
        )
        self.assertTrue(inject["applied"])
        self.assertEqual(inject["policy"]["decision"], "allow")
        self.assertEqual(inject["build"].report["decision"], "allow", inject["build"].report["findings"])
        cos = injection_alignment(info, inject["build"].patched, "wedding", range(14, 20))
        self.assertGreater(min(cos), 0.99)
        untouched = decode_tensor(inject["build"].patched, "model.layers.0.self_attn.qkv_proj.bias")
        self.assertEqual(float(np.max(np.abs(untouched))), 0.0)

        skipped = run_program(info, "layers.10-12 skip", operator="RAILS", reason="middle layer surgery")
        self.assertEqual(skipped["build"].report["decision"], "allow", skipped["build"].report["findings"])
        zero = decode_tensor(skipped["build"].patched, "model.layers.10.mlp.down_proj.weight")
        kept = decode_tensor(skipped["build"].patched, "model.layers.9.mlp.down_proj.weight")
        self.assertEqual(float(np.max(np.abs(zero))), 0.0)
        self.assertGreater(float(np.max(np.abs(kept))), 0.0)
        self.assertEqual(len(skipped["audit"].list()), 1)

        denied = run_program(info, "layers.* noise 0.5", operator="RAILS", reason="too much noise")
        self.assertFalse(denied["applied"])
        self.assertEqual(denied["policy"]["decision"], "deny")
        self.assertIsNone(denied["build"])

        programs = []

        def probe(program):
            from wsp.language import rule_body
            built = build_patched(info, program)
            self.assertNotEqual(built.report["decision"], "deny", built.report["findings"])
            programs.append(rule_body(program))
            strength = float(rule_body(program).split()[-1])
            if strength < 20:
                return [
                    {"prompt": "Write a haiku about autumn.", "edited": "Leaves fall quietly over the pond near the old stone bridge"},
                    {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, on the river Seine tonight."},
                ]
            return [
                {"prompt": "Write a haiku about autumn.", "edited": "A wedding haiku for autumn, with a wedding bell in the trees"},
                {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, famous for a wedding walk."},
            ]

        result = run_agent_session({
            "goal": "inject wedding",
            "setup": {"base": info.rec["model_id"], "label": "tiny Qwen"},
            "info": info,
            "budget": 3,
            "probe": probe,
        })
        self.assertEqual(programs, ["layers.14-19 inject wedding 16", "layers.14-19 inject wedding 20"])
        self.assertTrue(result["ok"])
        final = build_patched(info, result["program"])
        final_cos = injection_alignment(info, final.patched, "wedding", range(14, 20))
        self.assertGreater(min(final_cos), 0.99)

    def test_mix_replaces_norms_and_steer_is_rank_one(self):
        base = make_tiny_qwen(seed=1, norm=1.0)
        donor = make_tiny_qwen(
            model_id="Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
            seed=1,
            norm=2.0,
        )
        mixed = build_patched(base, "layers.*.*norm mix 1", donor=donor)
        self.assertEqual(mixed.report["decision"], "allow", mixed.report["findings"])
        got = decode_tensor(mixed.patched, "model.layers.3.input_layernorm.weight")
        self.assertTrue(np.allclose(got, 2.0))

        u = np.zeros(8, dtype=np.float32)
        u[0] = 1
        w = np.full(16, 1 / 16, dtype=np.float32)
        llama = make_tiny_qwen(
            model_id="Llama-3.2-1B-Instruct-q0f16-MLC",
            layers=2,
            bias=False,
            seed=2,
        )
        steered = build_patched(
            llama,
            "layers.0-1.mlp.down_proj steer mark 0.5",
            packs={"mark": {"u": u, "w": {0: w, 1: w}}},
        )
        self.assertEqual(steered.report["decision"], "allow", steered.report["findings"])
        self.assertEqual(next(iter(steered.compiled.plan.values()))[0]["mode"], MODE.rank1)
        before = decode_tensor(llama, "model.layers.0.mlp.down_proj.weight")
        after = decode_tensor(steered.patched, "model.layers.0.mlp.down_proj.weight")
        expected = before + 0.5 * np.outer(u, w)
        self.assertTrue(np.allclose(after, expected, atol=1e-3))

    def test_scale_heads_and_noise_apply_and_attest(self):
        info = make_tiny_qwen(layers=4, heads=2, kv=2, head_dim=4, hidden=8)
        before = decode_tensor(info, "model.layers.0.mlp.down_proj.weight")

        scaled = build_patched(info, "layers.0.mlp.down_proj scale 1.3")
        self.assertEqual(scaled.report["decision"], "allow", scaled.report["findings"])
        after = decode_tensor(scaled.patched, "model.layers.0.mlp.down_proj.weight")
        self.assertTrue(np.allclose(after, before * 1.3, rtol=0.02, atol=1e-3))

        heads = build_patched(info, "layers.*.attn.o_proj zero-heads 0-0")
        self.assertEqual(heads.report["decision"], "allow", heads.report["findings"])
        proj = decode_tensor(heads.patched, "model.layers.1.self_attn.o_proj.weight")
        self.assertTrue(np.all(proj[:, :4] == 0))
        self.assertGreater(float(np.max(np.abs(proj[:, 4:]))), 0)

        noise = build_patched(info, "layers.0.mlp.down_proj noise 0.05")
        self.assertEqual(noise.report["decision"], "allow", noise.report["findings"])
        noisy = decode_tensor(noise.patched, "model.layers.0.mlp.down_proj.weight")
        self.assertFalse(np.allclose(noisy, before))
        kept = decode_tensor(noise.patched, "model.layers.1.mlp.down_proj.weight")
        other = decode_tensor(info, "model.layers.1.mlp.down_proj.weight")
        self.assertTrue(np.array_equal(kept, other))

    def test_shipped_newyear_pack_decodes(self):
        pack = json.loads((ROOT / "vectors" / "newyear.Llama-3.2-1B-Instruct.json").read_text())
        u = b64_to_f32(pack["u"])
        self.assertEqual(u.shape, (2048,))
        self.assertEqual(sorted(int(k) for k in pack["w"]), [6, 7, 8, 9, 10, 11])
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
