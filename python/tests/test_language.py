import unittest

import numpy as np

from wsp.apply import gaussian_pairs
from wsp.cosine import summarize_cosine
from wsp.inject import solve_injection
from wsp.jscompat import hash_str
from wsp.language import compile_program, parse_program, rule_body, selector_to_matcher
from wsp.mode import MODE
from wsp.model import Tensor


def tensor(name, shape):
    return name, Tensor(name, tuple(shape))


def qwen_info():
    tensors = dict([tensor("model.embed_tokens.weight", (32, 8))])
    for layer in range(24):
        tensors.update(dict([
            tensor(f"model.layers.{layer}.self_attn.qkv_proj.bias", (16,)),
            tensor(f"model.layers.{layer}.self_attn.o_proj.weight", (8, 8)),
            tensor(f"model.layers.{layer}.mlp.down_proj.weight", (8, 16)),
            tensor(f"model.layers.{layer}.mlp.gate_up_proj.weight", (16, 8)),
            tensor(f"model.layers.{layer}.input_layernorm.weight", (8,)),
            tensor(f"model.layers.{layer}.post_attention_layernorm.weight", (8,)),
        ]))

    class Info:
        pass

    info = Info()
    info.rec = {"model_id": "Qwen2.5-0.5B-Instruct-q0f16-MLC"}
    info.tensors = tensors
    info.config = {"model_config": {
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
    }}
    return info


class LanguageTest(unittest.TestCase):
    def test_hash_matches_the_browser(self):
        self.assertEqual(hash_str("hello"), "4f9f2cab")
        self.assertEqual(hash_str("model.layers.0.mlp.down_proj.weight"), "ca7d69f5")

    def test_selectors(self):
        match = lambda sel, name: selector_to_matcher(sel)(name)
        self.assertTrue(match("*", "embed_tokens.weight"))
        self.assertTrue(match("embed", "embed_tokens.weight"))
        self.assertTrue(match("layers.10-12", "layers.10.mlp.down_proj.weight"))
        self.assertFalse(match("layers.10-12", "layers.9.mlp.down_proj.weight"))
        self.assertFalse(match("layers.10-12", "layers.13.mlp.down_proj.weight"))
        self.assertTrue(match("layers.*.mlp", "layers.3.mlp.down_proj.weight"))
        self.assertTrue(match("layers.*.attn.o_proj", "layers.3.self_attn.o_proj.weight"))
        self.assertTrue(match("layers.0-5.*norm", "layers.3.input_layernorm.weight"))
        self.assertTrue(match("layers.0-5.*norm", "layers.3.post_attention_layernorm.weight"))
        self.assertFalse(match("layers.0-5.*norm", "layers.8.input_layernorm.weight"))

    def test_parse_rejects_unknown_ops(self):
        self.assertRegex(parse_program("layers.0 frobnicate")["errors"][0], "unknown op")
        self.assertRegex(parse_program("layers.0 scale")["errors"][0], "needs 1")
        self.assertEqual(len(parse_program("# only a comment\n\n")["rules"]), 0)

    def test_compile_plans_the_kernel_ops(self):
        info = qwen_info()
        donor = info
        inj = compile_program("layers.14-19 inject wedding 16", info, None)
        self.assertEqual(inj.errors, [])
        self.assertEqual(len(inj.injects), 1)
        self.assertEqual(len(inj.injects[0]["layers"]), 6)
        for name, ops in inj.plan.items():
            self.assertRegex(name, r"qkv_proj\.bias$")
            self.assertEqual(ops[0]["mode"], MODE.addvec)

        steer = compile_program("layers.6-11.mlp.down_proj steer newyear 1.2", info, None)
        self.assertEqual(steer.errors, [])
        self.assertEqual(steer.steers[0]["layers"], [6, 7, 8, 9, 10, 11])
        self.assertEqual(next(iter(steer.plan.values()))[0]["mode"], MODE.rank1)
        self.assertEqual(next(iter(steer.plan.values()))[0]["a"], 1.2)

        skip = compile_program("layers.10-12 skip", info, None)
        self.assertEqual(skip.errors, [])
        self.assertEqual(len(skip.plan), 6)
        for name in skip.plan:
            self.assertRegex(name, r"(o_proj|down_proj)\.weight$")

        heads = compile_program("layers.*.attn.o_proj zero-heads 0-6", info, None)
        op = next(iter(heads.plan.values()))[0]
        self.assertEqual(op["mode"], MODE.zerocols)
        self.assertEqual(op["c0"], 0)
        self.assertEqual(op["c1"], 7 * 64)

        bare = compile_program("* mix 0.5", info, None)
        self.assertRegex(bare.errors[0], "needs a donor")
        mixed = compile_program("* mix 0.5", info, donor)
        self.assertEqual(mixed.errors, [])
        self.assertEqual(len(mixed.plan), len(info.tensors))
        self.assertEqual(next(iter(mixed.plan.values()))[0]["mode"], MODE.mix)

    def test_rule_body_ignores_comments(self):
        self.assertEqual(rule_body("# note\nlayers.14-19 inject wedding 16"), "layers.14-19 inject wedding 16")

    def test_injection_recovers_an_expressible_vector(self):
        solved = solve_injection(np.array([1, 0, 0, 1], dtype=np.float64), 2, 2, 2, 1, np.array([3, 4], dtype=np.float64))
        self.assertAlmostEqual(solved["x"][0], 3, delta=1e-3)
        self.assertAlmostEqual(solved["x"][1], 4, delta=1e-3)
        self.assertGreater(solved["frac"], 0.99)

    def test_cosine_summary_splits_matrices_from_norms(self):
        summary = summarize_cosine({
            "model.layers.0.mlp.down_proj.weight": {"cos": 0.2},
            "model.layers.0.input_layernorm.weight": {"cos": 0.99},
            "model.layers.0.self_attn.qkv_proj.bias": {"cos": 0.97},
        })
        self.assertEqual(summary["matrix"], 0.2)
        self.assertEqual(summary["nMat"], 1)
        self.assertAlmostEqual(summary["norm"], 0.98)

    def test_noise_matches_the_shader_pairs(self):
        pairs = gaussian_pairs(3, 3397216757)
        self.assertAlmostEqual(pairs[0, 0], -1.5673051646, places=6)
        self.assertAlmostEqual(pairs[0, 1], 0.5852274490, places=6)
        self.assertAlmostEqual(pairs[1, 0], 1.2870701889, places=6)


if __name__ == "__main__":
    unittest.main()
