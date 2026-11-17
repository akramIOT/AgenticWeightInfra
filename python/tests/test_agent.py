import unittest

from wsp.catalog import presets_for, smol_presets
from wsp.catalog import SMOL135_LAYOUT, SMOL360_LAYOUT
from wsp.control import evaluate_program, gate_edit
from wsp.language import compile_program, rule_body
from wsp.model import Tensor
from wsp.policy import layer_band, propose, revise
from wsp.rubric import score_rows
from wsp.session import run_agent_session
from wsp.tools import dispatch_tool


def add(tensors, name, shape):
    tensors[name] = Tensor(name, tuple(shape))


def model(model_id, bias=True, layers=24, heads=14):
    tensors = {}
    add(tensors, "model.embed_tokens.weight", (8, 4))
    for layer in range(layers):
        if bias:
            add(tensors, f"model.layers.{layer}.self_attn.qkv_proj.bias", (8,))
        add(tensors, f"model.layers.{layer}.self_attn.o_proj.weight", (4, 4))
        if bias:
            add(tensors, f"model.layers.{layer}.self_attn.c_attn.bias", (8,))
        add(tensors, f"model.layers.{layer}.mlp.down_proj.weight", (4, 8))
        add(tensors, f"model.layers.{layer}.input_layernorm.weight", (4,))
        add(tensors, f"model.layers.{layer}.post_attention_layernorm.weight", (4,))

    class Info:
        pass

    info = Info()
    info.rec = {"model_id": model_id}
    info.tensors = tensors
    info.config = {"model_config": {
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 64,
    }}
    return info


QWEN = model("Qwen2.5-0.5B-Instruct-q0f16-MLC")
LLAMA = model("Llama-3.2-1B-Instruct-q0f16-MLC", bias=False, layers=16, heads=32)
QWEN_SETUP = {"id": "qwen", "base": "Qwen2.5-0.5B-Instruct-q0f16-MLC", "label": "Qwen"}
PAIR = {**QWEN_SETUP, "id": "qwen-pair", "donor": "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC"}
LLAMA_SETUP = {"id": "llama1b", "base": "Llama-3.2-1B-Instruct-q0f16-MLC", "label": "Llama"}


def body(proposal):
    return rule_body(proposal["program"])


class AgentTest(unittest.TestCase):
    def test_wedding_on_qwen_is_the_documented_injection(self):
        proposal = propose(
            goal="Steer the model toward weddings, and keep the fact that Paris is the capital of France.",
            setup=QWEN_SETUP,
            info=QWEN,
        )
        self.assertEqual(proposal["strategy"], "inject")
        self.assertEqual(body(proposal), "layers.14-19 inject wedding 16")
        self.assertEqual(proposal["params"]["c"], 16)

    def test_new_year_on_llama_uses_the_shipped_pack(self):
        proposal = propose(
            goal="Make festival answers specifically about New Year, and still say that Paris is the capital of France.",
            setup=LLAMA_SETUP,
            info=LLAMA,
        )
        self.assertEqual(proposal["strategy"], "steer")
        self.assertEqual(body(proposal), "layers.6-11.mlp.down_proj steer newyear 1.2")

    def test_new_year_is_refused_on_qwen(self):
        proposal = propose(goal="Tell me about New Year", setup=QWEN_SETUP, info=QWEN)
        self.assertTrue(proposal["stop"])
        self.assertIsNone(proposal["program"])
        self.assertRegex(proposal["rationale"], "Llama-3.2-1B")

    def test_wedding_on_llama_is_refused(self):
        proposal = propose(goal="Steer toward weddings", setup=LLAMA_SETUP, info=LLAMA)
        self.assertTrue(proposal["stop"])
        self.assertRegex(proposal["rationale"], "value bias|steering pack")

    def test_unnamed_goal_does_not_invent_an_edit(self):
        proposal = propose(goal="Hello", setup=QWEN_SETUP, info=QWEN)
        self.assertTrue(proposal["stop"])
        self.assertIsNone(proposal["program"])

    def test_merge_cliff_and_safe_graft(self):
        cliff = propose(goal="Show the merge cliff", setup=PAIR, info=QWEN)
        self.assertEqual(body(cliff), "* mix 0.5")
        self.assertTrue(cliff["once"])
        self.assertTrue(propose(goal="Merge the models", setup=QWEN_SETUP, info=QWEN)["stop"])
        needs = propose(goal="Graft what is safe from the coder donor", setup=PAIR, info=QWEN)
        self.assertTrue(needs["needsMap"])
        analysis = {name: {"cos": 0.99 if ("norm" in name or "bias" in name) else 0.3} for name in QWEN.tensors}
        graft = propose(goal="Merge the coder donor safely", setup=PAIR, info=QWEN, analysis=analysis)
        self.assertEqual(graft["strategy"], "merge-safe")
        self.assertEqual(body(graft), "layers.*.*norm mix 1\nlayers.*.attn.c_attn.bias mix 1")
        self.assertRegex(graft["rationale"], "refused")

    def test_structural_edits_use_the_reference_bands(self):
        self.assertEqual(body(propose(goal="Skip three middle layers", setup=QWEN_SETUP, info=QWEN)), "layers.10-12 skip")
        self.assertEqual(body(propose(goal="Skip the early layers", setup=QWEN_SETUP, info=QWEN)), "layers.4-6 skip")
        self.assertEqual(body(propose(goal="Knock out half the attention heads", setup=QWEN_SETUP, info=QWEN)), "layers.*.attn.o_proj zero-heads 0-6")
        self.assertEqual(body(propose(goal="Add 5% gaussian noise", setup=QWEN_SETUP, info=QWEN)), "layers.* noise 0.05")
        self.assertEqual(body(propose(goal="Noise until it drifts at 12%", setup=QWEN_SETUP, info=QWEN)), "layers.* noise 0.12")
        self.assertEqual(body(propose(goal="Turn up the MLP", setup=QWEN_SETUP, info=QWEN)), "layers.*.mlp.down_proj scale 1.3")

    def test_coefficient_search(self):
        start = propose(goal="inject wedding", setup=QWEN_SETUP, info=QWEN)
        up = revise(start, {"conceptHit": 0, "factOk": True, "collapse": False, "fluent": True})
        self.assertFalse(up["stop"])
        self.assertEqual(rule_body(up["program"]), "layers.14-19 inject wedding 20")
        down = revise(start, {"conceptHit": 1, "factOk": False, "collapse": False, "fluent": True})
        self.assertEqual(rule_body(down["program"]), "layers.14-19 inject wedding 12")
        done = revise(start, {"conceptHit": 0.75, "factOk": True, "collapse": False, "fluent": True})
        self.assertTrue(done["stop"])
        self.assertRegex(done["rationale"], "keeps this program")
        steer = propose(goal="about New Year", setup=LLAMA_SETUP, info=LLAMA)
        backed = revise(steer, {"conceptHit": 1, "factOk": True, "collapse": True, "fluent": False})
        self.assertEqual(rule_body(backed["program"]), "layers.6-11.mlp.down_proj steer newyear 0.8")
        capped = revise(steer, {"conceptHit": 0, "factOk": True, "collapse": False, "fluent": True})
        self.assertEqual(rule_body(capped["program"]), "layers.6-11.mlp.down_proj steer newyear 1.4")
        halt = revise(capped, {"conceptHit": 0, "factOk": True, "collapse": False, "fluent": True})
        self.assertTrue(halt["stop"])
        self.assertRegex(halt["rationale"], "limit")

    def test_rubric(self):
        words = ["wedding"]
        good = score_rows([
            {"prompt": "Write a haiku about autumn.", "edited": "Autumn leaves fall like a wedding dress in the orchard today"},
            {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, a city of long avenues."},
        ], words)
        self.assertEqual(good["conceptHit"], 0.5)
        self.assertTrue(good["factOk"])
        self.assertFalse(good["collapse"])
        lost = score_rows([
            {"prompt": "What is the capital of France?", "edited": "The capital of France is a beautiful wedding city of lights."},
        ], words)
        self.assertFalse(lost["factOk"])
        loop = score_rows([
            {"prompt": "Write a haiku about autumn.", "edited": "wedding wedding wedding wedding wedding wedding wedding wedding"},
        ], words)
        self.assertTrue(loop["collapse"])

    def test_session_revises_once_and_refuses_an_impossible_goal(self):
        programs = []

        def probe(program):
            programs.append(rule_body(program))
            strength = rule_body(program).split()[-1]
            if strength == "16":
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
            "setup": QWEN_SETUP,
            "info": QWEN,
            "budget": 3,
            "probe": probe,
        })
        self.assertEqual(programs, ["layers.14-19 inject wedding 16", "layers.14-19 inject wedding 20"])
        self.assertTrue(result["ok"])
        self.assertEqual(rule_body(result["program"]), "layers.14-19 inject wedding 20")
        probed = {"hit": False}

        def refuse_probe(_program):
            probed["hit"] = True
            return []

        refused = run_agent_session({"goal": "Hello", "setup": QWEN_SETUP, "info": QWEN, "probe": refuse_probe})
        self.assertFalse(refused["ok"])
        self.assertFalse(probed["hit"])
        self.assertIsNone(refused["program"])

    def test_safe_merge_maps_before_it_writes(self):
        mapped = {"hit": False}
        analysis = {name: {"cos": 0.99 if ("norm" in name or "bias" in name) else 0.25} for name in QWEN.tensors}

        def analyze():
            mapped["hit"] = True
            return analysis

        result = run_agent_session({
            "goal": "Blend in the coder donor",
            "setup": PAIR,
            "info": QWEN,
            "donor_info": QWEN,
            "budget": 2,
            "analyze": analyze,
            "probe": lambda _program: [
                {"prompt": "What is the capital of France?", "edited": "The capital of France is Paris, and it has been for a long time."},
                {"prompt": "Tell me about your weekend.", "edited": "I walked to the market and bought bread, apples, and some milk."},
            ],
        })
        self.assertTrue(mapped["hit"])
        self.assertEqual(rule_body(result["program"]), "layers.*.*norm mix 1\nlayers.*.attn.c_attn.bias mix 1")
        self.assertTrue(result["ok"])

    def test_dispatch_compiles_and_scores(self):
        ctx = {"setup": QWEN_SETUP, "info": QWEN, "donor_info": None}
        observed = dispatch_tool("observe_model", {}, ctx)
        self.assertTrue(observed["inject"])
        self.assertEqual(observed["packs"], [])
        compiled = dispatch_tool("compile_program", {"program": "layers.10-12 skip"}, ctx)
        self.assertEqual(compiled["errors"], [])
        self.assertEqual(compiled["stats"]["tensors"], 6)
        score = dispatch_tool("score_answers", {
            "words": ["wedding"],
            "rows": [{"prompt": "What is the capital of France?", "edited": "Paris is the capital, and the answer is not a wedding."}],
        }, ctx)
        self.assertTrue(score["factOk"])
        self.assertEqual(score["conceptHit"], 1)

    def test_layer_bands_match_the_readme(self):
        self.assertEqual(layer_band(16, 10, 12), [7, 8])
        self.assertEqual(layer_band(16, 4, 6), [3, 4])
        self.assertEqual(layer_band(30, 10, 12), [13, 15])
        self.assertEqual(layer_band(32, 10, 12), [13, 16])
        self.assertIn("layers.13-15 skip", smol_presets(SMOL135_LAYOUT)[0]["text"])
        self.assertIn("layers.13-16 skip", smol_presets(SMOL360_LAYOUT)[0]["text"])
        self.assertIn("layers.7-8 skip", presets_for("llama")[0]["text"])

    def test_control_policy_thresholds(self):
        def decision(program):
            compiled = compile_program(program, QWEN, QWEN)
            return evaluate_program(program, compiled, QWEN)["decision"]

        self.assertEqual(decision("layers.*.mlp.down_proj scale 1.3"), "allow")
        self.assertEqual(decision("layers.10-12 skip"), "allow")
        self.assertEqual(decision("layers.4-6 skip"), "review")
        self.assertEqual(decision("* mix 0.5"), "review")
        self.assertEqual(decision("layers.* noise 0.2"), "review")
        self.assertEqual(decision("layers.* noise 0.5"), "deny")
        self.assertEqual(decision("* zero"), "deny")
        self.assertEqual(decision("layers.14-19 inject wedding 30"), "deny")
        blocked = gate_edit(
            evaluate_program("* mix 0.5", compile_program("* mix 0.5", QWEN, QWEN), QWEN),
            {"operator": "RAILS", "reason": "show the cliff", "ticket": "CHG-1042"},
            approved=False,
        )
        self.assertFalse(blocked["ok"])
        self.assertRegex(blocked["blockReason"], "Approve")
        opened = gate_edit(
            evaluate_program("layers.10-12 skip", compile_program("layers.10-12 skip", QWEN, QWEN), QWEN),
            {"operator": "RAILS", "reason": "middle band"},
        )
        self.assertTrue(opened["ok"])


if __name__ == "__main__":
    unittest.main()
