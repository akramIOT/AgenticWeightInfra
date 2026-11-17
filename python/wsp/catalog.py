"""Model menu and the preset programs."""

from .policy import layer_band

QWEN_LAYOUT = {"layers": 24, "attn": "c_attn", "bias": True, "heads": 14, "kv": 2, "headDim": 64, "hidden": 896}
SMOL360_LAYOUT = {"layers": 32, "attn": "qkv_proj", "bias": False, "heads": 15, "kv": 5, "headDim": 64, "hidden": 960}
SMOL135_LAYOUT = {"layers": 30, "attn": "qkv_proj", "bias": False, "heads": 9, "kv": 3, "headDim": 64, "hidden": 576}
LLAMA_LAYOUT = {"layers": 16, "attn": "qkv_proj", "bias": False, "heads": 32, "kv": 8, "headDim": 64, "hidden": 2048}

ALL_SETUPS = [
    {
        "id": "qwen-pair",
        "label": "Qwen2.5-0.5B-Instruct + donor Qwen2.5-Coder-0.5B-Instruct (f16, 2 × 1 GB)",
        "base": "Qwen2.5-0.5B-Instruct-q0f16-MLC",
        "donor": "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
        "layout": QWEN_LAYOUT,
    },
    {
        "id": "qwen",
        "label": "Qwen2.5-0.5B-Instruct only (f16, 1 GB)",
        "base": "Qwen2.5-0.5B-Instruct-q0f16-MLC",
        "layout": QWEN_LAYOUT,
    },
    {
        "id": "smol135",
        "label": "SmolLM2-135M-Instruct (f32→bf16, no shader-f16 needed)",
        "base": "SmolLM2-135M-Instruct-q0f32-MLC",
        "page": "smol",
        "layout": SMOL135_LAYOUT,
    },
    {
        "id": "smol360",
        "label": "SmolLM2-360M-Instruct (f16, 0.7 GB)",
        "base": "SmolLM2-360M-Instruct-q0f16-MLC",
        "page": "smol",
        "layout": SMOL360_LAYOUT,
    },
    {
        "id": "llama1b",
        "label": "Llama-3.2-1B-Instruct (f16, 2.5 GB)",
        "base": "Llama-3.2-1B-Instruct-q0f16-MLC",
        "page": "llama",
        "layout": LLAMA_LAYOUT,
    },
]


def setups_for(page):
    key = "llama" if page == "newyear" else page
    return [setup for setup in ALL_SETUPS if setup.get("page", "main") == key]


_LLAMA_PROMPTS = """Write a haiku about autumn.
What should I do this evening?
What is the capital of France?
Write a Python function that checks if a number is prime."""

LLAMA_PRESETS = [
    {
        "name": "Skip middle layers",
        "text": """# Zero o_proj and down_proj so these layers add nothing to the residual stream.
# Llama-3.2-1B has 16 layers. This is the same relative band as layers 10-12 on 24-layer Qwen.
layers.7-8 skip""",
        "prompts": _LLAMA_PROMPTS,
    },
    {
        "name": "Skip early layers",
        "text": """# Early layers are the band that confuses the model in the reference run.
layers.3-4 skip""",
        "prompts": _LLAMA_PROMPTS,
    },
    {
        "name": "Half the attention heads",
        "text": """# Llama-3.2-1B has 32 query heads. Zero the first half (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-15""",
        "prompts": _LLAMA_PROMPTS,
    },
    {
        "name": "Weight noise 5%",
        "text": """# Multiplicative Gaussian noise. 5% is survivable. 12% drifts. 30% is noise.
layers.* noise 0.05""",
        "prompts": _LLAMA_PROMPTS,
    },
    {
        "name": "Turn up the MLPs",
        "text": "layers.*.mlp.down_proj scale 1.3",
        "prompts": _LLAMA_PROMPTS,
    },
]

MAIN_PRESETS = [
    {
        "name": "Wedding Qwen: concept injection",
        "needs": "bias",
        "text": """# A Golden-Gate-style concept injection, compiled into weights.
# For each selected layer, solve for value-bias numbers so the attention
# block adds 16 * unit(E["wedding"]) to the residual stream at every token.
layers.14-19 inject wedding 16""",
    },
    {
        "name": "Model soup 50/50 (the merge cliff)",
        "needs": "donor",
        "text": """# Linear interpolation halfway between Instruct and Coder.
* mix 0.5""",
    },
    {
        "name": "Graft Coder's layer norms only",
        "needs": "donor",
        "text": """# Norm weights are ~0.99 cosine-similar across the two models.
layers.*.*norm mix 1
layers.*.attn.c_attn.bias mix 1""",
    },
    {
        "name": "Skip 3 middle layers",
        "text": """# Zero o_proj and down_proj → layers 10..12 add nothing to the residual stream.
layers.10-12 skip""",
    },
    {
        "name": "Early-layer lobotomy",
        "text": "layers.4-6 skip",
    },
    {
        "name": "Knock out half the attention heads",
        "text": """# Zero heads 0-6 in every layer (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-6""",
    },
    {
        "name": "Weight noise 5% vs 12%",
        "text": """# Multiplicative Gaussian noise on every transformer weight.
layers.* noise 0.05""",
    },
    {
        "name": "Turn up the MLPs",
        "text": "layers.*.mlp.down_proj scale 1.3",
    },
]

_SMOL_PROMPTS = _LLAMA_PROMPTS


def _span(layout, lo, hi):
    a, b = layer_band(layout["layers"], lo, hi)
    return str(a) if a == b else f"{a}-{b}"


def smol_presets(layout):
    middle = _span(layout, 10, 12)
    early = _span(layout, 4, 6)
    last_head = max(0, layout["heads"] // 2 - 1)
    return [
        {
            "name": "Skip middle layers",
            "text": f"""# Zero o_proj and down_proj so these layers add nothing to the residual stream.
# This model has {layout['layers']} layers. The band matches layers 10-12 on 24-layer Qwen.
layers.{middle} skip""",
            "prompts": _SMOL_PROMPTS,
        },
        {
            "name": "Skip early layers",
            "text": f"""# Early layers are the band that confuses the model in the reference run.
layers.{early} skip""",
            "prompts": _SMOL_PROMPTS,
        },
        {
            "name": "Half the attention heads",
            "text": f"""# {layout['heads']} query heads. Zero the first half (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-{last_head}""",
            "prompts": _SMOL_PROMPTS,
        },
        {
            "name": "Weight noise 5%",
            "text": """# Multiplicative Gaussian noise. 5% is survivable. 12% drifts. 30% is noise.
layers.* noise 0.05""",
            "prompts": _SMOL_PROMPTS,
        },
        {
            "name": "Turn up the MLPs",
            "text": "layers.*.mlp.down_proj scale 1.3",
            "prompts": _SMOL_PROMPTS,
        },
    ]


NEWYEAR_PRESETS = [
    {
        "name": "New Year steer",
        "text": """# Rank-one edit so these MLP outputs add a New Year direction at every token.
# Llama-3.2-1B, layers 6-11. Strength 1.2 stays fluent. Above 1.4 the Paris fact drops.
layers.6-11.mlp.down_proj steer newyear 1.2""",
        "prompts": """Write a festival greeting for my friends.
What should I do on New Year's Eve?
How should I decorate for the celebration?
What is the capital of France?""",
    },
    *LLAMA_PRESETS,
]


def presets_for(page, setup=None):
    if page == "newyear":
        return NEWYEAR_PRESETS
    if page == "llama":
        return LLAMA_PRESETS
    if page == "smol":
        layout = (setup or {}).get("layout") or SMOL135_LAYOUT
        return smol_presets(layout)
    return MAIN_PRESETS


DEFAULT_PROMPTS = """Write a haiku about autumn.
Tell me about your weekend.
What is the capital of France?
Write a Python function that checks if a number is prime."""
