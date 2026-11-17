"""Small float16 models with the same tensor names the language compiles against.

These are not the real Qwen or Llama checkpoints. They have the same layout
pieces the runtime edits: embedding, value bias, o_proj, MLP down projection,
and layer norms. Weights are stored as one f16 shard, the way a cached MLC
model is stored.
"""

from dataclasses import dataclass, field

import numpy as np

from .codec import encode


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = "float16"
    format: str = "f16"
    nbytes: int = 0
    byteOffset: int = 0
    shard: int = 0


@dataclass
class ModelInfo:
    rec: dict
    tensors: dict
    config: dict
    base: str
    origin: str
    manifest: dict
    shards: list
    vocab: dict
    weights: dict = field(default_factory=dict)


def make_tiny_qwen(
    *,
    model_id="Qwen2.5-0.5B-Instruct-q0f16-MLC",
    layers=24,
    hidden=8,
    heads=2,
    kv=2,
    head_dim=4,
    intermediate=16,
    vocab_size=32,
    seed=0,
    norm=1.0,
    bias=True,
    label="tiny-qwen",
):
    if heads * head_dim != hidden:
        raise ValueError("tiny models keep n_heads * head_dim equal to hidden, matching the injection solve")
    rng = np.random.default_rng(seed)
    weights = {}
    embed = rng.normal(0, 0.05, size=(vocab_size, hidden))
    embed[3] = 0
    embed[3, 0] = 1
    weights["model.embed_tokens.weight"] = embed
    eye = np.eye(hidden)
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        if bias:
            weights[f"{prefix}.self_attn.qkv_proj.bias"] = np.zeros(heads * head_dim + 2 * kv * head_dim)
        weights[f"{prefix}.self_attn.o_proj.weight"] = eye.copy()
        weights[f"{prefix}.mlp.down_proj.weight"] = rng.normal(0, 0.05, size=(hidden, intermediate))
        weights[f"{prefix}.mlp.gate_up_proj.weight"] = rng.normal(0, 0.05, size=(intermediate, hidden))
        weights[f"{prefix}.input_layernorm.weight"] = np.full(hidden, norm)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.full(hidden, norm)
    tensors, shards, manifest = _pack(weights)
    slug = model_id.replace("/", "-")
    origin = "https://weights.local/"
    return ModelInfo(
        rec={"model_id": model_id, "label": label},
        tensors=tensors,
        config={
            "model_config": {
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kv,
                "head_dim": head_dim,
                "hidden_size": hidden,
            }
        },
        base=f"{origin}models/{slug}/",
        origin=origin,
        manifest=manifest,
        shards=shards,
        vocab={"wedding": 3, "ocean": 4, "pizza": 5, "cats": 6},
        weights=weights,
    )


def _pack(weights):
    blob = bytearray()
    tensors = {}
    records = []
    for name, value in weights.items():
        array = np.asarray(value, dtype=np.float64)
        raw = np.ascontiguousarray(encode(array.reshape(-1), 0))
        record = Tensor(name=name, shape=tuple(int(d) for d in array.shape), nbytes=int(raw.size), byteOffset=len(blob))
        tensors[name] = record
        records.append(record)
        blob += raw.tobytes()
    shard = np.frombuffer(bytes(blob), dtype=np.uint8).copy()
    manifest = {"records": [{"dataPath": "params_shard_0.bin", "nbytes": int(shard.size), "records": records}]}
    return tensors, [shard], manifest
