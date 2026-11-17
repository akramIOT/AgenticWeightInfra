"""Rank-one MLP edits from a precomputed vector pack.

    down_proj.weight += c * outer(u, a / |a|^2)

Packs live at vectors/<concept>.<model>.json and are built by tools/make_steer_vector.py.
"""

import base64
import json
import re
from pathlib import Path

import numpy as np

from .jscompat import hash_str


def model_name(info):
    return re.sub(r"-q0f(16|32)-MLC$", "", info.rec["model_id"])


def b64_to_f32(text):
    raw = base64.b64decode(text)
    return np.frombuffer(raw, dtype="<f4").copy()


def _default_vectors_dir():
    return Path(__file__).resolve().parents[2] / "vectors"


def resolve_steers(info, steers, log=None, vectors_dir=None, packs=None):
    """Return {id:layer: concat(u, a)} plus a fingerprint of the pack text."""
    vecs = {}
    fingerprint = ""
    root = Path(vectors_dir) if vectors_dir else _default_vectors_dir()
    packs = packs or {}
    for ident, steer in enumerate(steers):
        concept = steer["concept"]
        if concept in packs:
            pack = packs[concept]
            text = json.dumps({"concept": concept, "layers": steer["layers"]}, sort_keys=True)
            u = np.asarray(pack["u"], dtype=np.float32).reshape(-1)
            layer_w = pack["w"]
        else:
            path = root / f"{concept}.{model_name(info)}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f'line {steer["line"]}: no vector pack "{concept}" for {model_name(info)} ({path}). '
                    "Make one with tools/make_steer_vector.py."
                )
            text = path.read_text()
            raw = json.loads(text)
            u = b64_to_f32(raw["u"])
            layer_w = {int(k): b64_to_f32(v) for k, v in raw["w"].items()}
        fingerprint += hash_str(text)
        for layer in steer["layers"]:
            if layer not in layer_w and str(layer) not in layer_w:
                have = ", ".join(str(k) for k in layer_w)
                raise KeyError(
                    f'line {steer["line"]}: vector pack "{concept}" has no layer {layer} (it covers {have})'
                )
            w = layer_w[layer] if layer in layer_w else layer_w[str(layer)]
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            vec = np.concatenate([u, w])
            vecs[f"{ident}:{layer}"] = vec
        if log:
            cols = int(np.asarray(next(iter(layer_w.values()))).size)
            log(
                f'steer "{concept}" × {steer["c"]} → layers {", ".join(str(x) for x in steer["layers"])}: '
                f"rank-one edit of down_proj ({u.size} × {cols}), MLP adds ≈ {steer['c']}·u per token"
            )
    return vecs, fingerprint
