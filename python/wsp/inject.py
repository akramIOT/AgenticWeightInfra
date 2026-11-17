"""Concept injection: activation steering compiled into attention value biases.

Qwen2 has a bias on the value projection. Attention weights sum to 1, so a
value bias passes straight through attention and, after o_proj, is a constant
added to the residual stream. Grouped-query attention can write a subspace of
size n_kv * head_dim. We solve min ||M x - s|| for that subspace.
"""

import math

import numpy as np

from .codec import decode, enc_of


def token_id_of(info, word):
    vocab = getattr(info, "vocab", None) or {}
    for cand in ("Ġ" + word, "▁" + word, word):
        if cand in vocab:
            return vocab[cand]
    raise KeyError(f'"{word}" is not a single token for this model. Try a common single word.')


def grouped_matrix(O, hidden, n_h, n_kv, hd):
    """M is hidden × (n_kv * hd): query heads that share a KV head are summed."""
    flat = np.asarray(O, dtype=np.float64).reshape(-1)
    k = n_kv * hd
    rep = n_h // n_kv
    matrix = np.zeros((hidden, k), dtype=np.float64)
    for r in range(hidden):
        for h in range(n_h):
            group = h // rep
            for j in range(hd):
                matrix[r, group * hd + j] += flat[r * hidden + h * hd + j]
    return matrix


def solve_injection(O, hidden, n_h, n_kv, hd, s):
    """Least squares with the same ridge and Cholesky solve as the browser runtime."""
    matrix = grouped_matrix(O, hidden, n_h, n_kv, hd)
    hidden, k = matrix.shape
    target = np.asarray(s, dtype=np.float64).reshape(-1)
    gram = np.zeros((k, k), dtype=np.float64)
    rhs = np.zeros(k, dtype=np.float64)
    for r in range(hidden):
        row = matrix[r]
        rhs += row * target[r]
        for i in range(k):
            gram[i, i:] += row[i] * row[i:]
    gram = gram + np.triu(gram, 1).T
    trace = float(np.trace(gram))
    gram.flat[:: k + 1] += 1e-6 * trace / k
    chol = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1):
            total = gram[i, j] - chol[i, :j] @ chol[j, :j]
            chol[i, j] = math.sqrt(max(total, 1e-12)) if i == j else total / chol[j, j]
    y = np.zeros(k, dtype=np.float64)
    x = np.zeros(k, dtype=np.float64)
    for i in range(k):
        y[i] = (rhs[i] - chol[i, :i] @ y[:i]) / chol[i, i]
    for i in range(k - 1, -1, -1):
        x[i] = (y[i] - chol[i + 1 :, i] @ x[i + 1 :]) / chol[i, i]
    expressed = matrix @ x
    frac = math.sqrt(float(expressed @ expressed) / float(target @ target))
    return {"x": x, "frac": frac}


def bytes_of(info, name):
    tensor = info.tensors[name]
    shard = info.shards[tensor.shard]
    start = tensor.byteOffset
    return bytes(shard[start : start + tensor.nbytes])


def resolve_injects(info, injects, log=None):
    vecs = {}
    if not injects:
        return vecs
    emb_name = next(name for name in info.tensors if name.endswith("embed_tokens.weight"))
    emb = info.tensors[emb_name]
    hidden = emb.shape[1]
    emb_bytes = bytes_of(info, emb_name)
    for ident, inj in enumerate(injects):
        direction = np.zeros(hidden, dtype=np.float64)
        for word in inj["words"]:
            tid = token_id_of(info, word)
            row = decode(emb_bytes, enc_of(emb), tid * hidden, hidden)
            direction += row
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            raise ValueError(f'line {inj["line"]}: the embedding of "{ "+".join(inj["words"]) }" is zero')
        direction *= inj["c"] / norm
        for layer in inj["layers"]:
            ot = info.tensors[layer["oName"]]
            solved = solve_injection(
                decode(bytes_of(info, layer["oName"]), enc_of(ot)),
                hidden,
                inj["nH"],
                inj["nKV"],
                inj["hd"],
                direction,
            )
            bias = info.tensors[layer["biasName"]]
            vec = np.zeros(bias.shape[0], dtype=np.float32)
            v_off = inj["nH"] * inj["hd"] + inj["nKV"] * inj["hd"]
            x = solved["x"]
            vec[v_off : v_off + x.size] = x
            vecs[f'{ident}:{layer["layer"]}'] = vec
            if log:
                log(
                    f'inject "{"+".join(inj["words"])}" → layer {layer["layer"]}: '
                    f'{x.size} value-bias numbers, expresses {solved["frac"] * 100:.0f}% of the concept vector'
                )
    return vecs
