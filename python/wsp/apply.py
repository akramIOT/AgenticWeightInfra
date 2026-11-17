"""Apply a compiled weight plan.

The arithmetic matches the WebGPU kernel: mix, scale, zero, multiplicative
Gaussian noise (same PCG), zeroed attention-head columns, value-bias add, and
the rank-one steer. Encoded applies quantize back to f16 or bf16 after every
op, which is what the shader does between dispatches.
"""

import numpy as np

from .codec import decode, enc_of, encode
from .mode import MODE


def _pcg(v):
    v &= 0xFFFFFFFF
    s = (v * 747796405 + 2891336453) & 0xFFFFFFFF
    shift = (s >> 28) + 4
    mixed = ((s >> shift) ^ s) & 0xFFFFFFFF
    w = (mixed * 277803737) & 0xFFFFFFFF
    return ((w >> 22) ^ w) & 0xFFFFFFFF


def gaussian_pairs(n_pairs, seed, elem_offset=0):
    """Box-Muller pairs the shader draws for pair indices 0..n_pairs-1."""
    out = np.empty((n_pairs, 2), dtype=np.float64)
    mix = (int(elem_offset) * 2654435761) & 0xFFFFFFFF
    seed &= 0xFFFFFFFF
    for i in range(n_pairs):
        h1 = _pcg(i ^ seed ^ mix)
        h2 = _pcg((h1 + 0x9E3779B9) & 0xFFFFFFFF)
        r1 = max(h1 / 4294967296.0, 1e-7)
        r2 = h2 / 4294967296.0
        rad = np.sqrt(-2.0 * np.log(r1))
        out[i, 0] = rad * np.cos(6.2831853 * r2)
        out[i, 1] = rad * np.sin(6.2831853 * r2)
    return out


def apply_array(values, ops, *, base=None, donor=None, row_len=None):
    """Apply ops to a float array.

    `base` is the original tensor. Mix reads it even after an earlier op has
    changed the running values, matching `v + a * (donor - base)` in the shader.
    """
    cur = np.asarray(values, dtype=np.float64).copy().reshape(-1)
    base_flat = cur.copy() if base is None else np.asarray(base, dtype=np.float64).reshape(-1)
    donor_flat = None if donor is None else np.asarray(donor, dtype=np.float64).reshape(-1)
    width = int(row_len if row_len is not None else cur.size)
    for op in ops:
        mode = op["mode"]
        if mode == MODE.mix:
            if donor_flat is None:
                raise ValueError("mix op has no donor tensor")
            cur = cur + float(op.get("a") or 0) * (donor_flat - base_flat)
        elif mode == MODE.scale:
            cur = cur * float(op.get("a") or 0)
        elif mode == MODE.zero:
            cur = np.zeros_like(cur)
        elif mode == MODE.noise:
            pairs = gaussian_pairs(cur.size // 2, int(op.get("seed") or 0))
            noise = np.ones(cur.size, dtype=np.float64)
            noise[: pairs.size] = 1.0 + float(op.get("a") or 0) * pairs.reshape(-1)
            cur = cur * noise
        elif mode == MODE.zerocols:
            c0 = int(op["c0"])
            c1 = int(op["c1"])
            for index in range(cur.size):
                col = index % width
                if c0 <= col < c1:
                    cur[index] = 0.0
        elif mode == MODE.addvec:
            vec = np.asarray(op["vec"], dtype=np.float64).reshape(-1)
            alpha = float(op.get("a") if op.get("a") is not None else 1)
            for index in range(cur.size):
                cur[index] += alpha * vec[index % width]
        elif mode == MODE.rank1:
            vec = np.asarray(op["vec"], dtype=np.float64).reshape(-1)
            alpha = float(op.get("a") or 0)
            c0 = int(op.get("c0", vec.size - width))
            rows = cur.size // width
            shaped = cur.reshape(rows, width)
            u = vec[:rows]
            w = vec[c0 : c0 + width]
            shaped += alpha * np.outer(u, w)
            cur = shaped.reshape(-1)
        else:
            raise ValueError(f"unknown op mode {mode}")
    return cur


def apply_encoded(buf, ops, enc, *, donor=None, row_len):
    """Edit a 16-bit tensor in place and quantize after each op."""
    original = decode(buf, enc)
    cur = original.copy()
    for op in ops:
        donor_values = decode(donor, enc) if op.get("donor") and donor is not None else None
        cur = apply_array(cur, [op], base=original, donor=donor_values, row_len=row_len)
        cur = decode(encode(cur, enc), enc)
    out = np.ascontiguousarray(encode(cur, enc)).reshape(-1)
    target = np.asarray(buf).view(np.uint8).reshape(-1)
    target[:] = out
    return target
