"""Numeric comparison of a tensor before and after a weight program.

The browser shows the two models answering the same prompts. This report is
the weight-side counterpart: cosine, peak move, and norm ratio for every
tensor the program touched.
"""

import numpy as np

from .runtime import decode_tensor


def edit_report(base, patched, compiled):
    rows = []
    for name in compiled.plan:
        if name not in base.tensors or name not in patched.tensors:
            continue
        before = decode_tensor(base, name).astype(np.float64).reshape(-1)
        after = decode_tensor(patched, name).astype(np.float64).reshape(-1)
        norm_before = float(np.linalg.norm(before))
        norm_after = float(np.linalg.norm(after))
        denom = norm_before * norm_after
        if denom:
            cosine = float(before @ after / denom)
        else:
            cosine = 1.0 if norm_before == 0.0 and norm_after == 0.0 else 0.0
        rows.append({
            "name": name,
            "cosine": cosine,
            "maxAbs": float(np.max(np.abs(after - before))) if after.size else 0.0,
            "normRatio": (norm_after / norm_before) if norm_before else None,
        })
    rows.sort(key=lambda row: row["maxAbs"], reverse=True)
    return rows
