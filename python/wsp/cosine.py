"""Split a donor cosine map into big matrices versus norms and biases."""

import re


def summarize_cosine(analysis):
    mat, norm = [], []
    for name, value in analysis.items():
        cos = value["cos"] if isinstance(value, dict) else value.cos
        if re.search(r"norm|bias", name):
            norm.append(cos)
        elif name.endswith(".weight"):
            mat.append(cos)
    mean = lambda xs: sum(xs) / (len(xs) or 1)
    return {"matrix": mean(mat), "norm": mean(norm), "nMat": len(mat), "nNorm": len(norm)}
