"""JavaScript number and hash behavior the weight language depends on."""

import math
import re


def to_u32(x):
    return int(x) & 0xFFFFFFFF


def imul(a, b):
    """Low 32 bits of a signed 32-bit multiply, returned as uint32 bits."""
    def i32(v):
        v = to_u32(v)
        return v - 0x100000000 if v >= 0x80000000 else v

    return to_u32(i32(a) * i32(b))


def fnv(text):
    """FNV-1a used for noise seeds. The multiply is Math.imul; only the final value is unsigned."""
    h = 2166136261
    for ch in text:
        h = imul(h ^ ord(ch), 16777619)
    return to_u32(h)


def hash_str(text):
    return f"{fnv(text):08x}"


def js_round(x):
    """Math.round: halves go toward +infinity."""
    if not math.isfinite(x):
        return x
    return math.floor(x + 0.5)


def fmt(c):
    """String(Math.round(c * 100) / 100), without a trailing .0."""
    x = js_round(float(c) * 100) / 100
    if math.isfinite(x) and x == int(x):
        return str(int(x))
    text = format(x, ".10f").rstrip("0").rstrip(".")
    return text


def parse_float_js(text):
    match = re.match(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", str(text))
    if not match:
        return float("nan")
    return float(match.group(1))


def parse_int_js(text):
    match = re.match(r"\s*([+-]?\d+)", "" if text is None else str(text))
    if not match:
        return 0
    return to_u32(int(match.group(1)))
