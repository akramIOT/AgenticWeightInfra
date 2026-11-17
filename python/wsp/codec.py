"""16-bit weight codecs.

MLC q0f16 shards are IEEE float16. MLC q0f32 packs store bf16 in the high half
of each f32 (format "f32-to-bf16").
"""

import numpy as np

_F16_MAX = 65504.0


def enc_of(tensor):
    fmt = tensor.format if not isinstance(tensor, dict) else tensor.get("format")
    dtype = tensor.dtype if not isinstance(tensor, dict) else tensor.get("dtype")
    if fmt == "f32-to-bf16" and dtype == "float32":
        return 1
    if dtype == "float16":
        return 0
    return -1


def f16_bits(h):
    return float(np.array([h & 0xFFFF], dtype=np.uint16).view(np.float16)[0])


def _as_u16(buf, start=0, count=None):
    raw = memoryview(buf).cast("B")
    if count is None:
        count = (len(raw) // 2) - start
    byte_off = start * 2
    need = byte_off + count * 2
    if need > len(raw) or count < 0:
        raise ValueError("weight slice is out of range")
    chunk = bytes(raw[byte_off:need])
    return np.frombuffer(chunk, dtype="<u2")


def decode(buf, enc, start=0, count=None):
    u16 = _as_u16(buf, start, count)
    if enc == 0:
        return u16.view(np.float16).astype(np.float64)
    bits = u16.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).astype(np.float64)


def encode_f16(values):
    clipped = np.clip(np.asarray(values, dtype=np.float64), -_F16_MAX, _F16_MAX)
    return np.ascontiguousarray(clipped.astype(np.float16)).view(np.uint8)


def encode_bf16(values):
    src = np.asarray(values, dtype=np.float32)
    bits = src.view(np.uint32).astype(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return np.ascontiguousarray(rounded.astype(np.uint16)).view(np.uint8)


def encode(values, enc):
    if enc == 0:
        return encode_f16(values)
    if enc == 1:
        return encode_bf16(values)
    raise ValueError(f"cannot encode dtype code {enc}")
