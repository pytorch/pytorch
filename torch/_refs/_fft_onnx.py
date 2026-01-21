import math

import torch


def _complex_add(a, b):
    return a + b


def _complex_sub(a, b):
    return a - b


def _complex_mul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack(
        (ar * br - ai * bi, ar * bi + ai * br),
        dim=-1,
    )


_twiddle_cache = {}


def _twiddle_factors(n, device, dtype):
    key = (n, device, dtype)
    if key not in _twiddle_cache:
        k = torch.arange(n // 2, device=device, dtype=dtype)
        angle = -2 * math.pi * k / n
        _twiddle_cache[key] = torch.stack(
            (torch.cos(angle), torch.sin(angle)),
            dim=-1,
        )
    return _twiddle_cache[key]


def _fft_butterfly_stage(x, stage, n):
    """
    x: [..., n, 2]
    """
    m = 1 << (stage + 1)
    half = m >> 1

    # reshape [..., n] -> [..., n/m, m]
    shape = list(x.shape)
    shape[-2] = n // m
    shape.insert(-1, m)
    x = x.reshape(shape)

    even = x[..., :half, :]
    odd = x[..., half:, :]

    tw = _twiddle_factors(m, x.device, x.dtype)[:half]

    # broadcast twiddle
    while tw.ndim < odd.ndim:
        tw = tw.unsqueeze(0)

    odd = _complex_mul(odd, tw)

    out1 = _complex_add(even, odd)
    out2 = _complex_sub(even, odd)

    x = torch.cat((out1, out2), dim=-2)

    # restore [..., n, 2]
    return x.reshape(*x.shape[:-2], n, 2)


def _fft_1d_radix2(x, n):
    """
    x: [..., n, 2]
    """
    stages = int(math.log2(n))
    for stage in range(stages):
        x = _fft_butterfly_stage(x, stage, n)
    return x


def _rfftn_onnx_radix2(input, shape, dim):
    # ---- guards (mandatory) ----
    if any(not isinstance(n, (int, torch.SymInt)) for n in shape):
        raise RuntimeError("ONNX rfftn requires static shapes")

    for n in shape:
        if n & (n - 1) != 0:
            raise RuntimeError("ONNX rfftn requires power-of-2 sizes")

    # ---- real → complex ----
    x = torch.stack([input, torch.zeros_like(input)], dim=-1)

    # ---- FFT per dimension ----
    for n, d in zip(shape, dim):
        x = x.movedim(d, -2)
        x = _fft_1d_radix2(x, n)
        x = x.movedim(-2, d)

    # ---- onesided truncation ----
    last_dim = dim[-1]
    cutoff = shape[-1] // 2 + 1
    x = x.narrow(last_dim, 0, cutoff)

    return x
