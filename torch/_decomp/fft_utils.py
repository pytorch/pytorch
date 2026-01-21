import math

import torch


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def extract_static_fft_sizes(x, s, dim):
    if s is not None:
        sizes = s
    else:
        shape = x.shape
        sizes = [shape[d] for d in dim]

    for n in sizes:
        if not isinstance(n, (int, torch.SymInt)):
            return None
        if not is_power_of_2(n):
            raise RuntimeError(
                f"Only power-of-2 FFT sizes are supported for ONNX export, got size {n}."
            )

    return sizes


def _to_complex(x):
    zero = torch.zeros_like(x)
    return torch.stack([x, zero], dim=-1)


def _complex_add(a, b):
    return a + b


def _complex_sub(a, b):
    return a - b


def _complex_mul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def _twiddle_factors(n, device):
    k = torch.arange(n // 2, device=device)
    angle = -2 * math.pi * k / n
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _fft_butterfly_stage(x, stage, n, dim):
    m = 1 << (stage + 1)
    half = m >> 1
    shape = list(x.shape)
    shape[dim] = n // m
    shape.insert(dim + 1, m)
    x = x.reshape(shape)

    even = x[..., :half, :]
    odd = x[..., half:, :]

    tw = _twiddle_factors(m, x.device)
    tw = tw[:half]
    while tw.dim() < odd.dim():
        tw = tw.unsqueeze(0)

    odd = _complex_mul(odd, tw)
    out1 = _complex_add(even, odd)
    out2 = _complex_sub(even, odd)

    x = torch.cat([out1, out2], dim=dim + 1)
    return x.reshape(*x.shape[:dim], n, 2)


def _fft_1d(x, n, dim):
    stages = int(math.log2(n))
    for stage in range(stages):
        x = _fft_butterfly_stage(x, stage, n, dim)
    return x
