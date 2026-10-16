# CuTeDSL has no complex descriptor type, so kernels receive adjacent real storage.

import torch


def real_view(x: torch.Tensor) -> torch.Tensor:
    if x.dtype is torch.bool:
        return x.view(torch.uint8)
    return torch.view_as_real(x) if x.is_complex() else x


def row_view(x: torch.Tensor) -> torch.Tensor:
    view = real_view(x)
    return view.flatten(-2) if x.is_complex() else view


def flat_view(x: torch.Tensor) -> torch.Tensor:
    return real_view(x).reshape(-1)


def real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype is torch.bool:
        return torch.uint8
    if dtype is torch.complex32:
        return torch.float16
    if dtype is torch.complex64:
        return torch.float32
    if dtype is torch.complex128:
        return torch.float64
    return dtype
