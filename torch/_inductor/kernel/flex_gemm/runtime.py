# mypy: allow-untyped-defs
from __future__ import annotations

import os
from typing import Any, TypeAlias

import torch
import torch._vendor.quack.gemm_config as quack_gemm_config
from torch._inductor.runtime.cache_dir_utils import cache_dir


GemmConfigKey: TypeAlias = tuple[tuple[str, Any], ...]


def inductor_quack_cache_dir() -> str:
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    return os.path.join(cache_dir(), "quack")


def check_matrix(name: str, tensor: torch.Tensor) -> None:
    """Require a 2-D or 3-D CUDA tensor for FlexGEMM runtime dispatch."""
    if tensor.ndim not in (2, 3):
        raise NotImplementedError(f"FlexGEMM currently supports only 2-D or 3-D {name}")
    if not tensor.is_cuda:
        raise RuntimeError(f"FlexGEMM requires CUDA {name}")


def check_same_device(a: torch.Tensor, b: torch.Tensor, *rest: torch.Tensor) -> None:
    """Require all runtime tensors to live on the same CUDA device."""
    device = a.device
    if b.device != device or any(tensor.device != device for tensor in rest):
        raise RuntimeError("FlexGEMM inputs must be on the same device")


def check_broadcast_shape(
    name: str, shape: torch.Size, expected_shape: tuple[int, ...]
) -> None:
    """Require a tensor shape to broadcast exactly to the GEMM output shape."""
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(shape), expected_shape)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{name} shape must broadcast to {expected_shape}, got {tuple(shape)}"
        ) from exc
    if broadcast_shape != expected_shape:
        raise RuntimeError(
            f"{name} shape must broadcast to {expected_shape}, got {tuple(shape)}"
        )


def check_matrix_major_layout(name: str, tensor: torch.Tensor) -> None:
    """Require row-major or column-major dense matrix strides."""
    if tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
        raise NotImplementedError(
            f"FlexGEMM requires {name} to be row- or column-major"
        )


def check_epilogue_arg_kinds(epilogue_arg_kinds: tuple[str, ...]) -> None:
    """Require each epilogue arg kind to be row, col, or tile."""
    for kind in epilogue_arg_kinds:
        if kind not in ("tile", "row", "col"):
            raise NotImplementedError(
                f"FlexGEMM supports only tile/row/col args, got {epilogue_arg_kinds}"
            )


def infer_epilogue_arg_kind(a: torch.Tensor, b: torch.Tensor, arg: torch.Tensor) -> str:
    """Infer a captured epilogue tensor's broadcast kind from its shape."""
    m, n = a.shape[-2], b.shape[-1]
    if tuple(arg.shape) == (m, n):
        return "tile"
    if tuple(arg.shape) == (1, n):
        return "row"
    if tuple(arg.shape) == (m, 1):
        return "col"
    raise NotImplementedError(
        "FlexGEMM captured tensor args must match the GEMM output "
        "shape or broadcast as [1, N] / [M, 1]"
    )


def validate_epilogue_arg_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    arg: torch.Tensor,
    kind: str,
) -> None:
    """Require a captured epilogue tensor shape to match its declared kind."""
    m, n = a.shape[-2], b.shape[-1]
    expected_shapes = {
        "tile": (m, n),
        "row": (1, n),
        "col": (m, 1),
    }
    if tuple(arg.shape) != expected_shapes[kind]:
        raise RuntimeError(
            f"{kind} epilogue arg shape must be {expected_shapes[kind]}, "
            f"got {tuple(arg.shape)}"
        )


def resolve_epilogue_arg_kinds(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate declared epilogue arg kinds or infer them from tensor shapes."""
    if epilogue_arg_kinds and len(epilogue_arg_kinds) != len(epilogue_args):
        raise RuntimeError("epilogue_arg_kinds must match epilogue_args length")
    check_epilogue_arg_kinds(epilogue_arg_kinds)
    if not epilogue_arg_kinds:
        return tuple(infer_epilogue_arg_kind(a, b, arg) for arg in epilogue_args)
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        validate_epilogue_arg_shape(a, b, arg, kind)
    return epilogue_arg_kinds


def split_epilogue_args(
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
    """Group epilogue tensors into QuACK row, col, and tile argument lists."""
    row_args = []
    col_args = []
    tile_args = []
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        match kind:
            case "row":
                row_args.append(arg)
            case "col":
                col_args.append(arg.squeeze(-1).unsqueeze(0))
            case "tile":
                tile_args.append(arg.unsqueeze(0))
    return tuple(row_args), tuple(col_args), tuple(tile_args)


def normalize_c(
    C: torch.Tensor | None, expected_shape: tuple[int, ...], beta: float
) -> torch.Tensor | None:
    """Return the effective C tensor that QuACK should read for alpha/beta GEMMs."""
    if C is None:
        return None
    check_broadcast_shape("C", C.shape, expected_shape)
    if beta == 0:
        return None
    broadcast_C = (
        C if tuple(C.shape) == expected_shape else torch.broadcast_to(C, expected_shape)
    )
    check_matrix("C", broadcast_C)
    check_matrix_major_layout("C", broadcast_C)
    return broadcast_C


def dispatch_gemm_act(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor | None,
    out: torch.Tensor,
    epilogue_key: str,
    epilogue_arg_kinds: tuple[str, ...],
    row_args: tuple[torch.Tensor, ...],
    col_args: tuple[torch.Tensor, ...],
    tile_args: tuple[torch.Tensor, ...],
    alpha: float,
    beta: float,
    config,
    device_capacity_override: tuple[int, int] | None = None,
) -> None:
    """Dispatch one dense FlexGEMM call to the vendored QuACK GEMM kernel."""
    from torch._vendor.quack.gemm_act import gemm_act as gemm_act_dispatch

    quack_a = a.unsqueeze(0) if a.ndim == 2 else a
    quack_b = b.mT.unsqueeze(0) if b.ndim == 2 else b.mT
    quack_C = None if C is None else C.unsqueeze(0) if C.ndim == 2 else C
    quack_out = out.unsqueeze(0) if out.ndim == 2 else out

    gemm_act_dispatch(
        quack_a,
        quack_b,
        None,  # D
        quack_C,
        quack_out,
        None,  # tile_count_semaphore
        None,  # cu_seqlens_m
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_arg_kinds=epilogue_arg_kinds,
        tensor_epilogue_rowvec_biases=row_args,
        tensor_epilogue_colvec_biases=col_args,
        tensor_epilogue_tile_biases=tile_args,
        alpha=alpha,
        beta=beta,
        use_tma_gather=config.use_tma_gather,
        device_capacity_override=device_capacity_override,
    )


def gemm_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_fn,
    epilogue_key: str,
    *,
    C: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    config_key: GemmConfigKey | None = None,
    device_capacity_override: tuple[int, int] | None = None,
    quack_cache_dir: str | None = None,
) -> torch.Tensor:
    """Run a dense GEMM through QuACK with a CuTeDSL epilogue.

    Args:
        a: Left operand with shape ``[M, K]`` or ``[B, M, K]``.
        b: Right operand with shape ``[K, N]`` or ``[B, K, N]``.
        epilogue_fn: CuTeDSL epilogue callable applied to the accumulator tile.
        epilogue_key: Stable cache key component for the epilogue.
        C: Optional bias/addend broadcastable to the output shape.
        alpha: Scale applied to the GEMM accumulator.
        beta: Scale applied to ``C`` when ``C`` is present.
        out_dtype: Optional output dtype. Defaults to ``a.dtype``.
        out: Optional preallocated output tensor with shape ``[M, N]`` or ``[B, M, N]``.
        epilogue_args: Optional tensor args captured by the epilogue.
        epilogue_arg_kinds: Explicit ``tile``, ``row``, or ``col`` kind per arg.
        config_key: Optional explicit QuACK config key selected by Inductor autotune.
        device_capacity_override: Parent-computed capability for compile-only workers.
        quack_cache_dir: Optional scoped cache root for Inductor-generated QuACK work.

    Returns:
        Tensor with shape ``[M, N]`` or ``[B, M, N]``.
    """
    check_matrix("a", a)
    check_matrix("b", b)
    check_matrix_major_layout("a", a)
    check_matrix_major_layout("b", b)
    if a.ndim != b.ndim:
        raise RuntimeError("FlexGEMM inputs must both be 2-D or both be 3-D")
    if a.ndim == 3 and a.shape[0] != b.shape[0]:
        raise RuntimeError("FlexGEMM batched inputs must have the same batch size")
    if a.shape[-1] != b.shape[-2]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({a.shape} and {b.shape})"
        )
    expected_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    expected_dtype = a.dtype if out_dtype is None else out_dtype
    effective_C = normalize_c(C, expected_shape, beta)
    if out is not None:
        check_matrix("out", out)
        check_matrix_major_layout("out", out)
        if tuple(out.shape) != expected_shape:
            raise RuntimeError(
                f"out shape must be {expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != expected_dtype:
            raise RuntimeError(f"out dtype must be {expected_dtype}, got {out.dtype}")
    if a.ndim == 3 and epilogue_args:
        raise NotImplementedError("FlexGEMM batched args are not supported yet")
    if epilogue_args and effective_C is not None:
        # TODO: Route this through the flex frontend so validated A/B/C metadata
        # can be reused here.
        raise NotImplementedError("FlexGEMM args cannot be combined with C yet")
    if epilogue_args and (alpha != 1.0 or beta != 1.0):
        raise NotImplementedError(
            "FlexGEMM args cannot be combined with non-default alpha/beta yet"
        )
    tensors = (C, out, *epilogue_args)
    check_same_device(a, b, *(tensor for tensor in tensors if tensor is not None))
    inferred_arg_kinds = resolve_epilogue_arg_kinds(
        a, b, epilogue_args, epilogue_arg_kinds
    )
    for index, arg in enumerate(epilogue_args):
        check_matrix_major_layout(f"epilogue_args[{index}]", arg)
    row_args, col_args, tile_args = split_epilogue_args(
        epilogue_args, inferred_arg_kinds
    )

    from torch._vendor.quack.gemm_act import register_tensor_epilogue_fn

    register_tensor_epilogue_fn(epilogue_key, epilogue_fn)
    out = (
        torch.empty(expected_shape, device=a.device, dtype=expected_dtype)
        if out is None
        else out
    )
    from torch._inductor.template_heuristics.flex_gemm import (
        candidate_gemm_configs_for_device,
    )
    from torch._vendor.quack.cache import cache_dir_override

    with cache_dir_override(quack_cache_dir):
        dispatch_gemm_act(
            a,
            b,
            effective_C,
            out,
            epilogue_key,
            inferred_arg_kinds,
            row_args,
            col_args,
            tile_args,
            alpha,
            beta,
            config=(
                quack_gemm_config.GemmConfig(**dict(config_key))
                if config_key is not None
                else candidate_gemm_configs_for_device(a.device)[0]
            ),
            device_capacity_override=device_capacity_override,
        )
    return out
