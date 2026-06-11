# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


_QUACK_DEFAULT_TILE_M = 128
_QUACK_DEFAULT_TILE_N = 128
_QUACK_DEFAULT_CLUSTER_M = 1
_QUACK_DEFAULT_CLUSTER_N = 1


def _check_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise NotImplementedError(f"FlexGEMM currently supports only 2-D {name}")
    if not tensor.is_cuda:
        raise RuntimeError(f"FlexGEMM requires CUDA {name}")


def _check_same_device(a: torch.Tensor, b: torch.Tensor, *rest: torch.Tensor) -> None:
    device = a.device
    if b.device != device or any(tensor.device != device for tensor in rest):
        raise RuntimeError("FlexGEMM inputs must be on the same device")


def _check_matrix_major_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.stride(-1) != 1 and tensor.stride(-2) != 1:
        raise NotImplementedError(
            f"FlexGEMM requires {name} to be row- or column-major"
        )


def _check_epilogue_arg_kinds(epilogue_arg_kinds: tuple[str, ...]) -> None:
    for kind in epilogue_arg_kinds:
        if kind not in ("tile", "row", "col"):
            raise NotImplementedError(
                f"FlexGEMM supports only tile/row/col args, got {epilogue_arg_kinds}"
            )


def _infer_epilogue_arg_kind(
    a: torch.Tensor, b: torch.Tensor, arg: torch.Tensor
) -> str:
    m, n = a.shape[0], b.shape[1]
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


def _validate_epilogue_arg_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    arg: torch.Tensor,
    kind: str,
) -> None:
    m, n = a.shape[0], b.shape[1]
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


def _epilogue_arg_kinds(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    if epilogue_arg_kinds and len(epilogue_arg_kinds) != len(epilogue_args):
        raise RuntimeError("epilogue_arg_kinds must match epilogue_args length")
    _check_epilogue_arg_kinds(epilogue_arg_kinds)
    if not epilogue_arg_kinds:
        return tuple(_infer_epilogue_arg_kind(a, b, arg) for arg in epilogue_args)
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds):
        _validate_epilogue_arg_shape(a, b, arg, kind)
    return epilogue_arg_kinds


def _split_epilogue_args(
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
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


def gemm_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    *,
    C: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    out_dtype: torch.dtype | None = None,
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    epilogue_source: str | None = None,
    tuned: bool = False,
) -> torch.Tensor:
    """Run a dense GEMM through QuACK with a CuTeDSL epilogue.

    Args:
        a: Left operand with shape ``[M, K]``.
        b: Right operand with shape ``[K, N]``.
        epilogue_fn: CuTeDSL epilogue callable applied to the accumulator tile.
        epilogue_key: Stable cache key component for the epilogue.
        C: Optional bias/addend with shape ``[M, N]``.
        alpha: Scale applied to the GEMM accumulator.
        beta: Scale applied to ``C`` when ``C`` is present.
        out_dtype: Optional output dtype. Defaults to ``a.dtype``.
        epilogue_args: Optional tensor args captured by the epilogue.
        epilogue_arg_kinds: Explicit ``tile``, ``row``, or ``col`` kind per arg.
        epilogue_source: Optional source string included in the epilogue cache key.
        tuned: Whether to use QuACK autotuned config selection. Not supported yet.

    Returns:
        Tensor with shape ``[M, N]``.
    """
    _check_matrix("a", a)
    _check_matrix("b", b)
    _check_matrix_major_layout("a", a)
    _check_matrix_major_layout("b", b)
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({a.shape} and {b.shape})"
        )
    if C is not None:
        _check_matrix("C", C)
        _check_matrix_major_layout("C", C)
        if tuple(C.shape) != (a.shape[0], b.shape[1]):
            raise RuntimeError(
                f"C shape must be {(a.shape[0], b.shape[1])}, got {tuple(C.shape)}"
            )
    if epilogue_args and C is not None:
        # TODO: Route this through the flex frontend so validated A/B/C metadata
        # can be reused here.
        raise NotImplementedError("FlexGEMM args cannot be combined with C yet")
    if epilogue_args and (alpha != 1.0 or beta != 1.0):
        raise NotImplementedError(
            "FlexGEMM args cannot be combined with non-default alpha/beta yet"
        )
    if tuned:
        raise NotImplementedError(
            "FlexGEMM tuned=True requires the QuACK autotune wrapper follow-up"
        )

    tensors = (C, *epilogue_args) if C is not None else epilogue_args
    _check_same_device(a, b, *(tensor for tensor in tensors if tensor is not None))
    inferred_arg_kinds = _epilogue_arg_kinds(a, b, epilogue_args, epilogue_arg_kinds)
    for index, arg in enumerate(epilogue_args):
        _check_matrix_major_layout(f"epilogue_args[{index}]", arg)
    row_args, col_args, tile_args = _split_epilogue_args(
        epilogue_args, inferred_arg_kinds
    )
    passes_tensor_epilogue_args = bool(epilogue_args)

    if epilogue_source is not None:
        from torch._vendor.quack._compile_payload import set_epilogue_source_cache_key

        set_epilogue_source_cache_key(epilogue_fn, epilogue_source)

    from torch._vendor.quack.gemm_act import gemm_act as gemm_act_dispatch

    a_quack = a
    b_quack = b.mT
    c_quack = C
    out = torch.empty(
        (1, a.shape[0], b.shape[1]),
        device=a.device,
        dtype=a.dtype if out_dtype is None else out_dtype,
    )
    gemm_act_dispatch(
        a_quack.unsqueeze(0),
        b_quack.unsqueeze(0),
        None,
        None if c_quack is None else c_quack.unsqueeze(0),
        out,
        None,
        None,
        _QUACK_DEFAULT_TILE_M,
        _QUACK_DEFAULT_TILE_N,
        _QUACK_DEFAULT_CLUSTER_M,
        _QUACK_DEFAULT_CLUSTER_N,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=False,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_uses_c=passes_tensor_epilogue_args,
        tensor_epilogue_arg_kinds=inferred_arg_kinds,
        tensor_epilogue_rowvec_biases=row_args,
        tensor_epilogue_colvec_biases=col_args,
        tensor_epilogue_tile_biases=tile_args,
        alpha=alpha,
        beta=beta,
    )
    return out.squeeze(0)
