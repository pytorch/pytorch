# mypy: allow-untyped-defs
from __future__ import annotations

import os
from typing import NamedTuple

import sympy

import torch
from torch.utils._ordered_set import OrderedSet


class GemmConfigKey(NamedTuple):
    tile_m: int
    tile_n: int
    tile_k: int | None
    num_warps: int | None
    pingpong: bool
    is_dynamic_persistent: bool
    cluster_m: int
    cluster_n: int
    cluster_k: int
    swap_ab: bool
    max_swizzle_size: int
    device_capacity: int
    use_tma_gather: bool


_QUACK_DEFAULT_CONFIG_KEY = GemmConfigKey(
    128, 256, None, None, False, True, 2, 1, 1, False, 8, 10, False
)
_QUACK_SKINNY_CONFIG_KEY = GemmConfigKey(
    128, 192, None, None, False, True, 2, 1, 1, False, 8, 10, False
)
_QUACK_LARGE_CONFIG_KEY = GemmConfigKey(
    256, 256, None, None, False, True, 2, 2, 1, False, 8, 10, False
)
_QUACK_LARGE_RECT_CONFIG_KEY = GemmConfigKey(
    256, 256, None, None, False, True, 2, 1, 1, False, 8, 10, False
)
_QUACK_DENSE_CONFIG_PRIORITY_KEYS = (
    _QUACK_DEFAULT_CONFIG_KEY,
    _QUACK_SKINNY_CONFIG_KEY,
    _QUACK_LARGE_RECT_CONFIG_KEY,
    _QUACK_LARGE_CONFIG_KEY,
    GemmConfigKey(256, 192, None, None, False, True, 2, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 128, None, None, False, False, 1, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 256, None, None, False, True, 1, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 256, None, None, False, False, 1, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 128, None, None, False, True, 2, 1, 1, False, 8, 10, False),
    GemmConfigKey(256, 128, None, None, False, True, 2, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 224, None, None, False, True, 1, 1, 1, False, 8, 10, False),
    GemmConfigKey(128, 160, None, None, False, True, 1, 1, 1, False, 8, 10, False),
)
_QUACK_DENSE_CONFIG_PRIORITY = {
    key: priority for priority, key in enumerate(_QUACK_DENSE_CONFIG_PRIORITY_KEYS)
}


def inductor_quack_cache_dir() -> str:
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return os.path.join(cache_dir(), "quack")


def check_matrix(name: str, tensor: torch.Tensor) -> None:
    """Require a 2-D CUDA tensor for FlexGEMM runtime dispatch."""
    if tensor.ndim != 2:
        raise NotImplementedError(f"FlexGEMM currently supports only 2-D {name}")
    if not tensor.is_cuda:
        raise RuntimeError(f"FlexGEMM requires CUDA {name}")


def check_same_device(a: torch.Tensor, b: torch.Tensor, *rest: torch.Tensor) -> None:
    """Require all runtime tensors to live on the same CUDA device."""
    device = a.device
    if b.device != device or any(tensor.device != device for tensor in rest):
        raise RuntimeError("FlexGEMM inputs must be on the same device")


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


def validate_epilogue_arg_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    arg: torch.Tensor,
    kind: str,
) -> None:
    """Require a captured epilogue tensor shape to match its declared kind."""
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


def gemm_config_key(config) -> GemmConfigKey:
    """Project a QuACK GEMM config to a lossless generated-code key."""
    return GemmConfigKey(
        config.tile_m,
        config.tile_n,
        config.tile_k,
        config.num_warps,
        config.pingpong,
        config.is_dynamic_persistent,
        config.cluster_m,
        config.cluster_n,
        config.cluster_k,
        config.swap_ab,
        config.max_swizzle_size,
        config.device_capacity,
        config.use_tma_gather,
    )


def gemm_config_order(config) -> tuple[int, int, int, int, int, int]:
    """Rank QuACK configs by measured preference before stable tie-breakers."""
    config_key = gemm_config_key(config)
    priority = _QUACK_DENSE_CONFIG_PRIORITY.get(
        config_key, len(_QUACK_DENSE_CONFIG_PRIORITY)
    )
    return (
        priority,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        int(config.is_dynamic_persistent),
    )


def preferred_gemm_config_keys_from_dims(m, n) -> tuple[GemmConfigKey, ...]:
    """Choose Blackwell-measured config keys from GEMM output dimensions."""
    from torch._inductor.virtualized import V

    guard_or_false = V.graph.sizevars.guard_or_false
    if guard_or_false(sympy.Le(m, n)):
        min_dim, max_dim = m, n
    elif guard_or_false(sympy.Lt(n, m)):
        min_dim, max_dim = n, m
    else:
        return (_QUACK_DEFAULT_CONFIG_KEY,)
    if guard_or_false(sympy.Lt(min_dim, 512)):
        return (_QUACK_SKINNY_CONFIG_KEY, _QUACK_DEFAULT_CONFIG_KEY)
    if guard_or_false(sympy.And(sympy.Eq(min_dim, 1024), sympy.Eq(max_dim, 1024))):
        return (_QUACK_SKINNY_CONFIG_KEY, _QUACK_DEFAULT_CONFIG_KEY)
    if guard_or_false(
        sympy.And(
            sympy.Ge(max_dim, 4096), sympy.Ge(min_dim, 768), sympy.Lt(min_dim, 1024)
        )
    ):
        return (_QUACK_LARGE_CONFIG_KEY, _QUACK_DEFAULT_CONFIG_KEY)
    if guard_or_false(sympy.And(sympy.Ge(max_dim, 4096), sympy.Eq(min_dim, 1024))):
        return (_QUACK_LARGE_RECT_CONFIG_KEY, _QUACK_DEFAULT_CONFIG_KEY)
    if guard_or_false(sympy.Ge(min_dim, 2048)):
        return (_QUACK_LARGE_CONFIG_KEY, _QUACK_DEFAULT_CONFIG_KEY)
    return (_QUACK_DEFAULT_CONFIG_KEY,)


def candidate_gemm_configs_for_device(device: torch.device):
    """Return all device-compatible QuACK configs before shape-specific ranking."""
    from torch._vendor.quack.gemm_config import get_all_configs

    device_capacity = torch.cuda.get_device_capability(device)[0]
    if device_capacity == 11:
        device_capacity = 10
    configs = sorted(
        (
            config
            for config in get_all_configs()
            if config.device_capacity == device_capacity
            and not config.swap_ab
            and config.cluster_k == 1
            and not config.use_tma_gather
        ),
        key=gemm_config_order,
    )
    if not configs:
        raise RuntimeError(
            f"FlexGEMM found no QuACK configs for CUDA device capability "
            f"SM{device_capacity}0"
        )
    return configs


def default_gemm_config_key(device: torch.device, m, n) -> GemmConfigKey:
    """Return the untuned default QuACK config key for generated code."""
    configs = candidate_gemm_configs_for_device(device)
    config_keys = OrderedSet(gemm_config_key(config) for config in configs)
    for key in preferred_gemm_config_keys_from_dims(m, n):
        if key in config_keys:
            return key
    return gemm_config_key(configs[0])


def gemm_config_from_key(config_key: GemmConfigKey):
    """Resolve a generated-code key to an exact QuACK config without CUDA queries."""
    from torch._vendor.quack.gemm_config import get_all_configs

    config_key = GemmConfigKey(*config_key)
    for config in get_all_configs():
        if gemm_config_key(config) == config_key:
            return config
    raise RuntimeError(f"FlexGEMM found no QuACK config for key {config_key}")


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

    gemm_act_dispatch(
        a.unsqueeze(0),
        b.mT.unsqueeze(0),
        None,
        None if C is None else C.unsqueeze(0),
        out.unsqueeze(0),
        None,
        None,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_uses_c=bool(epilogue_arg_kinds),
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
        a: Left operand with shape ``[M, K]``.
        b: Right operand with shape ``[K, N]``.
        epilogue_fn: CuTeDSL epilogue callable applied to the accumulator tile.
        epilogue_key: Stable cache key component for the epilogue.
        C: Optional bias/addend with shape ``[M, N]``.
        alpha: Scale applied to the GEMM accumulator.
        beta: Scale applied to ``C`` when ``C`` is present.
        out_dtype: Optional output dtype. Defaults to ``a.dtype``.
        out: Optional preallocated output tensor with shape ``[M, N]``.
        epilogue_args: Optional tensor args captured by the epilogue.
        epilogue_arg_kinds: Explicit ``tile``, ``row``, or ``col`` kind per arg.
        config_key: Optional explicit QuACK config key selected by Inductor autotune.
        device_capacity_override: Parent-computed capability for compile-only workers.
        quack_cache_dir: Optional scoped cache root for Inductor-generated QuACK work.

    Returns:
        Tensor with shape ``[M, N]``.
    """
    check_matrix("a", a)
    check_matrix("b", b)
    check_matrix_major_layout("a", a)
    check_matrix_major_layout("b", b)
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({a.shape} and {b.shape})"
        )
    expected_shape = (a.shape[0], b.shape[1])
    expected_dtype = a.dtype if out_dtype is None else out_dtype
    if C is not None:
        check_matrix("C", C)
        check_matrix_major_layout("C", C)
        if tuple(C.shape) != expected_shape:
            raise RuntimeError(
                f"C shape must be {expected_shape}, got {tuple(C.shape)}"
            )
    if out is not None:
        check_matrix("out", out)
        check_matrix_major_layout("out", out)
        if tuple(out.shape) != expected_shape:
            raise RuntimeError(
                f"out shape must be {expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != expected_dtype:
            raise RuntimeError(f"out dtype must be {expected_dtype}, got {out.dtype}")
    if epilogue_args and C is not None:
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
    from torch._vendor.quack.cache import cache_dir_override

    with cache_dir_override(quack_cache_dir):
        dispatch_gemm_act(
            a,
            b,
            C,
            out,
            epilogue_key,
            inferred_arg_kinds,
            row_args,
            col_args,
            tile_args,
            alpha,
            beta,
            config=(
                gemm_config_from_key(config_key)
                if config_key is not None
                else candidate_gemm_configs_for_device(a.device)[0]
            ),
            device_capacity_override=device_capacity_override,
        )
    return out
