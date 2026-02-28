# mypy: allow-untyped-defs
from __future__ import annotations

import copy
import functools
import logging
import math
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Hashable

import torch
from torch.utils._ordered_set import OrderedSet

from ...utils import prefix_is_reduction
from ..hints import (
    _NUM_THREADS_PER_WARP,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    ReductionHint,
    TileHint,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from ..runtime_utils import (
    conditional_product,
    last_power_of_2,
    next_power_of_2,
    triton_config_to_hashable,
)
from ..triton_compat import Config as TritonConfig
from .registry import register_triton_heuristic


Config = cast(Any, TritonConfig)


log = logging.getLogger(__name__)


class InductorConfig(Config):
    """Inductor-specific Triton config with additional control flags"""

    def __init__(self, *args, dynamic_scale_rblock=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_scale_rblock = dynamic_scale_rblock


def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""
    seen: OrderedSet[Hashable] = OrderedSet()
    pruned_configs = []

    for cfg in configs:
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):
        if numel is None:
            continue
        block = cfg[f"{label}BLOCK"]
        if numel == 1:
            assert block == 1, (
                f"TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
            )
        max_block = TRITON_MAX_BLOCK[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        assert max_block % block == 0, (
            f"TritonKernel.indexing assumes {label}BLOCK divides {max_block_str}"
            f" but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg})."
        )


def check_max_block(cfg: dict[str, int]):
    """
    Check that block sizes are within the maximum allowed.
    """
    for var, val in cfg.items():
        block_suffix = "BLOCK"
        if block_suffix in var:
            prefix = var.removesuffix(block_suffix)
            max_block = TRITON_MAX_BLOCK[prefix]
            assert val <= max_block, (
                f"'{var}' too large. Maximum: {max_block}. Actual: {val}."
            )


def _num_warps(num_warps, max_num_warps=8, min_num_warps=2, register_intensive=False):
    # On AMD GPU each warp has 64 lanes which is double the size on NV GPU,
    # therefore using half the number of warps here correspondingly.
    if torch.version.hip:
        max_num_warps = (max_num_warps + 1) // 2
        min_num_warps = (min_num_warps + 1) // 2
    # persistent reduction is register intensive
    if register_intensive:
        max_num_warps = max_num_warps // 2
    return next_power_of_2(min(max(num_warps, min_num_warps), max_num_warps))


def _check_max_grid_x(size_hints, x, num_warps):
    # Check if maxGridSize is exceeded - if so then must scale XBLOCK further
    max_grid_x = 2147483647
    max_block_x = TRITON_MAX_BLOCK["X"]
    warp_size = (
        64 if torch.version.hip else 32
    )  # TODO: query warp size once #129663 is merged
    num_blocks = (size_hints["x"] + x - 1) // x

    if torch.version.hip:
        # HIP has a 2^31-1 limit on total threads (num_blocks * num_warps * warp_size)
        while (
            (num_blocks * num_warps * warp_size) > max_grid_x
            and x < size_hints["x"]
            and x < max_block_x
        ):
            x *= 2
            num_blocks = num_blocks // 2
    else:
        # NVIDIA has a 2^31-1 limit on number of blocks in grid (not total threads)
        while num_blocks > max_grid_x and x < size_hints["x"] and x < max_block_x:
            x *= 2
            num_blocks = num_blocks // 2

    if num_blocks > max_grid_x:
        raise AssertionError(
            "Reduction config exceeds cudaDeviceProp maxGridSize. Please raise a pytorch issue"
        )
    return x, num_blocks


def _conditional_product_optional(*values: int | None) -> int:
    return conditional_product(*[value for value in values if value is not None])


def triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
    num_warps=None,
    matrix_instr=None,
    waves_per_eu=None,
) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """
    # Ideally we want to read this from some device config

    maxGridSize = [2147483647, 65535, 65535]

    target = _conditional_product_optional(x, y, z)
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    if y:
        y = min(y, size_hints["y"])
    if z:
        z = min(z, size_hints["z"])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < min(size_hints["x"], TRITON_MAX_BLOCK["X"]) and (
        x * maxGridSize[0] < size_hints["x"]
        or _conditional_product_optional(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < min(size_hints["y"], TRITON_MAX_BLOCK["Y"])
        and (
            y * maxGridSize[1] < size_hints["y"]
            or _conditional_product_optional(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < min(size_hints["z"], TRITON_MAX_BLOCK["Z"])
        and (
            z * maxGridSize[2] < size_hints["z"]
            or _conditional_product_optional(x, y, z) < target
        )
    ):
        z *= 2

    # Calculate num_warps if they are not hard passed to config
    if num_warps is None:
        num_warps = _num_warps(
            _conditional_product_optional(x, y, z) // num_elements_per_warp,
            min_num_warps=1,
        )
    # we are going to arrive at 2 warps only if bs was too small due to
    # numel being too small. However to workaround some ptx bugs we still
    # want at least 4 warps if there's enough elements per thread
    # given that this is a rare situation, don't expect this to affect perf
    # in general
    # see https://github.com/pytorch/pytorch/pull/97950
    if _conditional_product_optional(x, y, z) >= 128 and not torch.version.hip:
        num_warps = max(num_warps, 4)
    xnumel = size_hints["x"]
    ynumel = size_hints.get("y")
    znumel = size_hints.get("z")

    # Increase x to satisfy min_elem_per_thread requirements.
    base_block = _conditional_product_optional(x, y, z)
    block_size = max(
        base_block,
        min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
    )
    x *= math.ceil(block_size / base_block)

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)
    x = min(x, size_hints["x"])

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    check_max_block(cfg)
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    config = Config(cfg, num_warps=num_warps, num_stages=num_stages)

    if torch.version.hip:
        if matrix_instr is not None:
            config.kwargs["matrix_instr_nonkdim"] = matrix_instr
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu

    return config


def cached_autotune(*args, **kwargs):
    from ..triton_heuristics import cached_autotune as _cached_autotune

    return _cached_autotune(*args, **kwargs)


def get_total_reduction_numel(numels: dict[str, int]) -> int:
    return conditional_product(
        *[numel for prefix, numel in numels.items() if prefix_is_reduction(prefix)]
    )


def autotune_hints_to_configs(
    hints: OrderedSet[AutotuneHint],
    size_hints,
    block_size: int,
    device_props: DeviceProperties,
) -> list[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """
    xyz_options: tuple[tuple[int, int | None, int | None], ...] = ()
    configs: list[Config] = []
    for hint in hints:
        if hint == AutotuneHint.ONE_ELEMENT_PER_THREAD:
            if len(size_hints) == 1:
                xyz_options = ((block_size // 4, None, None),)
            elif len(size_hints) == 2:
                xyz_options = ((block_size // 4, 1, None), (1, block_size // 4, None))
            elif len(size_hints) == 3:
                xyz_options = (
                    (block_size // 4, 1, 1),
                    (1, block_size // 4, 1),
                    (1, 1, block_size // 4),
                )
            configs.extend(
                triton_config(
                    size_hints,
                    *xyz,
                    num_elements_per_warp=(
                        device_props.warp_size if device_props.warp_size else 32
                    ),
                )
                for xyz in xyz_options
            )

    return configs


def _get_nd_reduction_numels(r: int, size_hints: dict[str, int]) -> dict[str, int]:
    """
    Converts a linear reduction numel to ND, in row major order.
    This order is often desirable as it presents opportunities to coalesce memory
    accesses.
    For example, if r = 64 and size_hints = [32,32], this function returns [32, 2].
    This unraveling works because both r and size_hints are powers of 2.
    """
    # Shrink r to size_hints.
    r = min(r, get_total_reduction_numel(size_hints))
    num_reduction_dims = len(
        [prefix for prefix in size_hints if prefix_is_reduction(prefix)]
    )

    remaining = r
    rnumels = {}
    for idx in range(num_reduction_dims - 1, -1, -1):
        prefix = f"r{idx}_"
        max_size = min(size_hints[prefix], TRITON_MAX_BLOCK[prefix.upper()])
        dim = min(max_size, remaining)
        assert remaining % dim == 0, (
            f"Expected dimension '{dim}' to divide remaining size '{remaining}'"
        )
        rnumels[prefix] = dim
        remaining //= dim

    # Sanity check the results.
    final_numel = conditional_product(*rnumels.values())
    assert r == final_numel, (
        f"Expected ND reduction size ({rnumels}) to have {r} elements."
    )
    assert all(rnumels[prefix] <= size_hints[prefix] for prefix in rnumels), (
        f"rnumels exceed size_hints. {rnumels} > {size_hints}"
    )

    return rnumels


def triton_config_reduction(
    size_hints,
    x: int,
    r: int,
    num_stages=1,
    num_warps=None,
    register_intensive=False,
    waves_per_eu=None,
    dynamic_scale_rblock=True,
    reduction_hint=None,
    min_num_warps=None,
) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])

    def total_numel() -> int:
        return conditional_product(x, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2

    if num_warps is None:
        if reduction_hint == ReductionHint.INNER:
            # r is contiguous, ensure at least 8 elements per thread
            # xblock is usually 1-2, default to giving each thread more work
            num_warps = r // 128
        else:
            num_warps = total_numel() // 128

    max_num_warps = 16 if r <= 8192 else 32
    if min_num_warps is not None:
        _num_warps_func = functools.partial(_num_warps, min_num_warps=min_num_warps)
    else:
        _num_warps_func = _num_warps

    num_warps = _num_warps_func(
        num_warps, max_num_warps=max_num_warps, register_intensive=register_intensive
    )

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

    for prefix in sorted(rnumels):
        while total_numel() > target:
            if rnumels[prefix] == 1:
                break
            rnumels[prefix] //= 2

    cfg = _get_config({"x": x, **rnumels})
    check_max_block(cfg)
    check_config(cfg, xnumel=size_hints["x"])
    config = InductorConfig(
        cfg,
        num_warps=num_warps,
        num_stages=num_stages,
        dynamic_scale_rblock=dynamic_scale_rblock,
    )

    if torch.version.hip:
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu

    return config


def _get_config(numels: dict[str, int]) -> dict[str, int]:
    """
    Convert numels ("x", "r0_", etc.) to block sizes ("XBLOCK", "R0_BLOCK"), etc.
    """

    return {prefix.upper() + "BLOCK": numel for prefix, numel in numels.items()}


def _handle_combo_kernel_per_subkernel_blocks(
    size_hints: dict[str, int],
    inductor_meta: dict[str, Any] | None,
    triton_meta: dict[str, Any] | None,
    filename: str | None = None,
    reduction_hint: bool = False,
    tile_hint: Any = None,
    min_elem_per_thread: int = 0,
) -> list[Config] | None:
    """
    Handle per-subkernel config generation for combo kernels.

    Each sub-kernel gets its own block sizes (XBLOCK_0, XBLOCK_1, etc.) generated
    using the same heuristics as standalone Triton kernels. The final config uses
    the maximum num_warps and num_stages across all sub-kernels.

    Returns:
        List of configs if combo kernel with combo_grid_meta and per-subkernel
        blocks enabled, None otherwise.
    """
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for combo kernel heuristics")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    combo_meta = inductor_meta.get("combo_grid_meta")
    if combo_meta is None or "heuristic_0" not in combo_meta:
        return None

    num_kernels = combo_meta["num_kernels"]
    inductor_meta_clean = {
        k: v for k, v in inductor_meta.items() if k != "combo_grid_meta"
    }

    combined_kwargs: dict[str, int] = {}
    all_num_warps: list[int] = []
    all_num_stages: list[int] = []
    unique_warp_stage_pairs: OrderedSet[tuple[int, int]] = OrderedSet()

    from ..triton_heuristics import persistent_reduction, pointwise, reduction

    for i in range(num_kernels):
        subkernel_heuristic = combo_meta[f"heuristic_{i}"]
        size_hints_i = combo_meta[f"size_hints_{i}"]

        if subkernel_heuristic == "pointwise":
            cfg = pointwise(
                size_hints_i,
                triton_meta=triton_meta,
                tile_hint=TileHint.SQUARE
                if combo_meta[f"tile_hint_{i}"] == "TileHint.SQUARE"
                else TileHint.DEFAULT,
                filename=filename,
                min_elem_per_thread=min_elem_per_thread,
                inductor_meta=inductor_meta_clean,
                return_configs=True,
            )[0]
            skip_rblock = False
        elif subkernel_heuristic == "reduction":
            cfg = reduction(
                size_hints_i,
                reduction_hint=reduction_hint,
                triton_meta=triton_meta,
                filename=filename,
                inductor_meta=inductor_meta_clean,
                return_configs=True,
            )[0]
            skip_rblock = False
        elif subkernel_heuristic == "persistent_reduction":
            cfg = persistent_reduction(
                size_hints_i,
                reduction_hint=reduction_hint,
                triton_meta=triton_meta,
                filename=filename,
                inductor_meta=inductor_meta_clean,
                return_configs=True,
            )[0]
            skip_rblock = True  # persistent reduction embeds RBLOCK in kernel body
        else:
            raise ValueError(f"Unknown heuristic: {subkernel_heuristic}")

        for key, value in cfg.kwargs.items():
            if skip_rblock and key.startswith("R") and "BLOCK" in key:
                continue
            combined_kwargs[f"{key}_{i}"] = value

        all_num_warps.append(cfg.num_warps)
        all_num_stages.append(cfg.num_stages)
        unique_warp_stage_pairs.add((cfg.num_warps, cfg.num_stages))

    unique_warp_stage_pairs.add((max(all_num_warps), max(all_num_stages)))

    return [
        Config(
            combined_kwargs,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps, num_stages in unique_warp_stage_pairs
    ]


def triton_config_tiled_reduction(
    size_hints, x, y, r, num_stages=1, register_intensive=False, waves_per_eu=None
):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    y = min(y, size_hints["y"])

    def total_numel() -> int:
        return conditional_product(x, y, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2
    while y < size_hints["y"] and total_numel() < target:
        y *= 2

    cfg = _get_config({"x": x, "y": y, **rnumels})
    num_warps = _num_warps(total_numel() // 256, min_num_warps=1)
    num_warps = _num_warps(
        num_warps, max_num_warps=16, register_intensive=register_intensive
    )
    check_config(cfg, xnumel=size_hints["x"], ynumel=size_hints["y"])
    check_max_block(cfg)
    config = Config(cfg, num_warps=num_warps, num_stages=num_stages)
    if torch.version.hip:
        if waves_per_eu is not None:
            config.kwargs["waves_per_eu"] = waves_per_eu
    return config


def make_matmul_triton_config(sizes: dict[str, int], num_warps: int, num_stages: int):
    cfg = _get_config(sizes)
    check_max_block(cfg)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs: list[Config]):
    tma_min_block_sizes: dict[str, int]
    if (tma_min_block_sizes := inductor_meta.get("tma_min_block_sizes")) and configs:
        # Rn blocks are not provided to the kernel for persistent reductions
        if inductor_meta.get("persistent_reduction"):
            tma_min_block_sizes = {
                block_type: block_size
                for block_type, block_size in tma_min_block_sizes.items()
                if not prefix_is_reduction(block_type.lower())
            }

        assert all(
            block_type in configs[0].kwargs for block_type in tma_min_block_sizes
        )

        # Add a config that is guaranteed to compile
        example_config = configs[0]
        config_block_sizes = {**example_config.kwargs}
        config_block_sizes.update(tma_min_block_sizes)
        new_configs = [
            Config(
                config_block_sizes,
                num_warps=example_config.num_warps,
                num_stages=example_config.num_stages,
                maxnreg=example_config.maxnreg,
                pre_hook=example_config.pre_hook,
            )
        ]
        # Remove configs that will not compile
        for c in configs:
            if all(
                c.kwargs.get(block_type) >= min_block_value
                for block_type, min_block_value in tma_min_block_sizes.items()
            ):
                new_configs.append(c)

        log.debug(
            "Filtering configs for TMA API restrictions. Input configs size: %d. Output configs size: %d",
            len(configs),
            len(new_configs),
        )
        return new_configs
    return configs


def _config_helper(bmm: bool, persistent: bool):
    _base_mm_configs = [
        ({"x": 32, "y": 32, "r": 16}, 2, 1),
        ({"x": 32, "y": 32, "r": 128}, 4, 2),
        ({"x": 32, "y": 64, "r": 32}, 8, 5),
        ({"x": 64, "y": 32, "r": 32}, 8, 5),
        ({"x": 64, "y": 32, "r": 128}, 4, 5),
        ({"x": 64, "y": 64, "r": 16}, 4, 2),
        ({"x": 64, "y": 64, "r": 32}, 4, 2),
        ({"x": 64, "y": 64, "r": 64}, 8, 3),
        ({"x": 64, "y": 64, "r": 128}, 4, 5),
        ({"x": 64, "y": 128, "r": 32}, 4, 3),
        ({"x": 64, "y": 128, "r": 32}, 8, 4),
        ({"x": 64, "y": 128, "r": 64}, 4, 3),
        ({"x": 64, "y": 128, "r": 128}, 4, 4),
        ({"x": 128, "y": 64, "r": 32}, 4, 3),
        ({"x": 128, "y": 64, "r": 32}, 8, 4),
        ({"x": 128, "y": 128, "r": 32}, 8, 2),
        ({"x": 128, "y": 128, "r": 32}, 4, 3),
        ({"x": 128, "y": 128, "r": 64}, 4, 3),
        ({"x": 128, "y": 128, "r": 64}, 8, 5),
    ]
    out = []
    for sizes, w, s in _base_mm_configs:
        d = dict(sizes)
        if persistent:
            d.pop("r", None)
        if bmm:
            d["z"] = 1
        out.append((d, w, s))

    # Deduplicate by converting dicts to immutable frozensets
    deduped = {(frozenset(d.items()), w, s): (d, w, s) for d, w, s in out}

    return list(deduped.values())


triton_native_mm_configs = _config_helper(bmm=False, persistent=False)
triton_native_persistent_mm_configs = _config_helper(bmm=False, persistent=True)
triton_native_bmm_configs = _config_helper(bmm=True, persistent=False)
triton_native_persistent_bmm_configs = _config_helper(bmm=True, persistent=True)


def _get_tiling_scores(
    inductor_meta: dict[str, Any],
    size_hints: dict[str, int],
) -> dict[str, float]:
    """
    Retrieve the tiling scores, providing suitable defaults if they are missing.
    """
    return inductor_meta.get("tiling_scores") or dict.fromkeys(size_hints, 1)


def _reduction_configs(
    *,
    size_hints: dict[str, int],
    inductor_meta: dict[str, Any] | None,
    triton_meta: dict[str, Any] | None,
    num_dynamic=0,
) -> list[Config]:
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for reduction configs")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    reduction_hint = inductor_meta.get("reduction_hint")

    # Convert reductions to 1D, to simplify heuristics.
    rnumel = get_total_reduction_numel(size_hints)

    # Is max autotune enabled
    max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
        "max_autotune_pointwise"
    )

    register_intensive = False
    MAX_R0_BLOCK = 2048
    loads_and_red = inductor_meta.get("num_load", 0) + inductor_meta.get(
        "num_reduction", 0
    )
    if size_hints["x"] >= 1024 and loads_and_red >= 10:
        # A heuristics to reduce R0_BLOCK if a kernel potentially need many registers.
        # Consider load and reduction since load need move data into registers and
        # reduction needs an accumulator.
        #
        # The magic numbers are a bit arbitrary.
        #
        # We cannot rely on dynamically scaling down R0_BLOCK later, since sometimes
        # triton makes it to use less registers with worse perf. Check:
        # https://github.com/pytorch/pytorch/issues/126463
        #
        # The heuristic is a very simple one since registers can be reused. But
        # hopefully it can be a good enough indicator.
        MAX_R0_BLOCK = 1024
        register_intensive = True

    if triton_meta.get("native_matmul"):
        if len(size_hints) == 3:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_mm_configs
            ]
        elif len(size_hints) == 4:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_bmm_configs
            ]
        else:
            raise NotImplementedError("native matmul only supports mm/bmm pattern")

    def make_config(
        x,
        r,
        num_warps=None,
        num_stages=1,
        register_intensive=False,
        dynamic_scale_rblock=True,
        waves_per_eu=None,
    ):
        # For 3D case with tiling scores, create an adapted version
        if "y" in size_hints:
            tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
            return adapt_config_for_tiling(
                size_hints,
                tiling_scores,
                x,
                r,
                num_warps=num_warps,
                num_stages=num_stages,
                register_intensive=register_intensive,
                waves_per_eu=waves_per_eu,
            )
        else:
            # For other cases, use the original function
            return triton_config_reduction(
                size_hints,
                x,
                r,
                num_warps=num_warps,
                num_stages=num_stages,
                register_intensive=register_intensive,
                waves_per_eu=waves_per_eu,
                dynamic_scale_rblock=dynamic_scale_rblock,
                reduction_hint=reduction_hint,
            )

    def outer_config_opt():
        # Default to 64 for vectorized loads
        max_x_block, x_block = 256, 64
        load_factor = inductor_meta.get("num_load", 0)
        x = size_hints["x"]
        num_warps = None

        # Try to use all SMs with small x
        if x <= 1024:
            x_block = max(min(x // 128, 8), 2)
            outer_r_block = min(rnumel, 64)
        # Lower bound x = 1024, 1024 // 16 = 128 around # of SMs
        elif x // 4096 <= 8:
            x_block = 16
            outer_r_block = 512 // x_block
        elif num_dynamic > 1:
            # Lots of compute with multiple dynamic shape per loop iteration
            # Larger RBLOCK minimizes loop iteration
            outer_r_block = max(min((rnumel // 64), 64), 8)
        elif num_dynamic == 1:
            # Dynamic shapes introduce a lot register pressure for indexing
            outer_r_block = (
                1
                if load_factor >= 3
                else min(next_power_of_2(max(rnumel, 128) // 128), 8)
            )
        else:
            x_block = max(min(max_x_block, next_power_of_2(x // 4096)), x_block)
            if load_factor < 4 or rnumel <= 128:
                outer_r_block = 512 // x_block
            else:
                # Heavier reductions contain a lot more overhead per loop iteration
                # We minimize the overhead by enlarging r block
                if rnumel >= 2048:
                    outer_r_block = 64
                else:
                    outer_r_block = 32
                x_block = min(x_block, 32)
                num_warps = 4

        # Set register intensive to true by default as we try to maximize tiles with heuristic
        return make_config(
            x_block,
            outer_r_block,
            num_warps=num_warps,
            register_intensive=register_intensive,
        )

    contiguous_config = make_config(
        2 if rnumel <= 2048 else 1,  # 1024 or less is persistent
        min(rnumel, MAX_R0_BLOCK),
        register_intensive=register_intensive,
    )
    tiny_config = make_config(
        2 * (256 // rnumel) if rnumel <= 256 else 1,
        min(rnumel, MAX_R0_BLOCK),
        register_intensive=register_intensive,
    )

    outer_config = make_config(64, 8, register_intensive=register_intensive)
    # TODO (paulzhan): Test heuristic on AMD and internal testing
    # for correctness
    if not torch.version.hip:
        outer_config = outer_config_opt()

    configs = []

    if inductor_meta.get("add_persistent_rblock") and loads_and_red <= 8:
        xnumel = max(4096 // rnumel, 1)
        c = make_config(
            xnumel,
            min(rnumel, 32768),
            register_intensive=register_intensive,
            dynamic_scale_rblock=False,
        )
        configs.append(c)

    result_configs = []

    # For 3d tiling, default to more autotuning initially
    if "y" in size_hints:
        pass
    elif max_autotune_enabled:
        pass  # skip all these cases
    elif reduction_hint == ReductionHint.INNER:
        return configs + [contiguous_config]
    elif reduction_hint == ReductionHint.OUTER:
        return configs + [outer_config]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return configs + [tiny_config]

    # We continue here under the following conditions:
    # - max_autotune_enabled is True
    # - max_autotune_enabled is False and reduction_hint is NOT one of the above cases
    result_configs = configs + [
        contiguous_config,
        outer_config,
        tiny_config,
        make_config(64, 64),
        make_config(8, 512),
        # halve the XBLOCK/Rn_BLOCK compared to outer_config
        # TODO: this may only be beneficial when each iteration of the reduction
        # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
        make_config(64, 4, num_warps=8),
    ]

    if torch.version.hip:
        result_configs.extend(
            [
                make_config(1024, 8, num_warps=4, num_stages=1, waves_per_eu=2),
                make_config(512, 8, num_warps=4, num_stages=1, waves_per_eu=1),
            ]
        )

    return result_configs


@register_triton_heuristic("reduction", None)
def reduction_common(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
    return_configs=False,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _handle_combo_kernel_per_subkernel_blocks(
        size_hints,
        inductor_meta,
        triton_meta,
        filename=filename,
        reduction_hint=reduction_hint,
    )
    if configs is not None:
        return cached_autotune(
            None,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.REDUCTION,
            filename=filename,
        )

    assert triton_meta is not None

    num_dynamic = 0
    for k in triton_meta["signature"]:
        if "ks" in k:
            num_dynamic += 1

    configs = _reduction_configs(
        size_hints=size_hints,
        inductor_meta=inductor_meta,
        triton_meta=triton_meta,
        num_dynamic=num_dynamic,
    )

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)

    if return_configs:
        return configs

    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def match_target_block_product(
    size_hints,
    tiling_scores,
    target_block_product,
    min_block_size=1,
    min_red_block: int | None = 4,
):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """
    min_red_block = (
        min_block_size if min_red_block is None else max(min_red_block, min_block_size)
    )
    total_score = sum(tiling_scores.values())
    if total_score == 0:
        # just assume even score with no minimum block size
        min_block_size = 1
        tiling_scores = dict.fromkeys(tiling_scores.keys(), target_block_product)

    # First, give each coalescing dimension at least min_block_size
    block_sizes = {}
    relative_scores = {}
    curr_block_product = 1

    for dim, score in tiling_scores.items():
        if score == 0 and "r" not in dim:
            block_sizes[dim] = 1
            relative_scores[dim] = 0
            continue

        size = min_block_size if "r" not in dim else min_red_block
        block_sizes[dim] = size
        curr_block_product *= size
        relative_scores[dim] = score / total_score

    # Scale up dimensions by their relative scores until we reach the target
    while curr_block_product < target_block_product and relative_scores:
        dim, score = max(relative_scores.items(), key=lambda item: item[1])

        # Check if we've hit the max for this dimension
        if (
            block_sizes[dim] >= TRITON_MAX_BLOCK[dim.capitalize()]
            or block_sizes[dim] >= size_hints[dim]
        ):
            del relative_scores[dim]
            continue

        block_sizes[dim] *= 2
        relative_scores[dim] /= 2
        curr_block_product *= 2

    return block_sizes


def adapt_config_for_tiling(
    size_hints,
    tiling_scores,
    original_x,
    original_r,
    num_warps=None,
    num_stages=1,
    register_intensive=False,
    persistent_reduction=False,
    waves_per_eu=None,
) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """
    assert all(s in tiling_scores for s in size_hints)
    target_block_product = original_x * original_r
    block_sizes = match_target_block_product(
        size_hints, tiling_scores, target_block_product
    )

    return triton_config_tiled_reduction(
        size_hints,
        block_sizes["x"],
        block_sizes["y"],
        block_sizes["r0_"],
        num_stages=num_stages,
        register_intensive=register_intensive,
        waves_per_eu=waves_per_eu,
    )


def filter_reduction_configs_for_determinism(
    inductor_meta: dict[str, Any], configs: list[Config]
) -> list[Config]:
    """
    Filter configs for reduction so the numerics can be deterministic.

    Heuristics:
    - skip reduction configs with too small RBLOCK
    - skip reduction configs with XBLOCK==1 if we are confident it will not perform well
    - if there is a tie, pick the config with second largest RBLOCK
    - if there is still a tie, pick the config with second largest num_warps
    - if there is still a tie, pick the config with second largest XBLOCK
    """
    configs = unique_configs(configs)
    assert len(configs) > 0

    def _do_filter_due_to_inductor_config():
        return (
            inductor_meta.get("deterministic", False)
            or inductor_meta.get("force_filter_reduction_configs", False)
        ) or inductor_meta.get("are_deterministic_algorithms_enabled")

    if not _do_filter_due_to_inductor_config() or len(configs) == 1:
        # no filtering happening if NOT in deterministic mode
        return configs

    if log.isEnabledFor(logging.DEBUG):
        log.debug("reduction configs before filtering:")
        for c in configs:
            log.debug("%s", c)
            log.debug("")

    def _has_too_small_rblock(config):
        rblock = config.kwargs.get("R0_BLOCK")
        # too small RBLOCK is likely to be bad
        return rblock is not None and rblock <= 4

    def _nonpromising_xblock_1(config):
        # kernel like https://gist.github.com/shunting314/0b3281c087e79bc915fe45985ff9d7d5
        # without a load/store having contiguous rdim is unlikely to perform well with XBLOCK==1
        return config.kwargs["XBLOCK"] == 1 and not inductor_meta.get(
            "has_loadstore_with_contiguous_rdim", True
        )

    newconfigs = [*filter(lambda x: not _has_too_small_rblock(x), configs)]
    # accept the filtering only if there are configs left
    if len(newconfigs) > 0:
        configs = newconfigs

    newconfigs = [*filter(lambda x: not _nonpromising_xblock_1(x), configs)]
    if len(newconfigs) > 0:
        configs = newconfigs

    assert len(configs) > 0

    def _r0_block(c):
        return c.kwargs.get("R0_BLOCK", -1)

    def _xblock(c):
        return c.kwargs.get("XBLOCK", -1)

    def _num_warps(c):
        return c.num_warps

    def _pick_second_largest(accessor):
        nonlocal configs
        configs = sorted(configs, key=lambda x: accessor(x))
        if accessor(configs[0]) != accessor(configs[-1]):
            max_val = accessor(configs[-1])
            configs = [*filter(lambda x: accessor(x) != max_val, configs)]
            second_max_val = accessor(configs[-1])
            configs = [*filter(lambda x: accessor(x) == second_max_val, configs)]
        return configs

    def _pick_config():
        nonlocal configs
        assert len(configs) > 0
        if len(configs) == 1:
            return configs[0]

        # break tie by R0_BLOCK
        configs = _pick_second_largest(_r0_block)
        if len(configs) == 1:
            return configs[0]

        # break tie by num_warps
        configs = _pick_second_largest(_num_warps)
        if len(configs) == 1:
            return configs[0]

        # break tie by XBLOCK
        configs = _pick_second_largest(_xblock)

        # there is still a tie, pick the first one
        return configs[0]

    configs = [_pick_config()]

    if log.isEnabledFor(logging.DEBUG):
        log.debug("reduction configs after filtering:")
        for c in configs:
            log.debug("%s", c)
            log.debug("")
    return configs


def _persistent_reduction_configs(
    size_hints,
    reduction_hint=False,
    inductor_meta=None,
    triton_meta=None,
):
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for persistent reduction")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    xnumel = size_hints["x"]
    rnumel = get_total_reduction_numel(size_hints)

    MAX_PERSISTENT_BLOCK_NUMEL = 4096

    if triton_meta.get("native_matmul"):
        if len(size_hints) == 3:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_persistent_mm_configs
            ]
        elif len(size_hints) == 4:
            return [
                make_matmul_triton_config(sizes, num_warps, num_stages)
                for sizes, num_warps, num_stages in triton_native_persistent_bmm_configs
            ]
        else:
            raise NotImplementedError("native matmul only supports mm/bmm pattern")

    max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
        "max_autotune_pointwise"
    )

    if torch.version.hip:
        xblock_vals = [1, 4, 8, 16, 32, 64, 128, 256]
    else:
        xblock_vals = [1, 8, 32, 128]

    if "y" not in size_hints:
        configs = [
            triton_config_reduction(
                size_hints,
                xblock,
                rnumel,
                register_intensive=True,
                reduction_hint=reduction_hint,
            )
            for xblock in xblock_vals
            if xblock == 1
            or (rnumel * xblock <= MAX_PERSISTENT_BLOCK_NUMEL and xblock <= xnumel)
        ]
    else:
        configs = []
        tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
        x_y_scores = {dim: tiling_scores[dim] for dim in ("x", "y")}
        for target_block_size in xblock_vals:
            if target_block_size * rnumel > MAX_PERSISTENT_BLOCK_NUMEL:
                continue

            block_sizes = match_target_block_product(
                size_hints, x_y_scores, target_block_size
            )
            configs.append(
                triton_config_tiled_reduction(
                    size_hints, block_sizes["x"], block_sizes["y"], rnumel
                )
            )

    tiny_configs = [
        triton_config_reduction(
            size_hints,
            2 * (256 // rnumel) if rnumel <= 256 else 1,
            rnumel,
        )
    ]

    # defer to more autotuning, initially
    if "y" in size_hints:
        pass
    # TODO(jansel): we should be able to improve these heuristics
    elif not max_autotune_enabled:  # Do not filter configs when tuning
        if reduction_hint == ReductionHint.INNER and rnumel >= 256:
            if rnumel > 1024 or xnumel // 8 < 128 or inductor_meta.get("RSPLIT_SIZE"):
                configs = configs[:1]
            else:
                if not torch.cuda.is_available():
                    # TODO(Intel): CUDA uses num_warps = 1 to disable shared memory.
                    # We apply different configurations from #168335.
                    # We currently let cost model in Triton to decide whether to use shared memory.
                    loads_and_stores = inductor_meta.get(
                        "num_load", 0
                    ) + inductor_meta.get("num_store", 0)
                    x_block = 8
                    if xnumel // x_block < 128 or loads_and_stores >= 5:
                        x_block = 1
                    num_warps, min_num_warps, reduction_hint = None, None, None
                else:
                    x_block = min(1024 // rnumel, 8)
                    num_warps, min_num_warps = 1, 1
                configs = [
                    triton_config_reduction(
                        size_hints,
                        x_block,
                        rnumel,
                        register_intensive=True,
                        num_warps=num_warps,
                        min_num_warps=min_num_warps,
                        reduction_hint=reduction_hint,
                    )
                ]

        elif reduction_hint == ReductionHint.OUTER:
            configs = configs[-1:]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            configs = tiny_configs
    else:
        if torch.version.hip:
            # If autotune is enabled append tiny configs
            for conf in tiny_configs:
                if conf not in configs:
                    configs.append(conf)

    for c in configs:
        # we don't need Rn_BLOCK for persistent reduction
        for prefix in size_hints:
            if prefix_is_reduction(prefix):
                c.kwargs.pop(f"{prefix.upper()}BLOCK")

    return configs


@register_triton_heuristic("cooperative_reduction", None)
def cooperative_reduction_common(
    size_hints,
    reduction_hint,
    triton_meta,
    filename,
    inductor_meta,
):
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for cooperative reduction")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    # Cooperative reductions currently only support a single reduction dimension.
    assert len(size_hints) == 2, (
        "Cooperative reductions don't support tiling reduction dims"
    )
    xnumel, rnumel = size_hints["x"], size_hints["r0_"]

    # Note that we must never create more CTAs than there are SMs, because we
    # depend on synchronizing between the CTAs in x_grid_barrier, and that will
    # deadlock if some of the CTAs are not running. In order to maximize use of
    # the GPU, we want to create as many CTAs as possible, while keeping things
    # in powers of 2.
    target = last_power_of_2(triton_meta["device"].multi_processor_count)
    split = max(1, min(target // xnumel, TRITON_MAX_RSPLIT))
    assert rnumel >= split
    assert split <= TRITON_MAX_RSPLIT
    if inductor_meta["persistent_reduction"]:
        configs = _persistent_reduction_configs(
            {"x": xnumel, "r0_": rnumel // split},
            reduction_hint,
            inductor_meta,
            triton_meta,
        )
    else:
        configs = _reduction_configs(
            size_hints={"x": xnumel, "r0_": rnumel // split},
            inductor_meta=inductor_meta,
            triton_meta=triton_meta,
        )
    for config in configs:
        config.kwargs["RSPLIT"] = split
    # TODO(jansel): add more configs in max_autotune

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


@register_triton_heuristic("persistent_reduction", None)
def persistent_reduction_common(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
    return_configs=False,
):
    """Generate persistent reductions + mix-order if available"""
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for persistent reduction")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _handle_combo_kernel_per_subkernel_blocks(
        size_hints,
        inductor_meta,
        triton_meta,
        filename=filename,
        reduction_hint=reduction_hint,
    )
    if configs is not None:
        return cached_autotune(
            None,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
            filename=filename,
        )

    configs = _persistent_reduction_configs(
        size_hints, reduction_hint, inductor_meta, triton_meta
    )

    # This key is not added to the inductor meta as its clear from the heuristic
    # choice that it is persistent. Add it and remove it below so that persistent
    # configs can be filtered appropriately by _maybe_filter_configs_for_tma_restrictions
    persistent_reduction_key = "persistent_reduction"
    inductor_meta[persistent_reduction_key] = True
    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    inductor_meta.pop(persistent_reduction_key)

    max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
        "max_autotune_pointwise"
    )

    if inductor_meta.get("RSPLIT_SIZE"):
        new_configs = []
        rsplit_size = inductor_meta.get("RSPLIT_SIZE")
        rnumel_hint = size_hints["r0_"]
        min_x_block = 1
        if rnumel_hint <= 512:
            min_x_block = 4
        # If TMA tensor descriptors are in use, Triton requires the last dimension
        # of a descriptor's block_shape to cover at least 16 bytes.
        # Codegen records such minimums in `tma_min_block_sizes`.
        # Ensuring our RSPLIT-driven XBLOCK override does not violate them.
        required_x_block = 1
        tma_min_block_sizes_any = inductor_meta.get("tma_min_block_sizes")
        tma_min_block_sizes: dict[str, int] | None = (
            tma_min_block_sizes_any
            if isinstance(tma_min_block_sizes_any, dict)
            else None
        )
        if tma_min_block_sizes is not None:
            required_x_block = max(
                required_x_block, tma_min_block_sizes.get("XBLOCK", 1)
            )
        x_block = min(max(rsplit_size // 32, min_x_block, required_x_block), 16)
        for c in configs:
            c.kwargs["RSPLIT_SIZE"] = rsplit_size
            # small XBLOCK to use less registers/smem
            c.kwargs["XBLOCK"] = x_block

            num_iters = rsplit_size // x_block

            # With large rnumel, we have higher chance of out-of-shared memory
            # To avoid adding too much autotuning overhead, we just constrain NUM_STAGES
            # if rnumel is large
            MAX_NUM_STAGES = 2 if rnumel_hint > 8192 else 3
            c.kwargs["NUM_STAGES"] = min(max(num_iters // 4, 1), MAX_NUM_STAGES)

            if rnumel_hint <= 1024:
                c.num_warps //= 2
                c.num_warps = max(c.num_warps, 1)
                new_configs.append(c)

                if max_autotune_enabled:
                    # less warps so potentially each sm can run more thread blocks
                    # Inside each thread block, we handle the split sequentially,
                    # more thread blocks is beneficial here.
                    newc = copy.deepcopy(c)
                    newc.num_warps = 2
                    new_configs.append(newc)
            else:
                # more warps for larger rows
                new_configs.append(c)

                if max_autotune_enabled and c.num_warps < 32:
                    newc = copy.deepcopy(c)
                    newc.num_warps *= 2
                    new_configs.append(newc)

        configs = unique_configs(new_configs)

    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)

    if return_configs:
        return configs

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


@register_triton_heuristic("split_scan", None)
def split_scan_common(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Heuristic for TritonSplitScanKernel"""
    if triton_meta is None:
        raise NotImplementedError("Missing triton_meta for split scan")

    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _reduction_configs(
        size_hints=size_hints, inductor_meta=inductor_meta, triton_meta=triton_meta
    )

    # Fixup configs to enforce the minimum Rn_BLOCK size
    min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
    for cfg in configs:
        for var in list(cfg.kwargs.keys()):
            if var.startswith("R") and cfg.kwargs[var] < min_rblock:
                cfg.kwargs[var] = min_rblock

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.SPLIT_SCAN,
        filename=filename,
    )


@register_triton_heuristic("foreach", None)
def foreach(triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    configs = []

    # Naive autotuning path for num_warps
    if not (
        inductor_meta.get("max_autotune") or inductor_meta.get("max_autotune_pointwise")
    ):
        configs.append(Config({}, num_stages=1, num_warps=8))
    else:
        for warps in [1, 2, 4, 8]:
            configs.append(Config({}, num_stages=1, num_warps=warps))

    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
