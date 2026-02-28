# mypy: allow-untyped-defs
from __future__ import annotations

import logging
import math
from typing import Any, cast

import torch
from torch.utils._ordered_set import OrderedSet

from ...utils import prefix_is_reduction
from ..hints import (
    _NUM_THREADS_PER_WARP,
    AutotuneHint,
    DeviceProperties,
    TileHint,
    TRITON_MAX_BLOCK,
)
from ..runtime_utils import conditional_product, next_power_of_2
from ..triton_compat import Config as TritonConfig


Config = cast(Any, TritonConfig)


log = logging.getLogger(__name__)


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
