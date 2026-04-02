from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.external import autotune
from helion.autotuner.local_cache import get_helion_cache_dir
from helion.runtime.config import Config

import torch


_COMPILE_OPTIONS = "--enable-tvm-ffi"
_EXECUTOR_CACHE: dict[tuple[Any, ...], Any] = {}
_AUTOTUNE_CONFIG_CACHE: dict[tuple[Any, ...], Config] = {}
_AUTOTUNE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _tensor_alignment(tensor: torch.Tensor, dim: int) -> int:
    if dim < 0:
        dim += tensor.ndim

    contiguous_bytes = tensor.shape[dim] * tensor.element_size()
    alignment = 128
    while alignment > 1:
        if tensor.data_ptr() % alignment == 0 and contiguous_bytes % alignment == 0:
            return alignment
        alignment //= 2
    return 1


def _stride_order(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        sorted(range(tensor.ndim), key=lambda mode: (-tensor.stride(mode), mode))
    )


def _select(tensor: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, mode=mode))


def _autotune_cache_key(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[Any, ...]:
    return (
        lhs.device.type,
        lhs.device.index,
        lhs.shape,
        lhs.stride(),
        lhs.dtype,
        rhs.shape,
        rhs.stride(),
        rhs.dtype,
    )


def _autotune_disk_cache_dir() -> Path:
    user_path = os.environ.get("TORCH_BMM_OUTER_PRODUCT_CUTEDSL_CACHE_DIR")
    if user_path is not None:
        return Path(user_path)
    return get_helion_cache_dir() / "bmm_outer_product_cutedsl"


def _autotune_disk_cache_path(key: tuple[Any, ...]) -> Path:
    encoded_key = json.dumps(
        {"source": _AUTOTUNE_SOURCE_HASH, "key": [str(value) for value in key]},
        sort_keys=True,
    ).encode("utf-8")
    return (
        _autotune_disk_cache_dir() / f"{hashlib.sha256(encoded_key).hexdigest()}.json"
    )


def _load_disk_cached_autotune_config(key: tuple[Any, ...]) -> Config | None:
    try:
        return Config.load(_autotune_disk_cache_path(key))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _store_disk_cached_autotune_config(key: tuple[Any, ...], config: Config) -> None:
    try:
        config.save(_autotune_disk_cache_path(key))
    except OSError:
        return


@dataclass(frozen=True)
class BmmOuterProductCuteConfig:
    thr_m: int
    thr_n: int
    val_m: int
    val_n: int
    order: tuple[int, int] = (1, 0)

    @property
    def tile_m(self) -> int:
        return self.thr_m * self.val_m

    @property
    def tile_n(self) -> int:
        return self.thr_n * self.val_n

    def is_exact_tile(self, lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return lhs.shape[1] % self.tile_m == 0 and rhs.shape[2] % self.tile_n == 0


class BmmOuterProductCuteOp:
    def __init__(
        self,
        config: BmmOuterProductCuteConfig,
        use_residue: bool,
    ) -> None:
        self.config = config
        self.use_residue = use_residue

    def cache_key(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        out: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            lhs.shape,
            lhs.stride(),
            lhs.dtype,
            _tensor_alignment(lhs, -1),
            rhs.shape,
            rhs.stride(),
            rhs.dtype,
            _tensor_alignment(rhs, -1),
            out.shape,
            out.stride(),
            out.dtype,
            _tensor_alignment(out, -1),
            self.config.thr_m,
            self.config.thr_n,
            self.config.val_m,
            self.config.val_n,
            self.config.order,
            self.use_residue,
        )

    def to_cute_tensor(
        self,
        tensor: torch.Tensor,
        *,
        leading_dim: int,
        compact_shape_divisibility: dict[int, int],
    ) -> cute.Tensor:
        stride_order = _stride_order(tensor)
        cute_tensor = from_dlpack(
            tensor,
            assumed_align=_tensor_alignment(tensor, -1),
            enable_tvm_ffi=True,
        )
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        for mode, divisibility in compact_shape_divisibility.items():
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=mode,
                stride_order=stride_order,
                divisibility=divisibility,
            )
        return cute_tensor

    @cute.kernel
    def kernel(
        self,
        tiled_lhs: cute.Tensor,
        tiled_rhs: cute.Tensor,
        tiled_out: cute.Tensor,
        problem_m: int,
        problem_n: int,
        num_tiles_m: int,
        num_tiles_n: int,
        out_tv_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        linear_tile_idx, _, _ = cute.arch.block_idx()
        tiles_per_batch = num_tiles_m * num_tiles_n
        batch_idx = linear_tile_idx // tiles_per_batch
        tile_idx_in_batch = linear_tile_idx % tiles_per_batch
        tile_n_idx = tile_idx_in_batch // num_tiles_m
        tile_m_idx = tile_idx_in_batch % num_tiles_m
        tile_offset = (tile_m_idx * self.config.tile_m, tile_n_idx * self.config.tile_n)

        lhs_tile = _select(
            tiled_lhs[((None, None, None), (tile_m_idx, 0, batch_idx))], [0, 1]
        )
        rhs_tile = _select(
            tiled_rhs[((None, None, None), (0, tile_n_idx, batch_idx))], [0, 1]
        )
        out_tile = _select(
            tiled_out[((None, None, None), (tile_m_idx, tile_n_idx, batch_idx))], [0, 1]
        )

        if cutlass.const_expr(not self.use_residue):
            lhs_broadcast = cute.make_tensor(
                lhs_tile.iterator,
                cute.make_layout(
                    (self.config.tile_m, self.config.tile_n), stride=(1, 0)
                ),
            )
            rhs_broadcast = cute.make_tensor(
                rhs_tile.iterator,
                cute.make_layout(
                    (self.config.tile_m, self.config.tile_n), stride=(0, 1)
                ),
            )

            thr_lhs = cute.composition(lhs_broadcast, out_tv_layout)[(tidx, None)]
            thr_rhs = cute.composition(rhs_broadcast, out_tv_layout)[(tidx, None)]
            thr_out = cute.composition(out_tile, out_tv_layout)[(tidx, None)]
            thr_out.store(thr_lhs.load() * thr_rhs.load())
        else:
            coord_tile = cute.domain_offset(
                tile_offset,
                cute.make_identity_tensor((self.config.tile_m, self.config.tile_n)),
            )
            thr_coords = cute.flatten(
                cute.composition(coord_tile, out_tv_layout)[(tidx, None)]
            )
            for i in cutlass.range(cute.size(thr_coords), unroll_full=True):
                global_row = thr_coords[i][0]
                global_col = thr_coords[i][1]
                local_row = global_row - tile_offset[0]
                local_col = global_col - tile_offset[1]
                if global_row < problem_m and global_col < problem_n:
                    out_tile[local_row, local_col] = (
                        lhs_tile[local_row, 0] * rhs_tile[0, local_col]
                    )

    @cute.jit
    def __call__(
        self,
        lhs: cute.Tensor,
        rhs: cute.Tensor,
        out: cute.Tensor,
    ) -> None:
        # reorder batch mode to last dim
        lhs_view = cute.make_tensor(
            lhs.iterator, cute.select(lhs.layout, mode=[1, 2, 0])
        )
        rhs_view = cute.make_tensor(
            rhs.iterator, cute.select(rhs.layout, mode=[1, 2, 0])
        )
        out_view = cute.make_tensor(
            out.iterator, cute.select(out.layout, mode=[1, 2, 0])
        )

        thr_layout = cute.make_ordered_layout(
            (self.config.thr_m, self.config.thr_n), self.config.order
        )
        out_val_layout = cute.make_ordered_layout(
            (self.config.val_m, self.config.val_n), self.config.order
        )

        _, out_tv_layout = cute.make_layout_tv(thr_layout, out_val_layout)

        lhs_tiler = (self.config.tile_m, 1, 1)
        rhs_tiler = (1, self.config.tile_n, 1)
        out_tiler = (self.config.tile_m, self.config.tile_n, 1)

        tiled_lhs = cute.zipped_divide(lhs_view, lhs_tiler)
        tiled_rhs = cute.zipped_divide(rhs_view, rhs_tiler)
        tiled_out = cute.zipped_divide(out_view, out_tiler)
        problem_m = out_view.shape[0]
        problem_n = out_view.shape[1]
        num_tiles_m = cute.ceil_div(problem_m, self.config.tile_m)
        num_tiles_n = cute.ceil_div(problem_n, self.config.tile_n)
        num_batches = out_view.shape[2]

        self.kernel(
            tiled_lhs,
            tiled_rhs,
            tiled_out,
            problem_m,
            problem_n,
            num_tiles_m,
            num_tiles_n,
            out_tv_layout,
        ).launch(
            grid=(num_tiles_m * num_tiles_n * num_batches, 1, 1),
            block=(cute.size(thr_layout), 1, 1),
        )

    def interface(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            lhs.shape[0], lhs.shape[1], rhs.shape[2], dtype=lhs.dtype, device=lhs.device
        )
        key = self.cache_key(lhs, rhs, out)
        executor = _EXECUTOR_CACHE.get(key)

        if executor is None:
            executor = cute.compile(
                self,
                self.to_cute_tensor(
                    lhs,
                    leading_dim=1,
                    compact_shape_divisibility={}
                    if self.use_residue
                    else {1: self.config.tile_m},
                ),
                self.to_cute_tensor(
                    rhs,
                    leading_dim=2,
                    compact_shape_divisibility={}
                    if self.use_residue
                    else {2: self.config.tile_n},
                ),
                self.to_cute_tensor(
                    out,
                    leading_dim=2,
                    compact_shape_divisibility=(
                        {}
                        if self.use_residue
                        else {1: self.config.tile_m, 2: self.config.tile_n}
                    ),
                ),
                options=_COMPILE_OPTIONS,
            )
            _EXECUTOR_CACHE[key] = executor

        executor(lhs, rhs, out)
        return out


class BmmOuterProductCuteAutotuner:
    tunables = {
        "thr_m": PowerOfTwoFragment(1, 32, 2),
        "thr_n": PowerOfTwoFragment(1, 64, 4),
        "val_m": PowerOfTwoFragment(1, 8, 4),
        "val_n": PowerOfTwoFragment(1, 16, 8),
    }

    def compile(self, config: Config):
        cute_config = BmmOuterProductCuteConfig(
            thr_m=config["thr_m"],
            thr_n=config["thr_n"],
            val_m=config["val_m"],
            val_n=config["val_n"],
        )
        op = None

        def run(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            if cute_config.thr_m * cute_config.thr_n > 1024:
                raise ValueError(
                    f"Invalid thread block size: {cute_config.thr_m * cute_config.thr_n}"
                )
            nonlocal op
            if op is None:
                op = BmmOuterProductCuteOp(
                    cute_config,
                    use_residue=not cute_config.is_exact_tile(lhs, rhs),
                )
            return op.interface(lhs, rhs)

        return run

    def baseline(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return (lhs * rhs).contiguous()


_BMM_OUTER_PRODUCT_CUTE_AUTOTUNER = BmmOuterProductCuteAutotuner()


def _get_autotuned_config(lhs: torch.Tensor, rhs: torch.Tensor) -> Config:
    key = _autotune_cache_key(lhs, rhs)
    config = _AUTOTUNE_CONFIG_CACHE.get(key)
    if config is None:
        config = _load_disk_cached_autotune_config(key)
    if config is None:
        config = autotune(
            tunables=_BMM_OUTER_PRODUCT_CUTE_AUTOTUNER.tunables,
            compile_fn=_BMM_OUTER_PRODUCT_CUTE_AUTOTUNER.compile,
            baseline_fn=_BMM_OUTER_PRODUCT_CUTE_AUTOTUNER.baseline,
            args=(lhs, rhs),
            algorithm="LFBOPatternSearch",
            autotune_accuracy_check=True,
            autotune_ignore_errors=True,
        )
        _store_disk_cached_autotune_config(key, config)
    _AUTOTUNE_CONFIG_CACHE[key] = config
    return config


def bmm_outer_product(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return _BMM_OUTER_PRODUCT_CUTE_AUTOTUNER.compile(_get_autotuned_config(lhs, rhs))(
        lhs, rhs
    )
