# TMA pointwise execution for transposed and row-strided CUDA inputs.
# Contiguous inputs and unsupported layouts stay on ATen's CUDA kernels.

from __future__ import annotations

import os
from typing import Any, NamedTuple, TYPE_CHECKING

import cuda.bindings.driver as cuda  # noqa: TC002 - CuTeDSL resolves kernel annotations

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...utils import capability as cap
from ._layout import _layout_config, _OperandLayout, _TmaConfig
from ._limits import (
    _ELEMENTS_PER_THREAD,
    _FP16_ELEMENTS_PER_BANK,
    _FP16_LANES_PER_GROUP,
    _FP32_LANES_PER_GROUP,
    _MIN_ALIGNMENT,
    _SMEM_ALIGNMENT,
    _SWIZZLE_ATOM_BYTES,
    _SWIZZLE_BASE_LOG2,
    _SWIZZLE_SHIFT,
    _TILES_PER_BLOCK,
    _TMA_ALIGNMENT,
    _WARP_SIZE,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


_CONST_SLOTS = 3
_ENABLED = os.getenv("TORCH_NATIVE_CUTEDSL_POINTWISE_TUNING", "1") != "0"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_PLAN: dict[tuple[Any, ...], _Plan] = {}
_KERNELS: dict[tuple[Any, ...], Any] = {}


class _Plan(NamedTuple):
    config: _TmaConfig
    fn: Any


def _pointer(t: torch.Tensor) -> int:
    with torch._C.DisableTorchFunctionSubclass():
        return t.const_data_ptr()


def _layout_key(inputs: Sequence[torch.Tensor], address: int, smem_limit: int) -> tuple:
    layouts = tuple(
        _OperandLayout(
            tuple(t.shape),
            t.stride(),
            t.element_size(),
            _pointer(t) % _TMA_ALIGNMENT,
            t.is_contiguous(),
        )
        for t in inputs
    )
    return layouts, tuple(inputs[0].shape), address % _MIN_ALIGNMENT, smem_limit


def _tma_smem_layout(dtype: Any, box: int, transposed: bool) -> Any:
    # Transposed reads need a bank swizzle: 64-byte rows use two XOR bits and
    # 128-byte rows use three. The lane mapping below complements this swizzle.
    inner_bytes = box * dtype.width // 8
    if not transposed:
        swizzle_bits = 0
    elif inner_bytes <= 32:
        swizzle_bits = 1
    elif inner_bytes <= 64:
        swizzle_bits = 2
    else:
        swizzle_bits = 3
    swizzle = cute.make_swizzle(swizzle_bits, _SWIZZLE_BASE_LOG2, _SWIZZLE_SHIFT)
    # Repeat an eight-row swizzled atom over the box and its pipeline stages.
    atom_bytes = _SWIZZLE_ATOM_BYTES << swizzle_bits
    atom_elements = atom_bytes * 8 // (8 * dtype.width)
    if transposed:
        outer = cute.make_layout((atom_elements, 8), stride=(1, atom_elements))
        order = (1, 0, 2)
    else:
        outer = cute.make_layout((8, atom_elements), stride=(atom_elements, 1))
        order = (0, 1, 2)
    atom = cute.make_composed_layout(swizzle, 0, outer)
    return cute.tile_to_shape(atom, (box, box, _TILES_PER_BLOCK), order=order)


class _ElementwiseTma2D:
    def __init__(
        self,
        fn: Any,
        ninputs: int,
        nconsts: int,
        comp_type: Any,
        out_type: Any,
        in_type: Any,
        config: _TmaConfig,
    ) -> None:
        self.fn = fn
        self.ninputs = ninputs
        self.nconsts = nconsts
        self.comp_type = comp_type
        self.out_type = out_type
        self.in_type = in_type
        self.transposed = config.transposed
        self.box = config.box
        self.block = config.block

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOut: cute.Tensor,
        const0,
        const1,
        const2,
        stream: cuda.CUstream,
    ) -> None:
        tma_inputs = []
        layouts = []
        for k in cutlass.range_constexpr(self.ninputs):
            layout = _tma_smem_layout(self.in_type, self.box, self.transposed[k])
            info = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mIns[k], layout, (self.box, self.box)
            )
            tma_inputs.append(
                (info, cute.zipped_divide(info.tma_tensor, (self.box, self.box)))
            )
            layouts.append(layout)
        batch, rows, cols = mOut.shape
        self.kernel(
            tma_inputs,
            mOut,
            const0,
            const1,
            const2,
            layouts,
        ).launch(
            grid=[
                cute.ceil_div(cols, self.box * _TILES_PER_BLOCK),
                cute.ceil_div(rows, self.box),
                batch,
            ],
            block=[self.block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_inputs,
        mOut: cute.Tensor,
        const0,
        const1,
        const2,
        layouts: Any,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        fast_group, slow_tile, batch = cute.arch.block_idx()
        fast_tile = fast_group * _TILES_PER_BLOCK
        fast_tiles = mOut.shape[2] // const_expr(self.box)
        slow_tiles = mOut.shape[1] // const_expr(self.box)
        valid_stages = cutlass.min(fast_tiles - fast_tile, _TILES_PER_BLOCK)

        smem = cutlass.memory.SmemAllocator()
        buffers = [
            smem.allocate_tensor(
                self.in_type,
                layouts[k].outer,
                byte_alignment=_SMEM_ALIGNMENT,
                swizzle=layouts[k].inner,
            )
            for k in range(const_expr(self.ninputs))
        ]
        barriers = smem.allocate_array(cutlass.Int64, num_elems=_TILES_PER_BLOCK)
        tx_bytes = const_expr(
            self.ninputs * self.box * self.box * self.in_type.width // 8
        )

        # The first warp prefetches descriptors, then fills the input pipeline.
        # Each stage loads every tensor operand at coordinates determined by its
        # orientation relative to the output. One barrier tracks all of that
        # stage's TMA transfers. After the block synchronizes, threads wait for
        # each stage in turn, compute its elementwise results, and immediately
        # store them; later stages may still be transferring in the meantime.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for k in cutlass.range_constexpr(self.ninputs):
                cpasync.prefetch_descriptor(tma_inputs[k][0].atom)
            for stage in cutlass.range_constexpr(_TILES_PER_BLOCK):
                if stage < valid_stages:
                    barrier = barriers + stage
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(barrier, 1)
                        cute.arch.fence_proxy(kind="async.shared", space="cta")
                    for k in cutlass.range_constexpr(self.ninputs):
                        info, tiled = tma_inputs[k]
                        if const_expr(self.transposed[k]):
                            tile_coord = (
                                slow_tile,
                                batch * fast_tiles + fast_tile + stage,
                            )
                        else:
                            tile_coord = (
                                batch * slow_tiles + slow_tile,
                                fast_tile + stage,
                            )
                        g_tile = tiled[((None, None), tile_coord)]
                        s_tile = buffers[k][None, None, stage]
                        cutlass.block.block_copy(
                            info.atom,
                            cute.group_modes(g_tile, 0, 2),
                            cute.group_modes(s_tile, 0, 2),
                            tma_bar_ptr=barrier,
                        )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)
        cute.arch.barrier()

        lane = tidx % _WARP_SIZE
        warp_iteration = (tidx // _WARP_SIZE) * _ELEMENTS_PER_THREAD
        slow_base = slow_tile * const_expr(self.box)
        for stage in cutlass.range_constexpr(_TILES_PER_BLOCK):
            if stage < valid_stages:
                cute.arch.mbarrier_wait(barriers + stage, 0)
                fast_base = (fast_tile + stage) * const_expr(self.box)
                for i in cutlass.range_constexpr(_ELEMENTS_PER_THREAD):
                    wi = warp_iteration + i
                    if const_expr(self.in_type.width == 32):
                        fast = lane
                        slow = wi ^ (lane // _FP32_LANES_PER_GROUP)
                    else:
                        fast = lane
                        slow = wi ^ (
                            (lane // _FP16_LANES_PER_GROUP) * _FP16_ELEMENTS_PER_BANK
                        )
                    values = []
                    for k in cutlass.range_constexpr(self.ninputs):
                        values.append(self.comp_type(buffers[k][slow, fast, stage]))
                    if const_expr(self.nconsts == 0):
                        value = self.fn(*values)
                    elif const_expr(self.nconsts == 1):
                        value = self.fn(*values, const0)
                    elif const_expr(self.nconsts == 2):
                        value = self.fn(*values, const0, const1)
                    else:
                        value = self.fn(*values, const0, const1, const2)
                    mOut[batch, slow_base + slow, fast_base + fast] = self.out_type(
                        value
                    )


def _fake_tensor(
    dtype: Any, view: tuple[tuple[int, ...], tuple[int, ...]], align: int
) -> Any:
    shape, stride = view
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(stride), assumed_align=align
    )


def can_run(
    inputs: Sequence[torch.Tensor],
    shape: Sequence[int],
    out_dtype: torch.dtype,
    target: torch.Tensor | None = None,
    *,
    dtypes: tuple[torch.dtype, ...] = _DTYPES,
) -> bool:
    # The router owns Tensor/tracing and operator-specific scalar checks. This
    # validator owns device, dtype, output and storage checks once per call.
    if not _ENABLED:
        return False
    ref = inputs[0]
    if not cap.device_ok(ref, (10,)) or not cap.on_current_device(ref):
        return False
    if any(
        t.device != ref.device
        or t.dtype not in dtypes
        or t.dtype != ref.dtype
        or t.is_neg()
        or t.is_conj()
        for t in inputs
    ):
        return False
    if out_dtype not in (ref.dtype, torch.bool):
        return False
    if target is not None and (
        target.device != ref.device
        or target.dtype not in (ref.dtype, torch.bool)
        or not torch.can_cast(out_dtype, target.dtype)
        or tuple(target.shape) != tuple(shape)
        or not target.is_contiguous()
        or target.is_neg()
        or target.is_conj()
        or torch._C._is_cow_tensor(target)  # pyrefly: ignore[missing-attribute]
        or any(torch._C._overlaps(target, t) for t in inputs)
    ):
        return False
    address = 0 if target is None else _pointer(target)
    key = _layout_key(inputs, address, _hw.caps(ref.device).smem_per_block_optin)
    return _layout_config(*key) is not None


def _compile_plan(
    fn: Any,
    aten_name: str,
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    comp_type: type[cutlass.Numeric],
    config: _TmaConfig | None,
) -> _Plan:
    if config is None:
        raise AssertionError("pointwise override selected an unsupported TMA layout")
    in_type = torch2cute[inputs[0].dtype]
    out_type = cutlass.Int8 if out.dtype == torch.bool else torch2cute[out.dtype]
    op = _ElementwiseTma2D(
        fn=fn,
        ninputs=len(inputs),
        nconsts=len(consts),
        comp_type=comp_type,
        out_type=out_type,
        in_type=in_type,
        config=config,
    )
    fakes = [
        _fake_tensor(
            in_type, view, _TMA_ALIGNMENT if config.transposed[k] else _MIN_ALIGNMENT
        )
        for k, view in enumerate(config.input_views)
    ]
    fake_out = _fake_tensor(out_type, config.output_view, _MIN_ALIGNMENT)
    compile_args = (fakes, fake_out, *([comp_type(1)] * _CONST_SLOTS), _L.stream())
    caps = _hw.caps(inputs[0].device)
    kernel_key = (
        id(fn),
        len(consts),
        comp_type,
        in_type,
        out_type,
        config,
        inputs[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
    )
    compiled = cached_plan(
        _KERNELS,
        kernel_key,
        lambda: _L.compile_kernel(op, *compile_args),
        op=f"aten::{aten_name}",
    )
    return _Plan(config, compiled)


def run(
    fn: Any,
    aten_name: str,
    inputs: Sequence[torch.Tensor],
    out: torch.Tensor,
    consts: Sequence[Any],
    comp_type: type[cutlass.Numeric],
) -> None:
    caps = _hw.caps(inputs[0].device)
    layout_key = _layout_key(inputs, _pointer(out), caps.smem_per_block_optin)
    plan_key = (
        aten_name,
        id(fn),
        comp_type,
        layout_key,
        tuple(t.dtype for t in inputs),
        out.dtype,
        out.stride(),
        len(consts),
        inputs[0].device.index,
        caps.cc,
        caps.smem_per_block_optin,
    )
    plan = cached_plan(
        _PLAN,
        plan_key,
        lambda: _compile_plan(
            fn, aten_name, inputs, out, consts, comp_type, _layout_config(*layout_key)
        ),
    )
    output = out.view(torch.int8) if out.dtype == torch.bool else out
    runtime_inputs = [
        _L.read_only(t.as_strided(*plan.config.input_views[k]))
        for k, t in enumerate(inputs)
    ]
    runtime_out = output.as_strided(*plan.config.output_view)
    runtime_consts = list(consts) + [comp_type(1)] * (_CONST_SLOTS - len(consts))
    plan.fn(runtime_inputs, runtime_out, *runtime_consts, _L.stream())
