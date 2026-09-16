# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Small kernels adapted from the upstream FlyDSL examples."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr.typing import full


GEMM_M = 64
GEMM_N = 64
GEMM_K = 8
ELEMENTWISE_BLOCK = 256
RMS_N = 64
RMS_EPS = 1e-5


@flyc.kernel
def _gemm_kernel(
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    out: fx.Tensor,
):
    thread = fx.thread_idx.x
    block = fx.block_idx.x

    lhs = fx.rocdl.make_buffer_tensor(lhs)
    rhs = fx.rocdl.make_buffer_tensor(rhs)
    out = fx.rocdl.make_buffer_tensor(out)

    block_lhs = fx.slice(
        fx.zipped_divide(lhs, (GEMM_M, GEMM_K)),
        (None, block),
    )
    block_rhs = fx.slice(
        fx.zipped_divide(rhs, (GEMM_N, GEMM_K)),
        (None, block),
    )
    block_out = fx.slice(
        fx.zipped_divide(out, (GEMM_M, GEMM_N)),
        (None, block),
    )

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((2, 2, 1), (1, 2, 0)),
    )
    thread_mma = tiled_mma.thr_slice(thread)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    thread_copy_lhs = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(thread)
    thread_copy_rhs = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(thread)
    thread_copy_out = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(thread)

    lhs_fragment = thread_mma.make_fragment_A(block_lhs)
    rhs_fragment = thread_mma.make_fragment_B(block_rhs)
    out_fragment = thread_mma.make_fragment_C(block_out)

    fx.copy(
        copy_atom,
        thread_copy_lhs.partition_S(block_lhs),
        thread_copy_lhs.retile(lhs_fragment),
        pred=None,
    )
    fx.copy(
        copy_atom,
        thread_copy_rhs.partition_S(block_rhs),
        thread_copy_rhs.retile(rhs_fragment),
        pred=None,
    )
    out_fragment.fill(0)
    fx.gemm(mma_atom, out_fragment, lhs_fragment, rhs_fragment, out_fragment)
    fx.copy(
        copy_atom,
        thread_copy_out.retile(out_fragment),
        thread_copy_out.partition_S(block_out),
        pred=None,
    )


@flyc.jit
def gemm_launcher(
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    out: fx.Tensor,
):
    _gemm_kernel(lhs, rhs, out).launch(
        grid=(1, 1, 1),
        block=(256, 1, 1),
    )


@flyc.kernel
def _relu_kernel(
    inp: fx.Tensor,
    out: fx.Tensor,
    scale: fx.Float32,
    block_dim: fx.Constexpr[int],
):
    block = fx.block_idx.x
    thread = fx.thread_idx.x
    inp = fx.rocdl.make_buffer_tensor(inp)
    out = fx.rocdl.make_buffer_tensor(out)
    block_inp = fx.slice(
        fx.logical_divide(inp, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    block_out = fx.slice(
        fx.logical_divide(out, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    block_inp = fx.logical_divide(block_inp, fx.make_layout(1, 1))
    block_out = fx.logical_divide(block_out, fx.make_layout(1, 1))

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)
    register = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(copy_atom, fx.slice(block_inp, (None, thread)), register)
    value = fx.memref_load_vec(register)[0]
    activated = value.maximumf(fx.Float32(0.0)) * scale
    fx.memref_store_vec(full(1, activated, fx.Float32), register)
    fx.copy_atom_call(copy_atom, register, fx.slice(block_out, (None, thread)))


@flyc.jit
def relu_launcher(
    inp: fx.Tensor,
    out: fx.Tensor,
    elements: fx.Int32,
    scale: fx.Float32,
    block_dim: fx.Constexpr[int],
):
    blocks = (elements + block_dim - 1) // block_dim
    _relu_kernel(inp, out, scale, block_dim).launch(
        grid=(blocks, 1, 1),
        block=(block_dim, 1, 1),
    )


@flyc.kernel
def _rms_norm_kernel(
    inp: fx.Tensor,
    weight: fx.Tensor,
    out: fx.Tensor,
    columns: fx.Constexpr[int],
):
    row = fx.block_idx.x
    thread = fx.thread_idx.x
    inp = fx.rocdl.make_buffer_tensor(inp)
    weight = fx.rocdl.make_buffer_tensor(weight)
    out = fx.rocdl.make_buffer_tensor(out)

    row_inp = fx.logical_divide(
        fx.slice(inp, (row, None)),
        fx.make_layout(1, 1),
    )
    row_out = fx.logical_divide(
        fx.slice(out, (row, None)),
        fx.make_layout(1, 1),
    )
    divided_weight = fx.logical_divide(weight, fx.make_layout(1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)

    inp_register = fx.make_rmem_tensor(1, fx.Float32)
    weight_register = fx.make_rmem_tensor(1, fx.Float32)
    out_register = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(copy_atom, fx.slice(row_inp, (None, thread)), inp_register)
    fx.copy_atom_call(
        copy_atom,
        fx.slice(divided_weight, (None, thread)),
        weight_register,
    )
    value = fx.memref_load_vec(inp_register)[0]
    weight_value = fx.memref_load_vec(weight_register)[0]

    sum_of_squares = value * value
    for offset in (32, 16, 8, 4, 2, 1):
        sum_of_squares = sum_of_squares + sum_of_squares.shuffle_xor(
            offset,
            columns,
        )
    inverse_rms = fmath.rsqrt(
        sum_of_squares / fx.Float32(float(columns)) + fx.Float32(RMS_EPS)
    )
    normalized = value * inverse_rms * weight_value
    fx.memref_store_vec(full(1, normalized, fx.Float32), out_register)
    fx.copy_atom_call(
        copy_atom,
        out_register,
        fx.slice(row_out, (None, thread)),
    )


@flyc.jit
def rms_norm_launcher(
    inp: fx.Tensor,
    weight: fx.Tensor,
    out: fx.Tensor,
    rows: fx.Int32,
    columns: fx.Constexpr[int],
):
    _rms_norm_kernel(inp, weight, out, columns).launch(
        grid=(rows, 1, 1),
        block=(columns, 1, 1),
    )
