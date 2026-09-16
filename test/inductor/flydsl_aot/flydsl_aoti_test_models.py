# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx

import torch
from caffe2.test.inductor.flydsl_aot.flydsl_test_kernels import (
    _relu_kernel,
    ELEMENTWISE_BLOCK,
    gemm_launcher,
    GEMM_M,
    GEMM_N,
    relu_launcher,
    RMS_N,
    rms_norm_launcher,
)


RELU_SCALE = 0.75


@flyc.kernel
def _vector_add_kernel(
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    out: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    block = fx.block_idx.x
    thread = fx.thread_idx.x

    lhs = fx.rocdl.make_buffer_tensor(lhs)
    rhs = fx.rocdl.make_buffer_tensor(rhs)
    out = fx.rocdl.make_buffer_tensor(out)
    tiled_lhs = fx.slice(
        fx.logical_divide(lhs, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_rhs = fx.slice(
        fx.logical_divide(rhs, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_out = fx.slice(
        fx.logical_divide(out, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_lhs = fx.logical_divide(tiled_lhs, fx.make_layout(1, 1))
    tiled_rhs = fx.logical_divide(tiled_rhs, fx.make_layout(1, 1))
    tiled_out = fx.logical_divide(tiled_out, fx.make_layout(1, 1))

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)
    lhs_register = fx.make_rmem_tensor(1, fx.Float32)
    rhs_register = fx.make_rmem_tensor(1, fx.Float32)
    out_register = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(copy_atom, fx.slice(tiled_lhs, (None, thread)), lhs_register)
    fx.copy_atom_call(copy_atom, fx.slice(tiled_rhs, (None, thread)), rhs_register)
    value = fx.arith.addf(
        fx.memref_load_vec(lhs_register),
        fx.memref_load_vec(rhs_register),
    )
    fx.memref_store_vec(value, out_register)
    fx.copy_atom_call(copy_atom, out_register, fx.slice(tiled_out, (None, thread)))


@flyc.jit
def _vector_add_launcher(
    out: fx.Tensor,
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    elements: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    grid = (elements + block_dim - 1) // block_dim
    _vector_add_kernel(lhs, rhs, out, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )


@flyc.jit
def _two_stage_add_launcher(
    out: fx.Tensor,
    workspace: fx.Tensor,
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    elements: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    grid = (elements + block_dim - 1) // block_dim
    _vector_add_kernel(lhs, rhs, workspace, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )
    _vector_add_kernel(workspace, rhs, out, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )


class _BoundReluLauncher:
    @flyc.jit
    def launch(
        self,
        inp: fx.Tensor,
        out: fx.Tensor,
        elements: fx.Int32,
        block_dim: fx.Constexpr[int],
        *,
        scale: fx.Float32,
    ):
        blocks = (elements + block_dim - 1) // block_dim
        _relu_kernel(inp, out, scale, block_dim).launch(
            grid=(blocks, 1, 1),
            block=(block_dim, 1, 1),
        )


_bound_relu_launcher = _BoundReluLauncher()
_captured_vector_add = torch.library.wrap_flydsl(
    _vector_add_launcher,
    mutates_args={"out"},
)
_captured_two_stage_add = torch.library.wrap_flydsl(
    _two_stage_add_launcher,
    mutates_args={"out", "workspace"},
)
_captured_gemm = torch.library.wrap_flydsl(
    gemm_launcher,
    mutates_args={"out"},
)
_captured_relu = torch.library.wrap_flydsl(
    relu_launcher,
    mutates_args={"out"},
)
_captured_rms_norm = torch.library.wrap_flydsl(
    rms_norm_launcher,
    mutates_args={"out"},
)
_captured_bound_relu = torch.library.wrap_flydsl(
    _bound_relu_launcher.launch,
    mutates_args={"out"},
)


def _two_stage_add(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    workspace = torch.empty_like(lhs)
    out = torch.empty_like(lhs)
    _captured_two_stage_add(
        out,
        workspace,
        lhs,
        rhs,
        lhs.numel(),
        ELEMENTWISE_BLOCK,
    )
    return out


class VectorAddModel(torch.nn.Module):
    def forward(self, lhs, rhs):
        out = torch.empty_like(lhs)
        _captured_vector_add(
            out,
            lhs,
            rhs,
            lhs.numel(),
            256,
        )
        return out


class TwoStageAddModel(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.sin(_two_stage_add(lhs, rhs))


class BoundReluModel(torch.nn.Module):
    def forward(self, inp):
        out = torch.empty_like(inp)
        _captured_bound_relu(
            inp,
            out,
            inp.numel(),
            ELEMENTWISE_BLOCK,
            scale=RELU_SCALE,
        )
        return out


def _composed(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm_out = torch.empty(
        (GEMM_M, GEMM_N),
        device=lhs.device,
        dtype=lhs.dtype,
    )
    _captured_gemm(lhs, rhs, gemm_out)

    biased = gemm_out + bias
    flat_biased = biased.flatten()
    flat_activated = torch.empty_like(flat_biased)
    _captured_relu(
        flat_biased,
        flat_activated,
        flat_biased.numel(),
        RELU_SCALE,
        ELEMENTWISE_BLOCK,
    )
    activated = flat_activated.view(GEMM_M, GEMM_N)

    mixed = activated + torch.sin(activated) * 0.25
    normalized = torch.empty_like(mixed)
    _captured_rms_norm(
        mixed,
        weight,
        normalized,
        GEMM_M,
        RMS_N,
    )
    final = normalized * 1.5 - 0.5
    return final, gemm_out, activated, normalized


class ComposedModel(torch.nn.Module):
    def forward(self, lhs, rhs, bias, weight):
        return _composed(lhs, rhs, bias, weight)


class DynamicRMSNormModel(torch.nn.Module):
    def forward(self, inp, weight):
        out = torch.empty_like(inp)
        _captured_rms_norm(
            inp,
            weight,
            out,
            inp.shape[0],
            RMS_N,
        )
        return out
