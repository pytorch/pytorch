"""RMSNorm kernel entry points shared by JIT compilation and AOT export."""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.rmsnorm import RMSNorm, RMSNormBackward
from torch._vendor.quack.rmsnorm_config import RmsNormBwdConfig, RmsNormFwdConfig

from .rmsnorm_launch import NORMALIZED_SIZES


class RmsNormForward:
    def __init__(self, dtype, n):
        self.n = n
        self.dtype = dtype

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mO: cute.Tensor,
        mRstd: cute.Tensor,
        rows: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        stride = cute.assume(cutlass.Int64(self.n), divby=self.n)
        layout = cute.make_layout((rows, self.n), stride=(stride, 1))
        x = cute.make_tensor(mX.iterator, layout)
        out = cute.make_tensor(mO.iterator, layout)
        rstd = cute.make_tensor(mRstd.iterator, cute.make_layout((rows,), stride=(1,)))
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        norm = RMSNorm(
            self.dtype,
            self.n,
            config=RmsNormFwdConfig.from_analytical_heuristic(
                self.n, self.dtype.width, arch_major=arch.major
            ),
        )
        norm(x, mW, None, None, out, None, rstd, None, eps, stream)


class RmsNormBackward:
    def __init__(self, dtype, n, compute_dw):
        self.n = n
        self.dtype = dtype
        self.compute_dw = compute_dw

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdWPartial: cute.Tensor | None,
        mdW: cute.Tensor | None,
        rows: Int32,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        stride = cute.assume(cutlass.Int64(self.n), divby=self.n)
        layout = cute.make_layout((rows, self.n), stride=(stride, 1))
        x = cute.make_tensor(mX.iterator, layout)
        dout = cute.make_tensor(mdO.iterator, layout)
        dx = cute.make_tensor(mdX.iterator, layout)
        rstd = cute.make_tensor(mRstd.iterator, cute.make_layout((rows,), stride=(1,)))
        partial = None
        if cutlass.const_expr(self.compute_dw):
            partial = cute.make_tensor(
                mdWPartial.iterator,
                cute.make_layout((blocks, self.n), stride=(self.n, 1)),
            )
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        norm = RMSNormBackward(
            self.dtype,
            self.n,
            dout_dtype=self.dtype,
            num_acc=int(self.compute_dw),
            config=RmsNormBwdConfig.from_analytical_heuristic(
                self.n,
                self.dtype.width,
                self.dtype.width,
                arch_major=arch.major,
                num_acc=int(self.compute_dw),
            ),
        )
        norm(x, mW, dout, None, rstd, None, dx, partial, None, None, blocks, stream)
        if cutlass.const_expr(self.compute_dw):
            if blocks <= 32:
                self.weight_grad(partial, mdW, 32, 128).launch(
                    grid=[self.n // 32, 1, 1], block=[128, 1, 1], stream=stream
                )
            else:
                cols = 4 if self.n <= 1024 else 8 if self.n <= 4096 else 16
                self.weight_grad(partial, mdW, cols, 256).launch(
                    grid=[self.n // cols, 1, 1], block=[256, 1, 1], stream=stream
                )

    @cute.kernel
    def weight_grad(
        self,
        partial: cute.Tensor,
        out: cute.Tensor,
        cols: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane, warp = tidx % 32, tidx // 32
        col = bidx * cols + tidx % cols
        acc = Float32(0)
        for row in cutlass.range(tidx // cols, partial.shape[0], threads // cols):
            acc += partial[row, col]
        for i in cutlass.range_constexpr((32 // cols).bit_length() - 1):
            acc += cute.arch.shuffle_sync_bfly(acc, offset=(1 << i) * cols)
        smem = cutlass.utils.SmemAllocator()
        sums = smem.allocate_tensor(
            Float32, cute.make_layout((threads // 32, cols), stride=(cols, 1))
        )
        if lane < cols:
            sums[warp, lane] = acc
        cute.arch.barrier()
        if tidx < cols:
            for i in cutlass.range_constexpr(1, threads // 32):
                acc += sums[i, tidx]
            out[bidx * cols + tidx] = out.element_type(acc)


def kernel_spec(direction, dtype, n, has_weight, compute_dw=False, *, jit=False):
    if n not in NORMALIZED_SIZES:
        raise ValueError(f"unsupported shared RMSNorm width: {n}")
    dtype_name = dtype
    dtype = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
    }[dtype]
    # AOT needs only pointers. JIT descriptors also express the real tensor
    # ranks for TVM-FFI validation; both entry points construct layouts from rows.
    shape = (cute.sym_int(), n) if jit else (1,)
    x = cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=(1, 0) if jit else None, assumed_align=16
    )
    weight = (
        cute.runtime.make_fake_compact_tensor(dtype, (n,), assumed_align=16)
        if has_weight
        else None
    )
    out = cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=(1, 0) if jit else None, assumed_align=16
    )
    rstd = cute.runtime.make_fake_compact_tensor(
        Float32,
        (cute.sym_int(), 1) if jit else (1,),
        stride_order=(1, 0) if jit else None,
        assumed_align=4,
    )
    tensor_args = [{"name": "mX", "read_only": True}]
    if has_weight:
        tensor_args.append({"name": "mW", "read_only": True})
    prefix = f"rmsnorm_{direction}_{dtype_name}_n{n}_w{int(has_weight)}"
    scalar_args = [{"name": "rows", "ctype": "int32_t"}]
    if direction == "forward":
        fn = RmsNormForward(dtype, n)
        fake_args = [x, weight, out, rstd, Int32(0), Float32(0)]
        tensor_args += [{"name": "mO"}, {"name": "mRstd"}]
        scalar_args.append({"name": "eps", "ctype": "float"})
    else:
        partial = (
            cute.runtime.make_fake_compact_tensor(
                Float32,
                (cute.sym_int(), n) if jit else (1,),
                stride_order=(1, 0) if jit else None,
                assumed_align=16,
            )
            if compute_dw
            else None
        )
        dw = (
            cute.runtime.make_fake_compact_tensor(dtype, (n,), assumed_align=16)
            if compute_dw
            else None
        )
        fn = RmsNormBackward(dtype, n, compute_dw)
        fake_args = [x, weight, x, rstd, out, partial, dw, Int32(0), Int32(0)]
        tensor_args += [
            {"name": "mdO", "read_only": True},
            {"name": "mRstd", "read_only": True},
            {"name": "mdX"},
        ]
        if compute_dw:
            tensor_args += [{"name": "mdWPartial"}, {"name": "mdW"}]
        scalar_args.append({"name": "blocks", "ctype": "int32_t"})
        prefix += f"_dw{int(compute_dw)}"
    return {
        "kind": "cutedsl",
        "prefix": prefix,
        "fn": fn,
        "fake_args": [*fake_args, cute.runtime.make_fake_stream()],
        "tensor_args": tensor_args,
        "scalar_args": scalar_args,
    }


@instrumented_cutedsl_cache("aten::_fused_rms_norm")
def compile_rmsnorm_forward(dtype, n, has_weight, arch):
    spec = kernel_spec("forward", dtype, n, has_weight, jit=True)
    return cute.compile(
        spec["fn"],
        *spec["fake_args"],
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch[0]}{arch[1]}",
    )


@instrumented_cutedsl_cache("aten::_fused_rms_norm_backward")
def compile_rmsnorm_backward(dtype, n, has_weight, compute_dw, arch):
    spec = kernel_spec("backward", dtype, n, has_weight, compute_dw, jit=True)
    return cute.compile(
        spec["fn"],
        *spec["fake_args"],
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch[0]}{arch[1]}",
    )


def stream(device_index):
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(device_index))
