"""RMSNorm kernel entry points shared by JIT compilation and AOT export."""

from typing import Any, Literal, TYPE_CHECKING

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

import torch
from torch._native.cutedsl import launch
from torch._native.cutedsl.dtypes import torch2cute
from torch._native.cutedsl.reduce import reduce_rows
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._native.utils.tensor import row_alignment
from torch._vendor.quack.rmsnorm import RMSNorm, RMSNormBackward
from torch._vendor.quack.rmsnorm_config import RmsNormBwdConfig, RmsNormFwdConfig


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


class RmsNormForward:
    def __init__(self, dtype: type[cutlass.Numeric], n: int) -> None:
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
    ) -> None:
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
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        n: int,
        compute_dw: bool,
        dout_dtype: type[cutlass.Numeric],
    ) -> None:
        self.n = n
        self.dtype = dtype
        self.compute_dw = compute_dw
        self.dout_dtype = dout_dtype

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
    ) -> None:
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
            dout_dtype=self.dout_dtype,
            num_acc=int(self.compute_dw),
            config=RmsNormBwdConfig.from_analytical_heuristic(
                self.n,
                self.dtype.width,
                self.dout_dtype.width,
                arch_major=arch.major,
                num_acc=int(self.compute_dw),
            ),
        )
        norm(x, mW, dout, None, rstd, None, dx, partial, None, None, blocks, stream)
        if cutlass.const_expr(self.compute_dw):
            reduce_rows(partial, mdW, 32, 128).launch(
                grid=[cute.ceil_div(self.n, 32), 1, 1],
                block=[128, 1, 1],
                stream=stream,
            )


def kernel_spec(
    direction: Literal["forward", "backward"],
    dtype: str,
    n: int,
    has_weight: bool,
    compute_dw: bool = False,
    *,
    jit: bool = False,
    dout_dtype: str | None = None,
) -> dict[str, Any]:
    element_type = torch2cute[getattr(torch, dtype)]

    def tensor(
        element_type: type[cutlass.Numeric],
        shape: tuple[int | cute.SymInt, ...],
        stride_order: tuple[int, ...] | None = None,
    ) -> cute.Tensor:
        return launch.fake_compact(
            element_type,
            shape,
            stride_order=stride_order,
            align=row_alignment(n, element_type.width // 8),
        )

    # AOT needs only pointers. JIT descriptors also express the real tensor
    # ranks for TVM-FFI validation; both entry points construct layouts from rows.
    shape = (launch.sym(), n) if jit else (1,)
    stride_order = (1, 0) if jit else None
    x = tensor(element_type, shape, stride_order)
    weight = tensor(element_type, (n,)) if has_weight else None
    out = tensor(element_type, shape, stride_order)
    rstd = launch.fake_compact(
        Float32,
        (launch.sym(), 1) if jit else (1,),
        stride_order=(1, 0) if jit else None,
        align=4,
    )
    tensor_args = [{"name": "mX", "read_only": True}]
    if has_weight:
        tensor_args.append({"name": "mW", "read_only": True})
    prefix = f"rmsnorm_{direction}_{dtype}_n{n}_w{int(has_weight)}"
    scalar_args = [{"name": "rows", "ctype": "int32_t"}]
    if direction == "forward":
        fn = RmsNormForward(element_type, n)
        fake_args = [x, weight, out, rstd, Int32(0), Float32(0)]
        tensor_args += [{"name": "mO"}, {"name": "mRstd"}]
        scalar_args.append({"name": "eps", "ctype": "float"})
    else:
        dout_element_type = (
            element_type
            if dout_dtype is None
            else torch2cute[getattr(torch, dout_dtype)]
        )
        dout = tensor(dout_element_type, shape, stride_order)
        partial = (
            tensor(Float32, (launch.sym(), n) if jit else (1,), stride_order)
            if compute_dw
            else None
        )
        dw = tensor(element_type, (n,)) if compute_dw else None
        fn = RmsNormBackward(element_type, n, compute_dw, dout_element_type)
        fake_args = [x, weight, dout, rstd, out, partial, dw, Int32(0), Int32(0)]
        tensor_args += [
            {"name": "mdO", "read_only": True},
            {"name": "mRstd", "read_only": True},
            {"name": "mdX"},
        ]
        if compute_dw:
            tensor_args += [{"name": "mdWPartial"}, {"name": "mdW"}]
        scalar_args.append({"name": "blocks", "ctype": "int32_t"})
        prefix += f"_dw{int(compute_dw)}"
        if dout_element_type != element_type:
            prefix += f"_dout{dout_element_type.__name__}"
    return {
        "kind": "cutedsl",
        "prefix": prefix,
        "fn": fn,
        "fake_args": [*fake_args, cute.runtime.make_fake_stream()],
        "tensor_args": tensor_args,
        "scalar_args": scalar_args,
    }


@instrumented_cutedsl_cache(
    lambda direction, *args, **kwargs: (
        "aten::_fused_rms_norm"
        if direction == "forward"
        else "aten::_fused_rms_norm_backward"
    )
)
def compile_rmsnorm(
    direction: Literal["forward", "backward"],
    dtype: torch.dtype,
    n: int,
    has_weight: bool,
    arch: tuple[int, int],
    compute_dw: bool = False,
    dout_dtype: torch.dtype | None = None,
) -> "Function":
    spec = kernel_spec(
        direction,
        str(dtype).removeprefix("torch."),
        n,
        has_weight,
        compute_dw,
        jit=True,
        dout_dtype=(
            str(dout_dtype).removeprefix("torch.") if dout_dtype is not None else None
        ),
    )
    return launch.compile_kernel(
        spec["fn"],
        *spec["fake_args"],
        options=f"--gpu-arch=sm_{arch[0]}{arch[1]}",
    )
