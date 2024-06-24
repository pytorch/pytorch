# mypy: allow-untyped-defs
import os
from typing import Any, Dict, List, Optional, Tuple

import sympy

import torch
from torch._inductor.runtime.triton_heuristics import grid as default_grid

from .. import config
from ..codecache import CudaKernelParamCache, get_cpp_wrapper_cubin_path_name
from ..runtime.hints import instance_descriptor
from ..utils import _type_of, DeferredLineBase
from ..virtualized import V
from .common import KernelArgType, SizeArg, TensorArg, WorkspaceArg
from .cpp_utils import cexpr


def signature_of(arg: KernelArgType, *, size_dtype: str) -> str:
    if isinstance(arg, TensorArg):
        # TODO: Remove fp8 special handling when Triton supports PyTorch fp8 dtypes.
        # Related PR: https://github.com/openai/triton/pull/2279/
        if arg.dtype == torch.float8_e4m3fn:
            tye = "*fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            tye = "*fp8e5"
        elif arg.dtype == torch.float8_e4m3fnuz:
            tye = "*fp8e4b8"
        elif arg.dtype == torch.float8_e5m2fnuz:
            tye = "*fp8e5b16"
        else:
            tye = _type_of(arg.dtype)
        if V.graph.is_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_tye = tye.lstrip("*")
            if new_tye in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_tye
        else:
            return tye
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            # From triton/runtime/jit.py
            # `None` is nullptr.  Implicitly convert to *i8.
            return "*i8"
        elif isinstance(arg.expr, (float, sympy.Float)):
            return "fp32"
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return "*i8"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def signature_to_meta(
    signature: List[KernelArgType],
    *,
    size_dtype: str,
    indices: Optional[List[int]] = None,
) -> Dict[int, str]:
    if indices is None:
        indices = list(range(len(signature)))
    return {
        i: signature_of(arg, size_dtype=size_dtype)
        for i, arg in zip(indices, signature)
    }


def is_unaligned_buffer(arg: TensorArg):
    buf_name = arg.buffer
    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        return buf_name not in V.graph.aligned_inputs

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.get_buffer(buf_name)
        # output arg
        if not buffer:
            assert buf_name == V.kernel.output_node.name
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()

    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def config_of(
    args: List[KernelArgType],
    *,
    indices: Optional[List[int]] = None,
) -> Any:
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/openai/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize, alignment  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, float):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        if isinstance(x, WorkspaceArg):
            return V.graph.sizevars.statically_known_multiple_of(x.nbytes, alignment)  # type: ignore[arg-type]
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(arg, alignment=16, include_tensor=True)
        )
    else:
        divisible_by_16 = ()
    divisible_by_8 = tuple(
        i
        for i, arg in zip(indices, args)
        if is_aligned(arg, alignment=8, include_tensor=False)
    )

    equal_to_1 = tuple(
        i
        for i, arg in zip(indices, args)
        if isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )
    # ids_of_folded_args is set from equal_to_1
    # and None args by the Triton compiler
    ids_of_folded_args = tuple(equal_to_1)

    return instance_descriptor(
        divisible_by_16, equal_to_1, ids_of_folded_args, divisible_by_8
    )


class DeferredCudaKernelLine(DeferredLineBase):
    """
    When using cpp wrapper, CUDA kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred line to backfill those information
    """

    def __init__(
        self,
        kernel_name: str,
        line_template: str,
        keys: Tuple[str, ...],
    ):
        super().__init__(line_template)
        assert not isinstance(line_template, DeferredLineBase)
        self.kernel_name = kernel_name
        self.line_template = line_template
        self.keys = keys

    def __call__(self):
        params = CudaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in CudaKernelParamCache"
        for key in self.keys:
            assert (
                key in params
            ), f"{key} not found in CudaKernelParamCache[{self.kernel_name}]"
            if key == get_cpp_wrapper_cubin_path_name():
                assert os.path.exists(params[key]), f"{params[key]} does not exist"

        return self.line_template % tuple(params[key] for key in self.keys)

    def _new_line(self, line):
        return DeferredCudaKernelLine(self.kernel_name, line, self.keys)


class DeferredCudaDefaultGrid:
    """
    A marker to
    """

    def __init__(
        self,
        kernel_name: str,
        grid,
    ):
        self.kernel_name = kernel_name
        self.grid = grid

    def __iter__(self):
        # DeferredCudaDefaultGrid can be passed to the base class, WrapperCodeGen,
        # to genrete the autotune code block, and thus we need this iterator
        return iter(self.grid)

    def __call__(self):
        from .wrapper import SymbolicCallArg

        params = CudaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in CudaKernelParamCache"

        grid = [
            e.inner_expr if isinstance(e, SymbolicCallArg) else e for e in self.grid
        ]
        grid_fn = default_grid(*grid)
        block_cfg = {
            "XBLOCK": params["x_block"],
            "YBLOCK": params["y_block"],
            "ZBLOCK": params["z_block"],
        }
        return grid_fn(block_cfg)


class DeferredCudaGridLine(DeferredLineBase):
    """
    When using cpp wrapper, CUDA kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred line to backfill those information
    """

    def __init__(
        self,
        kernel_name: str,
        grid_var: str,
        grid,
        autotune_configs,
    ):
        super().__init__("")
        self.kernel_name = kernel_name
        self.grid_var = grid_var
        self.grid = grid
        self.autotune_configs = autotune_configs

    def __call__(self):
        params = CudaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in CudaKernelParamCache"

        if self.autotune_configs is not None:
            # This indicates the Triton kernel is a user-defined one.
            grid = None
            if len(self.grid) == 1:
                grid = self.grid[0]
            else:
                for i, c in enumerate(self.autotune_configs):
                    if all(arg == params["meta"][key] for key, arg in c.kwargs.items()):
                        grid = self.grid[i]
                        break
            assert grid is not None
        elif isinstance(self.grid, DeferredCudaDefaultGrid):
            grid = self.grid()
        else:
            grid = self.grid

        assert len(grid) != 0, "Grid can't be empty"
        grid_args_str = ", ".join(
            [cexpr(V.graph.sizevars.simplify(item)) for item in grid]
        )
        return f"Grid {self.grid_var} = Grid({grid_args_str});"

    def _new_line(self, line):
        return DeferredCudaGridLine(
            self.kernel_name, self.grid_var, self.grid, self.autotune_configs
        )
