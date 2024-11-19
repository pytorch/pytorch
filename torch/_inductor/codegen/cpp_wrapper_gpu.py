# mypy: allow-untyped-defs
import functools
import os
from itertools import chain, count, zip_longest
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import sympy

from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.triton_heuristics import grid as default_grid_fn

from ..codecache import CudaKernelParamCache
from ..ir import IRNode
from ..utils import DeferredLineBase, get_gpu_type
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides
from .cpp_utils import cexpr
from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import PythonWrapperCodegen, SymbolicCallArg


if TYPE_CHECKING:
    from ..graph import GraphLowering


class DeferredGpuKernelLine(DeferredLineBase):
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
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

    def get_kernel_path(self):
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
                return params[key]

        raise RuntimeError("Unable to find a path for kernel %s", self.kernel_name)

    def _new_line(self, line):
        return DeferredGpuKernelLine(self.kernel_name, line, self.keys)


class DeferredGpuDefaultGrid:
    """
    A container for the default grid, which may be used by DeferredCudaGridLine
    """

    def __init__(
        self,
        kernel_name: str,
        grid,
        grid_callable: Optional[Callable[..., Any]] = None,
        **grid_extra_kwargs,
    ):
        self.kernel_name = kernel_name
        self.grid = grid
        self.grid_callable = grid_callable
        self.grid_extra_kwargs = grid_extra_kwargs

    def _process_grid(self, grid: Union[List[Any], Tuple[Any, ...]]):
        if isinstance(grid, (list, tuple)):
            return [self._process_grid(e) for e in grid]
        else:
            return grid.inner_expr if isinstance(grid, SymbolicCallArg) else grid

    def __call__(self):
        grid = self.grid
        assert isinstance(grid, (list, tuple)), f"expected {grid=} to be a list"
        grid = self._process_grid(grid)
        assert self.grid_callable is not None, "grid_callable can't be None"
        if not self.grid_extra_kwargs:
            grid_fn = self.grid_callable(*grid)
        else:
            grid_fn = self.grid_callable(*grid, **self.grid_extra_kwargs)

        params = CudaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in CudaKernelParamCache"
        return grid_fn(params["meta"])


class DeferredGpuGridLine(DeferredLineBase):
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
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
        elif isinstance(self.grid, DeferredGpuDefaultGrid):
            grid = self.grid()
        else:
            grid = self.grid

        assert len(grid) != 0, "Grid can't be empty"
        grid_args_str = ", ".join(
            [cexpr(V.graph.sizevars.simplify(item)) for item in grid]
        )
        return f"    Grid {self.grid_var} = Grid({grid_args_str});"

    def _new_line(self, line):
        return DeferredGpuGridLine(
            self.kernel_name, self.grid_var, self.grid, self.autotune_configs
        )


class CppWrapperGpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = get_gpu_type()
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()

    @staticmethod
    def create(
        is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperGpu()

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        super().write_header()

        self.header.splice("#include <filesystem>")
        self.header.splice(self.device_codegen.abi_compatible_header())
        self.header.splice(
            maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        )

    @functools.lru_cache(None)  # noqa: B019
    def write_tma_descriptor_helpers_once(self):
        self.header.splice(self.device_codegen.tma_descriptor_helpers())

    def write_get_raw_stream(self, index, graph=None):
        name = f"stream{index}"
        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({self.device_codegen.aoti_get_stream()}({index}, (void**)&{name}));"
        )
        return name

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu=True,
    ):
        if not gpu:
            return CppWrapperCpu.define_kernel(
                self, kernel_name, kernel_body, metadata, gpu
            )

    def generate(self, is_inference):
        self.prefix.writeline("\n")
        if not V.graph.aot_mode:
            for kernel in chain(
                sorted(self.src_to_kernel.values()),
                sorted([entry[0] for entry in self.user_defined_kernel_cache.values()]),
            ):
                self.prefix.writeline(
                    maybe_hipify_code_wrapper(
                        f"static {self.device_codegen.cpp_kernel_type()} {kernel} = nullptr;"
                    )
                )
            self.prefix.writeline("\n")
        return super().generate(is_inference)

    def generate_user_defined_triton_kernel(
        self,
        kernel_name: str,
        raw_args: List[Any],
        grid: List[Any],
        configs,
        triton_meta,
        constexprs,
    ):
        # in C++ wrapper, we don't pass constexpr args, as they don't
        # get added as parameters to the PTX code compiled from the
        # user-defined Triton kernel (only non-constexpr args do)
        raw_args = [
            raw_arg for i, raw_arg in enumerate(raw_args) if i not in constexprs
        ]
        args = [self.val_to_arg_str(v) for v in raw_args]
        arg_types = [
            arg.get_dtype() if isinstance(arg, IRNode) else type(arg)
            for arg in raw_args
        ]
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args,
            grid=grid,
            gpu=True,
            triton=True,
            triton_meta=triton_meta,
            autotune_configs=configs,
        )

    def generate_tma_descriptor(self, desc):
        self.write_tma_descriptor_helpers_once()

        # generate data pointer for the source tensor
        source = self.generate_args_decl(
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
        )

        desc_name = desc.name
        self.writeline(f"alignas(64) CUtensorMap {desc_name};")

        # `source` is in the form of `&var_x`, where `var_x` is the data pointer
        # (CUdeviceptr); we dereference `source` and cast to `void*` to pass to
        # the data pointer of the source tensor ot the helper function
        # `init{1,2}DTMADescriptor`
        ptr = f"reinterpret_cast<void*>(*({source}))"
        dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.dims)
        block_dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.block_dims)
        element_size = self.val_to_arg_str(desc.element_size)
        fn = f"init{desc.rank}DTMADescriptor"
        args = f"&{desc_name}, {ptr}, {dims}, {block_dims}, {element_size}"
        self.writeline(f"{fn}({args});")

    @functools.lru_cache(None)  # noqa: B019
    def generate_load_kernel_once(
        self,
        kernel_name: str,
        graph: "GraphLowering",  # for per-graph caching
    ):
        keys = (get_cpp_wrapper_cubin_path_name(), "mangled_name", "shared_mem")
        kernel_var_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
        self.writeline(f"if ({kernel_var_name} == nullptr) {{")
        deferred_gpu_kernel_line = DeferredGpuKernelLine(
            kernel_name,
            (
                "    "
                + kernel_var_name
                + ' = loadKernel("%s", "%s", %s, this->cubin_dir_);'
                if V.graph.aot_mode
                else "    " + kernel_var_name + ' = loadKernel("%s", "%s", %s);'
            ),
            keys,
        )
        self.writeline(deferred_gpu_kernel_line)
        self.writeline("}")

        self.additional_files.append(deferred_gpu_kernel_line.get_kernel_path())
        return kernel_var_name

    def generate_args_decl(self, call_args, arg_types, arg_signatures):
        new_args: list[str] = []

        # Add more cases for other types as needed
        signature2dtype = {
            "i32": "int32_t",
            "i64": "int64_t",
            "fp32": "float",
        }

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore nvTmaDesc, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
                if arg.endswith(".item()"):
                    # Need to declare a scalar in this case
                    arg = arg[:-7]
                    self.codegen_tensor_item(
                        arg_type,
                        arg,
                        var_name,
                    )
                else:
                    device_ptr_type = self.device_codegen.cpp_device_ptr()
                    self.writeline(
                        maybe_hipify_code_wrapper(
                            f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                        )
                    )
            elif arg_type in (sympy.Integer, int):
                self.writeline(f"int {var_name} = {self.expr_printer(arg)};")
            elif arg_type in (sympy.Float, float):
                self.writeline(f"float {var_name} = {self.expr_printer(arg)};")
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in signature2dtype.keys()
            ):
                self.writeline(
                    f"{signature2dtype[arg_signature]} {var_name} = {self.expr_printer(arg)};"
                )
            else:
                self.writeline(f"auto {var_name} = {self.expr_printer(arg)};")
            new_args.append(f"&{var_name}")

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        return ", ".join(new_args)

    def generate_default_grid(
        self,
        kernel_name: str,
        grid: List[Any],
        gpu: bool = True,
        grid_callable: Optional[Callable[..., Any]] = default_grid_fn,
        **grid_extra_kwargs,
    ):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics. Because its computation needs
        to read kernel config after autotune, it is done in a deferred way
        using DeferredGpuDefaultGrid.
        """
        if not gpu:
            return grid
        return DeferredGpuDefaultGrid(
            kernel_name, grid, grid_callable, **grid_extra_kwargs
        )

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        gpu=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        assert arg_types is not None and len(call_args) == len(
            arg_types
        ), "call_args and arg_types do not match"

        if not gpu:
            # Even in CppWrapperGpu, we may see cpp kernels
            return CppWrapperCpu.generate_kernel_call(
                self,
                kernel_name,
                call_args,
                grid,
                device_index,
                gpu,
                triton,
                arg_types,
                raw_args,
                grid_fn,
                triton_meta,
                autotune_configs,
                grid_extra_kwargs,
            )

        if device_index is None:
            current_device = V.graph.get_current_device_or_throw()
            device_index = current_device.index
        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device_index, V.graph)
        )

        if triton:
            device_index, call_args = self.prepare_triton_kernel_call(
                device_index, call_args
            )
            kernel_var_name = self.generate_load_kernel_once(kernel_name, V.graph)

            # args with value 1 are added into equal_to_1 and constants
            # in triton_meta (in the Python codegen) which makes them
            # inlined in the PTX and compiled CUBIN
            arg_signatures = []
            if (
                triton_meta is not None
                and triton_meta.get("configs")
                and triton_meta.get("signature")
            ):
                equal_to_1 = triton_meta["configs"][0].equal_to_1
                call_args = [
                    arg for i, arg in enumerate(call_args) if i not in equal_to_1
                ]
                arg_types = [t for i, t in enumerate(arg_types) if i not in equal_to_1]
                # extract the arg signatures from triton_meta
                arg_signatures = triton_meta["signature"].values()
                arg_signatures = [
                    v for i, v in enumerate(arg_signatures) if i not in equal_to_1
                ]

            call_args_str = self.generate_args_decl(
                call_args, arg_types, arg_signatures
            )
            kernel_args_var = f"kernel_args_var_{next(self.kernel_callsite_id)}"
            self.writeline(f"void* {kernel_args_var}[] = {{{call_args_str}}};")

            grid_var = f"{kernel_name}_grid_{next(self.grid_id)}"
            self.writeline(
                DeferredGpuGridLine(kernel_name, grid_var, grid, autotune_configs)
            )

            kernel_var_name = (
                f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
            )
            # add debug printer code for all triton kernel related calls
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args, kernel_name, arg_types, None
            )
            with debug_printer_manager:
                self.writeline(f"if ({grid_var}.is_non_zero()) {{")
                self.writeline(
                    DeferredGpuKernelLine(
                        kernel_name,
                        r"    launchKernel({}, {}, {}, {}, %s, %s, {}, {});".format(
                            kernel_var_name,
                            f"{grid_var}.grid_x",
                            f"{grid_var}.grid_y",
                            f"{grid_var}.grid_z",
                            kernel_args_var,
                            stream,
                        ),
                        ("num_warps", "shared_mem"),
                    ),
                )
                self.writeline("}")
        else:
            casted = []
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                casted.append(f"({arg_type}){new_arg}")
            call_args_str = ", ".join(casted)
            self.writeline(f"kernels.{kernel_name}({call_args_str}, {stream});")

    def make_zero_buffer(self, name):
        return (
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get())){self.ending}"
        )
