# mypy: allow-untyped-defs
import functools
from itertools import chain, count
from typing import Any, List, Optional, TYPE_CHECKING

import sympy

from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name

from .. import config
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .codegen_device_driver import cuda_kernel_driver, cuda_kernel_header
from .cpp_utils import DTYPE_TO_CPP
from .cpp_wrapper_cpu import CppWrapperCpu
from .triton_utils import (
    DeferredCudaDefaultGrid,
    DeferredCudaGridLine,
    DeferredCudaKernelLine,
)
from .wrapper import user_defined_kernel_grid_fn_code

if TYPE_CHECKING:
    from ..graph import GraphLowering


class CppWrapperCuda(CppWrapperCpu):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self):
        self.device = "cuda"
        super().__init__()
        self.grid_id = count()
        self.cuda = True

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        super().write_header()

        self.header.splice("#include <filesystem>")
        if config.abi_compatible:
            self.header.splice(
                "#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>"
            )
        else:
            self.header.splice(maybe_hipify_code_wrapper(cuda_kernel_header()))
        self.header.splice(maybe_hipify_code_wrapper(cuda_kernel_driver()))

    def write_get_raw_stream(self, index, graph=None):
        name = f"stream{index}"
        self.writeline(maybe_hipify_code_wrapper(f"cudaStream_t {name};"))
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream({index}, (void**)&{name}));"
        )
        return name

    def generate(self, is_inference):
        self.prefix.writeline("\n")
        if not V.graph.aot_mode:
            for kernel in chain(
                sorted(self.src_to_kernel.values()),
                sorted([entry[0] for entry in self.user_defined_kernel_cache.values()]),
            ):
                self.prefix.writeline(
                    maybe_hipify_code_wrapper(f"static CUfunction {kernel} = nullptr;")
                )
            self.prefix.writeline("\n")
        return super().generate(is_inference)

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        cuda=True,
    ):
        """
        Override the default value of argument 'cuda' to True here. define_kernel can still be
        called with cuda=False because of a mix of cpu kernels and triton kernels.
        """
        return super().define_kernel(kernel_name, kernel_body, metadata, cuda)

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
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]

        # Similar to WrapperCodeGen.generate_user_defined_triton_kernel but only insert
        # into the autotune code block here
        grid_fn, code = user_defined_kernel_grid_fn_code(
            kernel_name, configs, grid, wrapper=self
        )
        super().generate_kernel_call(
            kernel_name,
            args,
            grid_fn=grid_fn,
            arg_types=arg_types,
            raw_args=raw_args,
            cuda=True,
            triton=True,
        )

        # super().generate_kernel_call only generates the autotune code block.
        # Calling self.generate_kernel_call generates the real kernel call in cpp.
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args,
            grid=grid,
            cuda=True,
            triton=True,
            triton_meta=triton_meta,
            autotune_configs=configs,
        )

    @functools.lru_cache(None)  # noqa: B019
    def generate_load_kernel_once(
        self,
        kernel_name: str,
        graph: "GraphLowering",  # for per-graph caching
    ):
        keys = (get_cpp_wrapper_cubin_path_name(), "mangled_name", "shared_mem")
        kernel_var_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
        self.writeline(f"if ({kernel_var_name} == nullptr) {{")
        self.writeline(
            DeferredCudaKernelLine(
                kernel_name,
                kernel_var_name + """ = loadKernel("%s", "%s", %s, this->cubin_dir_);"""
                if V.graph.aot_mode
                else kernel_var_name + """ = loadKernel("%s", "%s", %s);""",
                keys,
            )
        )
        self.writeline("}")
        return kernel_var_name

    def generate_args_decl(self, call_args, arg_types):
        new_args = []
        for arg, arg_type in zip(call_args, arg_types):
            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(arg_type, torch_dtype):
                if arg.endswith(".item()"):
                    # Need to declare a scalar in this case
                    ctype = DTYPE_TO_CPP[arg_type]
                    arg = arg[:-7]
                    if config.abi_compatible:
                        self.codegen_tensor_item(
                            arg_type,
                            arg,
                            var_name,
                        )
                    else:
                        self.writeline(f"{ctype} {var_name} = {arg}.item<{ctype}>();")
                else:
                    if config.abi_compatible:
                        self.writeline(
                            maybe_hipify_code_wrapper(f"CUdeviceptr {var_name};")
                        )
                        self.writeline(
                            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, reinterpret_cast<void**>(&{var_name})));"
                        )
                    else:
                        self.writeline(
                            maybe_hipify_code_wrapper(
                                f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());"
                            )
                        )
            elif arg_type in (sympy.Integer, int):
                self.writeline(f"int {var_name} = {self.expr_printer(arg)};")
            elif arg_type in (sympy.Float, float):
                self.writeline(f"float {var_name} = {self.expr_printer(arg)};")
            else:
                self.writeline(f"auto {var_name} = {self.expr_printer(arg)};")
            new_args.append(f"&{var_name}")

        return ", ".join(new_args)

    def generate_default_grid(self, kernel_name: str, grid_args: List[Any]):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics. Because its computation needs
        to read kernel config after autotune, it is done in a deferred way
        using DeferredCudaDefaultGrid.
        """
        super().generate_default_grid(kernel_name, grid_args)
        return DeferredCudaDefaultGrid(kernel_name, grid_args)

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
    ):
        """
        Override the default value of argument 'cuda' to True here. generate_kernel_call can still be
        called with cuda=False because of a mix of cpu kernels and triton kernels.
        """
        # Call parent class to create the autotune code block
        super().generate_kernel_call(
            kernel_name,
            call_args,
            grid,
            device_index,
            cuda,
            triton,
            arg_types,
            raw_args,
            grid_fn,
            triton_meta,
            autotune_configs,
        )
        if not cuda:
            # Next steps are irrelevant for CPU kernels
            return

        device_index, call_args = self.prepare_triton_kernel_call(
            device_index, call_args
        )
        kernel_var_name = self.generate_load_kernel_once(kernel_name, V.graph)

        # args with value 1 are added into equal_to_1 and constants
        # in triton_meta (in the Python codegen) which makes them
        # inlined in the PTX and compiled CUBIN
        if (
            triton_meta is not None
            and "configs" in triton_meta
            and triton_meta["configs"]
        ):
            equal_to_1 = triton_meta["configs"][0].equal_to_1
            call_args = [arg for i, arg in enumerate(call_args) if i not in equal_to_1]
            arg_types = [t for i, t in enumerate(arg_types) if i not in equal_to_1]

        call_args = self.generate_args_decl(call_args, arg_types)
        kernel_args_var = f"kernel_args_var_{next(self.kernel_callsite_id)}"
        self.writeline(f"void* {kernel_args_var}[] = {{{call_args}}};")
        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device_index, V.graph)
        )

        grid_var = f"{kernel_name}_grid_{next(self.grid_id)}"
        self.writeline(
            DeferredCudaGridLine(kernel_name, grid_var, grid, autotune_configs)
        )

        self.writeline(f"if ({grid_var}.is_non_zero()) {{")
        self.writeline(
            DeferredCudaKernelLine(
                kernel_name,
                "launchKernel({}, {}, {}, {},".format(
                    kernel_var_name,
                    f"{grid_var}.grid_x",
                    f"{grid_var}.grid_y",
                    f"{grid_var}.grid_z",
                )
                + "%s, %s,"
                + f"{kernel_args_var}, {stream});",
                ("num_warps", "shared_mem"),
            ),
        )
        self.writeline("}")
