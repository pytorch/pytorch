# mypy: allow-untyped-defs
import functools
import os
from itertools import chain, count
from typing import Any, List, Optional, TYPE_CHECKING

import sympy

from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.triton_heuristics import grid as default_grid

from .. import config
from ..codecache import CudaKernelParamCache
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .codegen_device_driver import cuda_kernel_driver, cuda_kernel_header
from .cpp_utils import DTYPE_TO_CPP
from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import SymbolicCallArg


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

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        if not cuda:
            return super().define_kernel(name, kernel, metadata, cuda)

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

    def generate_user_defined_triton_kernel(
        self, kernel_name, raw_args, grid, configs, triton_meta, constexprs
    ):
        # in C++ wrapper, we don't pass constexpr args, as they don't
        # get added as parameters to the PTX code compiled from the
        # user-defined Triton kernel (only non-constexpr args do)
        raw_args = [
            raw_arg for i, raw_arg in enumerate(raw_args) if i not in constexprs
        ]

        assert len(grid) != 0
        if len(grid) == 1:
            grid_decision = grid[0]
        else:
            meta = CudaKernelParamCache.get(kernel_name)
            assert meta is not None
            grid_decision = None
            for i, c in enumerate(configs):
                if all(arg == meta["meta"][key] for key, arg in c.kwargs.items()):
                    grid_decision = grid[i]
                    break
            assert grid_decision is not None

        args = [self.val_to_arg_str(v) for v in raw_args]
        arg_types = [
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            grid=grid_decision,
            cuda=True,
            triton=True,
            triton_meta=triton_meta,
        )

    @functools.lru_cache(None)  # noqa: B019
    def generate_load_kernel_once(
        self,
        name: str,
        mangled_name: str,
        cubin_path: str,
        shared_mem: int,
        graph: "GraphLowering",  # for per-graph caching
    ):
        if V.graph.aot_mode:
            self.writeline(f"if (kernels.{name} == nullptr) {{")
            self.writeline(
                f"""    kernels.{name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem}, this->cubin_dir_);"""
            )
            self.writeline("}")
        else:
            self.writeline(f"if ({name} == nullptr) {{")
            self.writeline(
                f"""    {name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem});"""
            )
            self.writeline("}")

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

    def generate_default_grid(self, name: str, grid: List[Any], cuda: bool = True):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics.
        """
        if not cuda:
            return grid
        assert isinstance(grid, list), f"expected {grid=} to be a list"
        grid = [e.inner_expr if isinstance(e, SymbolicCallArg) else e for e in grid]
        grid_fn = default_grid(*grid)
        params = CudaKernelParamCache.get(name)
        assert (
            params is not None
        ), f"cuda kernel parameters for {name} should already exist at this moment, only found {CudaKernelParamCache.get_keys()}"
        block_cfg = {
            "XBLOCK": params["x_block"],
            "YBLOCK": params["y_block"],
            "ZBLOCK": params["z_block"],
        }
        return grid_fn(block_cfg)

    def generate_kernel_call(
        self,
        name,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
    ):
        assert arg_types is not None and len(call_args) == len(
            arg_types
        ), "call_args and arg_types do not match"

        if not cuda:
            # Even in CppWrapperCuda, we may see cpp kernels
            return super().generate_kernel_call(
                name, call_args, grid, device_index, cuda, triton, arg_types
            )

        device_index, call_args = self.prepare_triton_kernel_call(
            device_index, call_args
        )
        params = CudaKernelParamCache.get(name)
        assert (
            params is not None
        ), f"cuda kernel parameters for {name} should already exist at this moment"
        mangled_name = params.get("mangled_name", None)
        assert mangled_name is not None, "missing mangled_name"
        cubin_path = params.get(get_cpp_wrapper_cubin_path_name(), None)
        assert cubin_path is not None and os.path.exists(
            cubin_path
        ), f"cubin file should already exist at this moment: {cubin_path}"
        shared_mem = params.get("shared_mem", 0)

        self.generate_load_kernel_once(
            name, mangled_name, cubin_path, shared_mem, V.graph
        )

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
        grid_name = f"{name}_grid_{next(self.grid_id)}"
        assert isinstance(
            grid, (list, tuple)
        ), f"expected grid to be a list or tuple but got: {grid=}"

        grid = [V.graph.sizevars.simplify(item) for item in grid]
        grid_uses_symbolic_shapes = any(item.free_symbols for item in grid)
        grid_args = [self.expr_printer(item) for item in grid]
        grid_args_str = ", ".join(grid_args)
        self.writeline(f"Grid {grid_name} = Grid({grid_args_str});")

        if grid_uses_symbolic_shapes:
            self.writeline(f"if ({grid_name}.is_non_zero()) {{")
        kernel_var_name = f"kernels.{name}" if V.graph.aot_mode else name
        launch_kernel_call = """launchKernel({}, {}, {}, {}, {}, {}, {}, {});""".format(
            kernel_var_name,
            f"{grid_name}.grid_x",
            f"{grid_name}.grid_y",
            f"{grid_name}.grid_z",
            params["num_warps"],
            params["shared_mem"],
            kernel_args_var,
            stream,
        )
        if grid_uses_symbolic_shapes:
            # TODO: Use codegen `do_indent()` to properly generate the indentation.
            # This works in this case as there's only one `if` condition.
            self.writeline("    " + launch_kernel_call)
            self.writeline("}")
        else:
            self.writeline(launch_kernel_call)
