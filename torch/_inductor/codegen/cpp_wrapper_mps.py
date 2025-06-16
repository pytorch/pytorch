from typing import Any, Optional

import sympy

import torch

from ..ir import GraphPartitionSignature
from ..virtualized import V
from .cpp_wrapper_gpu import CppWrapperGpu
from .wrapper import PythonWrapperCodegen


class CppWrapperMps(CppWrapperGpu):
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ) -> "CppWrapperMps":
        return CppWrapperMps()

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args: list[str],
        arg_types: Optional[list[type]] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Generates MPS kernel call code. It should look something like:
        ```
        auto mps_lib_0_func = mps_lib_0.getKernelFunction("generated_kernel");
        auto mps_lib_0_func_handle = AOTIMetalKernelFunctionHandle(mps_lib_0_func.get());
        mps_lib_0_func->runCommandBlock([&] {
            mps_lib_0_func->startEncoding();
            aoti_torch_mps_set_arg(mps_lib_0_func_handle, 0, buf0);
            aoti_torch_mps_set_arg(mps_lib_0_func_handle, 1, arg0_1);
            ...
            mps_lib_0_func->dispatch(9);
        });
        ```
        """
        assert arg_types is not None

        new_args = []
        for idx, (arg, arg_type) in enumerate(zip(call_args[:-2], arg_types[:-2])):
            if isinstance(arg_type, torch.dtype):
                new_args.append(
                    f"aoti_torch_mps_set_arg_tensor({kernel_name}_handle, {idx}, {arg});\n"
                )
            elif arg_type in (int, sympy.core.symbol.Symbol):
                new_args.append(
                    f"aoti_torch_mps_set_arg_int({kernel_name}_handle, {idx}, {arg});\n"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported arg type {arg_type} for arg {arg} for kernel {kernel_name}"
                )

        threads, group_size = call_args[-2], call_args[-1]
        if threads is None:
            raise NotImplementedError("No threads or group_size provided")
        elif group_size is None:
            new_args.append(f"{kernel_name}->dispatch({threads});\n")
        else:
            new_args.append(f"{kernel_name}->dispatch({threads}, {group_size});\n")

        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args[:-2],
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def wrap_kernel_call(self, name: str, call_args: list[str]) -> str:
        lib_name = name[: -len("_func")]
        calling_args = "        ".join(call_args)
        return f"""
    auto {name} = {lib_name}.getKernelFunction("generated_kernel");
    auto {name}_handle = AOTIMetalKernelFunctionHandle({name}.get());
    {name}->runCommandBlock([&] {{
        {name}->startEncoding();
        {calling_args}
    }});
        """

    @staticmethod
    def get_device_include_path(device: str) -> str:
        assert V.graph.aot_mode
        return (
            "#include <torch/csrc/inductor/aoti_include/mps.h>\n"
            "#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>"
        )
