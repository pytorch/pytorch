from typing import Optional

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
    ):
        return CppWrapperMps()

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        graph_name="",
        original_fxnode_name=None,
    ):
        """
        Generates kernel call code.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        new_args = []
        for idx, arg in enumerate(call_args[:-2]):
            new_args.append(
                f"aoti_torch_mps_set_arg({kernel_name}_handle, {idx}, {arg});\n"
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

    def wrap_kernel_call(self, name, call_args):
        lib_name = name[: -len("_func")]
        calling_args = "        ".join(call_args)
        return f"""
    auto {name} = {lib_name}.getKernelFunction("generated_kernel");
    AOTIMetalKernelFunctionOpaque* {name}_handle = new AOTIMetalKernelFunctionOpaque();
    {name}_handle->kernelFunction = {name};
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
