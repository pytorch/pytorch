from typing import Any, Optional

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from ..ir import GraphPartitionSignature
from ..virtualized import V
from .cpp_wrapper_cpu import CppWrapperCpu
from .cpp_wrapper_gpu import CppWrapperGpu
from .wrapper import KernelCallLine, PythonWrapperCodegen


class CppWrapperMps(CppWrapperGpu):
    """
    Generates cpp wrapper for running on MPS and calls metal kernels
    """

    def __init__(self) -> None:
        super().__init__()
        self._used_kernel_names: OrderedSet[str] = OrderedSet()

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
        *,
        device: Optional[torch.device] = None,
        triton: bool = True,
        arg_types: Optional[tuple[Any, ...]] = None,
        raw_keys: Optional[tuple[Any, ...]] = None,
        raw_args: Optional[tuple[Any, ...]] = None,
        triton_meta: Optional[dict[str, Any]] = None,
        graph_name: str = "",
        original_fxnode_name: Optional[str] = None,
    ) -> None:
        """
        Generates MPS kernel call code. It should look something like:
        ```
        get_mps_lib_0()->runCommandBlock([&] {
            get_mps_lib_0()->startEncoding();
            aoti_torch_mps_set_arg(get_mps_lib_0_handle(), 0, buf0);
            aoti_torch_mps_set_arg(get_mps_lib_0_handle(), 1, arg0_1);
            ...
            get_mps_lib_0()->dispatch(9);
        });
        ```
        """
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "cpu":
            # Even in CppWrapperGpu, we may see cpp kernels
            return CppWrapperCpu._generate_kernel_call_helper(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
            )

        assert device.type == "mps"

        assert arg_types is not None

        new_args = []
        for idx, (arg, arg_type) in enumerate(zip(call_args[:-2], arg_types[:-2])):
            if isinstance(arg_type, torch.dtype):
                new_args.append(
                    f"aoti_torch_mps_set_arg_tensor(get_{kernel_name}_handle(), {idx}, {arg});"
                )
            elif arg_type in (int, sympy.core.symbol.Symbol):
                new_args.append(
                    f"aoti_torch_mps_set_arg_int(get_{kernel_name}_handle(), {idx}, {arg});"
                )
            else:
                raise NotImplementedError(
                    f"Unsupported arg type {arg_type} for arg {arg} for kernel {kernel_name}"
                )

        threads, group_size = call_args[-2], call_args[-1]
        if threads is None:
            raise NotImplementedError("No threads or group_size provided")
        elif group_size is None:
            new_args.append(f"get_{kernel_name}()->dispatch({threads});\n")
        else:
            new_args.append(
                f"get_{kernel_name}()->dispatch({threads}, {group_size});\n"
            )

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
            self.write_mps_kernel_call(kernel_name, new_args)

    def write_mps_kernel_call(self, name: str, call_args: list[str]) -> None:
        # Initialization of the kernel function and kernel function handle
        # variables have already been done at the beginning, which was
        # codegen-ed in `codegen_mps_func_init`
        self.writeline(f"get_{name}()->runCommandBlock([&] {{")
        self.writeline(f"    get_{name}()->startEncoding();")
        for call_arg in call_args:
            self.writeline(f"    {call_arg}")
        self.writeline("});")

    @staticmethod
    def get_device_include_path(device: str) -> str:
        assert V.graph.aot_mode
        return (
            "#include <torch/csrc/inductor/aoti_include/mps.h>\n"
            "#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>"
        )

    def codegen_additional_funcs(self) -> None:
        """
        We want to codegen the mps kernel function variable initializations
        ahead of time.  This is so that if we reuse kernels within subgraphs, we
        don't need to worry about the scope in which we're initializing the
        variables. Instead we will just initialize the variables all at the top
        level.

        The kernel function variable initializations should look something like:
        ```
        const std::shared_ptr<at::native::mps::MetalKernelFunction> get_mps_lib_0() {
            static const auto func = mps_lib_0.getKernelFunction("generated_kernel");
            return func;
        }
        AOTIMetalKernelFunctionHandle get_mps_lib_0_handle() {
            static const auto handle = AOTIMetalKernelFunctionHandle(get_mps_lib_0().get());
            return handle;
        }
        ```
        """

        for line in self.lines:
            if not isinstance(line, KernelCallLine):
                continue
            if line.device.type != "mps":
                continue

            # Only add handle definition once
            if line.kernel_name not in self._used_kernel_names:
                self._used_kernel_names.add(line.kernel_name)

                self.prefix.writeline(
                    f"const std::shared_ptr<at::native::mps::MetalKernelFunction> get_{line.kernel_name}() {{"
                )
                self.prefix.writeline(
                    f'    static const auto func = {line.kernel_name}.getKernelFunction("generated_kernel");'
                )
                self.prefix.writeline("    return func;")
                self.prefix.writeline("}")

                self.prefix.writeline(
                    f"AOTIMetalKernelFunctionHandle get_{line.kernel_name}_handle() {{"
                )
                self.prefix.writeline(
                    f"    static const auto handle = AOTIMetalKernelFunctionHandle(get_{line.kernel_name}().get());"
                )
                self.prefix.writeline("    return handle;")
                self.prefix.writeline("}")
