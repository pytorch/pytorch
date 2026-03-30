from typing import Any

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
        self._lambda_counter: int = 0

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = None,
    ) -> "CppWrapperMps":
        return CppWrapperMps()

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args: list[str],
        *,
        device: torch.device | None = None,
        triton: bool = True,
        arg_types: tuple[Any, ...] | None = None,
        raw_keys: tuple[Any, ...] | None = None,
        raw_args: tuple[Any, ...] | None = None,
        triton_meta: dict[str, Any] | None = None,
        inductor_meta: dict[str, Any] | None = None,
        graph_name: str = "",
        original_fxnode_name: str | None = None,
    ) -> None:
        """
        Generates MPS kernel call code. It should look something like:
        ```
        auto mps_lib_0_lambda = [&](AOTIMetalKernelFunctionHandle handle) {
            aoti_torch_mps_start_encoding(handle);
            aoti_torch_mps_set_arg_tensor(handle, 0, buf0);
            aoti_torch_mps_set_arg_tensor(handle, 1, arg0_1);
            aoti_torch_mps_set_arg_tensor(handle, 2, arg1_1);
            aoti_torch_mps_dispatch_single(handle, static_cast<uint64_t>(10LL));
        };

        std::function<void(AOTIMetalKernelFunctionHandle)> mps_lib_0_func_wrapper = mps_lib_0_lambda;
        aoti_torch_mps_run_command_block(get_mps_lib_0_handle(), aoti_torch_mps_shared_callback, &mps_lib_0_func_wrapper);
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
                inductor_meta=inductor_meta,
            )

        assert device.type == "mps"

        assert arg_types is not None

        new_args = []
        for idx, (arg, arg_type) in enumerate(zip(call_args[:-2], arg_types[:-2])):
            if isinstance(arg_type, torch.dtype):
                new_args.append(f"aoti_torch_mps_set_arg_tensor(handle, {idx}, {arg});")
            elif arg_type in (int, sympy.core.symbol.Symbol):
                new_args.append(f"aoti_torch_mps_set_arg_int(handle, {idx}, {arg});")
            else:
                raise NotImplementedError(
                    f"Unsupported arg type {arg_type} for arg {arg} for kernel {kernel_name}"
                )

        threads, group_size = call_args[-2], call_args[-1]
        if threads is None:
            raise NotImplementedError("No threads or group_size provided")

        # Check if threads is a single value or an array-like structure
        threads_str = str(threads)
        is_single_value = (
            threads_str.startswith("{")
            and threads_str.endswith("}")
            and threads_str.count(",") == 0
        ) or not threads_str.startswith(("{", "["))

        if is_single_value:
            # Extract single value from braces if present
            if threads_str.startswith("{") and threads_str.endswith("}"):
                single_value = threads_str[1:-1].strip()  # Remove braces
            else:
                single_value = threads_str

            if group_size is None:
                new_args.append(
                    f"aoti_torch_mps_dispatch_single(handle, {single_value});"
                )
            else:
                # Extract group size value if it's also in braces
                group_size_str = str(group_size)
                if group_size_str.startswith("{") and group_size_str.endswith("}"):
                    group_size_value = group_size_str[1:-1].strip()
                else:
                    group_size_value = group_size_str
                new_args.append(
                    f"aoti_torch_mps_dispatch_single_with_group_size(handle, {single_value}, {group_size_value});"
                )
        else:
            # Handle array case - need to convert initializer list to array
            # Use kernel name to make variable names unique
            threads_var = f"{kernel_name}_threads_array"
            group_size_var = f"{kernel_name}_group_size_array"

            # Extract array size from the initializer list string
            def get_array_size(array_str: str) -> int:
                # Remove braces and whitespace
                content = array_str.strip()
                if content.startswith("{") and content.endswith("}"):
                    content = content[1:-1].strip()

                if not content:  # Empty array
                    return 0

                # Count elements by counting commas, accounting for nested structures
                depth = 0
                comma_count = 0
                for char in content:
                    if char in "({[<":
                        depth += 1
                    elif char in ")}]>":
                        depth -= 1
                    elif char == "," and depth == 0:
                        comma_count += 1

                return comma_count + 1  # Number of elements = commas + 1

            threads_size = get_array_size(threads_str)

            if group_size is None:
                new_args.append("{")
                new_args.append(f"    uint64_t {threads_var}[] = {threads};")
                new_args.append(
                    f"    aoti_torch_mps_dispatch_array(handle, {threads_var}, {threads_size});"
                )
                new_args.append("}")
            else:
                group_size_str = str(group_size)
                group_size_size = get_array_size(group_size_str)
                new_args.append("{")
                new_args.append(f"    uint64_t {threads_var}[] = {threads};")
                new_args.append(f"    uint64_t {group_size_var}[] = {group_size};")
                dispatch_args = f"handle, {threads_var}, {threads_size}, {group_size_var}, {group_size_size}"
                new_args.append(
                    f"    aoti_torch_mps_dispatch_array_with_group_size({dispatch_args});"
                )
                new_args.append("}")

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
        # Generate unique variable names to avoid duplicate declarations
        # when the same MPS lib is used multiple times
        unique_suffix = self._lambda_counter
        self._lambda_counter += 1

        lambda_name = f"{name}_lambda_{unique_suffix}"
        wrapper_name = f"{name}_func_wrapper_{unique_suffix}"

        # Generate the function call code (in current location)
        # Create lambda that captures by reference and pass its pointer through void*
        self.writeline(
            f"auto {lambda_name} = [&](AOTIMetalKernelFunctionHandle handle) {{"
        )
        self.writeline("    aoti_torch_mps_start_encoding(handle);")

        # Output call args directly since we're capturing by reference
        for call_arg in call_args:
            self.writeline(f"    {call_arg}")
        self.writeline("};")
        self.writeline("")

        # Pass lambda pointer through void*
        self.writeline(
            f"std::function<void(AOTIMetalKernelFunctionHandle)> {wrapper_name} = {lambda_name};"
        )
        self.writeline(
            f"aoti_torch_mps_run_command_block(get_{name}_handle(), aoti_torch_mps_shared_callback, &{wrapper_name});"
        )

    @staticmethod
    def get_device_include_path(device: str) -> str:
        assert V.graph.aot_mode
        return (
            "#include <torch/csrc/inductor/aoti_include/mps.h>\n"
            "#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>"
        )

    def codegen_additional_funcs(self) -> None:
        """
        Generate thread-safe lazy singleton pattern for MPS shader libraries with RAII cleanup.

        The generated code will look like:
        ```
        AOTIMetalKernelFunctionHandle get_mps_lib_0_handle() {
            static auto kernel_handle = []() {
                AOTIMetalShaderLibraryHandle lib_handle = nullptr;
                AOTIMetalKernelFunctionHandle kern_handle = nullptr;

                aoti_torch_mps_create_shader_library(mps_lib_0_source, &lib_handle);
                aoti_torch_mps_get_kernel_function(lib_handle, "generated_kernel", &kern_handle);

                // RAII wrapper with custom deleter
                auto lib_deleter = [](AOTIMetalShaderLibraryHandle h) {
                    if (h) aoti_torch_mps_delete_shader_library(h);
                };

                using LibDeleter = decltype(lib_deleter);
                using LibPtr = std::unique_ptr<AOTIMetalShaderLibraryOpaque, LibDeleter>;

                // Return pair of kernel handle and library smart pointer for cleanup
                return std::make_pair(kern_handle, LibPtr(lib_handle, lib_deleter));
            }();
            return kernel_handle.first;
        }
        ```
        """

        # Add shimified handles and functions
        shader_libraries: OrderedSet[str] = OrderedSet()
        for line in self.lines:
            if not isinstance(line, KernelCallLine):
                continue
            if line.device.type != "mps":
                continue

            # Extract library name from kernel name (e.g., "mps_lib_0" from kernel calls)
            if line.kernel_name not in self._used_kernel_names:
                self._used_kernel_names.add(line.kernel_name)
                shader_libraries.add(line.kernel_name)

        # NOTE: For shimified version, we expect the shader source constant to be generated
        # by the existing MPS shader generation process, but instead of instantiating the
        # DynamicMetalShaderLibrary directly, we'll use our shim functions.
        # The existing codegen should produce something like:
        # const char* mps_lib_0_source = R"MTL(...shader_source...)MTL";
        # instead of:
        # at::native::mps::DynamicMetalShaderLibrary mps_lib_0(R"MTL(...shader_source...)MTL");

        # Generate thread-safe lazy singleton with RAII for each library
        for lib_name in shader_libraries:
            self.prefix.splice(f"""
AOTIMetalKernelFunctionHandle get_{lib_name}_handle() {{
    static auto kernel_handle = []() {{
        AOTIMetalShaderLibraryHandle lib_handle = nullptr;
        AOTIMetalKernelFunctionHandle kern_handle = nullptr;

        aoti_torch_mps_create_shader_library({lib_name}_source, &lib_handle);
        aoti_torch_mps_get_kernel_function(lib_handle, "generated_kernel", &kern_handle);

        // RAII wrapper with custom deleter
        auto lib_deleter = [](AOTIMetalShaderLibraryHandle h) {{
            if (h) aoti_torch_mps_delete_shader_library(h);
        }};

        using LibDeleter = decltype(lib_deleter);
        using LibPtr = std::unique_ptr<AOTIMetalShaderLibraryOpaque, LibDeleter>;

        // Return pair of kernel handle and library smart pointer for cleanup
        return std::make_pair(kern_handle, LibPtr(lib_handle, lib_deleter));
    }}();
    return kernel_handle.first;
}}
""")
