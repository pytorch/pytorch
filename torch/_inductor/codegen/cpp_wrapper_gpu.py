# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import re
import sys
from itertools import count, zip_longest
from typing import Any, Optional, Union
from typing_extensions import Self

import sympy

import torch
from torch import dtype as torch_dtype
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch._inductor.runtime.runtime_utils import dynamo_timed

from .. import config
from ..codecache import CudaKernelParamCache
from ..ir import (
    GraphPartitionSignature,
    TensorBox,
    TMADescriptorExperimental,
    TMADescriptorStable,
)
from ..utils import cache_on_self, get_gpu_type, GPU_ALIGN_BYTES, IndentedBuffer
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, TritonScratchWorkspace
from .cpp_utils import cexpr
from .cpp_wrapper_cpu import CppWrapperCpu
from .multi_kernel import MultiKernelCall
from .triton_utils import should_unwrap_unspec_arg
from .wrapper import PythonWrapperCodegen, SymbolicCallArg


_cpp_string_literal_escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_cpp_string_literal_pattern = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    escaped = _cpp_string_literal_pattern.sub(
        lambda match: _cpp_string_literal_escapes[match.group(0)], s
    )
    return f'"{escaped}"'


@dataclasses.dataclass
class DeferredTritonCallWrapper:
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]

    def generate(self, wrapper: CppWrapperGpu):
        """
        Generate the GPU kernel definition, as well as load and launch code.
        """
        prefix = wrapper.prefix
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel will select one kernel after running the autotune block
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)
        params = CudaKernelParamCache.get(self.kernel_name)
        assert params, f"CudaKernelParamCache not populated for {self.kernel_name}"
        def_args = params["def_args"]
        arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]

        if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
            # extra_launcher_args should already be in def_args
            assert len(def_args) == len(arg_types) - len(
                inductor_meta["extra_launcher_args"]
            )
            arg_types = arg_types + [SymbolicCallArg] * len(
                inductor_meta["extra_launcher_args"]
            )

        if not V.graph.aot_mode:
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"static {wrapper.device_codegen.cpp_kernel_type()} {self.kernel_name} = nullptr;"
                )
            )
            kernel_var_name = self.kernel_name
        else:
            kernel_var_name = f"kernels_.{self.kernel_name}"

        # tensors can be RAIIAtenTensorHandle or ConstantHandle, so make them template types
        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(def_args, arg_types)
            if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg))
        ]
        if V.graph.aot_mode:
            template_types.append("typename kernels_type_")
        if template_types:
            prefix.writeline(f"template <{', '.join(template_types)}>")
        prefix.writeline(f"static inline void {self.wrapper_name}(")
        with prefix.indent():
            assert len(def_args) == len(arg_types), (def_args, arg_types)
            for name, arg_type in zip(def_args, arg_types):
                if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg)):
                    prefix.writeline(f"const {name}_type_& {name},")
                elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
                    prefix.writeline(f"int64_t {name},")
                elif arg_type is float:
                    prefix.writeline(f"float {name},")
                elif arg_type is bool:
                    prefix.writeline(f"bool {name},")
                else:
                    raise ValueError(f"Unexpected arg type {arg_type}")
            prefix.writeline("int32_t device_idx_,")
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"{wrapper.device_codegen.cpp_stream_type()} stream_,"
                )
            )
            if V.graph.aot_mode:
                prefix.writeline("kernels_type_& kernels_,")
            prefix.writeline(
                "const std::optional<std::string>& cubin_dir_ = std::nullopt"
            )
        prefix.writeline("){")
        with prefix.indent():
            if V.graph.aot_mode:
                # Emit the original Triton kernel for debugging purposes
                prefix.writeline("/*")
                prefix.splice(self.kernel_name_to_body[self.kernel_name])
                prefix.writeline("*/")
            self.generate_grid(prefix, inductor_meta, params)
            self.generate_load_kernel(prefix, kernel_var_name, params)
            self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
        prefix.writeline("}")

        if not config.aot_inductor.embed_kernel_binary:
            # Ensure the cubin file is included in the package
            V.graph.wrapper_code.additional_files.append(
                params[get_cpp_wrapper_cubin_path_name()]
            )

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):
        from ..runtime.triton_heuristics import GridExpr

        grid = GridExpr.from_meta(inductor_meta, params["config"], mode="cpp")
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""\
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(self, prefix, kernel_var_name, params):
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            embed_kernel_args = [f"__{params['inductor_meta']['kernel_name']}_start"]
            if torch.xpu.is_available():
                # XPU needs the end address of the kernel to calculate the size of the kernel binary.
                embed_kernel_args.append(
                    f"__{params['inductor_meta']['kernel_name']}_end"
                )

            load_kernel_args = (
                [
                    *embed_kernel_args,
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                ]
                if V.graph.aot_mode and config.aot_inductor.embed_kernel_binary
                else [
                    cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()]),
                    cpp_string_literal(params["mangled_name"]),
                    str(params["shared_mem"]),
                    "cubin_dir_",
                ]
            )
            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)}); "
            )
        prefix.writeline("}")

    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params):
        """
        Generate the GPU kernel launching code.
        This is where all the call args being sorted out and generated.
        If enable_kernel_profile is enabled, all args related information would be packed in this function.
        """
        triton_meta = params["triton_meta"]
        assert len(self.arg_types) == len(params["def_args"]), (
            self.arg_types,
            params["def_args"],
        )
        arg_type_loookup = dict(zip(params["def_args"], self.arg_types))
        # difference between Python and C++ wrapper: C++ wrapper strips out equal_to_1 constants
        call_args = [
            name for name in params["call_args"] if name not in triton_meta["constants"]
        ]
        arg_types = [arg_type_loookup[name] for name in call_args]
        arg_signatures = [triton_meta["signature"][name] for name in call_args]
        scratch_spaces = {
            name: params[name]
            for name in ["global_scratch", "profile_scratch"]
            if params.get(name, None) is not None
        }
        call_args_str = wrapper.generate_args_decl(
            prefix,
            call_args,
            arg_types,
            arg_signatures,
            scratch_spaces=scratch_spaces,
        )
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        launch_kernel_args = [
            kernel_var_name,
            "grid_0",
            "grid_1",
            "grid_2",
            str(params["num_warps"]),
            str(params["shared_mem"]),
            "kernel_args_",
            "stream_",
        ]
        if wrapper.device == "xpu":
            launch_kernel_args.append(str(params["threads_per_warp"]))

        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            normalized_kernel_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{kernel_var_name}")
            prefix.writeline("{")
            with prefix.indent():
                prefix.writelines(
                    [
                        f"std::unordered_map<std::string, C10IValueHandle> kwargs_{normalized_kernel_name};",
                        "",
                    ]
                )
                # Add launch args info
                record_launch_kernel_args = [
                    ("grid_0", "grid_0"),
                    ("grid_1", "grid_1"),
                    ("grid_2", "grid_2"),
                    ("num_warps", str(params["num_warps"])),
                    ("shared_mem", str(params["shared_mem"])),
                ]
                for k, v in record_launch_kernel_args:
                    arg_name = f"{normalized_kernel_name}_{k}"
                    prefix.writelines(
                        [
                            f"// Create c10::IValue for {k}",
                            f"C10IValueHandle tmp_{arg_name};",
                            f"aoti_torch_int64_to_ivalue({v}, &tmp_{arg_name});",
                            f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                            f'kwargs_{normalized_kernel_name}.emplace("{k}", RAII_{arg_name});',
                        ]
                    )

                # Add input info (This copies the logic from args_decl)
                signature2dtype = {
                    "i32": "int32_t",
                    "i64": "int64_t",
                    "fp32": "float",
                }

                def signature_is_tma_desc(sig):
                    if not sig:
                        return False
                    if sig == "nvTmaDesc":
                        return True
                    if sig.startswith("tensordesc<"):
                        return True
                    return False

                curr_arg_id = -1
                total_args = []
                ordered_argsname = []

                def write_dummy_scalar_ivalue(arg_name):
                    # We only care about the shape, therefore we create a dummy scalar here.
                    prefix.writelines(
                        [
                            f"// Create c10::IValue for arg_{curr_arg_id}",
                            f"C10IValueHandle tmp_{arg_name};",
                            f"aoti_torch_int64_to_ivalue(0, &tmp_{arg_name});",
                            f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                        ]
                    )
                    # pyrefly: ignore [bad-argument-type]
                    total_args.append(f"tmp_{arg_name}")

                def process_args_for_input_shape(arg, arg_type, arg_signature=None):
                    nonlocal curr_arg_id
                    curr_arg_id += 1
                    arg_name = f"{normalized_kernel_name}_arg_{curr_arg_id}"
                    # ignore tma descriptors, as host-side TMA descriptors need
                    # to be passed to the compiled Triton kernel by value
                    if isinstance(
                        arg_type, UnwrapUnspecArg
                    ) and not signature_is_tma_desc(arg_signature):
                        write_dummy_scalar_ivalue(arg_name)
                    elif isinstance(
                        arg_type, torch_dtype
                    ) and not signature_is_tma_desc(arg_signature):
                        # This is an at::Tensor.
                        prefix.writelines(
                            [
                                f"// Create c10::IValue for arg_{curr_arg_id}",
                                f"C10IValueHandle tmp_{arg_name};",
                                f"aoti_torch_tensor_to_ivalue({arg}, &tmp_{arg_name});",
                                f"RAIIC10IValueHandle RAII_{arg_name}(tmp_{arg_name});",
                            ]
                        )
                        # pyrefly: ignore [bad-argument-type]
                        total_args.append(f"tmp_{arg_name}")
                    elif (
                        isinstance(arg_type, type(SymbolicCallArg))
                        and arg_signature is not None
                        and arg_signature in signature2dtype
                    ) or arg_type in (sympy.Integer, int, sympy.Float, float):
                        write_dummy_scalar_ivalue(arg_name)
                    elif arg_signature and arg_signature.startswith("tensordesc<"):
                        # Skip tma related args
                        pass
                    else:
                        write_dummy_scalar_ivalue(arg_name)

                # Add input name and shape information
                for arg, arg_type, arg_signature in zip_longest(
                    call_args, arg_types, arg_signatures
                ):
                    # pyrefly: ignore [bad-argument-type]
                    ordered_argsname.append(f'"{arg}"')
                    process_args_for_input_shape(arg, arg_type, arg_signature)

                # Add input name into kwargs
                name_var = f"{normalized_kernel_name}_input_names"
                prefix.writelines(
                    [
                        "// Create c10::IValue for input names",
                        f"C10IValueHandle tmp_{name_var};",
                        f"std::vector<const char*> {name_var}({{{', '.join(ordered_argsname)}}});",
                        f"aoti_torch_strlist_to_ivalue({name_var}.data(), {len(ordered_argsname)}, &tmp_{name_var});",
                        f"RAIIC10IValueHandle RAII_{name_var}(tmp_{name_var});",
                        f'kwargs_{normalized_kernel_name}.emplace("Input Args", RAII_{name_var});',
                    ]
                )

                inputs_info_ = f"{normalized_kernel_name}_inputs_info_"
                # We pass in the non-RAII handles, since C10 doesn't automatically free them.
                # The RAII will make sure they get freed when they are out of scope.
                tmp_args = ",".join(total_args)
                prefix.writelines(
                    [
                        "// Aggregate all c10::IValue for inputs",
                        f"std::vector<C10IValueHandle> {inputs_info_}({{{tmp_args}}});",
                    ]
                )

                # Start recording Function
                prefix.writelines(
                    [
                        "",
                        (
                            "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                            f"record_{normalized_kernel_name}_"
                            f'("{kernel_var_name}", '
                            f"reinterpret_cast<IValueMapHandle>(&kwargs_{normalized_kernel_name}), "
                            f"{inputs_info_});"
                        ),
                        "",
                        f"launchKernel({', '.join(launch_kernel_args)});",
                    ]
                )
            prefix.writeline("}")
        else:
            prefix.writeline(f"launchKernel({', '.join(launch_kernel_args)});")


class CppWrapperGpu(CppWrapperCpu):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = get_gpu_type()
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()
        self.grid_id = count()
        self._kernel_name_to_body: dict[str, str] = {}
        self._triton_call_wrappers: dict[str, DeferredTritonCallWrapper] = {}
        self.autotune_input_prefix = "_REAL_AUTOTUNE_INPUT"

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperGpu()

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        super().write_header()
        self.header.splice(
            maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        )

    @cache_on_self
    def write_tma_descriptor_helpers_once(self):
        self.header.splice(self.device_codegen.tma_descriptor_helpers())

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        name = f"stream{device_idx}"
        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({self.device_codegen.aoti_get_stream()}({device_idx}, (void**)&{name}));"
        )
        return name

    def get_autotuning_input_name(self, idx):
        return f"{self.autotune_input_prefix}_{idx}"

    def codegen_inputs(self):
        # See Note: [Input Alignment handling in Inductor]
        #
        # JIT Inductor does not guard on input alignment. It relies on copy_misaligned_inputs to
        # copy misaligned inputs to aligned buffers. For AOTInductor, we need to do the same in cpp.

        if config.is_fbcode():
            # TODO: This is added because FC. Remove this once the newly added shim symbols,
            # e.g. aoti_torch_clone_preserve_strides, have landed
            return super().codegen_inputs()

        if V.graph.aot_mode and V.graph.inputs_to_check:
            for idx in V.graph.inputs_to_check:
                input_name = V.graph.graph_input_names[idx]
                assert input_name in V.graph.graph_inputs, (
                    f"{input_name} not found in graph inputs"
                )
                value = V.graph.graph_inputs[input_name]
                assert isinstance(value, TensorBox), (
                    f"{input_name} is expected to be tensor but found as {type(value)}"
                )
                warn_msg = (
                    f"Input {idx} was compiled as {GPU_ALIGN_BYTES}-bytes aligned, "
                    "but it is not aligned at run time. Copying to an aligned tensor "
                    "to guarantee correctness, but expect a performance hit."
                )
                self.prefix.splice(
                    f"""
                    if ((reinterpret_cast<std::uintptr_t>({input_name}.data_ptr()) & ({GPU_ALIGN_BYTES} -1)) != 0) {{
                        AOTI_TORCH_WARN("{warn_msg}");
                        AtenTensorHandle {input_name}_aligned;
                        aoti_torch_clone_preserve_strides({input_name}, &{input_name}_aligned);
                        {input_name} = std::move(RAIIAtenTensorHandle({input_name}_aligned));
                    }}
                    """
                )

        super().codegen_inputs()

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ):
        if gpu:
            self._kernel_name_to_body[kernel_name] = kernel_body
            if config.triton.autotune_at_compile_time:
                # Call PythonWrapperCodegen to create the autotune code block
                PythonWrapperCodegen._define_kernel_helper(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
        else:
            return CppWrapperCpu._define_kernel_helper(
                self, kernel_name, kernel_body, metadata, gpu, cpp_definition
            )

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperGpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def finalize_prefix(self):
        """Define the triton kernels now that autotuning is finished"""
        old_prefix = self.prefix  # new content should go at start of prefix

        # Generating triton kernel callers can modify the prefix (cached dtypes),
        # so do this before running finalize_prefix(), but put the generated code
        # after the finalize_prefix() code.
        self.prefix = IndentedBuffer()
        for kernel in self._triton_call_wrappers.values():
            self.prefix.writeline("\n")
            kernel.generate(self)
        triton_prefix = self.prefix

        self.prefix = IndentedBuffer()
        super().finalize_prefix()

        self.prefix.splice(triton_prefix)

        self.prefix.writeline("\n")
        self.prefix.splice(old_prefix)

    def generate_tma_descriptor(self, desc):
        self.write_tma_descriptor_helpers_once()

        if isinstance(desc, TMADescriptorExperimental):
            self._generate_experimental_tma_descriptor(desc)
        else:
            assert isinstance(desc, TMADescriptorStable)
            self._generate_stable_tma_descriptor(desc)

    def _generate_experimental_tma_descriptor(self, desc):
        # generate data pointer for the source tensor
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        self.writeline(f"alignas(64) CUtensorMap {desc_name};")

        # `source` is in the form of `&var_x`, where `var_x` is the data pointer
        # (CUdeviceptr); we dereference `source` and cast to `void*` to pass to
        # the data pointer of the source tensor to the helper function
        # `init{1,2}DTMADescriptor`
        ptr = f"reinterpret_cast<void*>(*({source}))"
        dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.dims)
        block_dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.block_dims)
        element_size = self.val_to_arg_str(desc.element_size)
        fn = f"init{desc.rank}DTMADescriptor"
        args = f"&{desc_name}, {ptr}, {dims}, {block_dims}, {element_size}"
        self.writeline(f"{fn}({args});")

    def _generate_stable_tma_descriptor(self, desc):
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # these args are passed to initNDTMADescriptor, which is NOT a triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        # Pack the relevant information into a StableTMADescriptor struct.
        # See [Note: AOTI TMA Stable handling] for more details.
        self.writeline(f"alignas(64) StableTMADescriptor {desc_name};")

        def fill_array(name, values):
            for i, val in enumerate(values):
                self.writeline(f"{name}[{i}] = {val};")

        ptr = f"reinterpret_cast<void*>(*({source}))"
        rank = len(desc.tensor.get_size())

        fill_array(f"{desc_name}.block_shape", desc.block_shape)
        fill_array(f"{desc_name}.global_shape", desc.tensor.get_size())
        fill_array(f"{desc_name}.strides", desc.tensor.get_stride())

        element_size = self.val_to_arg_str(desc.tensor.get_dtype().itemsize)
        fn = "initTMADescriptor"
        args = ", ".join(
            str(x)
            for x in [
                f"&{desc_name}.m",
                ptr,
                element_size,
                rank,
                f"{desc_name}.block_shape",
                f"{desc_name}.global_shape",
                f"{desc_name}.strides",
            ]
        )
        self.writeline(f"{fn}({args});")

    def generate_args_decl(
        self,
        code: Union[IndentedBuffer, Self],
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=True,
        scratch_spaces: Optional[dict[str, int]] = None,
    ):
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """
        new_args: list[str] = []

        # Add more cases for other types as needed
        signature2dtype = {
            "i32": "int32_t",
            "i64": "int64_t",
            "fp32": "float",
        }

        def signature_is_tma_desc(sig):
            if not sig:
                return False
            if sig == "nvTmaDesc":
                return True
            if sig.startswith("tensordesc<"):
                return True
            return False

        def process_tma_stable_arg(arg, arg_type, arg_signature, var_name):
            # [Note: AOTI TMA Stable handling]
            # For most args, a single arg passed to the python triton interface
            # maps to a single arg in the cubin interface. However, for host-side
            # TMA descriptors, a single python arg turns into 1 + 2 * N args in the
            # cubin interface (where N is the rank).
            #
            # To do this: at TMA codegen time (for aoti), we generate a struct
            # (StableTMADescriptor) containing the necessary information; and then
            # when we call the function (i.e. here), we unpack the struct members.
            code.writeline(f"auto {var_name} = {cexpr(arg)};")

            result = []
            result.append(f"&{var_name}.m")

            # from https://github.com/triton-lang/triton/blob/16961b79bdac1b774b42d44e52fd55a266ec2866/third_party/nvidia/backend/driver.py#L111  # noqa: B950
            match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", arg_signature)
            assert match is not None
            shape = match.group(2)
            ndim = shape.count(",") + 1

            for i in range(ndim):
                result.append(f"&{var_name}.block_shape[{i}]")

            for i in range(ndim):
                result.append(f"&{var_name}.strides[{i}]")

            return result

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore tma descriptors, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and not signature_is_tma_desc(
                arg_signature
            ):
                self.codegen_tensor_item(
                    arg_type.dtype,
                    arg,
                    var_name,
                    indented_buffer=code,
                )
                new_args.append(f"&{var_name}")
            elif isinstance(arg_type, torch_dtype) and not signature_is_tma_desc(
                arg_signature
            ):
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            elif arg_type in (sympy.Float, float):
                code.writeline(f"float {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in signature2dtype
            ):
                code.writeline(
                    f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
                )
                new_args.append(f"&{var_name}")
            elif arg_signature and arg_signature.startswith("tensordesc<"):
                new_args.extend(
                    process_tma_stable_arg(arg, arg_type, arg_signature, var_name)
                )
            else:
                code.writeline(f"auto {var_name} = {cexpr(arg)};")
                new_args.append(f"&{var_name}")

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        for scratch_name, workspace_size in (scratch_spaces or {}).items():
            if (
                is_triton_kernel
                and (
                    scratch := self.device_codegen.cpp_scratch(
                        next(self.arg_var_id),
                        workspace=TritonScratchWorkspace(
                            size=workspace_size,
                            generate_dtype_str=(
                                lambda: self.codegen_dtype(torch.uint8)
                            ),
                        ),
                        prefix=scratch_name,
                    )
                )
                is not None
            ):
                scratch_def, scratch_var = scratch
                code.writelines([maybe_hipify_code_wrapper(x) for x in scratch_def])
                new_args.append(f"&{scratch_var}")

        return ", ".join(new_args)

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
        Override the default value of argument 'gpu' to True here.
        generate_kernel_call can still be called with gpu=False because of
        a mix of cpu kernels and gpu kernels.
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

        if (
            triton
            and config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Call PythonWrapperCodegen to create the autotune code block
            PythonWrapperCodegen._generate_kernel_call_helper(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_keys=raw_keys,
                raw_args=raw_args,
                triton_meta=triton_meta,
                original_fxnode_name=original_fxnode_name,
            )

        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device.index, graph_name)
        )

        if triton:
            call_args, arg_types = self.prepare_triton_wrapper_args(
                call_args,
                # pyrefly: ignore [bad-argument-type]
                arg_types,
            )
            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._triton_call_wrappers:
                self._triton_call_wrappers[wrapper_name] = DeferredTritonCallWrapper(
                    wrapper_name,
                    kernel_name,
                    self._kernel_name_to_body,
                    arg_types,
                )
            device_idx = "this->device_idx_" if V.graph.aot_mode else str(device.index)
            call_args.append(device_idx)
            call_args.append(stream)
            if V.graph.aot_mode:
                call_args.append("kernels")
                call_args.append("this->cubin_dir_")
            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args[: len(arg_types)], kernel_name, arg_types, None
            )
            with debug_printer_manager:
                self.writeline(f"{wrapper_name}({', '.join(call_args)});")
        else:
            casted = []
            # pyrefly: ignore [no-matching-overload]
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                # pyrefly: ignore [bad-argument-type]
                casted.append(f"({arg_type}){cexpr(new_arg)}")
            call_args_str = ", ".join(casted)
            self.writeline(f"kernels.{kernel_name}({call_args_str}, {stream});")

    @staticmethod
    def prepare_triton_wrapper_args(
        call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        assert len(call_args) == len(arg_types), (call_args, arg_types)
        new_args = []
        new_args_types = []
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg, str):
                if isinstance(arg_type, torch_dtype) and should_unwrap_unspec_arg(arg):
                    # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                    arg_type = UnwrapUnspecArg(dtype=arg_type)
                new_args.append(arg)
            elif isinstance(arg, bool):
                new_args.append(str(arg).lower())
            elif isinstance(arg, (int, float, SymbolicCallArg)):
                new_args.append(str(arg))
            else:
                new_args.append(cexpr(V.graph.sizevars.simplify(arg)))
            new_args_types.append(arg_type)
        return new_args, new_args_types

    def make_zero_buffer(self, name):
        return f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_zero_({name}.get()));"


@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor"""

    dtype: torch_dtype
