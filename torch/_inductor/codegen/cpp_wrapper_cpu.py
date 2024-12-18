# mypy: allow-untyped-defs
import functools
import math
import os
import sys
from itertools import count
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy
from sympy import Expr

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch._ops
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config, ir
from ..utils import _align, ALIGN_BYTES, cache_on_self, normalize_name
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, IndentedBuffer, Kernel
from .cpp_utils import cexpr, DEVICE_TO_ATEN, DTYPE_TO_ATEN, DTYPE_TO_CPP
from .triton_utils import should_unwrap_unspec_arg
from .wrapper import (
    EnterSubgraphLine,
    ExitSubgraphLine,
    PythonWrapperCodegen,
    SymbolicCallArg,
)


class CppWrapperCpu(PythonWrapperCodegen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        super().__init__()
        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False
        self.kernel_callsite_id = count()
        self.var_array_id = (
            count()
        )  # for different types of local array variable declarations
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars = OrderedSet[str]()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self.used_cached_devices = OrderedSet[str]()
        self.used_cached_dtypes = OrderedSet[str]()
        self.used_cached_layouts = OrderedSet[str]()
        self.used_cached_memory_formats = OrderedSet[str]()
        self.used_cond_predicate = OrderedSet[str]()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        # For GEMM kernels that must be initialized and are resolved at linking.
        self.initialized_kernels: Dict[str, Kernel] = {}
        self.device_codegen = get_device_op_overrides(self.device)

    @staticmethod
    def create(
        is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpu()

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        gpu=False,
        triton=False,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        """
        Generates kernel call code.

        gpu: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        assert not gpu, "CppWrapperCpu.generate_kernel_call does not support GPU"
        assert arg_types is not None and len(call_args) == len(
            arg_types
        ), "Mismatch call_args and arg_types in generate_kernel_call"
        new_args = []
        for idx, arg in enumerate(call_args):
            if "*" in arg_types[idx]:
                new_args.append(f"({arg_types[idx]})({arg}.data_ptr())")
            else:
                # arg is a scalar
                new_args.append(arg)
        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def write_constant(self, name, hashed):
        # include a hash so our code cache gives different constants different files
        self.header.writeline(f"// {name} {hashed}")

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if V.graph.aot_mode:
            self.header.splice(
                """
                #include <torch/csrc/inductor/aoti_runtime/interface.h>
                #include <torch/csrc/inductor/aoti_runtime/model.h>
                """
            )
            with open(
                os.path.join(os.path.dirname(__file__), "aoti_runtime", "interface.cpp")
            ) as f:
                self.header.splice(f.read())
        else:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                '''
                #include <pybind11/pybind11.h>
                namespace py = pybind11;

                class RAIIPyObject {
                public:
                    RAIIPyObject() : obj_(nullptr) {}
                    RAIIPyObject(PyObject* obj) : obj_(obj) {}
                    ~RAIIPyObject() {
                        Py_XDECREF(obj_);
                    }
                    RAIIPyObject& operator=(const RAIIPyObject& other) {
                        if (this != &other) {
                            Py_XDECREF(obj_);
                            obj_ = other.obj_;
                            Py_XINCREF(obj_);
                        }
                        return *this;
                    }
                    operator PyObject*() {
                        return obj_;
                    }
                    PyObject* get() {
                        return obj_;
                    }
                private:
                    PyObject* obj_;
                };

                #include <torch/csrc/inductor/aoti_runtime/device_utils.h>
                #include <torch/csrc/inductor/aoti_runtime/utils.h>
                using namespace torch::aot_inductor;
                """
            )

        self.header.splice(
            f"""
            #include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
            #include <torch/csrc/inductor/aoti_runtime/thread_local.h>
            #include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
            #include <torch/csrc/inductor/aoti_torch/generated/c_shim_{self.device}.h>

            #include <c10/util/generic_math.h>
            typedef at::Half half;
            typedef at::BFloat16 bfloat16;

            // Round up to the nearest multiple of {ALIGN_BYTES}
            [[maybe_unused]] static int64_t align(int64_t nbytes) {{
              return (nbytes + {ALIGN_BYTES} - 1) & -{ALIGN_BYTES};
            }}
            """
        )
        extend_aoti_c_shim_include = (
            f"torch/csrc/inductor/aoti_torch/generated/extend/c_shim_{self.device}.h"
        )
        extend_aoti_c_shim_path = os.path.join(
            os.path.dirname(torch.__file__),
            "include",
            extend_aoti_c_shim_include,
        )
        if os.path.exists(extend_aoti_c_shim_path):
            self.header.splice(f"#include <{extend_aoti_c_shim_include}>")

        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if config.profiler_mark_wrapper_call or enable_kernel_profile:
            # No C shim for profiling APIs, assuming profiling is a debugging feature which
            # does not provide any ABI compatibility promise.
            self.header.splice("#include <ATen/record_function.h>")

    @functools.lru_cache(None)  # noqa: B019
    def include_extra_header(self, header: str):
        # This is needed for cpp to python dtype conversion
        self.header.splice(f"#include <{header}>")

    def mark_output_type(self):
        # mark output type to unwrap tensor back to python scalar
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = {}
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        if V.graph.is_const_graph:
            # We do not write prefix for constant graph, it will be written by main module.
            return
        if V.graph.aot_mode:
            self.prefix.writeline("namespace torch::aot_inductor {")

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")

    def codegen_input_symbol_assignment(
        self,
        name: str,
        value: ir.TensorBox,
        bound_vars: OrderedSet[sympy.Symbol],
    ):
        code = self.prefix

        @functools.lru_cache(None)
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value in bound_vars:
                return
            if value.is_integer:
                decl = "int64_t"
            elif value.is_float:
                decl = "double"
            else:
                raise AssertionError("Unexpected symbol type")
            code.writeline(f"{decl} {value} = {name};")
            bound_vars.add(value)
        elif isinstance(value, ir.TensorBox):
            for dim, size in enumerate(value.get_size()):
                if isinstance(size, sympy.Symbol) and size not in bound_vars:
                    code.writeline(f"int64_t {size} = {sizeof(name)}[{dim}];")
                    bound_vars.add(size)
            for dim, stride in enumerate(value.get_stride()):
                if isinstance(stride, sympy.Symbol) and stride not in bound_vars:
                    code.writeline(f"int64_t {stride} = {strideof(name)}[{dim}];")
                    bound_vars.add(stride)
        else:
            raise AssertionError(f"Unknown value type: {type(value)}")

    def generate_input_output_runtime_checks(self):
        # In debug_compile mode, we generate checks to ensure the dtype/shape/stride of each
        # real input/output tensor match ones provided at compile time via sample
        # input/output.
        def gen_check(handle_kind, idx, name, tensor):
            # Wrap AtenTensorHandle with ConstantHandle for cleaner utility function access
            self.prefix.writeline(
                f"ConstantHandle {name} = ConstantHandle({handle_kind}[{idx}]);"
            )
            self.codegen_tensor_dtype_var_decl(self.prefix, name)
            expected_dtype_name = DTYPE_TO_ATEN[tensor.dtype]
            dtype_str = str(tensor.dtype).split(".")[-1]
            self.prefix.splice(
                f"""
                    int32_t {name}_expected_dtype = aoti_torch_dtype_{dtype_str}();
                    if ({name}_expected_dtype != {name}_dtype) {{
                        std::stringstream ss;
                        ss << "{handle_kind}[{idx}]: unmatched dtype, "
                           << "expected: " << {name}_expected_dtype << "({expected_dtype_name}), "
                           << "but got: " << {name}_dtype << "\\n";
                        throw std::runtime_error(ss.str());
                    }}
                """
            )
            self.codegen_input_size_var_decl(self.prefix, name)
            for dim_idx, d in enumerate(tensor.get_size()):
                if isinstance(d, (int, sympy.Integer)):
                    self.prefix.splice(
                        f"""
                            if ({d} != {name}_size[{dim_idx}]) {{
                                std::stringstream ss;
                                ss << "{handle_kind}[{idx}]: unmatched dim value at {dim_idx}, "
                                   << "expected: {d}, " << "but got: " << {name}_size[{dim_idx}]
                                   << "\\n";
                                throw std::runtime_error(ss.str());
                            }}
                        """
                    )
                else:
                    from torch.utils._sympy.value_ranges import bound_sympy

                    sym_range = bound_sympy(d, V.graph.sizevars.shape_env.var_to_range)
                    if not math.isinf(sym_range.lower):
                        self.prefix.splice(
                            f"""
                                if ({name}_size[{dim_idx}] < {sym_range.lower}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too small at {dim_idx}, "
                                       << "expected it to be >= {sym_range.lower}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(ss.str());
                                }}
                            """
                        )
                    if not math.isinf(sym_range.upper):
                        self.prefix.splice(
                            f"""
                                if ({name}_size[{dim_idx}] > {sym_range.upper}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too large at {dim_idx}, "
                                       << "expected to be <= {sym_range.upper}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(ss.str());
                                }}
                            """
                        )

            self.codegen_input_stride_var_decl(self.prefix, name)
            for stride_idx, s in enumerate(tensor.get_stride()):
                if not isinstance(s, (int, sympy.Integer)):
                    continue
                self.prefix.splice(
                    f"""
                        if ({s} != {name}_stride[{stride_idx}]) {{
                            std::stringstream ss;
                            ss << "{handle_kind}[{idx}]: unmatched stride value at {stride_idx}, "
                               << "expected: {s}, " << "but got: " << {name}_stride[{stride_idx}]
                               << "\\n";
                            throw std::runtime_error(ss.str());
                        }}
                    """
                )

        # force noinline to avoid any potential compilation slowdown due to aggressive
        # inline done by the host compiler
        self.prefix.splice(
            """
            AOTI_NOINLINE static void __check_inputs_outputs(
                AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles) {
            """
        )
        with self.prefix.indent():
            for idx, (name, tensor) in enumerate(V.graph.graph_inputs.items()):
                gen_check("input_handles", idx, name, tensor)
        self.prefix.writeline("}")

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            if V.graph.const_module:
                self.header.splice(V.graph.const_module.wrapper_code.header)
                self.prefix.splice(V.graph.const_code)

            if V.graph.is_const_graph:
                self.prefix.splice(
                    """
                    void AOTInductorModel::_const_run_impl(
                        std::vector<AtenTensorHandle>& output_handles,
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {
                    """
                )
            else:
                if not config.aot_inductor.use_runtime_constant_folding:
                    # If we do not split the constant graph, we'll just create
                    # an empty implementation when wrapping the main module.
                    self.prefix.splice(
                        """
                        void AOTInductorModel::_const_run_impl(
                            std::vector<AtenTensorHandle>& output_handles,
                            DeviceStreamType stream,
                            AOTIProxyExecutorHandle proxy_executor
                        ) {}

                        """
                    )

                run_impl_proto = """
                    void AOTInductorModel::run_impl(
                        AtenTensorHandle*
                            input_handles, // array of input AtenTensorHandle; handles
                                            // are stolen; the array itself is borrowed
                        AtenTensorHandle*
                            output_handles, // array for writing output AtenTensorHandle; handles
                                            // will be stolen by the caller; the array itself is
                                            // borrowed
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {
                    """
                if config.aot_inductor.debug_compile:
                    self.generate_input_output_runtime_checks()
                    run_impl_proto += """
                        __check_inputs_outputs(input_handles, output_handles);
                    """

                self.prefix.splice(run_impl_proto)
        else:
            # cpp entry function for JIT with cpp wrapper
            self.prefix.splice(
                """
                void inductor_entry_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles  // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed)
                ) {
                """
            )
        with self.prefix.indent():
            # assign inputs and outputs in both cases so the later codegen can be simplified
            if not V.graph.is_const_graph:
                if V.graph.aot_mode:
                    num_args = len(V.graph.graph_inputs)
                else:
                    # Weights are promoted in the JIT mode
                    num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
                    # release GIL to support multiple instances inference (in different threads of the same process)
                    self.prefix.splice("py::gil_scoped_release release;")

                self.prefix.splice(
                    f"""
                        auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {num_args});
                    """
                )

            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    # unwrap input tensor back to scalar
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype

                        dtype = may_get_constant_buffer_dtype(
                            V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                        )
                        assert (
                            dtype is not None
                        ), "Fails to get the dtype of the sympy.Expr"
                        self.codegen_tensor_item(
                            dtype, f"inputs[{idx}]", input_key, self.prefix
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {input_key} = std::move(inputs[{idx}]);"
                        )
                # debug printing for all input args to AOTI model
                debug_printer_manager = V.graph.wrapper_code.debug_printer
                debug_printer_manager.codegen_model_inputs_value_print(
                    input_args_to_print=[
                        input_key
                        for input_key in V.graph.graph_inputs.keys()
                        if input_key.startswith("arg")
                    ]
                )

            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            for idx, constants_key in enumerate(V.graph.constants.keys()):
                if V.graph.aot_mode:
                    # Weights are stored in constants_ and owned by RAIIAtenTensorHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    self.prefix.writeline(
                        f"""[[maybe_unused]] auto {constants_key} = constants_->at({idx});"""
                    )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"[[maybe_unused]] auto {constants_key} = std::move(inputs[{constants_idx}]);"
                    )

            self.codegen_inputs()

            if V.graph.aot_mode:
                if not V.graph.is_const_graph:
                    self.prefix.writeline("inputs.clear();")
                self.prefix.writeline(
                    "auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
                )

    def codegen_tensor_dtype_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"int32_t {name}_dtype;")
        code.writeline(
            "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype"
            f"({name}, &{name}_dtype));"
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_size = {name}.sizes();")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_stride = {name}.strides();")

    def codegen_model_kernels(self):
        self.prefix.writeline("namespace {")

        # Tell compiler we need to link with the non-mangled symbols
        for kernel in self.initialized_kernels.values():
            assert hasattr(
                kernel, "get_signature"
            ), f"{kernel} must have get_signature implemented"
            signature = kernel.get_signature()
            self.prefix.writeline(f'extern "C" {signature};')

        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = OrderedSet(self.src_to_kernel.values()) - OrderedSet(
            self.initialized_kernels.keys()
        )
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        for kernel in sorted(declare_kernel):
            self.prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"    {self.device_codegen.cpp_kernel_type()} {kernel}{{nullptr}};"
                )
            )
        for name, kernel in self.initialized_kernels.items():
            assert hasattr(
                kernel, "get_signature"
            ), f"{kernel} must have get_signature implemented"
            kernel_ptr = f"(*{name})"
            signature = kernel.get_signature().replace(name, kernel_ptr)
            self.prefix.writeline(f"    {signature} = torch::aot_inductor::{name};")
        self.prefix.writeline("};")
        self.prefix.writeline("}  // namespace")

    def codegen_model_constructor(self):
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """

        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        include_weights = (
            "true" if config.aot_inductor.package_constants_in_so else "false"
        )
        self.prefix.splice(
            f"""
            AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                               std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                               const std::string& device_str,
                                               std::optional<std::string> cubin_dir,
                                               bool include_weights)
                : AOTInductorModelBase({num_inputs}, {num_outputs}, {num_constants}, device_str, cubin_dir, {include_weights}) {{
            """
        )

        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                assert not isinstance(
                    inp, sympy.Expr
                ), f"input {name=} cannot be symbolic"
                self.write_input_output_info("inputs_info_", idx, name)

            all_cuda = all(
                V.graph.get_original_value_of_constant(name).is_cuda
                for name in V.graph.constants.keys()
                if name not in V.graph.folded_constants
            )
            for idx, name in enumerate(V.graph.constants.keys()):
                tensor = V.graph.get_original_value_of_constant(name)
                assert isinstance(tensor, torch.Tensor)
                self.prefix.writeline(f"""constants_info_[{idx}].name = "{name}";""")
                self.prefix.writeline(
                    f"constants_info_[{idx}].dtype = static_cast<int32_t>({self.codegen_dtype(tensor.dtype)});"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].offset = {tensor.storage_offset()};"
                )

                # If constants to serialize contain cpu tensors, we always align data_size it to 64.
                # When loading the constants, the valid data will depends on the size
                # not the data_size so there won't be correctness issue.
                data_size = (
                    torch.ops.mkldnn._nbytes(tensor)
                    if tensor.is_mkldnn
                    else tensor.untyped_storage().nbytes()
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].data_size = {data_size if all_cuda else _align(data_size)};"
                )

                from_folded = "true" if name in V.graph.folded_constants else "false"
                self.prefix.writeline(
                    f"constants_info_[{idx}].from_folded = {from_folded};"
                )

                if name in V.graph.folded_constants:
                    constant_type_str = "FoldedConstant"
                elif name.startswith("_tensor_constant"):
                    constant_type_str = "TensorConstant"
                elif any(
                    name == normalize_name(parameter_name)
                    for parameter_name, _ in V.graph.orig_gm.named_parameters()
                ):
                    constant_type_str = "Parameter"
                elif any(
                    name == normalize_name(buffer_name)
                    for buffer_name, _ in V.graph.orig_gm.named_buffers()
                ):
                    constant_type_str = "Buffer"
                else:
                    constant_type_str = "Unknown"
                self.prefix.writeline(
                    f"constants_info_[{idx}].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::{constant_type_str});"
                )

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].layout = static_cast<int32_t>({self.codegen_layout(tensor.layout)});"
                )

                if tensor.is_mkldnn:
                    opaque_metadata_tensor = torch.ops.mkldnn._get_mkldnn_serialized_md(
                        tensor
                    )
                    assert (
                        opaque_metadata_tensor.dim() == 1
                    ), "Expect opaque_metadata_tensor to be 1-D"

                    opaque_metadata_list = opaque_metadata_tensor.tolist()
                    opaque_metadata_str = self.codegen_shape_tuple(opaque_metadata_list)
                    self.prefix.writeline(
                        f"constants_info_[{idx}].opaque_metadata = {opaque_metadata_str};"
                    )
                if name in V.graph.dynamo_flat_name_to_original_fqn:
                    original_fqn = V.graph.dynamo_flat_name_to_original_fqn.get(
                        name, name
                    )
                elif name in V.graph.allocated_constant_name:
                    original_fqn = V.graph.allocated_constant_name[name]
                else:
                    raise AssertionError("original_fqn must be set for constant")
                self.prefix.writeline(
                    f"""constants_info_[{idx}].original_fqn = "{original_fqn}";"""
                )
            self.prefix.writeline("update_constants_map(std::move(constants_map));")
            self.prefix.writeline("update_constants_array(std::move(constants_array));")

            def escape_string(x):
                return (
                    x.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )

            self.prefix.writeline(
                f'in_spec_ = "{escape_string(config.aot_inductor.serialized_in_spec)}";'
            )
            self.prefix.writeline(
                f'out_spec_ = "{escape_string(config.aot_inductor.serialized_out_spec)}";'
            )

            for idx, output in enumerate(V.graph.graph_outputs):
                assert not isinstance(
                    output, sympy.Expr
                ), f"output {name=} cannot be symbolic"
                name = f"output{idx}"
                self.write_input_output_info("outputs_info_", idx, name)

            self.prefix.writeline(
                "this->kernels_ = std::make_unique<AOTInductorModelKernels>();"
            )

        self.prefix.writeline("}")

    def codegen_const_run_driver(self):
        """
        // Generated code example
        std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
            DeviceStreamType stream,
            AOTIProxyExecutorHandle proxy_executor,
            bool initialization
        ) {
            std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
            std::vector<AtenTensorHandle> output_handles;
            // build up output_handles over here.
            _const_run_impl(output_handles, stream, proxy_executor);
            // build up folded_constants_map
            return folded_constants_map;
        }
        """

        self.prefix.splice(
            """
            std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
                DeviceStreamType stream,
                AOTIProxyExecutorHandle proxy_executor,
                bool initialization
            ) {
            """
        )
        if not config.aot_inductor.use_runtime_constant_folding:
            self.prefix.splice(
                """
                    if (!initialization) {
                        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                                  << "aot_inductor.use_runtime_constant_folding=False\\n";
                    }
                    return {};
                }
                """
            )
            return

        with self.prefix.indent():
            # This is a mapping to the index of constant folding graph's output
            const_index_mapping: List[Optional[Tuple[int, str]]] = [None] * len(
                V.graph.const_output_index
            )
            for idx, (name, _) in enumerate(V.graph.constants.items()):
                if name in V.graph.const_output_index:
                    const_index_mapping[V.graph.const_output_index[name]] = (idx, name)  # type: ignore[call-overload]
            assert (
                None not in const_index_mapping
            ), "Not all constant gets mapped for constant folding graph."

            self.prefix.writeline(
                f"""
                std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
                folded_constants_map.reserve({len(const_index_mapping)});
                std::vector<AtenTensorHandle> output_handles({len(const_index_mapping)});
                """
            )

            self.prefix.splice(
                """
                // The below assignment of output_handles to constants is not used directly.
                // It's only used to memo the correspondence of handle and constants.
                """
            )

            for output_idx, (const_idx, _) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f"output_handles[{output_idx}] = constants_->at({const_idx});"
                )

            self.prefix.writeline(
                "_const_run_impl(output_handles, stream, proxy_executor);"
            )

            for output_idx, (_, const_name) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f'folded_constants_map["{const_name}"] = output_handles[{output_idx}];'
                )
            self.prefix.writeline("return folded_constants_map;")

        self.prefix.writeline("}")

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperCpu.generate", log_pt2_compile_event=True):
            if V.graph.aot_mode and not V.graph.is_const_graph:
                self.codegen_model_kernels()
                self.codegen_model_constructor()
                self.codegen_const_run_driver()
            self.write_wrapper_decl()
            return super().generate(is_inference)

    def finalize_prefix(self):
        cached_dtypes_buffer = IndentedBuffer()
        for dtype in self.used_cached_dtypes:
            cached_dtypes_buffer.writeline(f"CACHE_TORCH_DTYPE({dtype});")
        for device in self.used_cached_devices:
            cached_dtypes_buffer.writeline(f"CACHE_TORCH_DEVICE({device});")
        for layout in self.used_cached_layouts:
            cached_dtypes_buffer.writeline(f"CACHE_TORCH_LAYOUT({layout});")
        for memory_format in self.used_cached_memory_formats:
            cached_dtypes_buffer.writeline(
                f"CACHE_TORCH_MEMORY_FORMAT({memory_format});"
            )
        cached_dtypes_buffer.splice(self.prefix)
        self.prefix = cached_dtypes_buffer

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu=False,
    ):
        self.header.splice(f"\n{kernel_body}\n")

    def codegen_scalar_to_tensor(self, output: str):
        name = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
        self.wrapper_call.writeline(
            f"RAIIAtenTensorHandle {name} = scalar_to_tensor_handle({output});"
        )
        return name

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )

    @cache_on_self
    def get_output_refs(self):
        return [x.codegen_reference(self.wrapper_call) for x in V.graph.graph_outputs]

    def generate_return(self, output_refs: List[str]):
        cst_names = V.graph.constants.keys()
        output2idx: Dict[str, int] = {}
        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            if is_constant_buffer:
                # See NOTE(return_constant) above.
                self.wrapper_call.writeline(
                    f"aoti_torch_clone({output}, &output_handles[{idx}]);"
                )
            else:
                if output in output2idx:
                    src_idx = output2idx[output]
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = output_handles[{src_idx}];"
                    )
                else:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = {output}.release();"
                    )

            if output not in output2idx:
                output2idx[output] = idx

    def generate_before_suffix(self, result):
        if not V.graph.is_const_graph:
            if V.graph.aot_mode:
                result.writeline("} // AOTInductorModel::run_impl")
            else:
                result.writeline("} // inductor_entry_impl")

    def generate_end(self, result):
        if V.graph.aot_mode:
            if V.graph.is_const_graph:
                result.writeline("} // AOTInductorModel::_const_run_impl")
            else:
                result.writeline("} // namespace torch::aot_inductor\n\n\n")
            return

        # cpp entry function for JIT with cpp wrapper
        result.splice(
            f"""
            '''
            )

            inductor_entry = CppWrapperCodeCache.load_pybinding(
                ["std::vector<AtenTensorHandle>"], cpp_wrapper_src, "{self.device}", {len(V.graph.graph_outputs)})
            """
        )

        wrapper_body = "input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]"
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_impl function explicitly.
            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    input_tensors.extend(constants_tensor)
            """
        # Convert vector of at::Tensor to vector of AtenTensorHandle.
        # If we pass at::Tensor, the compilation will be too slow.
        wrapper_body += """
                    input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
        """
        # Release the inputs for memory reuse.
        wrapper_body += """
                    args.clear()
        """

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            outputs_str = "output_tensors"
        else:
            outputs = [
                (
                    f"output_tensors[{i}]"
                    if self.output_is_tensor[i]
                    else f"output_tensors[{i}].item()"
                )
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """
        )

    def get_c_shim_func_name(self, kernel):
        if kernel.startswith("aoti_torch_"):
            return kernel

        assert "::" in kernel, "Cpp kernel name: " + kernel + " does not contain '::'"
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]

        shim_fn = f"aoti_torch_{self.device}_{kernel_suffix}"
        return shim_fn

    def generate_c_shim_extern_kernel_call(self, kernel, args):
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
        with debug_printer_manager:
            shim_fn = self.get_c_shim_func_name(kernel)
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(args)}));"
            )

    def generate_c_shim_extern_kernel_alloc(self, extern_kernel, args):
        # registered output buffer name
        name = extern_kernel.name
        output_handle_name = f"{name}_handle"
        self.writeline(f"AtenTensorHandle {output_handle_name};")
        output_arg = f"&{output_handle_name}"
        self.generate_c_shim_extern_kernel_call(
            extern_kernel.get_kernel_name(), args + [output_arg]
        )
        self.writeline(f"RAIIAtenTensorHandle {name}({output_handle_name});")

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        if getattr(extern_kernel, "outputs", None):
            # ir.ExternKernelAlloc may have outputs if it returns a tuple
            self.generate_c_shim_fallback_kernel(extern_kernel, args)
        else:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(self, fallback_kernel, args):
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: handle integer output (e.g., as in attention)
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    assert (
                        output.indices[0][1] == idx
                    ), f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Expr):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {cexpr(output)};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError(f"unsupported type of {output=}")
        args = args + output_args
        self.generate_c_shim_extern_kernel_call(fallback_kernel.cpp_kernel_name, args)
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def generate_fallback_kernel(self, fallback_kernel, args):
        self.generate_c_shim_fallback_kernel(fallback_kernel, args)

    def generate_extern_kernel_out(
        self, kernel: str, out: str, out_view: Optional[str], args: List[str]
    ):
        if out_view:
            out_name = f"{out}_as_strided"
            self.writeline(f"auto {out_name} = {out_view};")
            args.insert(0, out_name)
        else:
            args.insert(0, out)

        self.generate_c_shim_extern_kernel_call(kernel, args)

    def generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
    ):
        # call the ABI shim function instead of the ATen one
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name)
        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        inputs_wrapped = [str(x) for x in inputs]
        line = f"{cpp_kernel_name}({output}, {','.join(inputs_wrapped)}"

        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        else:
            if src_is_tensor:
                if reduce:
                    line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
            else:
                assert (
                    reduce is None
                ), "Expect reduce to be None for aten.scatter_ with scalar src"
        line += ");"
        self.writeline(line)

    def generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the correponding
        # tensor prematurely deallocated, thus this std::vector().data() trick here.
        indices_str = (
            "std::vector<AtenTensorHandle>{" + (", ".join(indices)) + "}.data()"
        )
        args = [
            x,
            indices_str,
            str(len(indices)),
            values,
            accumulate,
        ]
        args.insert(0, x)  # set x as the output tensor, this fallback mutates x.
        self.writeline(self.wrap_kernel_call(kernel, args))

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def codegen_cpp_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        return cexpr(V.graph.sizevars.simplify(x) if simplify else x)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_cpp_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        # in the abi_compatible mode, outputs are returned via arguments
        return name

    def codegen_shape_tuple(self, shape: Sequence[Expr]) -> str:
        parts = [*map(self.codegen_sizevar, shape)]
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(f"int64_t {sym} = {cexpr(expr)};")

    def generate_numel_expr(self, kernel_name: str, tree, suffix: Optional[str] = None):
        expr = f"{kernel_name}_{tree.prefix}numel"
        if suffix is not None:
            expr += f"_{suffix}"
        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(f"int64_t {expr} = {cexpr(tree.numel)};")
        else:
            self.writeline(f"{expr} = {cexpr(tree.numel)};")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, tree.numel)

    def prepare_triton_kernel_call(self, device_index, call_args):
        def wrap_arg(arg):
            if isinstance(arg, str):
                # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                return arg + ".item()" if should_unwrap_unspec_arg(arg) else arg
            elif isinstance(arg, (int, float, bool, SymbolicCallArg)):
                return str(arg)
            else:
                return cexpr(V.graph.sizevars.simplify(arg))

        call_args = [wrap_arg(arg) for arg in call_args]

        if device_index is None:
            current_device = V.graph.get_current_device_or_throw()
            device_index = current_device.index

        return device_index, call_args

    def codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        self.codegen_tensor_item(node.inputs[0].get_dtype(), data, f"{node.sym}_raw")

        if len(node.keypath) == 0:
            self.writeline(f"auto {node.sym} = {node.sym}_raw;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"int64_t {node.sym} = {node.sym}_raw ? 1 : 0;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # TODO: assert divisibility here
            self.writeline(
                f"int64_t {node.sym} = {node.sym}_raw / {node.keypath[0].divisor};"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")

        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(node.sym))

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or isinstance(buffer, ir.TMADescriptor)
            else f"{buffer.get_name()}.reset();"
        )

    def make_free_by_names(self, names_to_del: List[str]):
        return " ".join(f"{name}.reset();" for name in names_to_del)

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"auto {new_name} = std::move({old_name});  // reuse"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>());'
        )

    def generate_start_graph(self):
        pass

    def generate_end_graph(self):
        pass

    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            # TODO: Add buf name directly into check_inf_and_nan.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan({buf}));"
            )

    def codegen_device(self, device):
        assert device.type in DEVICE_TO_ATEN, (
            device.type + " not found in DEVICE_TO_ATEN"
        )
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index if device.index else 0}"

    def codegen_dtype(self, dtype):
        dtype_str = str(dtype).split(".")[-1]
        self.used_cached_dtypes.add(dtype_str)
        return f"cached_torch_dtype_{dtype_str}"

    def codegen_layout(self, layout):
        layout_str = str(layout).split(".")[-1]
        self.used_cached_layouts.add(layout_str)
        return f"cached_torch_layout_{layout_str}"

    def codegen_memory_format(self, memory_format):
        memory_format_str = str(memory_format).split(".")[-1]
        self.used_cached_memory_formats.add(memory_format_str)
        return f"cached_torch_memory_format_{memory_format_str}"

    @functools.lru_cache(None)  # noqa: B019
    def codegen_int_array_var(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically=False,
        graph=None,  # for per-graph caching
    ):
        # Used for size/stride declaration
        #
        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        # As a result, the emitted int array declarations may appear in a later
        # position of the generated code, so the second pass codegen should not
        # reuse int array declarations generated in the first pass.
        # This is why writeline needs to explicitly passed in as a parameter.
        var = f"int_array_{next(self.int_array_id)}"
        ctype = "int64_t"
        if var not in self.declared_int_array_vars:
            self.declared_int_array_vars.add(var)
            if known_statically:
                writeline(f"static constexpr {ctype} {var}[] = {int_array};")
            else:
                writeline(f"const {ctype} {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
        )

    def make_allocation(self, name, device, dtype, shape, stride):
        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(orig_stride)
        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )
        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(orig_stride),
            graph=self.get_codegened_graph(),
        )
        device_type, device_id = device_str.split(",")
        device_idx = "this->device_idx_" if V.graph.aot_mode else device_id

        args = [
            str(len(shape)),
            size_array_var,
            stride_array_var,
            dtype_code,
            device_type,
            device_idx,
            f"&{name}_handle",
        ]

        self.wrapper_call.writeline(f"AtenTensorHandle {name}_handle;")
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));"
        )

        return f"RAIIAtenTensorHandle {name}({name}_handle);"

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
        args = [
            name,
            cexpr(offset),  # bytes not numel
            self.codegen_dtype(dtype),
            str(len(shape)),
            self.codegen_int_array_var(
                size, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            self.codegen_int_array_var(
                stride, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            f"&{tmp_name}",
        ]
        self.wrapper_call.writeline(f"AtenTensorHandle {tmp_name};")
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));"
        )
        return f"RAIIAtenTensorHandle({tmp_name})"

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        dim = str(len(size))
        original_offset = offset
        offset = self.codegen_sizevar(offset)
        call_strs = []
        final_tmp_name = None

        def create_reinterpret_call() -> Tuple[str, str]:
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(size),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(size),
                    graph=self.get_codegened_graph(),
                ),
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(stride),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(stride),
                    graph=self.get_codegened_graph(),
                ),
                offset,
            ]
            call_str = (
                f"auto {tmp_name} = reinterpret_tensor_wrapper({', '.join(args)});"
            )
            return tmp_name, call_str

        def create_dtypeview_call(reinterpret_call: str) -> Tuple[str, List[str]]:
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            call_strs = [f"AtenTensorHandle {tmp_AtenTensorHandle};"]
            dtype_name = str(dtype).split(".")[-1]
            device_name = data.layout.device.type
            get_dtype_function = f"aoti_torch_dtype_{dtype_name}"
            dtypeview_function = f"aoti_torch_{device_name}_view_dtype"
            call_strs.append(
                f"AOTI_TORCH_ERROR_CODE_CHECK({dtypeview_function}"
                f"({reinterpret_call}, {get_dtype_function}(), &{tmp_AtenTensorHandle}));"
            )
            tmp_RAIIAtenTensorHandle = (
                f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}_handle"
            )
            call_strs.append(
                f"RAIIAtenTensorHandle {tmp_RAIIAtenTensorHandle}({tmp_AtenTensorHandle});"
            )
            return tmp_RAIIAtenTensorHandle, call_strs

        if (
            size == data.layout.size
            and stride == data.layout.stride
            and original_offset == data.layout.offset
        ):
            # pure dtypeview
            if dtype is not None and dtype != data.dtype:
                tmp_output_name, tmp_call_strs = create_dtypeview_call(data.get_name())
                call_strs.extend(tmp_call_strs)
                final_tmp_name = tmp_output_name
            else:
                return data.get_name()
        else:
            # firstly create reinterpretview
            final_tmp_name, reinterpret_call = create_reinterpret_call()
            call_strs.append(reinterpret_call)

            if dtype is not None and dtype != data.dtype:
                # wrap it with dtypeview
                final_tmp_name, tmp_call_strs = create_dtypeview_call(final_tmp_name)
                call_strs.extend(tmp_call_strs)
            else:
                call_strs.append(
                    f"RAIIAtenTensorHandle {final_tmp_name}_raii({final_tmp_name});"
                )
                final_tmp_name = f"{final_tmp_name}_raii"

        for line in call_strs:
            writeline(line)

        # NB, the return handle here represents a temporary tensor, which will be automatically
        # released.
        # Here's a sample usage in the cpp wrapper code:
        # ```
        # aoti_torch_addmm_out(
        #     buf1,
        #     arg1_1,
        #     RAIIAtenTensorHandle(tmp_tensor_handle_0),
        #     buf0,
        #     1L,
        #     1L));
        # ```
        # RAIIAtenTensorHandle(tmp_tensor_handle_0) will be released after the call to addmm_out.
        # This could be problematic when it's used in a different pattern, for example:
        # ````
        # AtenTensorHandle tensor_args[] = {RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6};
        # aoti_torch_proxy_executor_call_function(..., tensor_args);
        # ````
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) will be invalid when it's used in the latter
        # kernel call.
        #
        # This is solved by updating the proxy_executor invocation to
        # ```
        # aoti_torch_proxy_executor_call_function(...,
        #     std::vector<AtenTensorHandle>{
        #         RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6
        #     }.data()
        # );
        # ```
        return final_tmp_name

    def codegen_device_copy(self, src, dst, non_blocking: bool):
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(expensive_copy_to_tensor_if_needed({dst}), {src}, {non_blocking}));"
        )

    def codegen_multi_output(self, name, value):
        # in the abi_compatible mode, outputs are retrieved by passing
        # output pointers, so we skip its codegen here.
        pass

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        assert len(subgraph.graph.graph_inputs) == len(outer_inputs)

        for (inner_input, inner_input_val), outer_input in zip(
            subgraph.graph.graph_inputs.items(), outer_inputs
        ):
            if not isinstance(inner_input_val, ir.TensorBox):
                continue

            # in ABI-compatible mode, we copy the underlying at::Tensor of the conditional
            # input (outer_input) into another at::Tensor to be used as a subgraph input
            # (inner_input) in the nested scope. we can't std::move here, as the codegened
            # outer input may be an expression / rvalue (e.g., reinterpret_view(x)), so we
            # can't necessarily std::move it back to the origin (x).
            self.writeline(f"AtenTensorHandle {inner_input}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({outer_input}, &{inner_input}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);")

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            src = inner_output.codegen_reference()
            # in ABI-compatible mode, we need to std::move subgraph output (inner_output)
            # to the conditional output (outer_output), as RAIIAtenTensorHandle's copy
            # constructor is deleted.
            src = f"std::move({src})"
            # in case the outer_output carried a value
            # before (e.g., in the while_loop codegen)
            self.writeline(f"{outer_output}.reset();")
            self.writeline(f"{outer_output} = {src};")

    def codegen_invoke_subgraph(self, invoke_subgraph):
        raise NotImplementedError(
            "codegen invoke_subgraph is not implemented for cpp wrapper"
        )

    def codegen_conditional(self, conditional):
        outer_inputs = [f"{buf.codegen_reference()}" for buf in conditional.operands]
        outer_outputs = []
        for out in conditional.outputs:
            # in ABI-compatible mode, ir.MultiOutput is not codegened,
            # hence pre-declare output variables directly and separately
            self.writeline(f"RAIIAtenTensorHandle {out.get_name()};")
            outer_outputs.append(out.get_name())

        if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
            # in ABI-compatible mode, we need to use the ABI shim function
            # to extract a C++ bool from the unrelying scalar bool Tensor
            predicate = f"{conditional.predicate.get_name()}_scalar"
            if predicate not in self.used_cond_predicate:
                self.codegen_tensor_item(
                    torch.bool,
                    conditional.predicate.codegen_reference(),
                    predicate,
                )
                self.used_cond_predicate.add(predicate)
        else:
            # the predicate is not a Tensor: SymBool or Python bool
            predicate = conditional.predicate.codegen_reference()

        self.writeline(f"if ({predicate}) {{")
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("} else {")
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("}")

    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # PythonWrapperCode `codegen_subgraph` function. We should perhaps
        # support lifting of subgraphs as functions for cpp wrapper as well.
        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"// subgraph: {subgraph.name}")
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            self.pop_codegened_graph()

    def codegen_while_loop(self, while_loop):
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]
        cond_result_name = f"{name}_cond_result"
        self.writeline(f"RAIIAtenTensorHandle {cond_result_name};")

        cond_outer_inputs = []
        for inp, out in zip(outer_carried_inputs, while_loop.outputs):
            # in ABI-compatible mode, the carried inputs are codegened
            # as buffers outside the while loop and set to the initial
            # values. at the end of each while_loop iteration, they
            # will be assined the carried values.
            out_name = out.get_name()
            self.writeline(f"AtenTensorHandle {out_name}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({inp}, &{out_name}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {out_name}({out_name}_handle);")
            cond_outer_inputs.append(out_name)

        # additional inputs will be assinged within the while_loop
        # iteration directly from the corresponding outer graph buffers
        cond_outer_inputs.extend(outer_additional_inputs)

        cond_outer_outputs = [cond_result_name]
        body_outer_inputs = list(cond_outer_inputs)
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        self.writeline("while (1) {")
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
        self.codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )

        cond_result = f"{cond_result_name}_scalar"
        self.codegen_tensor_item(torch.bool, cond_result_name, cond_result)
        self.writeline(f"if (!{cond_result}) break;")

        self.writeline(ExitSubgraphLine(self))
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        self.codegen_subgraph(
            while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
        )
        self.writeline(ExitSubgraphLine(self))
        self.writeline("}")

    def generate_extern_kernel_args_decl_if_needed(
        self,
        op_overload,
        raw_args,
        output_args: Optional[List[str]] = None,
        raw_outputs: Optional[List[ir.Buffer]] = None,
    ):
        arg_types = [x.real_type for x in op_overload._schema.arguments]
        return_types = [x.type for x in op_overload._schema.returns]

        new_tensor_args = []
        new_int_args = []

        def fill_args(arg, arg_type):
            static_arg_types = (
                torch.FloatType,
                torch.BoolType,
                torch.StringType,
                torch.Type,
                torch.DeviceObjType,
            )
            inductor_tensor_buffers = (
                ir.Buffer,
                ir.ReinterpretView,
            )

            if isinstance(arg_type, torch.TensorType):
                assert isinstance(arg, inductor_tensor_buffers), f"got {type(arg)}"
                new_tensor_args.append(f"{arg.codegen_reference()}")
            elif isinstance(arg_type, torch.IntType):
                # int
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.SymIntType):
                # SymInt
                expr = arg.node.expr if isinstance(arg, torch.SymInt) else arg
                new_int_args.append(cexpr(expr))
            elif isinstance(arg_type, torch.NumberType):
                # Scalar of type int
                assert isinstance(arg, (int, float, bool))
                # Only treat int Scalar as dynamic
                if isinstance(arg, int):
                    new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.ListType):
                assert isinstance(arg, (list, tuple))

                # List[Tensor]
                if isinstance(arg_type.getElementType(), torch.TensorType):
                    new_tensor_args.extend([f"{a.codegen_reference()}" for a in arg])
                # List[Optional[Tensor]]
                elif isinstance(
                    arg_type.getElementType(), torch.OptionalType
                ) and isinstance(
                    arg_type.getElementType().getElementType(), torch.TensorType
                ):
                    new_tensor_args.extend(
                        [f"{a.codegen_reference()}" for a in arg if a is not None]
                    )
                # List[int]
                elif isinstance(arg_type.getElementType(), torch.IntType):
                    new_int_args.extend([str(a) for a in arg])
                # List[SymInt]
                elif isinstance(arg_type.getElementType(), torch.SymIntType):
                    expressions = [
                        a.node.expr if isinstance(a, torch.SymInt) else a for a in arg
                    ]
                    new_int_args.extend([cexpr(expr) for expr in expressions])
                # List[Scalar]
                elif isinstance(arg_type.getElementType(), torch.NumberType):
                    # Only treat int Scalar as dynamic
                    is_int_type = [isinstance(a, int) for a in arg]
                    if any(is_int_type):
                        assert all(
                            is_int_type
                        ), "AOTInductor only supports int scalars of the same type"
                        new_int_args.extend([str(a) for a in arg])
                else:
                    assert isinstance(
                        arg_type.getElementType(), static_arg_types  # type: ignore[arg-type]
                    ), f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
            else:
                assert isinstance(
                    arg_type, static_arg_types  # type: ignore[arg-type]
                ), f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"

        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(arg, return_type, is_mutated_output: bool):
            if isinstance(return_type, torch.TensorType):
                if not is_mutated_output:
                    self.writeline(f"AtenTensorHandle {arg}_handle;  // output buffer")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));"
                    )
                    self.writeline(f"RAIIAtenTensorHandle {arg}({arg}_handle);")
                new_tensor_args.append(f"{arg}")
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError("NYI support for return type: SymInt")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.SymIntType
            ):
                raise NotImplementedError("NYI support for return type: List[SymInt]")
            else:
                raise AssertionError(f"Unsupported return type found: {return_type}")

        # TODO: Only support tensor(s) returns for now, SymInt is not implemented yet
        for return_type in return_types:
            if isinstance(return_type, (torch.TensorType)):
                pass
            elif isinstance(return_type, torch.OptionalType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            elif isinstance(return_type, torch.ListType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            else:
                raise NotImplementedError(
                    f"return type {return_type} is not yet supported."
                )

        for output_arg, raw_output_arg in zip(output_args, raw_outputs):  # type: ignore[arg-type]
            assert output_arg is not None, "Optional return types are not yet supported"
            if isinstance(output_arg, (list, tuple)):
                for out in output_arg:
                    fill_output_arg(
                        out,
                        torch.TensorType.get(),
                        isinstance(raw_output_arg, ir.MutationOutput),
                    )
            else:
                fill_output_arg(
                    output_arg,
                    torch.TensorType.get(),
                    isinstance(raw_output_arg, ir.MutationOutput),
                )

        return new_tensor_args, new_int_args

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        def extract_output_name(out):
            if out is None:
                return None
            elif isinstance(out, (ir.MultiOutput, ir._CollectiveKernel)):
                return out.get_name()
            elif isinstance(out, ir.MutationOutput):
                mutated_buf_names = out.get_mutation_names()
                assert (
                    isinstance(mutated_buf_names, list) and len(mutated_buf_names) == 1
                ), "Expect only one mutated buffer in MutationOutput"
                return mutated_buf_names[0]
            elif isinstance(out, (list, tuple)):
                return type(out)(extract_output_name(o) for o in out)
            else:
                raise AssertionError(f"Unexpected output: {type(out)}")

        # output_args has the same pytree structure as outputs
        if op_overload and not op_overload._schema.returns:
            # kernel does not return a value
            output_args = []
        elif outputs is None:
            # outputs is not specified, the default is to write to buf_name
            output_args = [buf_name]
        else:
            output_args = extract_output_name(outputs)
            if isinstance(output_args, str):
                output_args = [output_args]

        if V.graph.aot_mode:
            assert op_overload is not None
            assert raw_args is not None
            assert output_args is not None

            return self.generate_fallback_kernel_with_runtime_lookup_aot(
                op_overload,
                raw_args,
                output_args,
                outputs,
            )
        else:
            return self.generate_fallback_kernel_with_runtime_lookup_jit(
                buf_name,
                python_kernel_name,
                cpp_kernel_name,
                codegen_args,
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

    def generate_scoped_gil_acquire(self, declarations_before_scope, lines_in_scope):
        scoped_lines = IndentedBuffer()
        for declaration in declarations_before_scope:
            scoped_lines.writeline(declaration)

        scoped_lines.writeline("{")
        with scoped_lines.indent():
            scoped_lines.writeline("py::gil_scoped_acquire acquire;")
            scoped_lines.writelines(lines_in_scope.split("\n"))
        scoped_lines.writelines("}")
        return scoped_lines._lines

    def load_custom_op_wrapper(self):
        # TODO: need to support control flow
        if self.custom_op_wrapper_loaded:
            return

        lines = """
RAIIPyObject codecache_module(PyImport_ImportModule("torch._inductor.codecache"));
if (codecache_module.get() == NULL) {
    throw std::runtime_error("Failed to load torch._inductor.codecache");
}
custom_op_wrapper = PyObject_GetAttrString(codecache_module, "custom_op_wrapper");
if (custom_op_wrapper.get() == NULL) {
    throw std::runtime_error("Failed to load torch._inductor.codecache.custom_op_wrapper");
}"""

        declarations_before_scope = ["RAIIPyObject custom_op_wrapper;"]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

        self.custom_op_wrapper_loaded = True

    def generate_float_value(self, val):
        assert isinstance(val, float)
        if val == float("inf"):
            return "std::numeric_limits<float>::infinity()"
        elif val == float("-inf"):
            return "-std::numeric_limits<float>::infinity()"
        elif val == float("nan"):
            return "std::numeric_limits<float>::quiet_NaN()"
        else:
            return f"{val}"

    def generate_py_arg(self, py_args_var, idx, raw_arg, arg_type):
        def generate_py_arg_inner(lines, raw_arg, arg_type):
            def add_py_newref():
                if sys.version_info < (3, 10):
                    # Py_NewRef is only available since Python 3.10
                    self.include_extra_header("torch/csrc/utils/pythoncapi_compat.h")

            if raw_arg is None:
                # Py_None is a singleton, so we have to explicitly incref it here
                lines.append("Py_INCREF(Py_None);\n")
                return "Py_None"
            elif isinstance(arg_type, torch.TensorType):
                # Store AtenTensorHandle as void*
                base_handle = raw_arg.codegen_reference()
                (
                    tmp_raii_handle_var,
                    tmp_raii_handle_var_decl,
                ) = self.create_tmp_raii_handle_var(base_handle)
                if tmp_raii_handle_var:
                    lines.append(tmp_raii_handle_var_decl)
                    base_handle = tmp_raii_handle_var
                return f"PyCapsule_New(reinterpret_cast<void*>({base_handle}.get()), NULL, NULL)"
            elif isinstance(arg_type, torch.OptionalType):
                return generate_py_arg_inner(lines, raw_arg, arg_type.getElementType())
            elif isinstance(arg_type, torch.IntType):
                # int
                return f"PyLong_FromLongLong({raw_arg})"
            elif isinstance(arg_type, torch.SymIntType):
                # SymInt
                expr = (
                    raw_arg.node.expr if isinstance(raw_arg, torch.SymInt) else raw_arg
                )
                return f"PyLong_FromLongLong({cexpr(expr)})"
            elif isinstance(arg_type, torch.FloatType):
                return f"PyFloat_FromDouble({self.generate_float_value(raw_arg)})"
            elif isinstance(arg_type, torch.BoolType):
                return f"PyBool_FromLong({1 if raw_arg else 0})"
            elif isinstance(arg_type, torch.StringType):
                return f'PyUnicode_FromString("{raw_arg}")'
            elif isinstance(arg_type, torch.NumberType):
                # Union[bool, int, float, complex]
                # torch/_prims_common/__init__.py
                if isinstance(raw_arg, int):
                    return f"PyLong_FromLongLong({raw_arg})"
                elif isinstance(raw_arg, float):
                    return f"PyFloat_FromDouble({self.generate_float_value(raw_arg)})"
                elif isinstance(raw_arg, bool):
                    return f"PyBool_FromLong({1 if raw_arg else 0})"
                elif isinstance(raw_arg, complex):
                    return f"PyComplex_FromDoubles({raw_arg.real}, {raw_arg.imag})"
                elif isinstance(raw_arg, torch.SymInt):
                    expr = raw_arg.node.expr
                    return f"PyLong_FromLongLong({cexpr(expr)})"
                else:
                    raise NotImplementedError(
                        f"arg type {arg_type} with raw_arg {raw_arg}, {type(raw_arg)} is not yet supported by custom_op_wrapper"
                    )
            elif isinstance(raw_arg, torch.device):
                # device
                self.include_extra_header("torch/csrc/Device.h")
                device_str, device_index = self.codegen_device(raw_arg).split(", ")
                return f"THPDevice_New(c10::Device(static_cast<c10::DeviceType>({device_str}), {device_index}))"
            elif isinstance(raw_arg, torch.dtype):
                # dtype
                add_py_newref()
                self.include_extra_header("torch/csrc/DynamicTypes.h")
                return f"Py_NewRef(torch::getTHPDtype(static_cast<c10::ScalarType>({self.codegen_dtype(raw_arg)})))"
            elif isinstance(raw_arg, torch.layout):
                # memory layout
                add_py_newref()
                self.include_extra_header("torch/csrc/DynamicTypes.h")
                return f"Py_NewRef(torch::getTHPLayout(static_cast<c10::Layout>({self.codegen_layout(raw_arg)})))"
            elif isinstance(raw_arg, torch.memory_format):
                # memory_format
                add_py_newref()
                self.include_extra_header("torch/csrc/utils/tensor_memoryformats.h")
                return (
                    "Py_NewRef(torch::utils::getTHPMemoryFormat(static_cast<c10::MemoryFormat>("
                    f"{self.codegen_memory_format(raw_arg)})))"
                )
            else:
                raise NotImplementedError(
                    f"arg type {arg_type} is not yet supported by custom_op_wrapper"
                )

        lines = []
        if isinstance(arg_type, torch.ListType):
            assert isinstance(raw_arg, (list, tuple)), str(raw_arg) + " is not a list"
            lines.append(
                f"PyObject* {py_args_var}_{idx} = PyList_New({len(raw_arg)});\n"
            )
            for i, elem in enumerate(raw_arg):
                lines.append(
                    f"PyList_SetItem({py_args_var}_{idx}, {i}, {generate_py_arg_inner(lines, elem, arg_type.getElementType())});\n"
                )
            lines.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {py_args_var}_{idx});\n"
            )
        else:
            lines.append(
                f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
            )
        return "".join(lines)

    def generate_fallback_kernel_with_runtime_lookup_jit(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        output_args: Optional[List[str]] = None,
        raw_outputs: Optional[List[ir.Buffer]] = None,
    ):
        # In the JIT mode, because of the ABI-compatible requirement, we can't directly call
        # c10::Dispatcher to find the custom op and call it. Instead, we go back to Python
        # to invoke this custom op.
        self.load_custom_op_wrapper()

        assert output_args is not None, "output_args should not be None"
        num_args = len(raw_args)
        py_args_var = f"py_args_{next(self.arg_var_id)}"
        # First arg is always the python op name
        lines = f"""
RAIIPyObject {py_args_var}(PyTuple_New({num_args + 1}));
if ({py_args_var}.get() == NULL) {{
throw std::runtime_error("PyTuple_New {py_args_var} failed");
}}
PyTuple_SetItem({py_args_var}, 0, PyUnicode_FromString("{python_kernel_name}"));
"""

        assert op_overload is not None, "op_overload should not be None"

        for idx, (raw_arg, schema_arg) in enumerate(
            zip(raw_args, op_overload._schema.arguments)
        ):
            lines += self.generate_py_arg(
                py_args_var, idx + 1, raw_arg, schema_arg.real_type
            )

        lines += f"""
// Call the custom op in Python
RAIIPyObject py_{buf_name}(PyObject_CallObject(custom_op_wrapper, {py_args_var}));
if (py_{buf_name}.get() == NULL) {{
if (PyErr_Occurred()) {{
return;
}}
throw std::runtime_error("PyObject_CallObject {python_kernel_name} failed");
}}"""

        if len(output_args) == 1:
            # result is a single tensor
            lines += f"""
{output_args[0]} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(py_{buf_name}.get(), NULL));"""
        else:
            # result is a tuple of tensors
            for idx, output_arg in enumerate(output_args):
                if output_arg is None:
                    continue
                lines += f"""
{output_arg} =
reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(PyList_GET_ITEM(py_{buf_name}.get(), {idx}), NULL));"""

        if raw_outputs:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
                if output_arg is not None
                and not isinstance(raw_output_arg, ir.MutationOutput)
            ]
        else:
            declarations_before_scope = [
                f"RAIIAtenTensorHandle {output_arg};"
                for output_arg in output_args  # type: ignore[arg-type]
                if output_arg is not None
            ]
        scope_gil_acquire = self.generate_scoped_gil_acquire(
            declarations_before_scope, lines
        )
        self.writelines(scope_gil_acquire)

    def generate_fallback_kernel_with_runtime_lookup_aot(
        self,
        op_overload,
        raw_args,  # contains both args and flatten kwargs
        output_args: Optional[List[str]] = None,
        raw_outputs: Optional[List[ir.Buffer]] = None,
    ):
        (
            tensor_call_args,
            int_call_args,
        ) = self.generate_extern_kernel_args_decl_if_needed(
            op_overload,
            raw_args,
            output_args,
            raw_outputs,
        )

        tensor_call_args_str = ", ".join(tensor_call_args)
        int_call_args_str = ", ".join(int_call_args)

        extern_kernel_node_index = len(V.graph.extern_kernel_nodes) - 1

        self.writeline(
            f"aoti_torch_proxy_executor_call_function(proxy_executor, "
            f"{extern_kernel_node_index}, "
            f"{len(int_call_args)}, "
            f"std::vector<int64_t>{{{int_call_args_str}}}.data(), "
            f"{len(tensor_call_args)}, "
            f"std::vector<AtenTensorHandle>{{{tensor_call_args_str}}}.data());"
        )

    def generate_reset_kernel_saved_flags(self):
        pass

    def generate_save_uncompiled_kernels(self):
        pass

    def c_type_for_prim_type(self, val, type_) -> str:
        if isinstance(type_, torch.OptionalType):
            return f"{self.c_type_for_prim_type(val, type_.getElementType())}*"
        elif isinstance(type_, torch.TensorType):
            return "AtenTensorHandle"
        elif isinstance(type_, (torch.IntType, torch.SymIntType)):
            return "int64_t"
        elif isinstance(
            type_, (torch.BoolType, torch.SymBoolType, torch.EnumType)
        ) or repr(type_) in ("ScalarType", "Layout"):
            return "int32_t"
        elif isinstance(type_, torch.FloatType):
            return "double"
        elif isinstance(type_, torch.NumberType):
            if isinstance(val, bool):
                return "int32_t"
            elif isinstance(val, (int, float)):
                return "double"
            elif val is None:
                # This could happen when val is an optional value
                return "double"
            else:
                raise AssertionError(
                    f"Unexpected type in c_type_for_prim_type: {type_=}"
                )
        elif isinstance(type_, torch.StringType):
            return "const char*"
        else:
            raise AssertionError(f"Unexpected type in c_type_for_prim_type: {type_=}")

    def val_to_arg_str_for_prim_type(self, val, type_) -> str:
        # TODO: not using type_ as the first step of refactoring. Will update this later.
        if isinstance(val, bool):
            return "1" if val else "0"
        elif isinstance(val, int):
            # uint64_t is long on Linux, but long long on MacOS and Windows
            return f"{val}LL" if sys.platform in ["darwin", "win32"] else f"{val}L"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(
            val, (ir.Buffer, ir.ReinterpretView, ir.StorageBox, ir.TensorBox)
        ):
            return val.codegen_reference()
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        elif isinstance(val, torch.layout):
            return self.codegen_layout(val)
        elif isinstance(val, torch.memory_format):
            return self.codegen_memory_format(val)
        elif isinstance(val, float):
            return self.generate_float_value(val)
        elif isinstance(val, (list, tuple)):
            # FIXME: This happens because type_ is not always properly set to torch.ListType
            return f"{{{', '.join(self.val_to_arg_str(x, None) for x in val)}}}"
        elif isinstance(val, SymTypes):
            return cexpr(val.node.expr)
        elif isinstance(val, sympy.Expr):
            return cexpr(val)
        else:
            return repr(val)

    def val_to_arg_str(self, val, type_=None) -> str:
        if val is None:
            # None needs special care. It either represent nullopt or an empty tensor
            if type_ is None or isinstance(type_, torch.OptionalType):
                if type_ is not None and isinstance(
                    type_.getElementType(),
                    (
                        torch.ListType,
                        torch.TupleType,
                        torch.DeviceObjType,
                    ),
                ):
                    return "0, 0"
                else:
                    return "0"  # nullptr is not available in C
            elif isinstance(type_, torch.TensorType):
                # create an empty tensor, the equivalent of at::Tensor()
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"AtenTensorHandle {var_name}_handle;")
                self.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{var_name}_handle));"
                )
                self.writeline(f"RAIIAtenTensorHandle {var_name}({var_name}_handle);")
                return var_name
            else:
                raise AssertionError("Can not map None to a known data type")

        if isinstance(type_, torch.OptionalType):
            element_type = type_.getElementType()
            if not isinstance(element_type, torch.TensorType):
                var_name = f"var_{next(self.arg_var_id)}"
                if isinstance(
                    element_type,
                    (torch.ListType, torch.TupleType, torch.DeviceObjType),
                ):
                    # type_ is something like Optional[List] or Optional[Device]
                    arg_str = self.val_to_arg_str(val, element_type)
                    # For datatypes with auxiliary info, we need to hoist out the extra arguments.
                    # NOTE: This only works if there is one additional argument, though it can easily be generalized.
                    main_value, aux = arg_str.rsplit(", ")
                    self.writeline(f"auto {var_name} = {main_value};")
                    return f"&{var_name}, {aux}"
                else:
                    self.writeline(
                        f"{self.c_type_for_prim_type(val, element_type)} {var_name} = {self.val_to_arg_str(val, element_type)};"
                    )
                    return f"&{var_name}"
            else:
                # type_ is Optional[Tensor]
                # Similar to other data type, use pointer to denote optional tensor arg in v2 C shim
                base_handle = self.val_to_arg_str(val, element_type)
                (
                    tmp_raii_handle_var,
                    tmp_raii_handle_var_decl,
                ) = self.create_tmp_raii_handle_var(base_handle)
                if tmp_raii_handle_var:
                    self.writeline(tmp_raii_handle_var_decl)
                    base_handle = tmp_raii_handle_var
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"AtenTensorHandle {var_name} = {base_handle}.get();")
                return f"&{var_name}"

        elif isinstance(type_, torch.ListType):
            assert isinstance(
                val, (list, tuple)
            ), f"{val} does not match with arg type {type_}"
            element_type = type_.getElementType()
            var_name = f"var_array_{next(self.var_array_id)}"
            if len(val) == 0:
                # Zero-size array is not supported in the C or C++ standard, so
                # we declare a null pointer for it.
                self.writeline(
                    f"const {self.c_type_for_prim_type(None, element_type)}* {var_name} = nullptr;"
                )
            else:
                result = f"{{{', '.join(self.val_to_arg_str(x, element_type) for x in val)}}}"
                self.writeline(
                    f"const {self.c_type_for_prim_type(val[0], element_type)} {var_name}[] = {result};"
                )
            # Need to pass the array length because we can't use std::vector
            return f"{var_name}, {len(val)}"

        return self.val_to_arg_str_for_prim_type(val, type_)

    def create_tmp_raii_handle_var(self, base_handle):
        if base_handle.startswith(("wrap_with_raii_handle_if_needed",)):
            # wrap_with_raii_handle_if_needed creates a temp RAIIAtenTensorHandle, so we need to
            # explicitly store it. Otherwise, it will be destroyed before the fallback kernel call.
            tmp_var_name = f"var_{next(self.arg_var_id)}"
            return (
                tmp_var_name,
                f"RAIIAtenTensorHandle {tmp_var_name} = {base_handle};\n",
            )
        else:
            return "", ""
