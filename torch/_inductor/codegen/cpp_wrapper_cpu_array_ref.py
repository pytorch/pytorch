# mypy: allow-untyped-defs
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch._ops
from .. import config, ir
from ..utils import sympy_product
from ..virtualized import V
from .cpp_utils import DTYPE_TO_CPP
from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import (
    BufferLike,
    EnterSubgraphLine,
    ExitSubgraphLine,
    MemoryPlanningLine,
    MemoryPlanningState,
    PythonWrapperCodegen,
)


BufferName = str

# Default thread stack sizes vary by platform:
# - Linux: 8 MB
# - macOS: 512 KB
# - Windows: 1 MB
# Just pick something comfortably smaller than the smallest for now.
MAX_STACK_ALLOCATION_SIZE = 1024 * 100


class CppWrapperCpuArrayRef(CppWrapperCpu):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels

    This class is forked from CppWrapperCpu, with a difference that tensors may be
    represented as ArrayRef, see torch/csrc/inductor/aoti_runtime/arrayref_tensor.h
    """

    def __init__(self):
        super().__init__()
        assert self.device == "cpu", "ArrayRefTensor only supported on CPU!"
        self.allow_stack_allocation = config.aot_inductor.allow_stack_allocation
        self.stack_allocated_buffers: dict[BufferName, BufferLike] = {}

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpuArrayRef()

    @staticmethod
    def get_input_cpp_type(input):
        assert config.aot_inductor.use_minimal_arrayref_interface

        if isinstance(input, sympy.Expr):
            from ..graph import may_get_constant_buffer_dtype

            dtype = may_get_constant_buffer_dtype(input)
            assert dtype is not None, f"Failed to get the dtype of sympy.Expr: {input}"
            return DTYPE_TO_CPP[dtype]
        return f"ArrayRefTensor<{DTYPE_TO_CPP[input.get_dtype()]}>"

    @staticmethod
    def get_device_include_path(device: str) -> str:
        assert device == "cpu", "ArrayRef only supported on CPU!"
        if V.graph.aot_mode:
            return "#include <torch/csrc/inductor/aoti_include/array_ref.h>"
        return "#include <torch/csrc/inductor/cpp_wrapper/array_ref.h>"

    def codegen_input_numel_asserts(self):
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            numel = buf.get_numel()
            self.prefix.writeline(f"assert_numel({name}, {numel});")

    def generate_extern_kernel_alloc(self, *args, **kwargs):
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_extern_kernel_alloc(*args, **kwargs)

    def generate_extern_kernel_out(self, *args, **kwargs):
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_extern_kernel_out(*args, **kwargs)

    def generate_fallback_kernel(self, node: ir.FallbackKernel) -> None:
        # Disable stack allocation for extern kernels.
        self.allow_stack_allocation = False
        super().generate_fallback_kernel(node)

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
        assert not triton, (
            "CppWrapperCpuArrayRef.generate_kernel_call does not support GPU"
        )
        assert arg_types is not None and len(call_args) == len(arg_types), (
            "Mismatch call_args and arg_types in generate_kernel_call"
        )
        new_args = []
        for idx, arg in enumerate(call_args):
            if "*" in arg_types[idx]:
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"auto* {var_name} = get_data_ptr_wrapper({arg});")
                new_args.append(f"({arg_types[idx]})({var_name})")
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

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            if (
                config.aot_inductor.use_minimal_arrayref_interface
                and not V.graph.is_const_graph
            ):
                input_cpp_types = ", ".join(
                    f"{CppWrapperCpuArrayRef.get_input_cpp_type(x)}"
                    for x in V.graph.graph_inputs.values()
                )
                output_arrayref_types = ", ".join(
                    f"ArrayRefTensor<{DTYPE_TO_CPP[x.get_dtype()]}>"
                    for x in V.graph.graph_outputs
                )

                self.prefix.splice(
                    f"""
                    using AOTInductorModelInputs = std::tuple<{input_cpp_types}>;
                    using AOTInductorModelOutputs = std::tuple<{output_arrayref_types}>;
                    """
                )

            if V.graph.const_module:
                self.header.splice(V.graph.const_module.wrapper_code.header)

                assert V.graph.const_wrapper_code is not None
                self.prefix.splice(V.graph.const_wrapper_code)

                assert V.graph.const_kernel_code is not None
                self.kernel_declarations.splice(V.graph.const_kernel_code)

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

                self.generate_input_output_runtime_checks()
                run_impl_proto += """
                    __check_inputs_outputs(input_handles, output_handles);
                """

                if config.aot_inductor.use_minimal_arrayref_interface:
                    self.prefix.splice(
                        """
                        template <>
                        AOTInductorModelOutputs AOTInductorModel::run_impl_minimal_arrayref_interface<
                          AOTInductorModelInputs, AOTInductorModelOutputs>(
                            const AOTInductorModelInputs& inputs,
                            DeviceStreamType stream,
                            AOTIProxyExecutorHandle proxy_executor
                        ) {
                        """
                    )
                    self.suffix.splice(run_impl_proto)
                    self.suffix.splice(
                        """
                            AOTInductorModelInputs inputs;
                            convert_handles_to_inputs(input_handles, inputs);
                            auto outputs = run_impl_minimal_arrayref_interface<AOTInductorModelInputs, AOTInductorModelOutputs>(
                                inputs, stream, proxy_executor);
                            // NOTE: outputs is full of ArrayRef to thread_local storage. If in the future we need this
                            // interface to perform well for a DSO using the minimal arrayref interface, all we need
                            // to do is provide ThreadLocalCachedTensor for each one!
                            convert_outputs_to_handles(outputs, output_handles);
                        }
                    """
                    )

                    self.suffix.splice(
                        """
                        extern "C" AOTIRuntimeError AOTInductorModelRunMinimalArrayrefInterface(
                            AOTInductorModelHandle model_handle,
                            const AOTInductorModelInputs& inputs,
                            AOTInductorModelOutputs& outputs) {
                          auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
                          CONVERT_EXCEPTION_TO_ERROR_CODE({
                              outputs = model->run_impl_minimal_arrayref_interface<AOTInductorModelInputs, AOTInductorModelOutputs>(
                                  inputs,
                                  (torch::aot_inductor::DeviceStreamType)nullptr,
                                  nullptr);
                          })
                        }
                    """
                    )
                else:
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
            if not config.aot_inductor.use_minimal_arrayref_interface:
                if not V.graph.is_const_graph:
                    if V.graph.aot_mode:
                        num_args = len(V.graph.graph_inputs)
                    else:
                        # Weights are promoted in the JIT mode
                        num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
                        # release GIL to support multiple instances inference (in different threads of the same process)
                        self.prefix.splice("py::gil_scoped_release_simple release;")

                    self.prefix.splice(
                        f"""
                            auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {num_args});
                        """
                    )

            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    if config.aot_inductor.use_minimal_arrayref_interface:
                        self.prefix.writeline(
                            f"auto {input_key} = std::get<{idx}>(inputs);"
                        )
                        continue
                    # unwrap input tensor back to scalar
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype

                        dtype = may_get_constant_buffer_dtype(
                            V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                        )
                        assert dtype is not None, (
                            "Fails to get the dtype of the sympy.Expr"
                        )
                        self.codegen_tensor_item(
                            dtype, f"inputs[{idx}]", input_key, self.prefix
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {input_key} = std::move(inputs[{idx}]);"
                        )

            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            for idx, constants_key in enumerate(V.graph.constants.keys()):
                if V.graph.aot_mode:
                    # Weights are stored in constants_ and owned by RAIIAtenTensorHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    self.prefix.writeline(
                        f"""auto {constants_key} = constants_->at({idx});"""
                    )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"auto {constants_key} = std::move(inputs[{constants_idx}]);"
                    )

            self.codegen_inputs()

            if V.graph.aot_mode:
                if not V.graph.is_const_graph:
                    if config.aot_inductor.use_minimal_arrayref_interface:
                        # TODO: input shape checking for regular tensor interface as well?
                        self.codegen_input_numel_asserts()
                    else:
                        self.prefix.writeline("inputs.clear();")
                self.prefix.writeline(
                    "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
                )

    def generate_return(self, output_refs: list[str]):
        cst_names = V.graph.constants.keys()
        arr_iface = (
            not V.graph.is_const_graph
            and config.aot_inductor.use_minimal_arrayref_interface
        )  # For brevity.

        def use_thread_local_cached_output_tensor(idx, output):
            cached_output_name = f"cached_output_{next(self.cached_output_id)}"
            cache_type = "Array" if arr_iface else "Tensor"
            self.wrapper_call.writeline(
                f"thread_local ThreadLocalCachedOutput{cache_type}<std::decay_t<decltype({output})>> "
                f"{cached_output_name}({output});"
            )
            if arr_iface:
                self.wrapper_call.writeline(
                    f"{cached_output_name}.copy_data_from({output});"
                )
                output_entry = f"std::get<{idx}>(output_arrayref_tensors)"
                element_type = f"std::decay_t<decltype({output_entry}.data()[0])>"
                self.wrapper_call.writeline(
                    f"{output_entry} = {cached_output_name}.arrayref_tensor<{element_type}>();"
                )
            else:
                self.wrapper_call.writeline(
                    f"{cached_output_name}.copy_data_from({output});"
                )
                self.wrapper_call.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&output_handles[{idx}]));"
                )
                self.wrapper_call.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors({cached_output_name}.tensor(), "
                    f"output_handles[{idx}]));"
                )

        if arr_iface:
            self.wrapper_call.writeline(
                "AOTInductorModelOutputs output_arrayref_tensors;"
            )

        output2idx: dict[str, int] = {}
        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                assert isinstance(output_storage, (ir.BaseView, ir.MutableBox))
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            output_is_tensor_handle_expr = (
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "RAIIAtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "AtenTensorHandle> || "
                f"std::is_same_v<std::decay_t<decltype({output})>,"
                "ConstantHandle>"
            )
            self.wrapper_call.writeline(
                f"if constexpr ({output_is_tensor_handle_expr}) {{"
            )
            with self.wrapper_call.indent():
                if arr_iface:
                    cached_output_name = f"cached_output_{next(self.cached_output_id)}"
                    self.wrapper_call.writeline(
                        f"thread_local RAIIAtenTensorHandle {cached_output_name};"
                    )
                    if is_constant_buffer:
                        # NOTE(return_constant): In some rare cases where we return
                        # a constant, we have to return a copy of this constant,
                        # because (1) constants are not owned by the Model instance
                        # (2) constants remain the same cross inference runs,
                        # assuming they are not updated at runtime Basically, we
                        # cannot release or transfer the ownership of any original
                        # constant to the user.
                        self.wrapper_call.writeline(
                            f"AtenTensorHandle {cached_output_name}_tmp;"
                        )
                        self.wrapper_call.writeline(
                            f"aoti_torch_clone({output}, &{cached_output_name}_tmp);"
                        )
                        self.wrapper_call.writeline(
                            f"{cached_output_name} = {cached_output_name}_tmp;"
                        )
                    else:
                        self.wrapper_call.writeline(
                            f"{cached_output_name} = {output}.release();"
                        )
                    self.wrapper_call.writeline(
                        f"convert_handle_to_arrayref_tensor({cached_output_name}, "
                        f"std::get<{idx}>(output_arrayref_tensors));"
                    )
                else:
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
            self.wrapper_call.writeline("} else {")
            with self.wrapper_call.indent():
                use_thread_local_cached_output_tensor(idx, output)
            self.wrapper_call.writeline("}")

            if output not in output2idx:
                output2idx[output] = idx
        if arr_iface:
            self.wrapper_call.writeline("return output_arrayref_tensors;")

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)
        # TODO: integrate memory planning & stack allocation?
        self.allow_stack_allocation = False

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()

        while (
            self.lines
            and isinstance(self.lines[-1], MemoryPlanningLine)
            # TODO: this seems legit, NullLine has no node
            and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
        ):
            # these lines will be pointless
            self.lines.pop()

        # codegen allocations in two passes
        planning_states = [MemoryPlanningState()]
        past_planning_states = []
        for i in range(len(self.lines)):
            line = self.lines[i]
            if isinstance(line, MemoryPlanningLine):
                self.lines[i] = line.plan(planning_states[-1])
            elif isinstance(line, EnterSubgraphLine):
                planning_states.append(MemoryPlanningState())
            elif isinstance(line, ExitSubgraphLine):
                past_planning_states.append(planning_states.pop())
        past_planning_states.append(planning_states.pop())
        assert len(planning_states) == 0

        # conservatively use the sum of all allocated buffer sizes
        # in potentially nested scopes as the total allocated size
        total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

        self.allow_stack_allocation = (
            self.allow_stack_allocation is not False
            and config.aot_inductor.allow_stack_allocation
            and total_allocated_buffer_size <= MAX_STACK_ALLOCATION_SIZE
        )

    def can_stack_allocate_buffer(self, buffer):
        return (
            self.allow_stack_allocation
            and buffer.get_device().type == "cpu"
            and self.can_prove_buffer_has_static_shape(buffer)
            and ir.is_contiguous_strides_for_shape(
                buffer.get_stride(), buffer.get_size()
            )
        )

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or (V.graph.aot_mode and buffer.get_name() in self.stack_allocated_buffers)
            or (
                config.aot_inductor.use_minimal_arrayref_interface
                and V.graph.aot_mode
                and buffer.get_name() in V.graph.graph_inputs
            )
            else f"{buffer.get_name()}.reset();"
        )

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            buffer if self.can_stack_allocate_buffer(buffer) else None,
            buffer.get_is_pinned(),
        )

    def make_allocation(
        self,
        name,
        device,
        dtype,
        shape,
        stride,
        buffer_if_can_stack_allocate=None,
        is_pinned=False,
    ):
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
        if buffer_if_can_stack_allocate is not None:
            self.stack_allocated_buffers[name] = buffer_if_can_stack_allocate
            cpp_type = DTYPE_TO_CPP[dtype]
            numel = buffer_if_can_stack_allocate.get_numel()
            # Note: we don't zero storage because empty_strided doesn't zero either.
            self.wrapper_call.writeline(f"{cpp_type} {name}_storage[{numel}];")
            args = [
                f"{name}_storage",
                size_array_var,
                stride_array_var,
                device_type,
                device_idx,
            ]
            return f"ArrayRefTensor<{cpp_type}> {name}({', '.join(args)});"

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
        pinned_str = "_pinned" if is_pinned else ""
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided{pinned_str}({', '.join(args)}));"
        )

        return f"RAIIAtenTensorHandle {name}({name}_handle);"

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            if old_name in self.stack_allocated_buffers:
                self.stack_allocated_buffers[new_name] = new
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
        )
        if reinterpret_view in self.stack_allocated_buffers:
            self.stack_allocated_buffers[new_name] = new
            # The only way to get into this case is via an exact buffer reuse, since all
            # other options result in a new tensor handle.
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  // reuse"

    def _assert_safe_to_use_borrow_arrayref_tensor_as_tensor(self):
        # Borrowing arguments to shim functions is only safe because we know
        # that the arguments can't be stack-allocated. Otherwise, to be sure
        # we can't return a dangling pointer, we need to either 1) be
        # certain that the shim function cannot return an alias of a
        # borrowed argument, or 2) be certain that the returned Tensor from
        # the shim function cannot escape.
        assert self.is_safe_to_use_borrow_arrayref_tensor_as_tensor(), (
            "borrowing arguments to shim functions is unsafe with "
            "stack allocation on! (see comment above this assertion)"
        )

    def is_safe_to_use_borrow_arrayref_tensor_as_tensor(self):
        return not self.allow_stack_allocation and not self.stack_allocated_buffers

    def generate_c_shim_extern_kernel_call(
        self, kernel: str, args: list[str], device: str, **_
    ) -> None:
        # In the abi_compatible mode, we call fallback aten ops through a C shim layer
        # Setting self.allow_stack_allocation to False because the exchange between
        # ArrayRefTensor and at::Tensor is still fragile.
        self.allow_stack_allocation = False

        wrapped_args = []
        for arg in args:
            # We only really *need* borrow_arrayref_tensor_as_tensor for
            # ArrayRefTensors. The code flowing into here uses `0` for nullptr, which
            # borrow_arrayref_tensor_as_tensor would blindly coerce to int, so just
            # avoid wrapping integers.  Name matching is to find tensor is hacky, but
            # fixing all the ArrayRefTensor issues is not a priority for now.
            if isinstance(arg, str) and arg.startswith(
                ("buf", "arg", "wrap_with_raii_handle_if_needed")
            ):
                self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
                arg = f"borrow_arrayref_tensor_as_tensor({arg})"
            wrapped_args.append(arg)

        super().generate_c_shim_extern_kernel_call(
            kernel, wrapped_args, device, debug_args=args
        )

    def generate_scatter_fallback(self, node: ir.ScatterFallback):
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_scatter_fallback(node)

    def _generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
        device,
    ):
        reduce = self._get_scatter_reduce_enum(reduce)

        # call the ABI shim function instead of the ATen one
        self.add_device_include(device)
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name, device)

        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
        inputs_wrapped = [
            (f"borrow_arrayref_tensor_as_tensor({x})" if isinstance(x, str) else str(x))
            for x in inputs
        ]
        line = f"{cpp_kernel_name}(borrow_arrayref_tensor_as_tensor({output}), {','.join(inputs_wrapped)}"

        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        else:
            if src_is_tensor:
                if reduce:
                    line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
            else:
                assert reduce is None, (
                    "Expect reduce to be None for aten.scatter_ with scalar src"
                )
        line += ");"
        self.writeline(line)

    def generate_index_put_fallback(self, node: ir.IndexPutFallback) -> None:
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_index_put_fallback(node)

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        self._assert_safe_to_use_borrow_arrayref_tensor_as_tensor()
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the corresponding
        # tensor prematurely deallocated, thus the temporary array trick here.
        indices_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle",
            [f"borrow_arrayref_tensor_as_tensor({i})" for i in indices],
        )
        args = [
            f"borrow_arrayref_tensor_as_tensor({x})",
            indices_str,
            str(len(indices)),
            f"borrow_arrayref_tensor_as_tensor({values})",
            accumulate,
        ]
        args.insert(
            0, f"borrow_arrayref_tensor_as_tensor({x})"
        )  # set x as the output tensor, this fallback mutates x.
        self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator],
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False
        super().generate_fallback_kernel_with_runtime_lookup(
            buf_name, python_kernel_name, get_args, op_overload, raw_args, outputs
        )

    def codegen_device_copy(self, src, dst, non_blocking: Union[bool, str]):
        # aoti_torch_tensor_copy_ takes AtenTensorHandle as input,
        # while stack-allocation results in ArrayRefTensor
        # so disable stack allocation here
        self.allow_stack_allocation = False
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(expensive_copy_to_tensor_if_needed({dst}), {src}, {non_blocking}));"
        )

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        """Returns a newly-created, temporary RAII tensor handle containing the
        reinterpreted tensor data.  Callers of this function are responsible for saving
        the handle if persistent access is needed."""
        dim = str(len(size))

        def create_reinterpret_call() -> str:
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
            return f"wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper({', '.join(args)}))"

        def create_new_tensor_handle() -> tuple[str, list[str]]:
            # Calling reset() on ArrayRefTensor does nothing, since the array is
            # const-allocated on the stack.  Thus, it's safe to return a reference to
            # the original array.
            if (name := data.get_name()) in self.stack_allocated_buffers:
                return name, []

            tmp_AtenTensorHandle = f"tmp_{name}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [
                f"AtenTensorHandle {tmp_AtenTensorHandle};",
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle({data.get_name()}, &{tmp_AtenTensorHandle}));",
            ]
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        if (
            size == data.layout.size
            and stride == data.layout.stride
            and offset == data.layout.offset
            and (dtype is None or dtype == data.dtype)
        ):
            final_tensor_str, call_strs = create_new_tensor_handle()
            for line in call_strs:
                writeline(line)
            return final_tensor_str

        return super().codegen_reinterpret_view(
            data, size, stride, offset, writeline, dtype
        )

    def val_to_arg_str(self, val, type_=None) -> str:
        if (
            val is not None
            and isinstance(type_, torch.OptionalType)
            and isinstance(type_.getElementType(), torch.TensorType)
        ):
            # Handle optional tensors as a special case, as in the parent class.
            base_handle = self.val_to_arg_str(val, torch.TensorType)
            if config.aot_inductor.use_minimal_arrayref_interface:
                if self.is_safe_to_use_borrow_arrayref_tensor_as_tensor():
                    base_handle = f"borrow_arrayref_tensor_as_tensor({base_handle})"
                else:
                    base_handle = f"copy_arrayref_tensor_to_tensor({base_handle})"
            return f"&temporary_reference({base_handle}.get())"

        return super().val_to_arg_str(val, type_)

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")

            # We know that item_ doesn't alias the input, so borrowing should be safe.
            tensor = f"borrow_arrayref_tensor_as_tensor({tensor})"

            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")

            # We know that item_ doesn't alias the input, so borrowing should be safe.
            tensor = f"borrow_arrayref_tensor_as_tensor({tensor})"

            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )
