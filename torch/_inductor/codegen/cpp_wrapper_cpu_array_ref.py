# mypy: allow-untyped-defs
from itertools import count
from typing import Dict, List, Optional, Tuple

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch._ops
from torch.utils._ordered_set import OrderedSet

from .. import config, ir
from ..virtualized import V
from .cpp_utils import cexpr, DTYPE_TO_CPP
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
        if not hasattr(self, "device"):
            self.device = "cpu"
        super().__init__()
        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.comment = "//"
        self.namespace = "at::"
        self.none_str = "nullptr"
        self.extern_call_ops = OrderedSet()
        self.size = "sizes()"
        self.stride = "strides()"
        self.supports_intermediate_hooks = False
        self.kernel_callsite_id = count()
        self.var_array_id = (
            count()
        )  # for different types of local array variable declarations
        self.int_array_id = count()  # for int array local variable declarations
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        self.expr_printer = cexpr
        self.allow_stack_allocation: Optional[bool] = None
        self.stack_allocated_buffers: Dict[BufferName, BufferLike] = {}

    @staticmethod
    def create(
        is_subgraph: bool, subgraph_name: str, parent_wrapper: PythonWrapperCodegen
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpuArrayRef()

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
            and config.allow_stack_allocation
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
            if isinstance(buffer.get_layout(), ir.MultiOutputLayout)
            or (V.graph.aot_mode and buffer.get_name() in self.stack_allocated_buffers)
            or (
                config.use_minimal_arrayref_interface
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
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, buffer_if_can_stack_allocate=None
    ):
        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(orig_stride)
        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )
        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call,
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
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));"
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
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call
        )
        if reinterpret_view in self.stack_allocated_buffers:
            self.stack_allocated_buffers[new_name] = new
        return f"{self.declare_maybe_reference}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def generate_c_shim_extern_kernel_call(self, kernel, args):
        # In the abi_compatible mode, we call fallback aten ops through a C shim layer
        # Setting self.allow_stack_allocation to False because the exchange between
        # ArrayRefTensor and at::Tensor is still fragile.
        self.allow_stack_allocation = False

        wrapped_args = []
        debug_printer_manager = V.graph.wrapper_code.debug_printer

        for x in args:
            pieces = x.split(", ")
            for piece in pieces:
                # We only really *need* convert_arrayref_tensor_to_tensor for
                # ArrayRefTensors. The code flowing into here uses `0` for nullptr,
                # which convert_arrayref_tensor_to_tensor would blindly coerce to int,
                # so just avoid wrapping integers.
                # Name matching is to find tensor is hacky, but fixing all the
                # ArrayRefTensor issues is not a priority for now.
                if isinstance(piece, str) and piece.startswith(
                    ("buf", "arg", "wrap_with_raii_handle_if_needed")
                ):
                    piece = f"convert_arrayref_tensor_to_tensor({piece})"
                wrapped_args.append(piece)

        debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
        with debug_printer_manager:
            shim_fn = self.get_c_shim_func_name(kernel)
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(wrapped_args)}));"
            )

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
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False

        # call the ABI shim function instead of the ATen one
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name)
        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        inputs_wrapped = [
            (
                f"convert_arrayref_tensor_to_tensor({x})"
                if isinstance(x, str)
                else str(x)
            )
            for x in inputs
        ]
        line = f"{cpp_kernel_name}(convert_arrayref_tensor_to_tensor({output}), {','.join(inputs_wrapped)}"

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
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False

        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the correponding
        # tensor prematurely deallocated, thus this std::vector().data() trick here.
        indices_str = (
            "std::vector<AtenTensorHandle>{"
            + (
                ", ".join(
                    [f"convert_arrayref_tensor_to_tensor({ind})" for ind in indices]
                )
            )
            + "}.data()"
        )
        args = [
            f"convert_arrayref_tensor_to_tensor({x})",
            indices_str,
            str(len(indices)),
            f"convert_arrayref_tensor_to_tensor({values})",
            accumulate,
        ]
        args.insert(
            0, f"convert_arrayref_tensor_to_tensor({x})"
        )  # set x as the output tensor, this fallback mutates x.

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        cpp_op_schema: str,
        cpp_kernel_key: str,
        cpp_kernel_overload_name: str = "",
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        # No stack allocation when there is a fallback op
        self.allow_stack_allocation = False

        def extract_output_name(out):
            if out is None:
                return None
            elif isinstance(out, (ir.MultiOutput, ir._CollectiveKernel)):
                return out.get_name()
            elif isinstance(out, (list, tuple)):
                return type(out)(extract_output_name(o) for o in out)
            else:
                raise AssertionError(f"Unexpected output: {type(out)}")

        # output_args has the same pytree structure as outputs
        output_args = None
        if outputs is None:
            # outputs is not specified, the default is to write to buf_name
            output_args = [buf_name]
        else:
            output_args = extract_output_name(outputs)
            if isinstance(output_args, str):
                output_args = [output_args]

        if V.graph.aot_mode:
            assert op_overload is not None
            assert raw_args is not None
            assert outputs is not None

            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_with_proxy_executor(
                cpp_kernel_key,
                op_overload,
                raw_args,
                output_args,
                outputs,
            )
        else:
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_jit(
                buf_name,
                python_kernel_name,
                cpp_kernel_name,
                codegen_args,
                cpp_op_schema,
                cpp_kernel_key,
                cpp_kernel_overload_name,
                op_overload,
                raw_args,
                output_args,
                outputs,
            )

    def codegen_device_copy(self, src, dst, non_blocking: bool):
        # aoti_torch_tensor_copy_ takes AtenTensorHandle as input,
        # while stack-allocation results in ArrayRefTensor
        # so disable stack allocation here
        self.allow_stack_allocation = False
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(expensive_copy_to_tensor_if_needed({dst}), {src}, {non_blocking}));"
        )

    def codegen_reinterpret_view(
        self, data, size_list, stride_list, offset, writer, dtype=None
    ) -> str:
        dim = str(len(size_list))
        original_offset = offset
        size = self.codegen_shape_tuple(size_list)
        stride = self.codegen_shape_tuple(stride_list)
        offset = self.codegen_sizevar(offset)
        call_strs = []
        final_tmp_name = None
        final_tmp_name_is_RAIIAtenTensorHandle = False

        def create_reinterpret_call() -> Tuple[str, str]:
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    size,
                    writer,
                    known_statically=self.is_statically_known_list_of_ints(size_list),
                    graph=self.get_codegened_graph(),
                ),
                self.codegen_int_array_var(
                    stride,
                    writer,
                    known_statically=self.is_statically_known_list_of_ints(stride_list),
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
            size_list == data.layout.size
            and stride_list == data.layout.stride
            and original_offset == data.layout.offset
        ):
            # pure dtypeview
            if dtype is not None and dtype != data.dtype:
                tmp_output_name, tmp_call_strs = create_dtypeview_call(data.get_name())
                call_strs.extend(tmp_call_strs)
                final_tmp_name = tmp_output_name
                final_tmp_name_is_RAIIAtenTensorHandle = True
            else:
                return f"{data.get_name()}"
        else:
            # firstly create reinterpretview
            final_tmp_name, reinterpret_call = create_reinterpret_call()
            call_strs.append(reinterpret_call)

            if dtype is not None and dtype != data.dtype:
                # wrap it with dtypeview
                final_tmp_name, tmp_call_strs = create_dtypeview_call(reinterpret_call)
                call_strs.extend(tmp_call_strs)

        if writer is None:
            writer = self

        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        writer.writelines(call_strs)

        if (
            self.can_stack_allocate_buffer(data)
            and self.is_statically_known_list_of_ints(size_list)
            and self.is_statically_known_list_of_ints(stride_list)
            and ir.is_contiguous_strides_for_shape(stride_list, size_list)
        ):
            return final_tmp_name

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
        if not final_tmp_name_is_RAIIAtenTensorHandle:
            return f"wrap_with_raii_handle_if_needed({final_tmp_name})"
        else:
            return final_tmp_name
