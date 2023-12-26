import collections
import contextlib
import dataclasses
import functools
import inspect
import operator
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sympy
from sympy import Expr

import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes

from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt

from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
    cache_on_self,
    get_benchmark_name,
    LineContext,
    sympy_product,
    sympy_str,
)
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta


pexpr = PythonPrinter().doprint


def buffer_reuse_key(node: ir.Buffer):
    return (
        node.get_device(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(node.layout.storage_size())),
    )


def is_int(s: str):
    # Cpp code gen adds L at the end of ints
    # Lets remove it for checking whether we have an int or not
    if s and s[-1] == "L":
        s = s[:-1]
    try:
        int(s)
    except ValueError:
        return False
    except TypeError:
        return False
    return True


def is_float(s: str):
    try:
        float(s)
    except ValueError:
        return False
    return True


def convert_arg_type(arg: torch.Argument):
    from .cpp import CONTAINER_PYTHON_TO_CPP, PYTHON_TO_CPP

    # use x.real_type instead of x.type so that we get ScalarType instead of int
    python_type = repr(arg.real_type)  # type: ignore[attr-defined]

    if python_type == "Tensor":
        # Conversions rules follow https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#func
        if arg.alias_info is not None and arg.alias_info.is_write:
            return f"at::{python_type}&"
        else:
            return f"at::{python_type} const&"

    if python_type in PYTHON_TO_CPP:
        cpp_type = PYTHON_TO_CPP[python_type]
        return cpp_type

    # Convert args of container types e.g. Optional[*]
    for py_container, cpp_container in CONTAINER_PYTHON_TO_CPP.items():
        container_match = re.findall(py_container + r"\[([a-zA-Z_]+)]", python_type)
        if len(container_match) == 1:
            contained_type = container_match[0]
            assert (
                contained_type in PYTHON_TO_CPP
            ), f"unsupported {py_container} type in convert_arg_type: {contained_type}"
            cpp_contained_type = PYTHON_TO_CPP[contained_type]
            return f"{cpp_container}<{cpp_contained_type}>"

    raise AssertionError(f"unsupport python_type: {python_type}")


def convert_return_type(ret: torch.Argument):
    # use x.real_type instead of x.type so that we get ScalarType instead of int
    python_type = repr(ret.real_type)  # type: ignore[attr-defined]
    python_to_cpp = {
        "Tensor": "at::Tensor",
        "List[Tensor]": "std::vector<at::Tensor>",
    }

    cpp_type = python_to_cpp.get(python_type, None)
    assert cpp_type is not None, f"NYI return type: {python_type}"
    # An output aliasing an input is returned by reference only when it's a
    # Tensor, not when it's a Tensor[]. For example, aten.split.Tensor's output
    # aliases the input tensor, but the op returns a vector by value.
    if python_type == "Tensor" and ret.alias_info is not None:
        cpp_type += "&"
    return cpp_type


def get_cpp_op_schema(kernel):
    args = kernel._schema.arguments
    returns = kernel._schema.returns

    num_returns = len(returns)
    assert num_returns > 0, "must have at least one return value"

    if num_returns == 1:
        cpp_return_value = convert_return_type(returns[0])
    elif num_returns > 1:
        tuple_returns = ", ".join([convert_return_type(r) for r in returns])
        cpp_return_value = f"std::tuple<{tuple_returns}>"

    cpp_arg_type = [f"{convert_arg_type(arg)} {arg.name}" for arg in args]
    return f"{cpp_return_value}({', '.join(cpp_arg_type)})"


def user_defined_kernel_grid_fn_code(name, configs, grids):
    output = IndentedBuffer()

    fn_name = f"grid_wrapper_for_{name}"
    output.writeline(f"def {fn_name}(meta):")
    with output.indent():
        if len(grids) == 1:
            output.writeline(f"return {grids[0]}")
        else:
            assert len(grids) > 1
            assert len(grids) == len(configs)
            seen = set()
            for grid, c in zip(grids, configs):
                guards = [f"meta['{name}'] == {val}" for name, val in c.kwargs.items()]
                guards = " and ".join(guards)
                statement = f"if {guards}: return {grid}"
                if statement in seen:
                    continue
                seen.add(statement)
                output.writeline(statement)

    return fn_name, output.getvalue()


@dataclasses.dataclass
class SymbolicCallArg:
    inner: Any
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


# Default thread stack sizes vary by platform:
# - Linux: 8 MB
# - macOS: 512 KB
# - Windows: 1 MB
# Just pick something comfortably smaller than the smallest for now.
MAX_STACK_ALLOCATION_SIZE = 1024 * 100


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[Any, List[FreeIfNotReusedLine]] = collections.defaultdict(
            list
        )
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key):
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key) -> "FreeIfNotReusedLine":
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key, item: "FreeIfNotReusedLine"):
        assert not item.is_reused
        self.reuse_pool[key].append(item)


@dataclasses.dataclass
class EnterDeviceContextManagerLine:
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if V.graph.aot_mode:
                # In AOT mode, we have a stream provided as a param. A stream is
                # associated with a device, so we never expect the device to change.
                # CUDAStreamGuard sets the stream and the device.
                if self.last_seen_device_guard_index is None:
                    if config.aot_inductor.abi_compatible:
                        code.writeline(
                            "AOTICudaStreamGuard stream_guard(stream, this->device_idx_);"
                        )
                    else:
                        code.writeline(
                            "at::cuda::CUDAStreamGuard stream_guard("
                            + "at::cuda::getStreamFromExternal(stream, this->device_idx_));"
                        )
                else:
                    assert (
                        self.last_seen_device_guard_index == self.device_idx
                    ), "AOTInductor only supports running on one CUDA device"
            else:
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"at::cuda::CUDAGuard device_guard({self.device_idx});"
                    )
                else:
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            device_cm_stack.enter_context(code.indent())
            code.writeline(V.graph.device_ops.set_device(self.device_idx))


class ExitDeviceContextManagerLine:
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if not V.graph.cpp_wrapper:
            device_cm_stack.close()


@dataclasses.dataclass
class MemoryPlanningLine:
    wrapper: "WrapperCodeGen"

    def plan(self, state: MemoryPlanningState) -> "MemoryPlanningLine":
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer):
        """Second pass to output code"""
        pass

    def __str__(self):
        """
        Emits a string representation that fits on one line.
        """
        args: List[str] = []
        for field in dataclasses.fields(self):
            if field.name == "wrapper":
                continue
            val = getattr(self, field.name)
            args.append(
                f"{field.name}={val.get_name() if field.type is ir.Buffer else val}"
            )
        return f"{type(self).__name__}({', '.join(args)})"


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        if self.node.get_device().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: ir.Buffer
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState):
        if isinstance(self.node.layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: ir.Buffer
    reused_as: ir.Buffer
    delete_old: bool = True

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )


class NullLine(MemoryPlanningLine):
    pass


class WrapperCodeGen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.src_to_kernel = {}
        self.kenel_numel_expr = set()
        self.lines = []
        self.declare = ""
        self.declare_maybe_reference = ""
        self.ending = ""
        self.open_bracket = "["
        self.closed_bracket = "]"
        self.comment = "#"
        self.namespace = ""
        self.none_str = "None"
        self.size = "size()"
        self.stride = "stride()"
        self.last_seen_device_guard_index = None
        self.supports_intermediate_hooks = True
        self.expr_printer = pexpr
        self.user_defined_kernel_cache: Dict[Tuple[Any, ...], str] = {}
        self.unbacked_symbol_decls = set()
        self.allow_stack_allocation = None
        self.stack_allocated_buffers = {}

        if not V.graph.is_const_graph or not V.graph.cpp_wrapper:
            self.write_header()
            self.write_prefix()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated = set()
        self.freed: Set[str] = set()

        # maps from reusing buffer to reused buffer
        self.reuses = dict()

        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

        @functools.lru_cache(None)
        def add_import_once(line):
            self.header.writeline(line)

        self.add_import_once = add_import_once
        self._metas = {}

    def write_constant(self, name, hashed):
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty, empty_strided
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = AsyncCompile()

            """
        )

    @cache_on_self
    def write_triton_header_once(self):
        self.header.splice(
            """
            import triton
            import triton.language as tl
            from torch._inductor.triton_heuristics import grid, start_graph, end_graph
            {}
            """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def add_meta_once(self, meta):
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self):
        return [x.codegen_reference(self.wrapper_call) for x in V.graph.graph_outputs]

    def mark_output_type(self):
        return

    def codegen_input_size_asserts(self):
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_shape_tuple(buf.get_size())
            stride = self.codegen_shape_tuple(buf.get_stride())
            self.prefix.writeline(f"assert_size_stride({name}, {size}, {stride})")

    def write_prefix(self):
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline(V.graph.device_ops.synchronize())
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()

    def write_get_raw_stream(self, index):
        self.write_triton_header_once()
        name = f"stream{index}"
        self.writeline(f"{name} = get_raw_stream({index})")
        return name

    def next_kernel_suffix(self):
        return f"{next(self._names_iter)}"

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(
            EnterDeviceContextManagerLine(device_idx, self.last_seen_device_guard_index)
        )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self):
        self.writeline(ExitDeviceContextManagerLine())

    def generate_return(self, output_refs):
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_before_suffix(self, result):
        return

    def generate_end(self, result):
        return

    def generate_fallback_kernel(self, fallback_kernel, args):
        self.generate_extern_kernel_alloc(fallback_kernel, args)

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            # view operation fallbacks cause issues since inductor
            # doesn't know the memory is still needed and might reuse it.
            ending = f".clone(){ending}"
        self.writeline(
            f"{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}"
        )
        if (
            self.supports_intermediate_hooks
            and config.generate_intermediate_hooks
            and origin_node is not None
        ):
            counters["inductor"]["intermediate_hooks"] += 1
            self.writeline(
                f"run_intermediate_hooks({origin_node.name!r}, {output_name})"
            )

    def generate_extern_kernel_out(self, output_view, codegen_reference, args, kernel):
        if output_view:
            args.append(f"out={output_view.codegen_reference()}")
        else:
            args.append(f"out={codegen_reference}")
        self.writeline(f"{kernel}({', '.join(args)})")

    def generate_user_defined_triton_kernel(self, kernel_name, grid, configs, args):
        grid, code = user_defined_kernel_grid_fn_code(kernel_name, configs, grid)
        # Must happen after free symbols are already codegened
        with self.prefix.indent():
            self.prefix.splice(code)

        stream_name = self.write_get_raw_stream(V.graph.scheduler.current_device.index)
        self.writeline(
            f"{kernel_name}.run({', '.join(args)}, grid={grid}, stream={stream_name})"
        )

    def generate_scatter_fallback(
        self, output, inputs, kernel, python_kernel_name, src_is_tensor, reduce, kwargs
    ):
        line = f"{kernel}({','.join(map(str, inputs))}"
        if kernel == "aten.scatter_":
            if reduce:
                line += f", reduce={repr(reduce)}"
        else:
            line += ", ".join([""] + kwargs)
        line += f"){self.ending}"
        self.writeline(line)

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        name,
        kernel,
        codegen_args,
        cpp_op_schema,
        cpp_kernel_key,
        cpp_kernel_overload_name="",
        op_overload=None,
        raw_args=None,
        outputs=None,
    ):
        self.writeline(f"{name} = {kernel}({', '.join(codegen_args)})")

    def generate_inf_and_nan_checker(self, node):
        # TODO: Add check for python too.
        pass

    @dynamo_timed
    def generate(self, is_inference):
        if config.profile_bandwidth:
            self.write_triton_header_once()
        result = IndentedBuffer()
        if (
            not V.graph.aot_mode
            or not V.graph.is_const_graph
            or not V.graph.cpp_wrapper
        ):
            result.splice(self.header)

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()

            # We disable planning during training because it presently increases peak memory consumption.
            if is_inference and config.memory_planning:
                self.memory_plan()
                # TODO: integrate memory planning & stack allocation?
                self.allow_stack_allocation = False
            else:
                self.memory_plan_reuse()

            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(
                    line,
                    (
                        EnterDeviceContextManagerLine,
                        ExitDeviceContextManagerLine,
                    ),
                ):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline(V.graph.device_ops.synchronize())

            if config.profile_bandwidth:
                self.generate_end_graph()

            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        self.finalize_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_before_suffix(result)
        result.splice(self.suffix)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def memory_plan(self):
        from .memory_planning import MemoryPlanner

        self.lines = MemoryPlanner(self).plan(self.lines)

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
        planning_state = MemoryPlanningState()
        for i in range(len(self.lines)):
            if isinstance(self.lines[i], MemoryPlanningLine):
                self.lines[i] = self.lines[i].plan(planning_state)

        self.allow_stack_allocation = (
            self.allow_stack_allocation is not False
            and config.allow_stack_allocation
            and planning_state.total_allocated_buffer_size <= MAX_STACK_ALLOCATION_SIZE
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"{self.declare}{name}_size = {name}.{self.size}{self.ending}")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(
            f"{self.declare}{name}_stride = {name}.{self.stride}{self.ending}"
        )

    def codegen_inputs(
        self, code: IndentedBuffer, graph_inputs: Dict[str, ir.TensorBox]
    ):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        # Assign all symbolic shapes needed to local variables
        needed = V.graph.sizevars.free_symbols()

        def is_expr(x):
            return isinstance(x[1], sympy.Expr)

        graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
        graph_inputs_tensors = list(
            filter(lambda x: not is_expr(x), graph_inputs.items())
        )

        for name, shape in graph_inputs_expr:
            shape = V.graph.sizevars.simplify(shape)
            if shape in needed:
                needed.remove(shape)
                code.writeline(f"{self.declare}{shape} = {name}{self.ending}")

        for name, value in graph_inputs_tensors:
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(
                        f"{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}"
                    )

        for name, value in graph_inputs_tensors:
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(
                        f"{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}"
                    )

    def append_precomputed_sizes_to_prefix(self):
        with self.prefix.indent():
            for sym, expr in V.graph.sizevars.inv_precomputed_replacements.items():
                self.prefix.writeline(
                    f"{self.declare}{sym} = {self.expr_printer(expr)}{self.ending}"
                )

    def finalize_prefix(self):
        pass

    def codegen_python_sizevar(self, x: Expr) -> str:
        return pexpr(V.graph.sizevars.simplify(x))

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        return f"{basename}[{index}]"

    def codegen_python_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_python_sizevar, shape))
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        return self.codegen_python_shape_tuple(shape)

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    str(dtype),
                    self.codegen_shape_tuple(shape),
                    self.codegen_shape_tuple(stride),
                ]
            )
        )

    def codegen_reinterpret_view(self, data, size, stride, offset, writer) -> str:
        size = self.codegen_shape_tuple(size)
        stride = self.codegen_shape_tuple(stride)
        offset = self.codegen_sizevar(offset)
        return f"reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset})"

    def codegen_device_copy(self, src, dst):
        self.writeline(f"{dst}.copy_({src})")

    def codegen_multi_output(self, name, value):
        self.writeline(f"{self.declare}{name} = {value}{self.ending}")

    def codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        if node.is_bool:
            self.writeline(f"{node.sym} = 1 if {data}.item() else 0")
        else:
            self.writeline(f"{node.sym} = {data}.item()")
        # No one should ever use this buffer, but for uniformity
        # define the variable and assign it None
        self.writeline(f"{node.get_name()} = None")

    def benchmark_compiled_module(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(
                        name, shape, stride, value.get_device(), value.get_dtype()
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline("return print_performance(fn, times=times, repeat=repeat)")

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        self.benchmark_compiled_module(output)

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(
                [
                    "from torch._inductor.wrapper_benchmark import compiled_module_main",
                    f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)",
                ]
            )

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        metadata_comment = f"{metadata}\n" if metadata else ""
        self.header.splice(f"\n\n{metadata_comment}{name} = {kernel}")

    def define_user_defined_triton_kernel(self, kernel, configs, kwargs):
        original_name = kernel.__name__

        # Distinguish between different functions using function id
        cache_key = [id(kernel.fn)]
        for arg in kwargs.values():
            if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                cache_key.append(arg.get_dtype())
            elif len(configs) > 0:
                # We need to key on non tensor arg only in autotune mode
                cache_key.append(arg)
        cache_key = tuple(cache_key)

        if cache_key in self.user_defined_kernel_cache:
            return self.user_defined_kernel_cache[cache_key]

        name = f"{original_name}_{len(self.user_defined_kernel_cache)}"
        # Add to the cache for the next use
        self.user_defined_kernel_cache[cache_key] = name

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")

        compile_wrapper.splice(
            """
            import triton
            import triton.language as tl
            from torch._inductor.utils import instance_descriptor
            from torch._inductor.triton_heuristics import user_autotune
            """,
            strip=True,
        )
        from .triton import TritonKernel

        if TritonKernel.gen_attr_descriptor_import():
            compile_wrapper.splice(TritonKernel.gen_attr_descriptor_import())
        compile_wrapper.newline()

        from .common import SizeArg, TensorArg

        signature: List[Union[TensorArg, SizeArg]] = []
        constants = {}
        for key, arg in kwargs.items():
            idx = kernel.arg_names.index(key)
            if idx in kernel.constexprs:
                constants[key] = arg
                continue
            if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                signature.append(
                    TensorArg(
                        key,
                        arg.codegen_reference(),
                        arg.get_dtype(),
                        # For ReinterpretView, we do not want to check alignment
                        not isinstance(arg, ReinterpretView),
                    )
                )
            else:
                signature.append(SizeArg(key, arg))
        index_dtype = "tl.int32"
        inductor_meta = {
            "kernel_name": name,
        }
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=index_dtype),
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": constants,
            "configs": [config_of(signature)],
        }
        configs = [
            {
                "kwargs": config.kwargs,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
            }
            for config in configs
        ]
        compile_wrapper.splice(
            f"""
            @user_autotune(
                configs={configs!r},
                inductor_meta={inductor_meta!r},
                triton_meta={triton_meta!r},
                filename=__file__
            )
            @triton.jit
            """
        )
        compile_wrapper.splice(kernel.src, strip=True)

        # Also include any possible kernel being called indirectly
        from triton import JITFunction

        symbols_included = {original_name}

        def traverse(cur_kernel):
            for symbol_name in cur_kernel.fn.__code__.co_names:
                if symbol_name in symbols_included:
                    continue
                if symbol_name in cur_kernel.fn.__globals__:
                    symbol = cur_kernel.fn.__globals__[symbol_name]
                    if isinstance(symbol, JITFunction):
                        compile_wrapper.newline()
                        compile_wrapper.writeline("@triton.jit")
                        compile_wrapper.splice(symbol.src, strip=True)
                        symbols_included.add(symbol_name)
                        traverse(symbol)
                    elif isinstance(symbol, (int, str, bool)):
                        compile_wrapper.newline()
                        compile_wrapper.writeline(f"{symbol_name} = {symbol!r}")
                        symbols_included.add(symbol_name)

        traverse(kernel)

        compile_wrapper.writeline("''')")
        _, lineno = inspect.getsourcelines(kernel.fn)
        srcfile = inspect.getsourcefile(kernel.fn)
        metadata = f"# Original path: {srcfile}:{lineno}"
        self.define_kernel(
            name,
            compile_wrapper.getvalue(),
            metadata,
        )
        return name

    def generate_numel_expr(self, kernel_name: str, tree):
        expr = f"{kernel_name}_{tree.prefix}numel"
        if expr not in self.kenel_numel_expr:
            self.kenel_numel_expr.add(expr)
            self.writeline(
                f"{self.declare}{expr} = {self.expr_printer(tree.numel)}{self.ending}"
            )
        else:
            self.writeline(f"{expr} = {self.expr_printer(tree.numel)}{self.ending}")
        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        return SymbolicCallArg(expr, tree.numel)

    def wrap_kernel_call(self, name, call_args):
        return f"{name}({', '.join(call_args)}){self.ending}"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        stack.enter_context(self.wrapper_call.indent())

    def generate_start_graph(self):
        self.wrapper_call.writeline("start_graph()")

    def generate_end_graph(self):
        self.wrapper_call.writeline("end_graph()")

    def generate_default_grid(self, name: str, grid_args: List[Any]):
        return grid_args

    def generate_kernel_call(
        self,
        name,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
    ):
        """
        Generates kernel call code.

        cuda: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if cuda:
            call_args_str = ", ".join(pexpr(item) for item in call_args)
            stream_name = self.write_get_raw_stream(
                V.graph.scheduler.current_device.index
            )
            if triton:
                grid_str = ", ".join(pexpr(item) for item in grid)
                self.writeline(
                    f"{name}.run({call_args_str}, grid=grid({grid_str}), stream={stream_name})"
                )
            else:
                stream_ptr = f"c_void_p({stream_name})"
                self.writeline(f"{name}.{name}({call_args_str}, {stream_ptr})")
        else:
            self.writeline(self.wrap_kernel_call(name, call_args))

    def writeline(self, line):
        self.lines.append(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    def val_to_cpp_arg_str(self, type_, val, is_legacy_abi) -> str:
        raise NotImplementedError()

    def val_to_arg_str(self, s):
        if isinstance(s, SymTypes):
            return pexpr(sympy.expand(repr(s)))
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                ref: Any

                def __repr__(self):
                    return self.ref

            return repr(type(s)(Shim(self.val_to_arg_str(a)) for a in s))
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        elif isinstance(s, (ComputedBuffer, InputBuffer, ReinterpretView)):
            return s.codegen_reference()
        else:
            return repr(s)

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return self.make_allocation(buffer.get_name(), device, dtype, shape, stride)

    def make_allocation(self, name, device, dtype, shape, stride):
        try:
            expected = tuple(ir.make_contiguous_strides_for(shape))
        except Exception:  # cannot determine truth value of Relational
            expected = None
        if stride == expected:
            return (
                f"{name} = empty("
                f"{self.codegen_shape_tuple(shape)}, "
                f"device='{device.type}', dtype={dtype})"
            )
        else:
            return (
                f"{name} = empty_strided("
                f"{self.codegen_shape_tuple(shape)}, "
                f"{self.codegen_shape_tuple(stride)}, "
                f"device='{device.type}', dtype={dtype})"
            )

    def make_tensor_alias(self, new_name, old_name, comment=""):
        return f"{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}"

    def make_buffer_free(self, buffer):
        return f"del {buffer.get_name()}"

    def make_free_by_names(self, names_to_del: List[str]):
        return f"del {', '.join(name for name in names_to_del)}"

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"

    def make_buffer_reuse(self, old, new, delete_old: bool):
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

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare_maybe_reference}{name} = {layout.view.codegen_reference()}{self.ending}  "
                f"{self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer):
        assert (
            buffer.get_workspace_size() == 0
        ), "Only support zero workspace size for now!"

        name = buffer.get_name()

        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)
        if isinstance(
            buffer,
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayout):
            return
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(
                layout.view, ir.ReinterpretView
            ), f"unexpected {type(layout.view)}: {layout.view}"
            self.codegen_allocation(layout.view.data)
            self.codegen_deferred_allocation(name, layout)
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        assert (
            buffer.get_workspace_size() == 0
        ), "Only support zero workspace size for now!"

        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, input_buffer, output_buffer=None):
        name = input_buffer.get_name()
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        ):
            return False

        return True

    def did_reuse(self, buffer, reused_buffer):
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

    def codegen_unbacked_symbol_decl(self, symbol):
        name = str(symbol)
        if name in self.unbacked_symbol_decls:
            return name
        else:
            # When in CppWrapperCodeGen, we should only generate the declaration once
            self.unbacked_symbol_decls.add(name)
            return self.declare + name

    @staticmethod
    def statically_known_int_or_none(x):
        try:
            val = V.graph._shape_env._maybe_evaluate_static(x)
            return int(x)
        except Exception:
            return None

    @staticmethod
    def statically_known_list_of_ints_or_none(lst):
        result = []
        for x in lst:
            num = WrapperCodeGen.statically_known_int_or_none(x)
            if num is None:
                return None
            result.append(num)
        return result

    @staticmethod
    def is_statically_known_list_of_ints(lst):
        return WrapperCodeGen.statically_known_list_of_ints_or_none(lst) is not None

    @staticmethod
    def static_shape_for_buffer_or_none(buffer):
        return WrapperCodeGen.statically_known_list_of_ints_or_none(buffer.get_size())

    @staticmethod
    def can_prove_buffer_has_static_shape(buffer):
        return WrapperCodeGen.static_shape_for_buffer_or_none(buffer) is not None


class CppWrapperCodeGen(WrapperCodeGen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        super().__init__()

        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.comment = "//"
        self.namespace = "at::"
        self.none_str = "at::Tensor()"
        self.extern_call_ops = set()
        self.size = "sizes()"
        self.stride = "strides()"
        self.call_func_name = "inductor_entry_cpp"
        self.cuda = False
        self.supports_intermediate_hooks = False
        self.outputs_need_copy = set()
        self.kernel_callsite_id = count()
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars = set()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self.used_cached_dtypes = set()
        self.cached_output_id = count()

        from .cpp import cexpr, CppPrinter

        self.expr_printer = cexpr

        # CppPrinter sometimes calls at::native functions which causes problems in
        # the ABI-compatible mode. Currently we are hitting this problem when codegen
        # Grid computation expressions, but we my need to fix other size computation
        # as well.
        class GridExprCppPrinter(CppPrinter):
            def _print_FloorDiv(self, expr):
                x, div = expr.args
                x = self.paren(self.doprint(x))
                div = self.paren(self.doprint(div))
                assert expr.is_integer, "Expect integers in GridExprPrinter"
                return f"({x}/{div})"

        self.grid_expr_printer = GridExprCppPrinter().doprint

    def generate_kernel_call(
        self,
        name,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
    ):
        """
        Generates kernel call code.

        cuda: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if cuda:
            return super().generate_kernel_call(
                name, call_args, grid, device_index, cuda, triton
            )
        else:
            if V.graph.aot_mode and config.aot_inductor.abi_compatible:
                from .cpp import DTYPE_TO_CPP

                new_args = []
                for arg in call_args:
                    var_name = f"var_{next(self.arg_var_id)}"
                    self.writeline(f"auto* {var_name} = get_data_ptr_wrapper({arg});")
                    dtype = V.graph.get_dtype(arg)
                    cpp_dtype = DTYPE_TO_CPP[dtype]
                    new_args.append(f"({cpp_dtype}*)({var_name})")
                self.writeline(self.wrap_kernel_call(name, new_args))
            else:
                self.writeline(self.wrap_kernel_call(name, call_args))

    def write_constant(self, name, hashed):
        # include a hash so our code cache gives different constants different files
        self.header.writeline(f"// {name} {hashed}")

    def write_header(self):
        if V.graph.aot_mode:
            for header_cpp_file in ("interface.cpp", "implementation.cpp"):
                with open(
                    os.path.join(
                        os.path.dirname(__file__), "aoti_runtime", header_cpp_file
                    )
                ) as f:
                    self.header.splice(f.read())
        else:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                '''
                """
            )

        if config.aot_inductor.abi_compatible:
            self.header.splice("#include <torch/csrc/inductor/aoti_torch/c/shim.h>")
        else:
            self.header.splice(
                """
                #include <ATen/ATen.h>
                #include <ATen/core/dispatch/Dispatcher.h>
                #include <ATen/native/BinaryOps.h>
                #include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
                #include <torch/csrc/inductor/inductor_ops.h>
                #include <torch/types.h>
                #include <ATen/ops/bernoulli_native.h>

                #define reinterpret_tensor torch::inductor::_reinterpret_tensor
                #define alloc_from_pool torch::inductor::_alloc_from_pool
                """
            )

        self.header.splice("#include <c10/util/generic_math.h>")

        from .memory_planning import ALIGN_BYTES

        # Round up to the nearest multiple of ALIGN_BYTES
        # ALIGN_BYTES must be a power of 2
        self.header.splice(
            f"""
            [[maybe_unused]] static int64_t align(int64_t nbytes) {{
              return (nbytes + {ALIGN_BYTES} - 1) & -{ALIGN_BYTES};
            }}
            """
        )

    def mark_output_type(self):
        # mark output type to unwrap tensor back to python scalar
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = dict()
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        if V.graph.aot_mode:
            self.prefix.writeline("namespace torch {")
            self.prefix.writeline("namespace aot_inductor {")

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")

    @staticmethod
    def get_input_cpp_type(input):
        assert config.use_minimal_arrayref_interface
        from .cpp import DTYPE_TO_CPP

        if isinstance(input, sympy.Expr):
            from ..graph import may_get_constant_buffer_dtype

            dtype = may_get_constant_buffer_dtype(input)
            assert dtype is not None, f"Failed to get the dtype of sympy.Expr: {input}"
            return DTYPE_TO_CPP[dtype]
        return f"ArrayRefTensor<{DTYPE_TO_CPP[input.get_dtype()]}>"

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            if config.use_minimal_arrayref_interface and not V.graph.is_const_graph:
                from .cpp import DTYPE_TO_CPP

                input_cpp_types = ", ".join(
                    f"{CppWrapperCodeGen.get_input_cpp_type(x)}"
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

            if V.graph.const_graph:
                self.header.splice(V.graph.const_graph.wrapper_code.header)
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
                if not config.use_runtime_constant_folding:
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
                if config.use_minimal_arrayref_interface:
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
            self.prefix.splice(
                f"""std::vector<at::Tensor> {self.call_func_name}(const std::vector<at::Tensor>& inputs) {{"""
            )
        with self.prefix.indent():
            # assign inputs and outputs in both cases so the later codegen can be simplified
            if not config.use_minimal_arrayref_interface:
                if V.graph.aot_mode:
                    if not V.graph.is_const_graph:
                        if config.aot_inductor.abi_compatible:
                            self.prefix.splice(
                                """
                                    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, num_inputs());
                                """
                            )
                        else:
                            # This looks dumb, but can avoid creating two versions of code in the AOTInductor runtime.
                            self.prefix.splice(
                                """
                                    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, num_inputs());
                                """
                            )
                else:
                    self.prefix.splice(
                        """
                            py::gil_scoped_release release;
                        """
                    )

            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    if config.use_minimal_arrayref_interface:
                        self.prefix.writeline(
                            f"auto {input_key} = std::get<{idx}>(inputs);"
                        )
                        continue
                    # unwrap input tensor back to scalar
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype
                        from .cpp import DTYPE_TO_CPP

                        dtype = may_get_constant_buffer_dtype(
                            V.graph.graph_inputs[input_key]
                        )
                        assert (
                            dtype is not None
                        ), "Fails to get the dtype of the sympy.Expr"
                        cpp_dtype = DTYPE_TO_CPP[dtype]
                        assert (
                            not config.aot_inductor.abi_compatible
                        ), "Need to add .item support for abi_compatible AOTInductor codegen"
                        self.prefix.writeline(
                            f"{cpp_dtype} {input_key} = inputs[{idx}].item<{cpp_dtype}>();"
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
                    if constants_key not in V.graph.used_constants:
                        continue
                    # Weights are stored in constants_ and owned by RAIIAtenTensorHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    if config.aot_inductor.abi_compatible:
                        self.prefix.writeline(
                            f"""auto {constants_key} = constants_->at({idx});"""
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {constants_key} = *tensor_handle_to_tensor_pointer("
                            + f"""constants_->at({idx}));"""
                        )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"auto {constants_key} = inputs[{constants_idx}];"
                    )

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)

            if V.graph.aot_mode:
                if not V.graph.is_const_graph:
                    if config.use_minimal_arrayref_interface:
                        # TODO: input shape checking for regular tensor interface as well?
                        self.codegen_input_numel_asserts()
                    else:
                        self.prefix.writeline("inputs.clear();")
                self.prefix.writeline(
                    "auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
                )

    def codegen_input_numel_asserts(self):
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            numel = buf.get_numel()
            self.prefix.writeline(f"assert_numel({name}, {numel});")

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        if config.aot_inductor.abi_compatible:
            code.writeline(f"int64_t* {name}_size;")
            code.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes({name}, &{name}_size));"
            )
        else:
            super().codegen_input_size_var_decl(code, name)

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        if config.aot_inductor.abi_compatible:
            code.writeline(f"int64_t* {name}_stride;")
            code.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides({name}, &{name}_stride));"
            )
        else:
            super().codegen_input_stride_var_decl(code, name)

    def codegen_model_kernels(self):
        self.prefix.writeline("namespace {")
        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = set(self.src_to_kernel.values())
        declare_kernel.update(self.user_defined_kernel_cache.values())
        declare_kernel.update(V.graph.const_kernels)
        for kernel in declare_kernel:
            self.prefix.writeline(f"    CUfunction {kernel}{{nullptr}};")
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
        self.prefix.splice(
            f"""
            AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                               std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                               std::optional<std::string> cubin_dir)
                : AOTInductorModelBase({num_inputs}, {num_outputs}, {num_constants}, cubin_dir) {{
            """
        )

        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                assert not isinstance(
                    inp, sympy.Expr
                ), f"input {name=} cannot be symbolic"
                self.write_input_output_info("inputs_info_", idx, name)

            for idx, (name, tensor) in enumerate(V.graph.constants.items()):
                assert isinstance(tensor, torch.Tensor)
                self.prefix.writeline(f"""constants_info_[{idx}].name = "{name}";""")
                self.prefix.writeline(
                    f"constants_info_[{idx}].dtype = static_cast<int32_t>({self.codegen_dtype(tensor.dtype)});"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].offset = {tensor.storage_offset()};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].data_size = {tensor.untyped_storage().nbytes()};"
                )
                from_folded = "true" if name in V.graph.folded_constants else "false"
                self.prefix.writeline(
                    f"constants_info_[{idx}].from_folded = {from_folded};"
                )

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
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
            AOTIProxyExecutorHandle proxy_executor
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
                AOTIProxyExecutorHandle proxy_executor
            ) {
            """
        )
        if not config.use_runtime_constant_folding:
            self.prefix.splice(
                """
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
                if name not in V.graph.const_output_index:
                    continue
                else:
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
        if V.graph.aot_mode and not V.graph.is_const_graph:
            self.codegen_model_kernels()
            self.codegen_model_constructor()
            self.codegen_const_run_driver()
        self.write_wrapper_decl()
        return super().generate(is_inference)

    def finalize_prefix(self):
        cached_dtypes_buffer = IndentedBuffer()
        if config.aot_inductor.abi_compatible:
            for dtype in self.used_cached_dtypes:
                cached_dtypes_buffer.writeline(f"CACHE_TORCH_DTYPE({dtype});")
        cached_dtypes_buffer.splice(self.prefix)
        self.prefix = cached_dtypes_buffer

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=False
    ):
        self.header.splice(f"\n{kernel}\n")

    def generate_return(self, output_refs):
        if V.graph.aot_mode:
            cst_names = V.graph.constants.keys()
            arr_iface = (
                not V.graph.is_const_graph and config.use_minimal_arrayref_interface
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
            for idx, output in enumerate(output_refs):
                if config.aot_inductor.abi_compatible:
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
                            cached_output_name = (
                                f"cached_output_{next(self.cached_output_id)}"
                            )
                            output_value_type = f"std::decay_t<decltype(std::get<{idx}>(output_arrayref_tensors).data()[0])>"
                            self.wrapper_call.writeline(
                                f"thread_local RAIIAtenTensorHandle {cached_output_name};"
                            )
                            if output in cst_names:
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
                            if output in cst_names:
                                # See NOTE(return_constant) above.
                                self.wrapper_call.writeline(
                                    f"aoti_torch_clone({output}, &output_handles[{idx}]);"
                                )
                            else:
                                self.wrapper_call.writeline(
                                    f"output_handles[{idx}] = {output}.release();"
                                )
                    self.wrapper_call.writeline("} else {")
                    with self.wrapper_call.indent():
                        use_thread_local_cached_output_tensor(idx, output)
                    self.wrapper_call.writeline("}")

                else:
                    assert (
                        not arr_iface
                    ), "minimal ArrayRef interface is only supported in ABI-compatible mode"
                    if output in cst_names:
                        output_expr = f"{output}.clone()"
                        # See NOTE(return_constant) above.
                    else:
                        output_expr = output
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>("
                        + f"new at::Tensor({output_expr}));"
                    )
            if arr_iface:
                self.wrapper_call.writeline("return output_arrayref_tensors;")

        else:
            self.wrapper_call.writeline(f"return {{{', '.join(output_refs)}}};\n}}")

    def generate_before_suffix(self, result):
        if V.graph.aot_mode and not V.graph.is_const_graph:
            result.writeline("} // AOTInductorModel::run_impl")

    def generate_end(self, result):
        if V.graph.aot_mode:
            if V.graph.is_const_graph:
                result.writeline("} // AOTInductorModel::_const_run_impl")
            else:
                result.writeline("} // namespace aot_inductor")
                result.writeline("} // namespace torch")
            return

        result.writeline("'''\n)")
        # get the hash of the wrapper code to name the extension
        wrapper_call_hash = codecache.code_hash(result.getvalue())
        result.splice(
            f"""
            module = CppWrapperCodeCache.load(cpp_wrapper_src, '{self.call_func_name}', '{wrapper_call_hash}', {self.cuda})
            """
        )

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            return_str = "return f(args_tensor)"
        else:
            outputs = [
                f"outputs[{i}]" if self.output_is_tensor[i] else f"outputs[{i}].item()"
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
            return_str = f"""
                    outputs = f(args_tensor)
                    return {outputs_str}
            """

        args_str = "args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]"
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_cpp function explicitly.
            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            args_str += f"""
                    constants_tensor = {constants_str}
                    args_tensor.extend(constants_tensor)
            """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    {args_str}
                    {return_str}
                return g
            call = _wrap_func(module.{self.call_func_name})
            """
        )

    def generate_c_shim_extern_kernel_call(self, kernel, args):
        # In the abi_compatible mode, we call fallback aten ops through a C shim layer
        self.allow_stack_allocation = False
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]
        shim_fn = f"aoti_torch_{kernel_suffix}"
        # HACK: val_to_arg_str jams multiple arguments together using a comma. If that
        # ever breaks, it needs to be reworked to be able to return multiple arguments,
        # and the split-on-comma code here needs to be removed.
        wrapped_args = []
        for x in args:
            pieces = x.split(", ")
            for piece in pieces:
                # We only really *need* convert_arrayref_tensor_to_tensor for
                # ArrayRefTensors. The code flowing into here uses `0` for nullptr,
                # which convert_arrayref_tensor_to_tensor would blindly coerce to int,
                # so just avoid wrapping integers.
                if not piece.isdigit():
                    piece = f"convert_arrayref_tensor_to_tensor({piece})"
                wrapped_args.append(piece)
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(wrapped_args)}));"
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
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)
        else:
            super().generate_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(self, fallback_kernel, args):
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
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
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError("unsupported type of {output=}")
        args = args + output_args
        assert (
            fallback_kernel.abi_compatible_kernel is not None
        ), f"abi_compatible_kernel is None for {fallback_kernel.python_kernel_name=}"
        self.generate_c_shim_extern_kernel_call(
            fallback_kernel.abi_compatible_kernel, args
        )
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def generate_fallback_kernel(self, fallback_kernel, args):
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_fallback_kernel(fallback_kernel, args)
        else:
            super().generate_fallback_kernel(fallback_kernel, args)

    def generate_extern_kernel_out(self, output_view, codegen_reference, args, kernel):
        if output_view:
            output_as_strided = f"{output_view.codegen_reference()}"
            output_name = f"{output_view.get_name()}_as_strided"
            self.writeline(f"auto {output_name} = {output_as_strided};")

            args.insert(0, output_name)
        else:
            args.insert(0, f"{codegen_reference}")

        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_extern_kernel_call(kernel, args)
        else:
            self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_user_defined_triton_kernel(self, kernel_name, grid, configs, args):
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

        self.generate_kernel_call(
            kernel_name,
            args,
            grid=grid_decision,
            device_index=V.graph.scheduler.current_device.index,
            cuda=True,
            triton=True,
        )

    def generate_scatter_fallback(
        self, output, inputs, kernel, python_kernel_name, src_is_tensor, reduce, kwargs
    ):
        # TODO: support other overload for cpp wrapper and remove the below assertions
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            # call the ABI shim function instead of the ATen one
            kernel = kernel.replace("at::", "aoti_torch_")
        line = f"{kernel}({output}, {','.join(map(str, inputs))}"
        if python_kernel_name == "aten.scatter_":
            if src_is_tensor:
                if reduce:
                    line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
            else:
                assert (
                    reduce is None
                ), "Expect reduce to be None for aten.scatter_ with scalar src"
        else:
            line += f", {','.join(kwargs)}"
        line += f"){self.ending}"
        self.writeline(line)

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.expr_printer(V.graph.sizevars.simplify(x))

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            # in the abi_compatible mode, outputs are returned via arguments
            return name
        else:
            return f"std::get<{index}>({basename})"

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def codegen_dynamic_scalar(self, node):
        from .cpp import DTYPE_TO_ATEN

        (data,) = (t.codegen_reference() for t in node.inputs)
        if node.is_bool:
            self.writeline(f"bool {node.sym} = {data}.item() ? 1 : 0;")
        else:
            convert_type = DTYPE_TO_ATEN[node.inputs[0].get_dtype()].replace(
                "at::k", "to"
            )
            self.writeline(f"auto {node.sym} = {data}.item().{convert_type}();")

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

    def make_free_by_names(self, names_to_del: List[str]):
        return " ".join(f"{name}.reset();" for name in names_to_del)

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        if config.aot_inductor.abi_compatible:
            return f"auto {new_name} = std::move({old_name});  // reuse"
        else:
            return super().codegen_exact_buffer_reuse(old_name, new_name, del_line)

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>());'
        )

    def write_triton_header_once(self):
        pass

    def generate_start_graph(self):
        pass

    def generate_end_graph(self):
        pass

    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            # TODO: Add buf name directly into check_inf_and_nan.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_check_inf_and_nan({buf}));"
            )

    def codegen_device(self, device):
        if config.aot_inductor.abi_compatible:
            return f"cached_torch_device_type_{device.type},{device.index if device.index else 0}"
        else:
            from .cpp import DEVICE_TO_ATEN

            return (
                f"c10::Device({DEVICE_TO_ATEN[device.type]}, {device.index})"
                if device.index is not None
                else f"{DEVICE_TO_ATEN[device.type]}"
            )

    def codegen_dtype(self, dtype):
        if config.aot_inductor.abi_compatible:
            dtype_str = str(dtype).split(".")[-1]
            self.used_cached_dtypes.add(dtype_str)
            return f"cached_torch_dtype_{dtype_str}"
        else:
            from .cpp import DTYPE_TO_ATEN

            return DTYPE_TO_ATEN[dtype]

    @functools.lru_cache(None)
    def codegen_int_array_var(
        self, int_array: str, writer=None, known_statically=False
    ):
        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        # As a result, the emitted int array declarations may appear in a later
        # position of the generated code, so the second pass codegen should not
        # reuse int array declarations generated in the first pass
        if writer is None:
            # The first pass codegen uses `self` as the writer
            writer = self

        var = f"int_array_{next(self.int_array_id)}"
        if var not in self.declared_int_array_vars:
            self.declared_int_array_vars.add(var)
            if known_statically:
                writer.writeline(f"static constexpr int64_t {var}[] = {int_array};")
            else:
                writer.writeline(f"int64_t {var}[] = {int_array};")
        return var

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
        device = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(orig_stride)
        if config.aot_inductor.abi_compatible:
            size_array_var = self.codegen_int_array_var(
                size,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(shape),
            )
            stride_array_var = self.codegen_int_array_var(
                stride,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(orig_stride),
            )
            device_type, device_id = device.split(",")
            device_idx = "this->device_idx_" if V.graph.aot_mode else device_id
            if buffer_if_can_stack_allocate is not None:
                from .cpp import DTYPE_TO_CPP

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

        if V.graph.aot_mode and device.startswith("c10::Device("):
            tensor_device = f"{device.split(',')[0]}, this->device_idx_)"
        else:
            tensor_device = device

        return (
            f"{self.declare}{name} = {self.namespace}empty_strided("
            f"{size}, {stride}, at::TensorOptions({tensor_device}).dtype({dtype_code})){self.ending}"
        )

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        if config.aot_inductor.abi_compatible:
            size = self.codegen_shape_tuple(shape)
            stride = self.codegen_shape_tuple(stride)
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            args = [
                name,
                pexpr(offset),  # bytes not numel
                self.codegen_dtype(dtype),
                str(len(shape)),
                self.codegen_int_array_var(size, self.wrapper_call),
                self.codegen_int_array_var(stride, self.wrapper_call),
                f"&{tmp_name}",
            ]
            self.wrapper_call.writeline(f"AtenTensorHandle {tmp_name};")
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));"
            )
            return f"RAIIAtenTensorHandle({tmp_name})"

        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    self.codegen_dtype(dtype),
                    self.codegen_shape_tuple(shape),
                    self.codegen_shape_tuple(stride),
                ]
            )
        )

    def codegen_reinterpret_view(
        self, data, size_list, stride_list, offset, writer
    ) -> str:
        dim = str(len(size_list))
        size = self.codegen_shape_tuple(size_list)
        stride = self.codegen_shape_tuple(stride_list)
        offset = self.codegen_sizevar(offset)

        if config.aot_inductor.abi_compatible:
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            # Because the memory planning is done in two passes (see the implementation
            # of self.generate), the writeline behavior is different in the two passes.
            if writer is None:
                writer = self

            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    size,
                    writer,
                    known_statically=self.is_statically_known_list_of_ints(size_list),
                ),
                self.codegen_int_array_var(
                    stride,
                    writer,
                    known_statically=self.is_statically_known_list_of_ints(stride_list),
                ),
                offset,
            ]

            def gen_reinterpret_call(writer, args):
                writer.writeline(
                    f"auto {tmp_name} = reinterpret_tensor_wrapper({', '.join(args)});"
                )

            if (
                self.can_stack_allocate_buffer(data)
                and self.is_statically_known_list_of_ints(size_list)
                and self.is_statically_known_list_of_ints(stride_list)
                and ir.is_contiguous_strides_for_shape(stride_list, size_list)
            ):
                gen_reinterpret_call(writer, args)
                return tmp_name

            gen_reinterpret_call(writer, args)

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
            return f"wrap_with_raii_handle_if_needed({tmp_name})"
        else:
            args = [data.get_name(), size, stride, offset]
            return f"reinterpret_tensor({', '.join(args)})"

    def codegen_device_copy(self, src, dst):
        if config.aot_inductor.abi_compatible:
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_tensor_copy_(expensive_copy_to_tensor_if_needed({src}), {dst}));"
            )
        else:
            self.writeline(f"{dst}.copy_({src});")

    def codegen_multi_output(self, name, value):
        # in the abi_compatible mode, outputs are retrieved by passing
        # output pointers, so we skip its codegen here.
        if not config.aot_inductor.abi_compatible:
            super().codegen_multi_output(name, value)

    def generate_extern_kernel_args_decl_if_needed(
        self, op_overload, raw_args, output_args
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
                new_int_args.append(str(arg))
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
                # List [int] or List[SymInt]
                elif isinstance(
                    arg_type.getElementType(), (torch.IntType, torch.SymIntType)
                ):
                    new_int_args.extend([str(a) for a in arg])
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

        def fill_output_arg(arg, return_type):
            if isinstance(return_type, torch.TensorType):
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

        for output_arg in output_args:
            assert output_arg is not None, "Optional return types are not yet supported"
            if isinstance(output_arg, (list, tuple)):
                for out in output_arg:
                    fill_output_arg(out, torch.TensorType.get())
            else:
                fill_output_arg(output_arg, torch.TensorType.get())

        return new_tensor_args, new_int_args

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        name,
        kernel,
        codegen_args,
        cpp_op_schema,
        cpp_kernel_key,
        cpp_kernel_overload_name="",
        op_overload=None,
        raw_args=None,
        outputs=None,
    ):
        if config.is_fbcode():
            assert op_overload is not None
            assert raw_args is not None
            assert outputs is not None

            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(
                name,
                cpp_kernel_key,
                op_overload,
                raw_args,
                outputs,
            )
        else:
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_oss(
                name,
                kernel,
                codegen_args,
                cpp_op_schema,
                cpp_kernel_key,
                cpp_kernel_overload_name,
            )

    def generate_extern_kernel_alloc_and_find_schema_if_needed_oss(
        self,
        name,
        kernel,
        codegen_args,
        cpp_op_schema,
        cpp_kernel_key,
        cpp_kernel_overload_name="",
    ):
        if cpp_kernel_key not in self.extern_call_ops:
            self.writeline(
                f"static auto op_{cpp_kernel_key} = c10::Dispatcher::singleton()"
            )
            self.writeline(
                f'\t.findSchemaOrThrow("{kernel}", "{cpp_kernel_overload_name}")'
            )
            self.writeline(f"\t.typed<{cpp_op_schema}>();")
            self.extern_call_ops.add(cpp_kernel_key)

        self.writeline(
            f"auto {name} = op_{cpp_kernel_key}.call({', '.join(codegen_args)});"
        )

    def generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(
        self,
        name,
        cpp_kernel_key,
        op_overload,
        raw_args,  # contains both args and flatten kwargs
        outputs,
    ):
        def extract_output_name(out):
            assert out is not None, "None, i.e. optional output is not supported"
            if isinstance(out, ir.MultiOutput):
                return out.get_name()
            elif isinstance(out, (list, tuple)):
                return type(out)(extract_output_name(o) for o in out)
            else:
                raise AssertionError(f"Unexpected output: {type(out)}")

        # output_args has the same pytree structure as outputs
        output_args = extract_output_name(outputs)
        if isinstance(output_args, str):
            output_args = [output_args]

        (
            tensor_call_args,
            int_call_args,
        ) = self.generate_extern_kernel_args_decl_if_needed(
            op_overload, raw_args, output_args
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

        self.extern_call_ops.add(cpp_kernel_key)

    def val_to_cpp_arg_str(self, type_, val, is_legacy_abi) -> str:
        if (
            config.aot_inductor.abi_compatible
            and not is_legacy_abi
            and isinstance(type_, torch.OptionalType)
        ):
            if val is None:
                return "0"  # nullptr is not available in C
            if isinstance(val, (bool, int, str, float)):
                var_name = f"var_{next(self.arg_var_id)}"
                self.writeline(f"auto {var_name} = {self.val_to_arg_str(val)};")
                return f"&{var_name}"
            if not isinstance(type_.getElementType(), torch.TensorType):
                return f"&{self.val_to_arg_str(val)}"

        return self.val_to_arg_str(val)

    def val_to_arg_str(self, val) -> str:
        if val is None:
            # When None is passed as an argument, it represents an optional that does not contain a value.
            if config.aot_inductor.abi_compatible:
                return "0"  # nullptr is not available in C
            return "c10::nullopt"
        elif isinstance(val, bool):
            if config.aot_inductor.abi_compatible:
                return "1" if val else "0"
            else:
                return "true" if val else "false"
        elif isinstance(val, int):
            return f"{val}L"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, (ComputedBuffer, InputBuffer, ReinterpretView)):
            return val.codegen_reference()
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        elif isinstance(val, float) and val in [float("inf"), float("-inf")]:
            if val == float("inf"):
                return "std::numeric_limits<float>::infinity()"
            else:
                return "-std::numeric_limits<float>::infinity()"
        elif isinstance(val, (list, tuple)):
            # FIXME handle embedded optional types?
            result = f"{{{', '.join(self.val_to_arg_str(x) for x in val)}}}"
            if config.aot_inductor.abi_compatible:
                static = self.is_statically_known_list_of_ints(val)
                # Need to pass the array length because we can't use std::vector
                return f"{self.codegen_int_array_var(result, known_statically=static)}, {len(val)}"
            else:
                return result
        else:
            return repr(val)


class CudaWrapperCodeGen(CppWrapperCodeGen):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self):
        super().__init__()
        self.grid_id = count()
        self.cuda = True

    def write_header(self):
        super().write_header()

        self.header.splice("#include <filesystem>")
        if not config.aot_inductor.abi_compatible:
            self.header.splice(
                """
                #include <c10/cuda/CUDAGuard.h>
                #include <c10/cuda/CUDAStream.h>
                """
            )

        self.header.splice(
            """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                CUresult code = EXPR;                          \\
                const char *msg;                               \\
                cuGetErrorString(code, &msg);                  \\
                if (code != CUDA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            namespace {

            struct Grid {
                Grid(uint32_t x, uint32_t y, uint32_t z)
                  : grid_x(x), grid_y(y), grid_z(z) {}
                uint32_t grid_x;
                uint32_t grid_y;
                uint32_t grid_z;

                bool is_non_zero() {
                    return grid_x > 0 && grid_y > 0 && grid_z > 0;
                }
            };

            }  // anonymous namespace

            static inline CUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CUmodule mod;
                CUfunction func;
                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    CUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
            """
        )

    def write_get_raw_stream(self, index):
        name = f"stream{index}"
        self.writeline(
            f"cudaStream_t {name} = at::cuda::getCurrentCUDAStream({index});"
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
                self.src_to_kernel.values(), self.user_defined_kernel_cache.values()
            ):
                self.prefix.writeline(f"static CUfunction {kernel} = nullptr;")
            self.prefix.writeline("\n")
        return super().generate(is_inference)

    @functools.lru_cache(None)
    def generate_load_kernel_once(
        self, name: str, mangled_name: str, cubin_path: str, shared_mem: int
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

    def generate_args_decl(self, call_args):
        dynamic_symbols = V.graph.sizevars.free_symbols()
        # TODO: only works for constant now, need type info
        new_args = []
        for arg in call_args:
            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(arg, (sympy.Integer, sympy.Symbol, SymbolicCallArg)):
                self.writeline(f"auto {var_name} = {arg};")
            elif isinstance(arg, sympy.Expr):
                self.writeline(f"auto {var_name} = {self.expr_printer(arg)};")
            elif is_int(arg):
                self.writeline(f"int {var_name} = {arg};")
            elif is_float(arg):
                self.writeline(f"float {var_name} = {arg};")
            elif any(str(arg) == s.name for s in dynamic_symbols):
                self.writeline(f"auto {var_name} = {arg};")
            elif arg == "nullptr":
                self.writeline(f"auto {var_name} = nullptr;")
            elif arg == "c10::nullopt":
                self.writeline(f"auto {var_name} = c10::nullopt;")
            else:
                if config.aot_inductor.abi_compatible:
                    self.writeline(f"CUdeviceptr {var_name};")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, reinterpret_cast<void**>(&{var_name})));"
                    )
                else:
                    self.writeline(
                        f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());"
                    )
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
        self, name, call_args, grid=None, device_index=None, cuda=True, triton=True
    ):
        if not cuda:
            # Even in CudaWrapperCodeGen, we may see cpp kernels
            return super().generate_kernel_call(
                name, call_args, grid, device_index, cuda, triton
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

        self.generate_load_kernel_once(name, mangled_name, cubin_path, shared_mem)

        call_args = self.generate_args_decl(call_args)
        kernel_args_var = f"kernel_args_var_{next(self.kernel_callsite_id)}"
        self.writeline(f"void* {kernel_args_var}[] = {{{call_args}}};")
        stream = (
            "stream" if V.graph.aot_mode else self.write_get_raw_stream(device_index)
        )
        grid_name = f"{name}_grid_{next(self.grid_id)}"
        assert isinstance(
            grid, (list, tuple)
        ), f"expected grid to be a list or tuple but got: {grid=}"

        grid = [V.graph.sizevars.simplify(item) for item in grid]
        grid_has_unbacked_symbols = any(free_unbacked_symbols(item) for item in grid)
        grid_args = [self.grid_expr_printer(item) for item in grid]
        grid_args_str = ", ".join(grid_args)
        self.writeline(f"Grid {grid_name} = Grid({grid_args_str});")

        if grid_has_unbacked_symbols:
            self.writeline(f"if ({grid_name}.is_non_zero()) {{")
        kernel_var_name = f"kernels.{name}" if V.graph.aot_mode else name
        self.writeline(
            "launchKernel({}, {}, {}, {}, {}, {}, {}, {});".format(
                kernel_var_name,
                f"{grid_name}.grid_x",
                f"{grid_name}.grid_y",
                f"{grid_name}.grid_z",
                params["num_warps"],
                params["shared_mem"],
                kernel_args_var,
                stream,
            )
        )
        if grid_has_unbacked_symbols:
            self.writeline("}")
