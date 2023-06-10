import collections
import contextlib
import dataclasses
import functools
import hashlib
import os
import re
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import sympy
from sympy import Expr

import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch.fx.experimental.symbolic_shapes import SymTypes
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..utils import (
    cache_on_self,
    get_benchmark_name,
    LineContext,
    sympy_dot,
    sympy_product,
)
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter


pexpr = PythonPrinter().doprint


def buffer_reuse_key(node: ir.Buffer):
    size = node.get_size()
    stride = node.get_stride()
    last_element = sympy_dot([s - 1 for s in size], stride)
    return (
        node.get_device(),
        node.get_dtype(),
        V.graph.sizevars.simplify(sympy_product(size)),
        # Detect gaps in tensor storage caused by strides
        V.graph.sizevars.size_hint(last_element),
    )


def is_int(s: str):
    try:
        int(s)
    except ValueError:
        return False
    return True


def is_float(s: str):
    try:
        float(s)
    except ValueError:
        return False
    return True


def convert_arg_type(python_type):
    from .cpp import CONTAINER_PYTHON_TO_CPP, PYTHON_TO_CPP

    if python_type == "Tensor":
        # Conversions rules follow https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#func
        return f"at::{python_type} const&"

    if python_type in PYTHON_TO_CPP:
        return PYTHON_TO_CPP[python_type]

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


def convert_return_type(python_type):
    # TODO: only support Tensor as func return type for now
    # TODO: support alias
    assert (
        python_type == "Tensor"
    ), f"only support tensor output for cpp_wrapper, but receive type {python_type}"
    return f"at::{python_type}"


def get_cpp_op_schema(kernel):
    # use x.real_type instead of x.type so that we get ScalarType instead of int
    arg_types = [repr(x.real_type) for x in kernel._schema.arguments]
    arg_names = [x.name for x in kernel._schema.arguments]
    # TODO: only support len(returns) == 1 for now.
    returns = [repr(x.type) for x in kernel._schema.returns]
    assert (
        len(returns) == 1
    ), f"only support 1 single output for cpp_wrapper, but {kernel.__name__} has {len(returns)} outputs"
    return_value = returns[0]
    cpp_return_value = convert_return_type(return_value)

    cpp_arg_type = [
        f"{convert_arg_type(arg_type)} {arg_name}"
        for arg_type, arg_name in zip(arg_types, arg_names)
    ]
    return f"{cpp_return_value}({', '.join(cpp_arg_type)})"


SUPPORTED_FALLBACK_CPP_WRAPPER = [
    "repeat_interleave.Tensor",
    "convert_element_type.default",  # can appear as a fallback if it has a complex input
]


@dataclasses.dataclass
class SymbolicCallArg:
    inner: Any

    def __str__(self):
        return str(self.inner)


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[
            Any, List["FreeIfNotReusedLine"]
        ] = collections.defaultdict(list)

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
class EnterCudaDeviceContextManagerLine:
    device_idx: int
    first_time: bool

    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if self.first_time:
                code.writeline(f"at::cuda::CUDAGuard device_guard({self.device_idx});")
            else:
                code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with torch.cuda._DeviceGuard({self.device_idx}):")
            device_cm_stack.enter_context(code.indent())
            code.writeline(
                f"torch.cuda.set_device({self.device_idx}) # no-op to ensure context"
            )


class ExitCudaDeviceContextManagerLine:
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


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

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
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
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
            self.wrapper.make_buffer_reuse(
                self.node,
                self.reused_as,
            )
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
        self.wrapper_call = IndentedBuffer()
        self.src_to_kernel = {}
        self.kenel_numel_expr = set()
        self.lines = []
        self.declare = ""
        self.ending = ""
        self.open_bracket = "["
        self.closed_bracket = "]"
        self.comment = "#"
        self.namespace = ""
        self.none_str = "None"
        self.size = "size()"
        self.stride = "stride()"
        self.first_device_guard = True
        self.supports_intermediate_hooks = True
        self.expr_printer = pexpr

        self.write_header()
        self.write_prefix()

        for name, value in V.graph.constants.items():
            # include a hash so our code cache gives different constants different files
            hashed = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
            self.header.writeline(f"{name} = None  # {hashed}")

        self.allocated = set()
        self.freed = set()

        # maps from reusing buffer to reused buffer
        self.reuses = dict()

        self.write_get_cuda_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_cuda_stream
        )

        @functools.lru_cache(None)
        def add_import_once(line):
            self.header.writeline(line)

        self.add_import_once = add_import_once
        self._metas = {}

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

                from torch import empty_strided, as_strided, device
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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
            from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
            """
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
        return [x.codegen_reference() for x in V.graph.graph_outputs]

    def mark_output_type(self):
        return

    def codegen_input_size_asserts(self):
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
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
                self.prefix.writeline("torch.cuda.synchronize()")
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()

    def write_get_cuda_stream(self, index):
        self.write_triton_header_once()
        name = f"stream{index}"
        self.writeline(f"{name} = get_cuda_stream({index})")
        return name

    def next_kernel_suffix(self):
        return f"{next(self._names_iter)}"

    def codegen_cuda_device_guard_enter(self, device_idx):
        self.writeline(
            EnterCudaDeviceContextManagerLine(device_idx, self.first_device_guard)
        )
        self.first_device_guard = False

    def codegen_cuda_device_guard_exit(self):
        self.writeline(ExitCudaDeviceContextManagerLine())

    def generate_return(self, output_refs):
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_end(self, result):
        return

    def generate_extern_kernel_alloc(self, output_name, kernel, args, origin_node):
        self.writeline(
            f"{self.declare}{output_name} = {kernel}({', '.join(args)}){self.ending}"
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

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        name,
        kernel,
        codegen_args,
        cpp_op_schema,
        cpp_kernel_key,
        cpp_kernel_overload_name="",
    ):
        self.writeline(f"{name} = {kernel}({', '.join(codegen_args)})")

    @dynamo_timed
    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)

        out_names = V.graph.get_output_names()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.write_triton_header_once()
                self.wrapper_call.writeline("start_graph()")

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

            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(
                    line,
                    (
                        EnterCudaDeviceContextManagerLine,
                        ExitCudaDeviceContextManagerLine,
                    ),
                ):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline("torch.cuda.synchronize()")

            if config.profile_bandwidth:
                self.wrapper_call.writeline("end_graph()")

            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def codegen_inputs(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.Buffer]):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            code.writeline(
                f"{self.declare}{name}_size = {name}.{self.size}{self.ending}"
            )
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            code.writeline(
                f"{self.declare}{name}_stride = {name}.{self.stride}{self.ending}"
            )
            return f"{name}_stride"

        # Assign all symbolic shapes needed to local variables
        needed = set(V.graph.sizevars.var_to_val.keys()) - set(
            V.graph.sizevars.replacements.keys()
        )

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

    def codegen_python_sizevar(self, x: Expr) -> str:
        return pexpr(V.graph.sizevars.simplify(x))

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, index: str) -> str:
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
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(
                        name, shape, stride, value.get_device(), value.get_dtype()
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(
                f"return print_performance(lambda: {call_str}, times=times, repeat=repeat)"
            )

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
                    "from torch._inductor.utils import compiled_module_main",
                    f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)",
                ]
            )

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        metadata_comment = f"{metadata}\n" if metadata else ""
        self.header.splice(f"\n\n{metadata_comment}{name} = {kernel}")

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
        return SymbolicCallArg(expr)

    def wrap_kernel_call(self, name, call_args):
        return f"{name}({', '.join(call_args)}){self.ending}"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        stack.enter_context(self.wrapper_call.indent())

    def generate_kernel_call(
        self, name, call_args, grid=None, device_index=None, cuda=True
    ):
        if cuda:
            call_args_str = ", ".join(pexpr(item) for item in call_args)
            grid_str = ", ".join(pexpr(item) for item in grid)
            stream_name = self.write_get_cuda_stream(
                V.graph.scheduler.current_device.index
            )
            self.writeline(
                f"{name}.run({call_args_str}, grid=grid({grid_str}), stream={stream_name})"
            )
        else:
            self.writeline(self.wrap_kernel_call(name, call_args))

    def writeline(self, line):
        self.lines.append(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    def val_to_str(self, s):
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

            return repr(type(s)(Shim(self.val_to_str(a)) for a in s))
        else:
            return repr(s)

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return (
            f"{buffer.get_name()} = empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"device='{device.type}', dtype={dtype})"
        )

    def make_buffer_free(self, buffer):
        return f"del {buffer.get_name()}"

    def make_buffer_reuse(self, old, new):
        assert old.get_dtype() == new.get_dtype()
        del_line = ""
        if old.get_name() not in V.graph.get_output_names():
            del_line = f"; {self.make_buffer_free(old)}"
        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return f"{self.declare}{new.get_name()} = {old.get_name()}{del_line}  {self.comment} reuse"

        return (
            f"{self.declare}{new.get_name()} = {self.namespace}as_strided({old.get_name()}, "
            f"{self.codegen_shape_tuple(new.get_size())}, "
            f"{self.codegen_shape_tuple(new.get_stride())}){del_line}  {self.comment} reuse"
        )

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare}{name} = {layout.view.codegen_reference()}{self.ending}  {self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer):
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
            if not layout.maybe_guard_aligned():
                V.graph.unaligned_buffers.add(name)
            self.codegen_allocation(layout.view.data)
            self.codegen_deferred_allocation(name, layout)
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        name = buffer.get_name()

        if not config.allow_buffer_reuse:
            self.writeline(self.make_buffer_free(buffer))
            return

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            self.writeline(self.make_buffer_free(buffer))
            return

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, buffer):
        name = buffer.get_name()
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
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


class CppWrapperCodeGen(WrapperCodeGen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        super().__init__()
        self.declare = "auto "
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

        from .cpp import cexpr

        self.expr_printer = cexpr

    def write_header(self):
        if V.graph.aot_mode:
            self.header.splice(
                """
                /* AOTInductor generated code */

                #include <ATen/ScalarOps.h>
                """
            )
        else:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                '''
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
        return

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        self.prefix.splice(
            f"""std::vector<at::Tensor> {self.call_func_name}(const std::vector<at::Tensor>& args) {{"""
        )
        with self.prefix.indent():
            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
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
                        self.prefix.writeline(
                            f"{cpp_dtype} {input_key} = args[{idx}].item<{cpp_dtype}>();"
                        )
                    else:
                        self.prefix.writeline(f"at::Tensor {input_key} = args[{idx}];")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)

            self.wrapper_call.splice(
                """
                c10::optional<at::Scalar> optional_scalar;
                c10::optional<c10::string_view> optional_string;
                torch::List<c10::optional<at::Scalar>> optional_list;
                """
            )

    def generate(self):
        self.write_wrapper_decl()
        return super().generate()

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=False
    ):
        self.header.splice(f"\n{kernel}\n")

    def generate_return(self, output_refs):
        self.wrapper_call.writeline(f"return {{{', '.join(output_refs)}}};\n}}")

    def generate_end(self, result):
        if V.graph.aot_mode:
            return

        result.writeline("'''\n)")
        # get the hash of the wrapper code to name the extension
        wrapper_call_hash = codecache.code_hash(self.wrapper_call.getvalue())
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
        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]
                    {return_str}
                return g
            call = _wrap_func(module.{self.call_func_name})
            """
        )

    def generate_extern_kernel_out(self, output_view, codegen_reference, args, kernel):
        if output_view:
            output_as_strided = f"{output_view.codegen_reference()}"
            output_name = f"{output_view.get_name()}_as_strided"
            self.writeline(f"auto {output_name} = {output_as_strided};")

            args.insert(0, output_name)
        else:
            args.insert(0, f"{codegen_reference}")
        self.writeline(self.wrap_kernel_call(kernel, args))

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.expr_printer(V.graph.sizevars.simplify(x))

    def codegen_tuple_access(self, basename: str, index: str) -> str:
        return f"std::get<{index}>({basename})"

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_layout(), ir.MultiOutputLayout)
            else f"{buffer.get_name()}.reset();"
        )

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>({{}}));'
        )

    def codegen_device(self, device):
        from .cpp import DEVICE_TO_ATEN

        return (
            f"at::device(c10::Device({DEVICE_TO_ATEN[device.type]}, {device.index}))"
            if device.index is not None
            else f"at::device({DEVICE_TO_ATEN[device.type]})"
        )

    def make_buffer_allocation(self, buffer):
        from .cpp import DTYPE_TO_ATEN

        # TODO: map layout here
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        device_str = self.codegen_device
        return (
            f"{self.declare}{buffer.get_name()} = {self.namespace}empty_strided("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"{self.codegen_device(device)}"
            f".dtype({DTYPE_TO_ATEN[dtype]})){self.ending}"
        )

    def generate_extern_kernel_alloc_and_find_schema_if_needed(
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
                f"""
    static auto op_{cpp_kernel_key} =
    c10::Dispatcher::singleton()
        .findSchemaOrThrow(
            \"{kernel}\",
            \"{cpp_kernel_overload_name}\")
        .typed<{cpp_op_schema}>();
            """
            )
            self.extern_call_ops.add(cpp_kernel_key)

        self.writeline(
            f"auto {name} = op_{cpp_kernel_key}.call({', '.join(codegen_args)});"
        )

    def val_to_str(self, val):
        from .cpp import DTYPE_TO_ATEN

        if val is None:
            return self.none_str
        elif isinstance(val, bool):
            return "true" if val else "false"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        elif isinstance(val, torch.dtype):
            return DTYPE_TO_ATEN[val]
        elif isinstance(val, float) and val in [float("inf"), float("-inf")]:
            if val == float("inf"):
                return "std::numeric_limits<float>::infinity()"
            else:
                return "-std::numeric_limits<float>::infinity()"
        elif isinstance(val, (list, tuple)):
            return f"{{{', '.join(list(map(self.val_to_str, val)))}}}"
        else:
            return repr(val)


class CudaWrapperCodeGen(CppWrapperCodeGen):
    """
    Generates cpp wrapper for running on GPU and calls CUDA kernels
    """

    def __init__(self):
        super().__init__()
        self.kernel_callsite_id = count()
        self.arg_var_id = count()
        self.cuda = True

    def write_header(self):
        super().write_header()
        self.prefix.splice(
            """
            #include <ATen/native/BinaryOps.h>
            #include <c10/util/Exception.h>
            #include <c10/cuda/CUDAGuard.h>

            #define AT_CUDA_DRIVER_CHECK_OVERRIDE(EXPR)                         \\
            do {                                                                \\
                CUresult __err = EXPR;                                          \\
                if (__err != CUDA_SUCCESS) {                                    \\
                    AT_ERROR("CUDA driver error: ", static_cast<int>(__err));   \\
                }                                                               \\
            } while (0)

            static inline CUfunction loadKernel(const std::string &filePath,
                    const std::string &funcName) {
                CUmodule mod;
                CUfunction func;
                AT_CUDA_DRIVER_CHECK_OVERRIDE(cuModuleLoad(&mod, filePath.c_str()));
                AT_CUDA_DRIVER_CHECK_OVERRIDE(cuModuleGetFunction(&func, mod, funcName.c_str()));
                return func;
            }

            static inline void launchKernel(
                    CUfunction func,
                    int gridX,
                    int gridY,
                    int gridZ,
                    int numWraps,
                    int sharedMemBytes,
                    void* args[],
                    int device_index) {
                AT_CUDA_DRIVER_CHECK_OVERRIDE(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWraps, 1, 1, sharedMemBytes,
                    at::cuda::getCurrentCUDAStream(device_index), args, nullptr));
            }
            """
        )

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        if not cuda:
            return super().define_kernel(name, kernel, metadata, cuda)

    def generate(self):
        self.prefix.writeline("\n")
        for kernel in self.src_to_kernel.values():
            self.prefix.writeline(f"static CUfunction {kernel} = nullptr;")
        self.prefix.writeline("\n")
        return super().generate()

    def generate_load_kernel(self, name, params):
        mangled_name = params.get("mangled_name", None)
        assert mangled_name is not None, "missing mangled_name"
        cubin_path = params.get("cubin_path", None)
        assert os.path.exists(
            cubin_path
        ), "cubin file should already exist at this moment"

        self.writeline(f"if ({name} == nullptr) {{")
        self.writeline(
            f"""     {name} = loadKernel("{cubin_path}", "{mangled_name}");"""
        )
        self.writeline("}")

    def generate_args_decl(self, call_args):
        # TODO: only works for constant now, need type info
        new_args = []
        for arg in call_args:
            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(
                arg,
                (
                    sympy.Integer,
                    sympy.Symbol,
                    SymbolicCallArg,
                ),
            ):
                self.writeline(f"auto {var_name} = {arg};")
            elif is_int(arg):
                self.writeline(f"int {var_name} = {arg};")
            elif is_float(arg):
                self.writeline(f"float {var_name} = {arg};")
            else:
                self.writeline(
                    f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({arg}.data_ptr());"
                )
            new_args.append(f"&{var_name}")

        return ", ".join(new_args)

    def generate_kernel_call(
        self, name, call_args, grid=None, device_index=None, cuda=True
    ):
        if not cuda:
            return super().generate_kernel_call(
                name, call_args, grid, device_index, cuda
            )

        params = CudaKernelParamCache.get(name)
        assert (
            params is not None
        ), f"cuda kernel parameters for {name} should already exist at this moment"

        self.generate_load_kernel(name, params)

        call_args = self.generate_args_decl(call_args)
        kernel_args_var = f"kernel_args_var_{next(self.kernel_callsite_id)}"
        self.writeline(f"void* {kernel_args_var}[] = {{{call_args}}};")
        self.writeline(
            "launchKernel({}, {}, {}, {}, {}, {}, {}, {});".format(
                name,
                params["grid_x"],
                params["grid_y"],
                params["grid_z"],
                params["num_warps"],
                params["shared_mem"],
                kernel_args_var,
                device_index,
            )
        )
