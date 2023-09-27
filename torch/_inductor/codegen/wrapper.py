import collections
import contextlib
import dataclasses
import functools
import os
import re
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import sympy
from sympy import Expr

import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch.fx.experimental.symbolic_shapes import SymTypes
from torch.fx.node import _get_qualified_name

from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
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


@dataclasses.dataclass
class SymbolicCallArg:
    inner: Any
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[Any, List[FreeIfNotReusedLine]] = collections.defaultdict(
            list
        )

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
        if config.allow_buffer_reuse and key in state:
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

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(self.wrapper.make_buffer_reuse(self.node, self.reused_as))


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
        self.optional_tensor_str = "None"
        self.size = "size()"
        self.stride = "stride()"
        self.last_seen_device_guard_index = None
        self.supports_intermediate_hooks = True
        self.expr_printer = pexpr
        # Not all the dynamic symbols will be used in the generated code. This
        # set contains those actually being defined by something like
        # "{self.declare_shape} s0 = ...". It ensures that we are not going to
        # emit queries for undefined symbols.
        self.defined_symbols = set()

        self.write_header()
        self.write_prefix()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated = set()
        self.freed = set()

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

                from torch import empty_strided, device
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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
                self.prefix.writeline("torch.cuda.synchronize()")
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
        self.writeline(f"{name} = get_cuda_stream({index})")
        return name

    def next_kernel_suffix(self):
        return f"{next(self._names_iter)}"

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(
            EnterCudaDeviceContextManagerLine(
                device_idx, self.last_seen_device_guard_index
            )
        )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self):
        self.writeline(ExitCudaDeviceContextManagerLine())

    def generate_return(self, output_refs):
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_end(self, result):
        return

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        self.writeline(
            f"{self.declare}{output_name} = {extern_kernel.kernel}({', '.join(args)}){self.ending}"
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

    def generate_scatter_fallback(
        self, output, inputs, kernel, fn, src_is_tensor, reduce, kwargs
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

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"{self.declare}{name}_size = {name}.{self.size}{self.ending}")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(
            f"{self.declare}{name}_stride = {name}.{self.stride}{self.ending}"
        )

    def codegen_inputs(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.Buffer]):
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
                self.defined_symbols.add(shape)
                needed.remove(shape)
                code.writeline(f"{self.declare}{shape} = {name}{self.ending}")

        for name, value in graph_inputs_tensors:
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    self.defined_symbols.add(shape)
                    needed.remove(shape)
                    code.writeline(
                        f"{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}"
                    )

        for name, value in graph_inputs_tensors:
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    self.defined_symbols.add(shape)
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

    def codegen_reinterpret_view(self, name, size, stride, offset, writer) -> str:
        size = self.codegen_shape_tuple(size)
        stride = self.codegen_shape_tuple(stride)
        offset = self.codegen_sizevar(offset)
        return f"reinterpret_tensor({name}, {size}, {stride}, {offset})"

    def codegen_device_copy(self, src, dst):
        self.writeline(f"{dst}.copy_({src})")

    def codegen_multi_output(self, name, value):
        self.writeline(f"{self.declare}{name} = {value}{self.ending}")

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
                    "from torch._inductor.wrapper_benchmark import compiled_module_main",
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
        return SymbolicCallArg(expr, tree.numel)

    def wrap_kernel_call(self, name, call_args):
        return f"{name}({', '.join(call_args)}){self.ending}"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline("from torch.profiler import record_function")
        self.wrapper_call.writeline(
            f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):"
        )
        stack.enter_context(self.wrapper_call.indent())

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

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"{self.declare}{new_name} = {old_name}{del_line}  {self.comment} reuse"

    def make_buffer_reuse(self, old, new):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ""
        if old_name not in V.graph.get_output_names():
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old_name, new.get_size(), new.get_stride(), 0, self.wrapper_call
        )
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare}{name} = {layout.view.codegen_reference()}{self.ending}  {self.comment} alias",
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

        layout = buffer.get_layout()
        if isinstance(layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            self.writeline(self.make_buffer_free(buffer))
            return

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


class CppWrapperCodeGen(WrapperCodeGen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        super().__init__()
        from ..ir import OptionalTensor

        self.declare = "auto "
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.comment = "//"
        self.namespace = "at::"
        self.none_str = "at::Tensor()"
        self.optional_tensor_str = repr(OptionalTensor())
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
                    self.writeline(f"void *{var_name}{self.ending}")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr({arg}, &{var_name}));"
                    )
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
                #define reinterpret_tensor torch::inductor::_reinterpret_tensor
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
        dtype: str,
        sizes: List[sympy.Expr],
    ):
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")
        self.prefix.writeline(f"""{info_kind}[{idx}].dtype = "{dtype}";""")
        self.prefix.writeline(f"{info_kind}[{idx}].shape.reserve({len(sizes)});")
        for size in sizes:
            if isinstance(size, sympy.Integer):
                self.prefix.writeline(
                    f"{info_kind}[{idx}].shape.push_back(make_static_dim({size}));"
                )
            else:
                size = V.graph.sizevars.simplify(size)
                # FIXME: handle non-Symbol cases later.
                assert isinstance(
                    size, sympy.Symbol
                ), f"expected {size=} to be a Symbol"
                self.prefix.writeline(
                    f"{info_kind}[{idx}].shape.push_back({size.name});"
                )

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            self.prefix.splice(
                """
                void AOTInductorModel::run_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles, // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed
                    cudaStream_t stream,
                    AOTIProxyExecutorHandle proxy_executor
                ) {
                """
            )
        else:
            self.prefix.splice(
                f"""std::vector<at::Tensor> {self.call_func_name}(const std::vector<at::Tensor>& inputs) {{"""
            )
        with self.prefix.indent():
            # assign inputs and outpus in both cases so the later codegen can be simplified
            if V.graph.aot_mode:
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
                    # Weights are stored in constants_ and owned by RAIIAtenTensorHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    if config.aot_inductor.abi_compatible:
                        self.prefix.writeline(
                            f"""auto {constants_key} = constants_->at("{constants_key}").get();"""
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {constants_key} = *tensor_handle_to_tensor_pointer("
                            + f"""constants_->at("{constants_key}").get());"""
                        )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"auto {constants_key} = inputs[{constants_idx}];"
                    )

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)

            if V.graph.aot_mode:
                self.prefix.writeline("inputs.clear();")
                dynamic_symbols = [
                    s
                    for s in V.graph.sizevars.free_symbols()
                    if s in self.defined_symbols
                ]
                for dim in dynamic_symbols:
                    self.prefix.writeline(
                        f'auto dim_{dim} = find_dynamic_dim("{dim}");'
                    )
                    self.prefix.writeline(f"dim_{dim}->set_value({dim});")

            if not config.aot_inductor.abi_compatible:
                self.wrapper_call.splice(
                    """
                    c10::optional<at::Scalar> optional_scalar;
                    c10::optional<c10::string_view> optional_string;
                    c10::optional<at::Layout> optional_layout;
                    c10::optional<at::Tensor> optional_tensor;
                    c10::List<c10::optional<at::Scalar>> optional_list;
                    """
                )

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

    def codegen_model_constructor(self):
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        auto s0 = make_dynamic_dim("s0", 2048);
        inputs_info_[0].name = "input0";
        inputs_info_[0].shape.reserve(2);
        inputs_info_[0].shape.push_back(make_static_dim(10));
        inputs_info_[0].shape.push_back(make_static_dim(64));
        inputs_info_[1].shape.reserve(2);
        inputs_info_[1].shape.push_back(s0);
        inputs_info_[1].shape.push_back(make_static_dim(64));
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].shape.reserve(2);
        outputs_info_[0].shape.push_back(s0);
        outputs_info_[0].shape.push_back(make_static_dim(10));
        }
        """

        def codegen_dynamic_dims():
            dynamic_symbols = V.graph.sizevars.free_symbols()
            for dim in dynamic_symbols:
                var_to_range = V.graph.sizevars.shape_env.var_to_range
                dim_range = var_to_range.get(dim, None)
                assert (
                    dim_range is not None
                ), f"Could not find dim_range for {dim=} from {var_to_range=}"
                self.prefix.writeline(
                    f'auto {dim.name} = make_dynamic_dim("{dim.name}", {dim_range.lower}, {dim_range.upper});'
                )

        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        self.prefix.splice(
            f"""
            AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map, std::optional<std::string> cubin_dir)
                : AOTInductorModelBase({num_inputs}, {num_outputs}, {num_constants}, cubin_dir) {{
            """
        )

        with self.prefix.indent():
            codegen_dynamic_dims()
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                assert not isinstance(
                    inp, sympy.Expr
                ), f"input {name=} cannot be symbolic"
                sizes = inp.get_size()
                dtype = V.graph.graph_inputs[name].get_dtype()
                self.write_input_output_info("inputs_info_", idx, name, dtype, sizes)

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

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
                )

            self.prefix.writeline("constants_ = constants_map;")

            for idx, output in enumerate(V.graph.graph_outputs):
                assert not isinstance(
                    output, sympy.Expr
                ), f"output {name=} cannot be symbolic"
                sizes = output.get_size()
                name = f"output{idx}"
                dtype = output.get_dtype()
                self.write_input_output_info("outputs_info_", idx, name, dtype, sizes)

        self.prefix.writeline("}")

    def generate(self):
        if V.graph.aot_mode:
            self.codegen_model_constructor()
        self.write_wrapper_decl()
        return super().generate()

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=False
    ):
        self.header.splice(f"\n{kernel}\n")

    def generate_return(self, output_refs):
        if V.graph.aot_mode:
            for idx, output in enumerate(output_refs):
                if config.aot_inductor.abi_compatible:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = {output}.release();"
                    )

                else:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>("
                        + f"new at::Tensor({output}));"
                    )
        else:
            self.wrapper_call.writeline(f"return {{{', '.join(output_refs)}}};\n}}")

    def generate_end(self, result):
        if V.graph.aot_mode:
            result.writeline("} // AOTInductorModel::run_impl")
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
        kernel = "aoti_torch_" + kernel.split("::")[-1]
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({kernel}({', '.join(args)}));")

    def generate_c_shim_extern_kernel_alloc_call(self, extern_kernel, args):
        output_args = []
        output_raii_handles = []
        output_name_base = extern_kernel.get_name()
        for idx, output in enumerate(extern_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                assert (
                    output.indices[0][1] == idx
                ), f"expected {output.indices[1]=} == {idx=} for {output_name_base=}"
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
        self.generate_c_shim_extern_kernel_call(extern_kernel.kernel, args)
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        if V.graph.aot_mode and config.aot_inductor.abi_compatible:
            self.generate_c_shim_extern_kernel_alloc_call(extern_kernel, args)
        else:
            super().generate_extern_kernel_alloc(extern_kernel, args)

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

    def generate_scatter_fallback(
        self, output, inputs, kernel, fn, src_is_tensor, reduce, kwargs
    ):
        # TODO: support other overload for cpp wrapper and remove the below assertions
        line = f"{kernel}({output}, {','.join(map(str, inputs))}"
        if fn == "aten.scatter_":
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

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_layout(), ir.MultiOutputLayout)
            else f"{buffer.get_name()}.reset();"
        )

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        if config.aot_inductor.abi_compatible:
            return f"auto {new_name} = std::move({old_name});  // reuse"
        else:
            return super().codegen_exact_buffer_reuse(old_name, new_name, del_line)

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>());'
        )

    def codegen_device(self, device):
        if config.aot_inductor.abi_compatible:
            return f"aoti_torch_device_type_{device.type}(),{device.index if device.index else 0}"
        else:
            from .cpp import DEVICE_TO_ATEN

            return (
                f"c10::Device({DEVICE_TO_ATEN[device.type]}, {device.index})"
                if device.index is not None
                else f"{DEVICE_TO_ATEN[device.type]}"
            )

    def codegen_dtype(self, dtype):
        if config.aot_inductor.abi_compatible:
            return f"aoti_torch_dtype_{str(dtype).split('.')[-1]}()"
        else:
            from .cpp import DTYPE_TO_ATEN

            return DTYPE_TO_ATEN[dtype]

    @functools.lru_cache(None)
    def codegen_int_array_var(self, int_array: str, writer=None):
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
            writer.writeline(f"int64_t {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        name = buffer.get_name()
        device = self.codegen_device(buffer.get_device())
        dtype = self.codegen_dtype(buffer.get_dtype())
        size = self.codegen_shape_tuple(tuple(buffer.get_size()))
        stride = self.codegen_shape_tuple(tuple(buffer.get_stride()))
        if config.aot_inductor.abi_compatible:
            device_type, device_id = device.split(",")
            args = [
                str(len(buffer.get_size())),
                self.codegen_int_array_var(size, self.wrapper_call),
                self.codegen_int_array_var(stride, self.wrapper_call),
                dtype,
                device_type,
                "this->device_idx_" if V.graph.aot_mode else device_id,
                f"&{name}_handle",
            ]
            self.wrapper_call.writeline(f"AtenTensorHandle {name}_handle;")
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));"
            )
            return f"RAIIAtenTensorHandle {name}({name}_handle);"
        else:
            if V.graph.aot_mode and device.startswith("c10::Device("):
                tensor_device = f"{device.split(',')[0]}, this->device_idx_)"
            else:
                tensor_device = device
            return (
                f"{self.declare}{name} = {self.namespace}empty_strided("
                f"{size}, {stride}, at::TensorOptions({tensor_device}).dtype({dtype}));"
            )

    def codegen_reinterpret_view(self, name, size, stride, offset, writer) -> str:
        dim = str(len(size))
        size = self.codegen_shape_tuple(size)
        stride = self.codegen_shape_tuple(stride)
        offset = self.codegen_sizevar(offset)

        if config.aot_inductor.abi_compatible:
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            # Because the memory planning is done in two passes (see the implementation
            # of self.generate), the writeline behavior is different in the two passes.
            if writer is None:
                writer = self
            args = [
                f"{name}",
                dim,
                self.codegen_int_array_var(size, writer),
                self.codegen_int_array_var(stride, writer),
                offset,
                f"&{tmp_name}",
            ]
            writer.writeline(f"AtenTensorHandle {tmp_name};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor({', '.join(args)}));"
            )
            return f"RAIIAtenTensorHandle({tmp_name})"
        else:
            args = [name, size, stride, offset]
            return f"reinterpret_tensor({', '.join(args)})"

    def codegen_device_copy(self, src, dst):
        if config.aot_inductor.abi_compatible:
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_tensor_copy_({src}, {dst}));"
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

            if isinstance(arg_type, torch.TensorType):
                assert isinstance(arg, (ir.InputBuffer, ir.ComputedBuffer))
                new_tensor_args.append(f"&{arg.name}")
            elif isinstance(arg_type, (torch.IntType, torch.SymIntType)):
                # int or SymInt
                assert isinstance(arg, int)
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
                    new_tensor_args.extend([f"&{a.name}" for a in arg])
                # List[Optional[Tensor]]
                elif isinstance(
                    arg_type.getElementType(), torch.OptionalType
                ) and isinstance(
                    arg_type.getElementType().getElementType(), torch.TensorType
                ):
                    new_tensor_args.extend([f"&{a.name}" for a in arg if a is not None])
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
                        arg_type.getElementType(), static_arg_types
                    ), f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
            else:
                assert isinstance(
                    arg_type, static_arg_types
                ), f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"

        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(arg, return_type):
            if isinstance(return_type, torch.TensorType):
                self.writeline(f"at::Tensor {arg};  // output buffer")
                new_tensor_args.append(f"&{output_arg}")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ):
                # TODO: handle tensor list return type
                raise NotImplementedError("NYI support for return type: List[Tensor]")
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError("NYI support for return type: SymInt")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.SymIntType
            ):
                raise NotImplementedError("NYI support for return type: List[SymInt]")
            else:
                raise AssertionError(f"Unsupport return type found: {return_type}")

        assert (
            len(output_args) == 1
        ), "Support for multiple returns is not implemented yet"
        for output_arg, return_type in zip(output_args, return_types):
            # TODO: check schema here
            # assume it's a tensor now
            if output_arg is not None:
                if isinstance(return_type, torch.OptionalType):
                    fill_output_arg(output_arg, return_type.getElementType())
                else:
                    fill_output_arg(output_arg, return_type)

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
    ):
        if config.is_fbcode():
            assert op_overload is not None
            assert raw_args is not None

            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(
                name,
                cpp_kernel_key,
                op_overload,
                raw_args,
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
    ):
        output_args = [name]

        (
            tensor_call_args,
            int_call_args,
        ) = self.generate_extern_kernel_args_decl_if_needed(
            op_overload, raw_args, output_args
        )

        tensor_args_var = f"tensor_args_var_{next(self.kernel_callsite_id)}"
        tensor_call_args_str = ", ".join(tensor_call_args)
        self.writeline(f"void* {tensor_args_var}[] = {{{tensor_call_args_str}}};")

        int_args_var = f"int_args_var_{next(self.kernel_callsite_id)}"
        int_call_args_str = ", ".join(int_call_args)
        self.writeline(f"int64_t {int_args_var}[] = {{{int_call_args_str}}};")

        extern_kernel_node_index = len(V.graph.extern_kernel_nodes) - 1

        self.writeline(
            f"aoti_torch_proxy_executor_call_function(proxy_executor, "
            f"{extern_kernel_node_index}, "
            f"{len(int_call_args)}, "
            f"{int_args_var}, "
            f"{len(tensor_call_args)}, "
            f"{tensor_args_var});"
        )

        self.extern_call_ops.add(cpp_kernel_key)

    def val_to_arg_str(self, val):
        if val is None:
            # When None is passed as an argument, it represents an optional that does not contain a value.
            return self.optional_tensor_str
        elif isinstance(val, bool):
            if config.aot_inductor.abi_compatible:
                return "1" if val else "0"
            else:
                return "true" if val else "false"
        elif isinstance(val, int):
            return f"{val}L"
        elif isinstance(val, str):
            return f'"{val}"'
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
            result = f"{{{', '.join(list(map(self.val_to_arg_str, val)))}}}"
            if config.aot_inductor.abi_compatible:
                # Need to pass the array length because we can't use std::vector
                return f"{self.codegen_int_array_var(result)}, {len(val)}"
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
            };

            }  // anonymous namespace

            static inline CUfunction loadKernel(
                    const std::string &filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes) {
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
        ), f"cubin file should already exist at this moment: {cubin_path}"

        shared_mem = params.get("shared_mem", 0)
        self.writeline(f"if ({name} == nullptr) {{")
        if V.graph.aot_mode:
            runtime_cubin_path_var = f"var_{next(self.arg_var_id)}"
            self.writeline("    if (this->cubin_dir_.has_value()) {")
            self.writeline(
                f"""        std::filesystem::path {runtime_cubin_path_var}{{this->cubin_dir_.value()}};"""
            )
            self.writeline(
                f"""        {runtime_cubin_path_var} /= "{os.path.basename(cubin_path)}";"""
            )
            self.writeline(
                f"""        {name} = loadKernel({runtime_cubin_path_var}.c_str(), "{mangled_name}", {shared_mem});"""
            )
            self.writeline("    } else {")
            self.writeline(
                f"""        {name} = loadKernel("{cubin_path}", "{mangled_name}", {shared_mem});"""
            )
            self.writeline("    }")
        else:
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
            elif any(str(arg) == s.name for s in dynamic_symbols):
                self.writeline(f"auto {var_name} = {arg};")
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
        ), f"cuda kernel parameters for {name} should already exist at this moment"
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
            return super().generate_kernel_call(
                name, call_args, grid, device_index, cuda, triton
            )

        params = CudaKernelParamCache.get(name)
        assert (
            params is not None
        ), f"cuda kernel parameters for {name} should already exist at this moment"

        self.generate_load_kernel(name, params)

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

        grid_args = [
            self.grid_expr_printer(V.graph.sizevars.simplify(item)) for item in grid
        ]
        grid_args_str = ", ".join(grid_args)
        self.writeline(f"Grid {grid_name} = Grid({grid_args_str});")

        self.writeline(
            "launchKernel({}, {}, {}, {}, {}, {}, {}, {});".format(
                name,
                f"{grid_name}.grid_x",
                f"{grid_name}.grid_y",
                f"{grid_name}.grid_z",
                params["num_warps"],
                params["shared_mem"],
                kernel_args_var,
                stream,
            )
        )
