# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import dis
import functools
import inspect
import logging
import operator
import random
import re
import tempfile
from itertools import count
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import sympy
from sympy import Expr

import torch
import torch._ops
import torch.utils._pytree as pytree
from torch import dtype as torch_dtype
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codegen.debug_utils import DebugPrinterManager
from torch._inductor.codegen.multi_kernel import MultiKernelState
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
    resolve_unbacked_bindings,
    SymTypes,
)
from torch.fx.node import _get_qualified_name
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import async_compile, config, ir
from ..codecache import output_code_log
from ..ir import IRNode, ReinterpretView
from ..runtime import triton_heuristics
from ..runtime.hints import DeviceProperties
from ..utils import (
    cache_on_self,
    DelayReplaceLine,
    get_benchmark_name,
    IndentedBuffer,
    LineContext,
    set_kernel_post_grad_provenance_tracing,
    sympy_product,
    sympy_str,
    sympy_subs,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    CodeGen,
    DeferredLine,
    PythonPrinter,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .cpp_utils import cexpr
from .triton_utils import config_of, should_unwrap_unspec_arg, signature_to_meta


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import triton

    from ..graph import GraphLowering
    from .wrapper_fxir import FxConverter


log = logging.getLogger(__name__)

pexpr = PythonPrinter().doprint


ReuseKey = tuple[torch.device, torch.dtype, str, bool]
BufferLike = Union[ir.Buffer, WorkspaceArg]
FxConversionFunc = Callable[["WrapperLine"], None]


def buffer_reuse_key(node: BufferLike) -> ReuseKey:
    storage_size = V.graph.get_allocation_storage_size(node)
    alignment = node.get_name() not in V.graph.unaligned_buffers
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        alignment,
    )


def can_match_buffer_size(input_buf: BufferLike, output_buf: BufferLike):
    # Return True if input_buf can be re-inplaced for output_buf.
    # This differs from `buffer_reuse_key` for general buffer reuse.
    if input_buf.get_device_or_error() != output_buf.get_device_or_error():
        return False

    if input_buf.get_dtype() != output_buf.get_dtype():
        return False

    input_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(input_buf)
    )
    output_size = V.graph.sizevars.simplify(
        V.graph.get_allocation_storage_size(output_buf)
    )

    if (
        # NB: this is symbolic so that we don't try to reuse a buffer
        # for s0 for s1, just because they happen to share the same
        # size hint
        sympy_str(input_size) == sympy_str(output_size)
    ) or (
        # statically known that 0.95 * input_size <= output_size <= input_size
        V.graph.sizevars.statically_known_geq(output_size, 0.95 * input_size)
        and V.graph.sizevars.statically_known_leq(output_size, input_size)
    ):
        return True

    return False


# TODO: Move to a well known place
TritonMetaParams = dict[str, int]
TritonGrid = Union[
    tuple[Union[int, sympy.Expr], ...], Callable[[TritonMetaParams], tuple[int, ...]]
]


def user_defined_kernel_grid_fn_code(
    name: str,
    configs: list[triton.Config],  # type: ignore[name-defined]
    grids: list[TritonGrid],
    wrapper: Optional[PythonWrapperCodegen] = None,
    original_fxnode_name: Optional[str] = None,
) -> tuple[str, str]:
    output = IndentedBuffer()

    def _convert_to_sympy_expr(item: Union[int, sympy.Expr]) -> sympy.Expr:
        return item if isinstance(item, sympy.Expr) else sympy.Integer(item)

    def determine_grid(
        grid: TritonGrid,
        example_grid: Optional[TritonGrid] = None,
    ):
        """
        This function return a tuple of two values: the first one is for the real grid
        which is used in the generated code; the second one is an example grid with
        concreate values which is used in the autotune block to run the generated
        kernels at compile time.
        """
        if wrapper is None or callable(grid):
            # return as-is when used in eager mode or when grid is callable
            return grid, grid
        # Grid contains ints/Expr, so utilize wrapper's expr printer for codegen
        sympy_grid = tuple(_convert_to_sympy_expr(g) for g in grid)
        if not example_grid:
            example_grid = sympy_grid
        return (
            wrapper.codegen_python_shape_tuple(sympy_grid),
            (
                wrapper.codegen_python_shape_tuple(
                    tuple(
                        wrapper.generate_example_arg_value(g, type(g))
                        for g in example_grid  # type: ignore[union-attr]
                    )
                )
                if config.triton.autotune_at_compile_time
                else None
            ),
        )

    def writeline(line: str, example_grid: Optional[str] = None):
        output.writeline(line)
        if (
            wrapper
            and config.triton.autotune_at_compile_time
            and name not in wrapper.kernel_autotune_names
        ):
            wrapper.kernel_autotune_calls.writeline(example_grid or line)

    fn_name = f"grid_wrapper_for_{name}"
    writeline(f"def {fn_name}(meta):")
    kernel_autotune_calls_indent = (
        wrapper.kernel_autotune_calls.indent()
        if wrapper and config.triton.autotune_at_compile_time
        else contextlib.nullcontext()
    )
    with output.indent(), kernel_autotune_calls_indent:
        if (
            config.triton.autotune_at_compile_time
            and original_fxnode_name
            and V.graph.autotuning_grids
            and original_fxnode_name in V.graph.autotuning_grids
        ):
            example_grids = V.graph.autotuning_grids[original_fxnode_name]
        else:
            example_grids = [None] * len(grids)
        if len(grids) == 1:
            grid, example_grid = determine_grid(grids[0], example_grids[0])
            writeline(f"return {grid}", f"return {example_grid}")
        else:
            assert len(grids) > 1
            assert len(grids) == len(configs)
            seen = OrderedSet[str]()
            # sort the configs from the largest # of kwargs to the smallest to
            # emit the grids in the order of (approximately) decreasing specificity
            # TODO(aakhundov): the sorting below is generally not sufficient, so
            # maybe we'll need to restrict the supported cases to identical kwarg
            # names in all autotuning configs.
            for grid, c, example_grid in sorted(
                zip(grids, configs, example_grids),
                key=lambda x: len(x[1].kwargs),
                reverse=True,
            ):
                if c.kwargs:
                    guards = [
                        f"meta['{name}'] == {val}" for name, val in c.kwargs.items()
                    ]
                    guards = " and ".join(guards)
                else:
                    guards = "True"  # for configs with empty kwargs
                grid, example_grid = determine_grid(grid, example_grid)
                statement = f"if {guards}: return {grid}"
                if statement in seen:
                    continue
                seen.add(statement)
                writeline(statement, f"if {guards}: return {example_grid}")

    return fn_name, output.getvalue()


def user_defined_triton_kernel_transitive_closure_source_code(kernel) -> str:
    """
    Given a triton kernel function pointer collect the transitive closure of
    its dependencies
    """
    compile_wrapper = IndentedBuffer()
    compile_wrapper.splice(kernel.src, strip=True)

    # Also include any possible kernel being called indirectly
    from triton import JITFunction  # type: ignore[name-defined, attr-defined]
    from triton.language import constexpr  # type: ignore[name-defined]

    # global constexpr vars handled above
    symbols_included = OrderedSet([kernel.__name__])

    def traverse(cur_kernel):
        # here we extract the unqualified names (i.e., not attributes and
        # without prepended module name) loaded in the kernel code, which
        # are matched with the co_names and __globals__ below to codegen
        # the respective imports necessary for the kernel compilation
        unqualified_loads = OrderedSet(
            inst.argval
            for inst in dis.Bytecode(cur_kernel.fn)
            if inst.opname == "LOAD_GLOBAL"
        )
        global_annotations = cur_kernel.fn.__globals__.get("__annotations__", {})
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
                elif isinstance(symbol, (int, str, bool, constexpr)):
                    compile_wrapper.newline()
                    if isinstance(symbol, constexpr):
                        symbol_str = f"tl.constexpr({symbol.value!r})"
                    else:
                        symbol_str = f"{symbol!r}"
                    if annotation := global_annotations.get(symbol_name):
                        if isinstance(annotation, type):
                            annotation_code = (
                                f": {annotation.__module__}.{annotation.__name__}"
                            )
                        else:
                            annotation_code = f": {annotation!r}"
                        compile_wrapper.writeline(
                            f"{symbol_name}{annotation_code} = {symbol_str}"
                        )
                    else:
                        compile_wrapper.writeline(f"{symbol_name} = {symbol_str}")
                    symbols_included.add(symbol_name)
                elif (
                    symbol_name in unqualified_loads
                    and symbol_name != "tl"  # already imported
                    and hasattr(symbol, "__module__")
                    # only codegen imports from triton; JITFunctions
                    # imported from other modules will be codegened
                    # in the separate branch above
                    and symbol.__module__.startswith("triton")
                ):
                    # a global symbol imported from triton is referenced
                    # without module qualification (i.e., `store` instead
                    # of `tl.store`): need to codegen an import
                    compile_wrapper.writeline(
                        f"from {symbol.__module__} import {symbol.__name__} as {symbol_name}"
                    )
                    symbols_included.add(symbol_name)

    traverse(kernel)
    return compile_wrapper.getvalue()


@dataclasses.dataclass
class SymbolicCallArg:
    inner: str
    # the original symbolic expression represented by inner
    inner_expr: sympy.Expr

    def __str__(self):
        return str(self.inner)


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: dict[ReuseKey, list[FreeIfNotReusedLine]] = (
            collections.defaultdict(list)
        )
        self.total_allocated_buffer_size: int = 0

    def __contains__(self, key: ReuseKey) -> bool:
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        assert not item.is_reused
        self.reuse_pool[key].append(item)


class WrapperLine:
    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        raise NotImplementedError("FX codegen not yet supported for type {type(self)}")


@dataclasses.dataclass
class EnterSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    graph: GraphLowering

    def __post_init__(self) -> None:
        self.wrapper.push_computed_sizes(self.wrapper.computed_sizes)

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.push_codegened_graph(self.graph)
        code.do_indent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_enter_subgraph


@dataclasses.dataclass
class CommentLine(WrapperLine):
    line: LineContext

    def codegen(self, code: IndentedBuffer) -> None:
        code.writeline(self.line)

    @staticmethod
    def codegen_fx(converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comment


@dataclasses.dataclass
class ExitSubgraphLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def __post_init__(self) -> None:
        self.wrapper.computed_sizes = self.wrapper.pop_computed_sizes()

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper.pop_codegened_graph()
        code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_exit_subgraph


@dataclasses.dataclass
class EnterDeviceContextManagerLine(WrapperLine):
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer) -> None:
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if V.graph.aot_mode:
                # In AOT mode, we have a stream provided as a param. A stream is
                # associated with a device, so we never expect the device to change.
                # CUDAStreamGuard sets the stream and the device.
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"{V.graph.device_ops.cpp_aoti_stream_guard()} stream_guard(stream, this->device_idx_);"
                    )
                else:
                    assert self.last_seen_device_guard_index == self.device_idx, (
                        "AOTInductor only supports running on one CUDA device"
                    )
            else:
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"{V.graph.device_ops.cpp_aoti_device_guard()} device_guard({self.device_idx});"
                    )
                else:
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_enter_device_context_manager


class ExitDeviceContextManagerLine(WrapperLine):
    def codegen(self, code: IndentedBuffer) -> None:
        if not V.graph.cpp_wrapper:
            code.do_unindent()

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_exit_device_context_manager


@dataclasses.dataclass
class ExternKernelAllocLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelAlloc

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs()]
        self.wrapper._generate_extern_kernel_alloc_helper(self.node, args)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_extern_kernel_alloc


@dataclasses.dataclass
class ExternKernelOutLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: ir.ExternKernelOut

    def codegen(self, code: IndentedBuffer) -> None:
        node = self.node
        args = [*node.codegen_args(), *node.codegen_kwargs(skip_out=True)]
        kernel_name = node.get_kernel_name()
        if (
            V.graph.cpp_wrapper
            and node.cpp_kernel_name == "torch::inductor::_mm_plus_mm"
        ):
            # For https://github.com/pytorch/pytorch/issues/128474
            kernel_name = "aoti_torch__mm_plus_mm_out"
        else:
            kernel_name = node.get_kernel_name()
        device = d.type if (d := node.get_device()) else V.graph.device_type
        # set provenance tracing kernel mapping for ExternKernel types
        if config.trace.enabled:
            set_kernel_post_grad_provenance_tracing(node, kernel_name, is_extern=True)
        self.wrapper._generate_extern_kernel_out_helper(
            kernel_name,
            node.codegen_reference(),
            node.output_view.codegen_reference() if node.output_view else None,
            args,
            device,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_extern_kernel_out


@dataclasses.dataclass
class FreeLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    node: Union[BufferLike, ir.TorchBindObject]

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        code.writeline(self.wrapper.make_buffer_free(self.node))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_free


@dataclasses.dataclass
class KernelCallLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    kernel_name: str
    call_args: tuple[Any, ...]
    raw_keys: tuple[Any, ...]
    raw_args: tuple[Any, ...]
    arg_types: list[str]
    triton: bool
    triton_meta: dict[str, Any]
    device: torch.device
    graph_name: str
    original_fxnode_name: str

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._generate_kernel_call_helper(
            self.kernel_name,
            self.call_args,
            triton=self.triton,
            arg_types=self.arg_types,
            raw_keys=self.raw_keys,
            raw_args=self.raw_args,
            triton_meta=self.triton_meta,
            device=self.device,
            graph_name=self.graph_name,
            original_fxnode_name=self.original_fxnode_name,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_kernel_call


@dataclasses.dataclass
class KernelDefinitionLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    kernel_name: str
    kernel_body: str
    metadata: Optional[str] = None
    gpu: bool = True
    cpp_definition: Optional[str] = None

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._define_kernel_helper(
            self.kernel_name,
            self.kernel_body,
            metadata=self.metadata,
            gpu=self.gpu,
            cpp_definition=self.cpp_definition,
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_kernel_definition


@dataclasses.dataclass
class MemoryPlanningLine(WrapperLine):
    wrapper: PythonWrapperCodegen

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        """Second pass to output code"""

    def __str__(self) -> str:
        """
        Emits a string representation that fits on one line.
        """
        args: list[str] = []
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
    node: BufferLike

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if config.allow_buffer_reuse and key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(self.wrapper, free_line.node, self.node)

        if self.node.get_device_or_error().type == "cpu":
            static_shape = self.wrapper.static_shape_for_buffer_or_none(self.node)
            if static_shape is not None:
                state.total_allocated_buffer_size += int(
                    functools.reduce(operator.mul, static_shape, 1)
                )

        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        line = self.wrapper.make_buffer_allocation(self.node)
        code.writeline(line)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_allocate


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: BufferLike
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if len(self.node.get_inputs_that_alias_output()) > 0:
            return self
        if isinstance(self.node.layout, ir.MultiOutputLayout):
            return self
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine(self.wrapper)
        if config.allow_buffer_reuse:
            state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(self.wrapper.make_buffer_free(self.node))

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_free_if_not_reused


@dataclasses.dataclass
class ReinterpretLine(MemoryPlanningLine):
    node: BufferLike
    reused_as: BufferLike
    layout: ir.Layout

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert isinstance(self.layout, ir.NonOwningLayout)
        assert isinstance(self.layout.view, ir.ReinterpretView)
        self.wrapper.codegen_deferred_allocation(
            self.reused_as.get_name(), self.layout.view
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_reinterpret


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: BufferLike
    reused_as: BufferLike
    delete_old: bool = True

    def plan(self, state: MemoryPlanningState) -> MemoryPlanningLine:
        if self.node.get_name() in V.graph.removed_buffers:
            assert self.reused_as.get_name() in V.graph.removed_buffers
            return NullLine(self.wrapper)
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            self.wrapper.make_buffer_reuse(self.node, self.reused_as, self.delete_old)
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_reuse


class NullLine(MemoryPlanningLine):
    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_null


@dataclasses.dataclass
class CommBufferLine(WrapperLine):
    wrapper: PythonWrapperCodegen  # type: ignore[name-defined] # noqa: F821
    node: ir.Buffer

    @property
    def size(self) -> int:
        from torch._inductor.utils import is_symbolic

        numel = self.node.get_numel()
        dtype = self.node.get_dtype()
        if is_symbolic(numel):
            raise AssertionError(
                f"The size of a comm buffer can't be symbolic: {self.node}"
            )
        return int(numel) * dtype.itemsize

    @property
    def comm_buffer_type(self) -> ir.CommBufferType:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.comm_buffer_type

    @property
    def group_name(self) -> str:
        layout = self.node.get_output_spec()
        assert isinstance(layout, ir.CommBufferLayout)
        return layout.group_name


@dataclasses.dataclass
class CommBufferAllocateLine(CommBufferLine):
    def codegen(self, code: IndentedBuffer) -> None:
        assert self.node.get_name() not in V.graph.removed_buffers
        name = self.node.get_name()
        device = self.node.get_device()
        dtype = self.node.get_dtype()
        shape = tuple(self.node.get_size())
        stride = tuple(self.node.get_stride())
        code.writeline(
            self.make_allocation_line(
                self.comm_buffer_type,
                self.group_name,
                self.wrapper,
                name,
                device,
                dtype,
                shape,
                stride,
            )
        )

    @staticmethod
    def make_allocation_line(
        comm_buffer_type, group_name, wrapper, name, device, dtype, shape, stride
    ):
        if comm_buffer_type == ir.CommBufferType.SYMM_MEM:
            return (
                f"{name} = empty_strided_p2p("
                f"{wrapper.codegen_shape_tuple(shape)}, "
                f"{wrapper.codegen_shape_tuple(stride)}, "
                f"{dtype}, "
                f'torch.device("cuda:{device.index}"), '
                f'group_name="{group_name}", '
                f"alloc_id={random.randint(0, 2**64 - 1)})"
            )
        else:
            raise NotImplementedError(
                f"Unsupported comm buffer type: {comm_buffer_type}"
            )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comm_buffer_allocate


@dataclasses.dataclass
class CommBufferFreeLine(CommBufferLine):
    def codegen(self, code: IndentedBuffer) -> None:
        line = self.wrapper.make_buffer_free(self.node)
        code.writeline(f"{line} # {self.comm_buffer_type.value} buffer free")

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_comm_buffer_free


@dataclasses.dataclass
class MultiOutputLine(WrapperLine):
    """
    Given a MultiOutputLayout buffer, indexes actual buffer(s) from the result.
    """

    wrapper: PythonWrapperCodegen
    result_name: str
    arg_name: str
    indices: Sequence[Any]

    def codegen(self, code: IndentedBuffer) -> None:
        def codegen_list_tuple_access(basename, indices):  # type: ignore[no-untyped-def]
            if len(indices) > 0:
                itype, i = indices[0]
                if issubclass(itype, list):
                    return codegen_list_tuple_access(f"{basename}[{i}]", indices[1:])
                elif issubclass(itype, tuple):
                    # cpp wrapper code needs to use std::get<> to access a tuple
                    tuple_access = self.wrapper.codegen_tuple_access(
                        basename, self.result_name, str(i)
                    )
                    return codegen_list_tuple_access(tuple_access, indices[1:])
                elif issubclass(itype, dict):
                    return codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
                else:
                    raise AssertionError("non supported index type: ", itype)
            else:
                return basename

        value = codegen_list_tuple_access(self.arg_name, self.indices)
        code.writeline(
            f"{self.wrapper.declare}{self.result_name} = {value}{self.wrapper.ending}"
        )

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_multi_output


@dataclasses.dataclass
class SymbolicCallArgLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    arg: SymbolicCallArg
    graph: GraphLowering

    def codegen(self, code: IndentedBuffer) -> None:
        self.wrapper._generate_symbolic_call_arg_helper(self.arg, self.graph)

    def codegen_fx(self, converter: FxConverter) -> FxConversionFunc:
        return converter._generate_symbolic_call_arg


BufferName = str
Line = Union[MemoryPlanningLine, LineContext]


class PythonWrapperCodegen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    supports_caching = True  # Whether the output code is cacheable.

    def __init__(self):
        super().__init__()
        self._names_iter: Iterator[int] = count()
        self.args_to_buffers: dict[
            str, Union[None, ir.TensorBox, ir.Buffer, ir.TorchBindObject]
        ] = {}
        self.imports = IndentedBuffer()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.kernel_declarations = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.kernel_autotune_defs = IndentedBuffer()
        self.kernel_autotune_calls = IndentedBuffer()
        self.subgraph_definitions = IndentedBuffer()
        self.kernel_autotune_names = OrderedSet[str]()
        # Map key is the kernel argument name; value is a tuple of the resulting example
        # tensor name with the kernel where that tensor was most recently used.
        self.kernel_autotune_example_args: dict[str, tuple[str, str]] = {}
        self.kernel_autotune_tmp_arg_idx: int = 0
        # If the generated source code is exactly the same, reuse the
        # pre-existing kernel for it
        self.src_to_kernel: dict[str, str] = {}
        self.kernel_numel_expr: OrderedSet[tuple[str, GraphLowering]] = OrderedSet()
        self.lines: list[Line] = []
        self.declare = ""
        self.declare_maybe_reference = ""
        self.ending = ""
        self.comment = "#"
        self.none_str = "None"
        self.move_begin = "std::move(" if V.graph.cpp_wrapper else ""
        self.move_end = ")" if V.graph.cpp_wrapper else ""
        self.last_seen_device_guard_index: Optional[int] = None
        self.supports_intermediate_hooks = True
        self.user_defined_kernel_cache: dict[tuple[Any, ...], tuple[str, Any]] = {}
        self.unbacked_symbol_decls = OrderedSet[str]()  # str of sympy.Symbol
        self.computed_sizes: OrderedSet[sympy.Symbol] = OrderedSet()
        self.launcher_fn_name = None
        # This function can be overridden to change the launcher name
        self.set_launcher_fn_name()

        # this is used for tracking which GraphLowering instance---parent graph
        # or (nested) subgraph---is currently codegened; the primary use case is
        # including the graph instance into a cache key to avoid cross-graph
        # caching during lowering of nested subgraphs
        self.codegened_graph_stack = []
        self.computed_sizes_stack = []

        self.write_header()
        self.write_prefix()
        self.write_kernel_autotune_defs_header()

        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                # include a hash so our code cache puts different constants into different files
                self.write_constant(name, hashed)

        self.allocated = OrderedSet[BufferName]()
        self.freed = OrderedSet[BufferName]()

        # maps from reusing buffer to reused buffer
        self.reuses: dict[BufferName, BufferName] = {}

        self.write_get_raw_stream = functools.lru_cache(None)(  # type: ignore[assignment]
            self.write_get_raw_stream
        )

        @functools.lru_cache(None)
        def add_import_once(line: str) -> None:
            self.imports.writeline(line)
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(line)

        self.add_import_once = add_import_once
        self._metas: dict[str, str] = {}
        self._meta_vars = OrderedSet[str]()
        self.multi_kernel_state = MultiKernelState()
        self.already_codegened_subgraphs = OrderedSet[str]()
        self.allocated_workspaces: dict[str, Any] = {}

        # intermediate tensor value printing utility
        self.debug_printer = DebugPrinterManager(
            debug_printer_level=config.aot_inductor.debug_intermediate_value_printer,
            use_array_ref=config.aot_inductor.allow_stack_allocation,
        )

        # Additional files that are dependent to the wrapper (ex. cubin files)
        self.additional_files = []

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return PythonWrapperCodegen()

    def set_launcher_fn_name(self) -> None:
        self.launcher_fn_name = "call"

    def write_constant(self, name: str, hashed: str) -> None:
        self.header.writeline(f"{name} = None  # {hashed}")

    def write_header(self) -> None:
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        aot_inductor_debug_utils = ""
        if int(config.aot_inductor.debug_intermediate_value_printer) > 0:
            aot_inductor_debug_utils = "from torch._inductor.codegen.debug_utils import _print_debugging_tensor_value_info"
        self.imports.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long, c_int
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from cmath import nanj
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align
                from torch import device, empty_strided
                from {async_compile.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.codegen.multi_kernel import MultiKernelCall
                {aot_inductor_debug_utils}
            """,
            strip=True,
        )
        self.header.splice(
            """
                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                assert_alignment = torch._C._dynamo.guards.assert_alignment
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()
            """,
            strip=True,
        )
        try:
            # Only add empty_strided_p2p() if distributed and SymmetricMemory
            # is available
            from torch._C._distributed_c10d import _SymmetricMemory  # noqa: F401

            self.header.splice(
                """
                empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
                """,
                strip=True,
            )
        except (AttributeError, ImportError):
            pass
        if config.annotate_training:
            self.header.writeline("from torch.cuda import nvtx")

    def include_extra_header(self, header: str):
        pass

    def write_kernel_autotune_defs_header(self) -> None:
        self.kernel_autotune_defs.splice(
            f"""
                import torch
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
            """
        )

    @cache_on_self
    def write_triton_header_once(self) -> None:
        import_str = f"""
            import triton
            import triton.language as tl
            from {triton_heuristics.__name__} import start_graph, end_graph
            """
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.splice(import_str)
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        if not V.graph.cpp_wrapper:
            self.imports.splice(import_str, strip=True)
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        if not V.graph.cpp_wrapper:
            self.imports.writeline(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )

    def add_meta_once(self, meta: TritonMetaParams) -> str:
        meta = repr(meta)
        if meta not in self._metas:
            var = f"meta{len(self._metas)}"
            self._metas[meta] = var
            self.header.writeline(f"{var} = {meta}")
            if config.triton.autotune_at_compile_time:
                self.kernel_autotune_calls.writeline(f"{var} = {meta}")
                self._meta_vars.add(var)
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self) -> list[str]:
        return [
            x.codegen_reference(self.wrapper_call) for x in self.get_graph_outputs()
        ]

    def mark_output_type(self) -> None:
        return

    def get_graph_inputs(
        self,
    ) -> dict[str, Union[ir.TensorBox, ir.TorchBindObject, sympy.Expr]]:
        return V.graph.graph_inputs

    def get_graph_outputs(self) -> list[IRNode]:
        return V.graph.graph_outputs

    def codegen_input_size_asserts(self) -> None:
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue

            # a graph partition may take an IRNode output from a previous partition
            if name not in V.graph.graph_input_names or isinstance(
                buf, ir.GeneratorState
            ):
                continue

            # comparing strides for 0 size tensor is tricky. Ignore them for now.
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_python_shape_tuple(buf.get_size())
            stride = self.codegen_python_shape_tuple(buf.get_stride())
            self.prefix.writeline(f"assert_size_stride({name}, {size}, {stride})")

    def codegen_input_nan_asserts(self) -> None:
        self.prefix.writeline("# make sure graph inputs are not nan/inf")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue

            line = f"assert not {name}.isnan().any().item()"
            self.prefix.writeline(line)
            line = f"assert not {name}.isinf().any().item()"
            self.prefix.writeline(line)

    def write_async_compile_wait(self) -> None:
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile
            """
        )

    def write_args(self, input_names: list[str]):
        lhs = ", ".join(input_names)
        if len(input_names) == 1:
            lhs += ","
        self.prefix.writeline(f"{lhs} = args")
        self.prefix.writeline("args.clear()")

    def write_launcher_fn_call_get_indent(self) -> int:
        if config.graph_partition:
            self.prefix.splice(
                """
                class Runner:
                    def __init__(self, partitions):
                        self.partitions = partitions

                    def recursively_apply_fns(self, fns):
                        new_callables = []
                        for fn, c in zip(fns, self.partitions):
                            new_callables.append(fn(c))
                        self.partitions = new_callables

                    def call(self, args):
                """
            )
            prefix_indent = 2
        else:
            self.prefix.splice(
                f"""
                def {self.launcher_fn_name}(args):
                """
            )
            prefix_indent = 1

        return prefix_indent

    def get_graph_input_names(self) -> list[str]:
        return V.graph.graph_input_names

    def write_prefix(self) -> None:
        assert self.launcher_fn_name is not None
        self.write_async_compile_wait()
        prefix_indent = self.write_launcher_fn_call_get_indent()

        with self.prefix.indent(prefix_indent):
            if config.triton.debug_sync_graph:
                self.prefix.writeline(V.graph.device_ops.synchronize())
            phase = V.graph.get_training_phase()
            if config.annotate_training:
                self.prefix.writeline(
                    f"training_annotation = nvtx._device_range_start('{phase}')"
                )

            if graph_input_names := self.get_graph_input_names():
                self.write_args(graph_input_names)

            self.codegen_inputs()
            self.codegen_input_size_and_nan_asserts()

    def codegen_input_size_and_nan_asserts(self) -> None:
        if config.size_asserts:
            self.codegen_input_size_asserts()
        if config.nan_asserts:
            self.codegen_input_nan_asserts()

    # this function (and below) takes the graph name as input so
    # that stream caching happens per graph instance. this
    # is important for nested subgraph codegening.
    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        self.write_get_raw_stream_header_once()
        name = f"stream{device_idx}"
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(
                f"{name} = get_raw_stream({device_idx})"
            )
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return name
        self.writeline(f"{name} = get_raw_stream({device_idx})")
        return name

    def get_codegened_graph(self):
        return self.codegened_graph_stack[-1]

    def push_codegened_graph(self, graph):
        self.codegened_graph_stack.append(graph)

    def pop_codegened_graph(self):
        return self.codegened_graph_stack.pop()

    def push_computed_sizes(self, computed_sizes):
        from copy import deepcopy

        return self.computed_sizes_stack.append(deepcopy(computed_sizes))

    def pop_computed_sizes(self):
        return self.computed_sizes_stack.pop()

    def next_kernel_suffix(self) -> str:
        return f"{next(self._names_iter)}"

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        self.writeline(
            EnterDeviceContextManagerLine(device_idx, self.last_seen_device_guard_index)
        )
        if config.triton.autotune_at_compile_time:
            # mimic logic of EnterDeviceContextManagerLine.codegen for the autotune code block
            self.write_triton_header_once()
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(device_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.set_device(device_idx)
            )
            self.kernel_autotune_calls.writeline(
                f"stream{device_idx} = get_raw_stream({device_idx})"
            )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self) -> None:
        self.writeline(ExitDeviceContextManagerLine())
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.do_unindent()

    def generate_return(self, output_refs: list[str]) -> None:
        if output_refs:
            self.wrapper_call.writeline("return (" + ", ".join(output_refs) + ", )")
        else:
            self.wrapper_call.writeline("return ()")

    def generate_before_suffix(self, result: IndentedBuffer) -> None:
        return

    def generate_after_suffix(self, result: IndentedBuffer) -> None:
        if config.graph_partition:
            all_partition_name_list = ", ".join(self.all_partition_names) + (
                "," if len(self.all_partition_names) == 1 else ""
            )

            result.splice(
                f"""
                runner = Runner(partitions=[{all_partition_name_list}])
                call = runner.call
                recursively_apply_fns = runner.recursively_apply_fns
                """
            )

    def generate_end(self, result: IndentedBuffer) -> None:
        return

    def generate_fallback_kernel(self, node: ir.FallbackKernel):
        self.writeline(ExternKernelAllocLine(self, node))

    def generate_extern_kernel_alloc(self, node: ir.ExternKernelAlloc):
        node.codegen_comment(self)
        self.writeline(ExternKernelAllocLine(self, node))
        if isinstance(node.layout, ir.Layout):
            node.codegen_size_asserts(self)

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        # If it's a NoneLayout then the extern_kernel should essentially be
        # treated as if it doesn't return anything
        no_return = isinstance(extern_kernel.layout, ir.NoneLayout)
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.get_kernel_name()
        ending = self.ending
        if config.memory_planning and "view_as_complex" in kernel_name:
            # view operation fallbacks cause issues since inductor
            # doesn't know the memory is still needed and might reuse it.
            ending = f".clone(){ending}"

        if no_return:
            self.writeline(f"{self.declare}{kernel_name}({', '.join(args)}){ending}")
        else:
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

    def generate_extern_kernel_out(
        self,
        node: ir.ExternKernelOut,
    ) -> None:
        node.codegen_comment(self)
        self.writeline(ExternKernelOutLine(self, node))

    def _generate_extern_kernel_out_helper(
        self,
        kernel: str,
        out: str,
        out_view: Optional[str],
        args: list[str],
        device: str,
    ) -> None:
        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(args, kernel, None, None, "extern")
        args.append(f"out={out_view if out_view else out}")
        with debug_printer_manager:
            self.writeline(f"{kernel}({', '.join(args)})")

    def _generate_tma_descriptor_call(self, desc, apply_size_hints=False):
        dims = desc.dims
        block_dims = desc.block_dims
        if apply_size_hints:
            dims = tuple(V.graph.sizevars.atomically_apply_size_hint(d) for d in dims)
            block_dims = tuple(
                V.graph.sizevars.atomically_apply_size_hint(d) for d in block_dims
            )

        ptr = f"{desc.tensor.codegen_reference()}.data_ptr()"
        # Explicitly call the Python version of val_to_arg_str
        dims = ", ".join(PythonWrapperCodegen.val_to_arg_str(self, dim) for dim in dims)
        block_dims = ", ".join(
            PythonWrapperCodegen.val_to_arg_str(self, dim) for dim in block_dims
        )
        element_size = PythonWrapperCodegen.val_to_arg_str(self, desc.element_size)
        prefix = "triton.tools.experimental_descriptor"
        fn = f"{prefix}.create_{desc.rank}d_tma_descriptor"
        args = f"{ptr}, {dims}, {block_dims}, {element_size}"
        call = f"{fn}({args})"
        return call

    def generate_tma_descriptor(self, desc):
        call = self._generate_tma_descriptor_call(desc)
        line = f"{desc.name} = {call}{self.ending}"
        self.writeline(line)

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
        line = f"{python_kernel_name}({','.join(map(str, inputs))}"
        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += ", ".join([""] + kwargs)
        else:
            if reduce:
                line += f", reduce={repr(reduce)}"
        line += ")"
        self.writeline(line)

    def generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        indices_str = f"[{', '.join(indices)}]"
        args = [x, indices_str, values, accumulate]
        self.writeline(self.wrap_kernel_call(kernel, args))

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: list[str],
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        self.writeline(f"{buf_name} = {python_kernel_name}({', '.join(codegen_args)})")

    def generate(self, is_inference):
        with dynamo_timed("PythonWrapperCodegen.generate"):
            return self._generate(is_inference)

    def get_wrapper_call_indent(self) -> int:
        if config.graph_partition:
            return 2
        else:
            return 1

    @contextlib.contextmanager
    def set_writeline(self, new: Callable[..., None]) -> Iterator[Callable[..., None]]:
        old = self.writeline
        try:
            self.writeline = new  # type: ignore[method-assign]
            yield new
        finally:
            self.writeline = old  # type: ignore[method-assign]

    def _write_multi_kernel_defs(self) -> None:
        kernel_defs = self.multi_kernel_state.kernel_defs
        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_defs.splice(kernel_defs)
        else:
            self.header.splice(kernel_defs)

    def _generate(self, is_inference):
        if config.profile_bandwidth:
            self.write_triton_header_once()

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()

            self.run_wrapper_ir_passes(is_inference)

            if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                self.generate_reset_kernel_saved_flags()

            # At this point, we shouldn't generate any new memory planning lines.
            # Override writeline to point at the wrapper call, in case it gets called.
            with self.set_writeline(self.wrapper_call.writeline):
                for line in self.lines:
                    if isinstance(line, WrapperLine):
                        line.codegen(self.wrapper_call)
                    else:
                        self.wrapper_call.writeline(line)

            self._write_multi_kernel_defs()

            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline(V.graph.device_ops.synchronize())

            if config.profile_bandwidth:
                self.generate_end_graph()

            if config.triton.store_cubin and not config.triton.autotune_at_compile_time:
                self.generate_save_uncompiled_kernels()

            if config.triton.autotune_at_compile_time:
                self.generate_and_run_autotune_block()

            # cpp_wrapper currently doesn't support nvtx
            if config.annotate_training and not config.cpp_wrapper:
                self.wrapper_call.writeline(
                    "nvtx._device_range_end(training_annotation)"
                )
            self.generate_return(output_refs)

        # Assemble the final code from sections.
        result = IndentedBuffer()
        result.splice(self.imports)
        result.writeline("")
        result.splice(self.header)
        # We do not want the cpp header for intermediate const graph. Headers would be
        # rendered by the main module instead.
        if V.graph.aot_mode and V.graph.cpp_wrapper and V.graph.is_const_graph:
            result = IndentedBuffer()

        # Add subgraph definitions to the result
        result.splice(self.subgraph_definitions)
        self.finalize_prefix()
        result.splice(self.prefix)

        wrapper_call_indent = self.get_wrapper_call_indent()

        with result.indent(wrapper_call_indent):
            result.splice(self.wrapper_call)

        self.generate_before_suffix(result)
        result.splice(self.suffix)
        self.generate_after_suffix(result)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return (
            result.getvaluewithlinemap(),
            self.kernel_declarations.getvaluewithlinemap(),
        )

    def generate_and_run_autotune_block(self):
        """
        Compose self.kernel_autotune_defs and self.kernel_autotune_calls into a single block of
        code and execute it to trigger Triton kernel compilation and auto-tuning
        """
        self.kernel_autotune_defs.splice(
            """
            async_compile.wait(globals())
            del async_compile
        """
        )
        scope = {}  # type: ignore[var-annotated]
        if config.triton.autotune_at_compile_time and V.graph.autotuning_inputs:
            scope = {
                self.get_autotuning_input_name(idx): v  # type: ignore[attr-defined]
                for idx, v in enumerate(V.graph.autotuning_inputs)
            }
        tuning_code = (
            self.kernel_autotune_defs.getvalue()
            + "\n"
            + self.kernel_autotune_calls.getvalue()
        )
        if output_code_log.level == logging.DEBUG:
            # Save the autotuning code block into a file
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                dir=cache_dir(), suffix=".py", delete=False
            ) as f:
                f.write(tuning_code.encode("utf-8"))
                file_path = f.name
            output_code_log.debug(
                "Auto-tuning code written to %s",
                file_path,
            )
        # Execute the code to autotune kernels
        try:
            exec(tuning_code, scope)
        except Exception as e:
            raise RuntimeError(f"Failed to run autotuning code block: {e}") from e

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
        # FIXME(rec): not used
        _total_allocated_buffer_size = sum(
            s.total_allocated_buffer_size for s in past_planning_states
        )

    def run_wrapper_ir_passes(self, is_inference: bool):
        # We disable planning during training because it presently increases peak memory consumption.
        if is_inference and config.memory_planning:
            self.memory_plan()
        else:
            self.memory_plan_reuse()

    def codegen_input_symbol_assignment(
        self,
        name: str,
        value: ir.TensorBox,
        bound_vars: OrderedSet[sympy.Symbol],
    ):
        code = self.prefix

        @functools.lru_cache(None)
        def sizeof(name):
            code.writeline(f"{name}_size = {name}.size()")
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            code.writeline(f"{name}_stride = {name}.stride()")
            return f"{name}_stride"

        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value in bound_vars:
                return
            code.writeline(f"{value} = {name}")
            bound_vars.add(value)
        elif isinstance(value, ir.TensorBox):
            for dim, size in enumerate(value.get_size()):
                if isinstance(size, sympy.Symbol) and size not in bound_vars:
                    code.writeline(f"{size} = {sizeof(name)}[{dim}]")
                    bound_vars.add(size)
            for dim, stride in enumerate(value.get_stride()):
                if isinstance(stride, sympy.Symbol) and stride not in bound_vars:
                    code.writeline(f"{stride} = {strideof(name)}[{dim}]")
                    bound_vars.add(stride)
        elif isinstance(value, ir.TorchBindObject):
            return
        elif isinstance(value, ir.GeneratorState):
            return
        else:
            if torch._inductor.config.graph_partition:
                pass
            else:
                raise AssertionError(f"Unknown value type: {type(value)}")

    def codegen_inputs(self):
        """Assign all symbolic shapes to locals"""
        bound_vars = OrderedSet[sympy.Symbol]()
        # There is a subtle case in the cpp wrapper codegen which requires generating
        # symbol inputs first followed by non-symbol ones.
        #
        # When a dynamic size constraint specified at the Export time is an expression,
        # we need to solve that expression to proper define a symbol in cpp. Thus we
        # are enforcing this iterating order here to make sure all plain size symbols
        # are defined first.
        graph_inputs = self.get_graph_inputs()
        inputs = [
            (k, v) for k, v in graph_inputs.items() if isinstance(v, sympy.Symbol)
        ] + [(k, v) for k, v in graph_inputs.items() if not isinstance(v, sympy.Symbol)]
        for name, value in inputs:
            self.codegen_input_symbol_assignment(name, value, bound_vars)

    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(f"{sym} = {pexpr(expr)}")

    def finalize_prefix(self):
        pass

    def codegen_cpp_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        raise RuntimeError("codegen_cpp_sizevar is only implemented for cpp_wrapper!")

    def codegen_python_sizevar(self, x: Expr, *, simplify: bool = True) -> str:
        return pexpr(x, simplify=simplify)

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        return f"{basename}[{index}]"

    def codegen_python_shape_tuple(self, shape: Sequence[Expr]) -> str:
        parts = [*map(self.codegen_python_sizevar, shape)]
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"

    def codegen_shape_tuple(self, shape: Sequence[Expr]) -> str:
        return self.codegen_python_shape_tuple(shape)

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    pexpr(offset),  # bytes not numel
                    str(dtype),
                    self.codegen_python_shape_tuple(shape),
                    self.codegen_python_shape_tuple(stride),
                ]
            )
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
        if (
            size == data.layout.size
            and stride == data.layout.stride
            and offset == data.layout.offset
        ):
            if dtype is not None and dtype != data.dtype:
                return f"aten.view.dtype({data.get_name()}, {dtype})"
            else:
                return f"{data.get_name()}"
        else:
            size = self.codegen_python_shape_tuple(size)
            stride = self.codegen_python_shape_tuple(stride)
            offset = self.codegen_sizevar(offset)
            if dtype is not None and dtype != data.dtype:
                return f"aten.view.dtype(reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset}), {dtype})"
            else:
                return (
                    f"reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset})"
                )

    def codegen_device_copy(self, src, dst, non_blocking: bool):
        self.writeline(f"{dst}.copy_({src}, {non_blocking})")

    def codegen_multi_output(self, node: ir.MultiOutput):
        result_name = node.get_name()
        arg_name = node.inputs[0].get_name()
        self.writeline(MultiOutputLine(self, result_name, arg_name, node.indices))

    def codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        if len(node.keypath) == 0:
            self.writeline(f"{node.sym} = {data}.item()")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"{node.sym} = 1 if {data}.item() else 0")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            self.writeline(f"{node.sym}_undivided = {data}.item()")
            self.writeline(
                f"assert {node.sym}_undivided % {node.keypath[0].divisor} == 0, "
                f"f'{{{node.sym}_undivided}} not divisible by {node.keypath[0].divisor}'"
            )
            self.writeline(
                f"{node.sym} = {node.sym}_undivided // {node.keypath[0].divisor}"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")
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

        def add_torchbind_input(name, value):
            import pickle

            assert isinstance(value, torch.ScriptObject)

            output.writeline(f"{name} = pickle.loads({pickle.dumps(value)!r})")

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

            if len(V.graph.torchbind_constants) > 0:
                output.writeline("import pickle")
                for name, torchbind_obj in V.graph.torchbind_constants.items():
                    # all the constants are global variables, that's why we need
                    # these 'global var_name' lines
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, torchbind_obj)

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, ir.TorchBindObject):
                    if len(V.graph.torchbind_constants) == 0:
                        # otherwise we have already imported the pickle package
                        output.writeline("import pickle")
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, value.get_real_obj())
                elif isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    # TODO: this fallback and those below actually will generate possibly
                    # invalid benchmark code, because it's not guaranteed 42
                    # is actually a valid value for the kernel in question.
                    # See https://github.com/pytorch/pytorch/issues/124686
                    add_expr_input(name, V.graph.sizevars.size_hint(value, fallback=42))
                elif isinstance(value, ir.GeneratorState):
                    add_expr_input(
                        name,
                        f"torch.cuda.default_generators[{value.device.index}].graphsafe_get_state()",
                    )
                else:
                    shape = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_size()
                    ]
                    stride = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_stride()
                    ]
                    add_fake_input(
                        name,
                        shape,
                        stride,
                        value.get_device(),
                        value.get_dtype(),
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
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ):
        self.writeline(
            KernelDefinitionLine(
                self,
                kernel_name,
                kernel_body,
                metadata=metadata,
                gpu=gpu,
                cpp_definition=cpp_definition,
            )
        )

    @staticmethod
    def _format_kernel_definition(
        kernel_name: str, kernel_body: str, metadata: Optional[str] = None
    ):
        metadata_comment = f"{metadata}\n" if metadata else ""
        body = f"\n\n{metadata_comment}{kernel_name} = {kernel_body}"
        return body

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ):
        if config.triton.autotune_at_compile_time:
            # Skip inserting comments for the autotune block as they may contain cpp style comments
            body = self._format_kernel_definition(
                kernel_name, kernel_body, metadata=None
            )
            self.kernel_autotune_defs.splice(body)
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return

        body = self._format_kernel_definition(
            kernel_name, kernel_body, metadata=metadata
        )
        self.header.splice(body)

    def define_subgraph_launcher_fn(self, fn_code: str):
        self.subgraph_definitions.splice(fn_code)

    def define_user_defined_triton_kernel(
        self,
        kernel,
        configs,
        kwargs,
        restore_value_args,
        reset_to_zero_args,
        grids: list[list[Union[int, sympy.Expr]]],
    ):
        from ..runtime.triton_heuristics import (
            config_to_dict,
            FixedGrid,
            PrecomputedGrid,
        )
        from .common import (
            ConstexprArg,
            KernelArgType,
            SizeArg,
            TensorArg,
            TMADescriptorArg,
        )
        from .triton import gen_common_triton_imports, TritonKernel

        original_name = kernel.__name__
        signature: list[KernelArgType] = []
        constants: dict[str, Any] = {}
        arg_indices: list[int] = []
        equal_to_1_args: list[str] = []

        def add_to_signature(idx, arg):
            signature.append(arg)
            arg_indices.append(idx)

        def add_arg(idx, arg, is_constexpr=False, equals_1=False, equals_none=False):
            if is_constexpr:
                if triton_version_uses_attrs_dict():
                    # tl.constexpr args appear in the signature in new versions of triton,
                    # but not in old versions of triton.
                    add_to_signature(idx, arg)

                if arg.name in kwargs:
                    # the arg may not appear in kwargs if it is an autotuned arg.
                    # in this case, it will be added in triton_heuristics after autotuning.
                    constants[arg.name] = kwargs[arg.name]

            else:
                # the only case where arg name isn't in kwargs, should be
                # when the arg is a constexpr.
                assert arg.name in kwargs

                if equals_1:
                    if triton_version_uses_attrs_dict():
                        # new versions of triton: add the equal-to-1 arg in the signature (labeled as "constexpr"),
                        #                         and add the arg as a constant.
                        # new versions of triton: add the equal-to-1 arg in the signature (labeled as, e.g., "i32"),
                        #                         and add the arg as a constant.
                        add_to_signature(idx, ConstexprArg(name=arg.name))
                    else:
                        add_to_signature(idx, arg)
                    constants[arg.name] = 1
                elif equals_none:
                    if triton_version_uses_attrs_dict():
                        # new versions of triton: add the none arg in the signature (as a constexpr arg) and as a constant
                        # old versions of triton: include the none arg as a constant (but not in the signature)
                        add_to_signature(idx, ConstexprArg(name=arg.name))
                    constants[arg.name] = None
                else:
                    add_to_signature(idx, arg)

        for idx, key in enumerate(kernel.arg_names):
            if idx in kernel.constexprs:
                add_arg(idx, ConstexprArg(name=key), is_constexpr=True)
                continue

            if key not in kwargs:
                continue

            arg = kwargs[key]

            if kwargs[key] is None:
                add_arg(idx, ConstexprArg(name=key), equals_none=True)
            else:
                if isinstance(arg, ir.TMADescriptor):
                    add_arg(
                        idx,
                        TMADescriptorArg(
                            name=key,
                        ),
                    )
                elif isinstance(arg, ir.Buffer):
                    add_arg(
                        idx,
                        TensorArg(
                            name=key,
                            buffer=arg.get_name(),
                            dtype=arg.get_dtype(),
                        ),
                    )
                elif isinstance(arg, ir.ReinterpretView):
                    # for ReinterpretView we use the underlying
                    # buffer name and note the (possibly non-zero)
                    # offset relative to the underlying buffer
                    add_arg(
                        idx,
                        TensorArg(
                            name=key,
                            buffer=arg.data.get_name(),
                            dtype=arg.get_dtype(),
                            offset=arg.layout.offset,
                        ),
                    )
                else:
                    equals_1 = isinstance(
                        arg, (int, sympy.Integer)
                    ) and V.graph.sizevars.statically_known_equals(
                        arg,
                        1,  # type: ignore[arg-type]
                    )
                    add_arg(idx, SizeArg(key, arg), equals_1=equals_1)

        triton_signature = signature_to_meta(
            signature,
            size_dtype=None,  # try to infer based on symints
            indices=arg_indices,
            argdefs=[ArgName(x) for x in kernel.arg_names],
        )
        triton_meta: dict[str, Any] = {
            "signature": triton_signature,
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            # Triton compiler includes equal_to_1 args into constants even
            # when they are not constexpr. otherwise there may be a segfault
            # during launching the Inductor-compiled Triton kernel.
            # TODO(aakhundov): add None args to constants, too. currently, this
            # causes CUDA errors in test_aot_inductor.test_triton_kernel_with_none_input.
            # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
            # https://github.com/triton-lang/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
            "constants": {
                **constants,
                **dict.fromkeys(equal_to_1_args, 1),
            },
            "configs": [
                config_of(
                    signature,
                    indices=arg_indices,
                )
            ],
        }

        if restore_value_args:
            triton_meta["restore_value"] = tuple(restore_value_args)

        if reset_to_zero_args:
            triton_meta["reset_to_zero"] = tuple(reset_to_zero_args)

        if len(grids) == 1:
            # compute the grid in the wrapper and pass it in as an arg
            inductor_meta: dict[str, Any] = FixedGrid.setup_grid_as_args()
            extra_launcher_call_args = [*map(sympy.sympify, grids[0])]
        else:

            def rename_sizes_for_launcher(expr: Union[int, sympy.Expr]) -> sympy.Expr:
                if isinstance(expr, sympy.Expr):
                    symbols = [*expr.free_symbols]
                    if not symbols:
                        return expr
                    symbols.sort(key=str)
                    for sym in symbols:
                        if sym in extra_launcher_args:
                            continue
                        extra_launcher_args[sym] = sympy.Symbol(
                            f"_launcher_s{len(extra_launcher_args)}"
                        )
                    return sympy_subs(expr, extra_launcher_args)
                assert isinstance(expr, int)
                return sympy.Integer(expr)

            extra_launcher_args: dict[sympy.Symbol, sympy.Symbol] = {}
            grids = [[*map(rename_sizes_for_launcher, grid)] for grid in grids]

            assert grids and len(grids) == len(configs)
            precomputed_grids = []
            for grid, cfg in sorted(
                zip(grids, configs), key=lambda x: len(x[1].kwargs), reverse=True
            ):
                precomputed_grids.append(
                    {
                        "config": config_to_dict(cfg),
                        "python": [*map(pexpr, grid)],
                        "cpp": [*map(cexpr, grid)],
                    }
                )
            inductor_meta = {
                "grid_type": PrecomputedGrid.__name__,
                "precomputed_grids": precomputed_grids,
                "extra_launcher_args": [*map(str, extra_launcher_args.values())],
            }
            extra_launcher_call_args = [*extra_launcher_args.keys()]

        # Distinguish between different functions using function id
        cache_key: Any = [id(kernel.fn)]
        if len(configs) > 0:
            for arg in kwargs.values():
                # We need to key on non tensor arg only in autotune mode
                if not isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                    cache_key.append(arg)
        cache_key.append(str(triton_meta))
        cache_key.extend(str(inductor_meta))
        cache_key = tuple(cache_key)
        if cache_key in self.user_defined_kernel_cache:
            return (
                *self.user_defined_kernel_cache[cache_key],
                extra_launcher_call_args,
            )

        name = f"{original_name}_{len(self.user_defined_kernel_cache)}"

        compile_wrapper = IndentedBuffer()
        if config.triton.unique_user_kernel_names:
            compile_wrapper.writeline(f"async_compile.triton({name!r}, '''")
        else:
            compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")

        inductor_meta["kernel_name"] = name
        inductor_meta.update(TritonKernel.inductor_meta_common())

        compile_wrapper.splice(gen_common_triton_imports())
        compile_wrapper.splice(
            f"""
            @triton_heuristics.user_autotune(
                configs={[*map(config_to_dict, configs)]!r},
                inductor_meta={inductor_meta!r},
                triton_meta={triton_meta!r},
                filename=__file__,
                custom_kernel=True,
            )
            @triton.jit
            """
        )
        kernel_src = user_defined_triton_kernel_transitive_closure_source_code(kernel)
        if config.triton.unique_user_kernel_names:
            # We replace the original_name with the unique name.
            kernel_src = kernel_src.replace(f"def {original_name}(", f"def {name}(")
        compile_wrapper.splice(kernel_src)

        current_device = V.graph.get_current_device_or_throw()
        compile_wrapper.writeline(f"''', device_str='{current_device.type}')")
        _, lineno = inspect.getsourcelines(kernel.fn)
        srcfile = inspect.getsourcefile(kernel.fn)
        metadata = f"# Original path: {srcfile}:{lineno}"
        self.define_kernel(
            name,
            compile_wrapper.getvalue(),
            metadata,
        )
        # Add to the cache for the next use
        self.user_defined_kernel_cache[cache_key] = (name, triton_meta)
        return name, triton_meta, extra_launcher_call_args

    def generate_numel_expr(self, kernel_name: str, tree, suffix: Optional[str] = None):
        expr = f"{kernel_name}_{tree.prefix}numel"
        if suffix is not None:
            expr += f"_{suffix}"

        # We can get symbolic expressions here, like s0*64
        # It is fine to have them here, but we need to handle them correctly as their own type
        # This is tricky to do, so we wrap in a custom type, distinct from scalars, but also from sympy*
        # scalars as well.
        # This is handled in `generate_args_decl` which has a correct comment of: TODO: only works for
        # constant now, need type info. I agree, this needs type info, and while this is not true type info
        # it suffices as a type hint for the purposes of producing the correct code for this type.
        arg = SymbolicCallArg(expr, tree.numel)
        self.writeline(SymbolicCallArgLine(self, arg, V.graph))

        return arg

    def _generate_symbolic_call_arg_helper(
        self, arg: SymbolicCallArg, graph: GraphLowering
    ) -> None:
        self.writeline(f"{arg.inner} = {pexpr(arg.inner_expr)}")

    def generate_workspace_allocation(self, ws: WorkspaceArg):
        name = ws.get_name()
        line = AllocateLine(self, ws)
        if ws.zero_mode == WorkspaceZeroMode.UNINITIALIZED:
            self.writeline(line)
        elif ws.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL:
            self.writeline(line)
            self.writeline(self.make_zero_buffer(name))
        elif ws.zero_mode == WorkspaceZeroMode.ZERO_PER_GRAPH:
            prior = self.allocated_workspaces.get(name)
            if prior:
                assert isinstance(prior, AllocateLine) and isinstance(
                    prior.node, WorkspaceArg
                )
                # expand existing allocation
                prior.node = WorkspaceArg.maximum(prior.node, ws)
            else:
                self.writeline(line)
                self.writeline(self.make_zero_buffer(name))
                self.allocated_workspaces[name] = line
        else:
            raise AssertionError(ws.zero_mode)

        if config.triton.autotune_at_compile_time:
            self.kernel_autotune_calls.writeline(
                PythonWrapperCodegen.make_allocation(
                    self,
                    name,
                    ws.device,
                    ws.dtype,
                    shape=(V.graph.sizevars.size_hint(ws.count),),
                    stride=(1,),
                )
            )
            if ws.zero_mode != WorkspaceZeroMode.UNINITIALIZED:
                self.kernel_autotune_calls.writeline(
                    PythonWrapperCodegen.make_zero_buffer(self, name)
                )

    def generate_workspace_deallocation(self, ws: WorkspaceArg):
        if ws.zero_mode != WorkspaceZeroMode.ZERO_PER_GRAPH:
            self.writeline(FreeIfNotReusedLine(self, ws))

    def make_zero_buffer(self, name):
        return f"{name}.zero_(){self.ending}"

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
        self.wrapper_call.writeline(f"end_graph({config.profile_bandwidth_output!r})")

    def generate_reset_kernel_saved_flags(self):
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    kernel.cuda_kernel_saved = False
            """
        )

    def generate_save_uncompiled_kernels(self):
        """
        Precompile and save the CUBINs of the Triton kernels that haven't
        been precompiled and saved as a side effect of running the generated
        JIT model (Python wrapper). This can happen when the model contains
        control flow: only one pass through the control flow operators covers
        the kernels that are saved, the remaining kernels are not launched,
        hence not saved. The main purpose of this codegen is to compile and
        save the Triton kernels outside the active control flow path for
        subsequent AOTInductor code generation and compilation.
        """
        self.wrapper_call.splice(
            f"""
            for kernel in globals().values():
                if isinstance(kernel, {triton_heuristics.__name__}.CachingAutotuner):
                    if not kernel.cuda_kernel_saved:
                        if len(kernel.launchers) == 0:
                            kernel.precompile()
                        kernel.save_gpu_kernel(
                            grid=(0, 0, 0),   # use dummy grid
                            stream="stream",  # use dummy stream
                            launcher=kernel.launchers[0],
                        )
            """
        )

    def prepare_triton_kernel_call(self, call_args):
        def wrap_arg(arg):
            if isinstance(arg, str):
                # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
                return arg + ".item()" if should_unwrap_unspec_arg(arg) else arg
            elif isinstance(arg, (int, float, bool, SymbolicCallArg)):
                return str(arg)
            else:
                return pexpr(V.graph.sizevars.simplify(arg))

        return [wrap_arg(arg) for arg in call_args]

    def generate_example_arg_value(self, arg, arg_type, raw_arg=None):
        if isinstance(arg_type, torch_dtype):
            if isinstance(raw_arg, ir.TMADescriptor):
                # first we generate the underlying buffer
                buf_name = raw_arg.tensor.get_name()
                buf = self.args_to_buffers[arg]
            elif self.args_to_buffers.get(arg):
                buf_name = arg
                buf = self.args_to_buffers[arg]
            else:
                assert raw_arg is not None, (
                    "V.graph.get_buffer(arg) and raw_arg can't be None at the same time"
                )
                buf_name = f"tmp_arg_{self.kernel_autotune_tmp_arg_idx}"
                buf = raw_arg
                self.kernel_autotune_tmp_arg_idx += 1

            assert buf is not None, f"Failed to find a buffer for arg {arg}"
            size = tuple(
                V.graph.sizevars.atomically_apply_size_hint(
                    e,
                    fallback=config.unbacked_symint_fallback,
                )
                for e in buf.get_size()
            )
            allocation_size = tuple(
                V.graph.sizevars.atomically_apply_size_hint(
                    e,
                    fallback=config.unbacked_symint_fallback,
                )
                for e in V.graph.get_allocation_size(buf)
            )
            stride = tuple(
                V.graph.sizevars.atomically_apply_size_hint(
                    e,
                    fallback=config.unbacked_symint_fallback,
                )
                for e in buf.get_stride()
            )
            device = buf.get_device()
            dtype = buf.get_dtype()
            offset = V.graph.sizevars.size_hint(
                buf.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            )
            value = f"generate_example_value({size}, {stride}, '{device}', {dtype}, {offset}, {allocation_size})"
            self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")

            if isinstance(raw_arg, ir.TMADescriptor):
                # generate another line initializing a host-side TMA
                # descriptor from the underlying buffer created above
                value = self._generate_tma_descriptor_call(
                    desc=raw_arg,
                    apply_size_hints=True,
                )
                buf_name = arg
                self.kernel_autotune_calls.writeline(f"{buf_name} = {value}")

            return buf_name
        elif issubclass(arg_type, sympy.Basic) or isinstance(arg, SymbolicCallArg):
            # arg is a symbol or symbolic expression
            if isinstance(arg, str):
                if arg in self._meta_vars:
                    return arg
                if raw_arg is None:
                    return "None"
                arg = raw_arg
            if isinstance(arg, SymbolicCallArg):
                arg = arg.inner_expr
            if arg in V.graph.sizevars.inv_precomputed_replacements:
                arg = V.graph.sizevars.inv_precomputed_replacements[arg]

            return str(
                V.graph.sizevars.atomically_apply_size_hint(
                    arg, fallback=config.unbacked_symint_fallback
                )
            )

        elif isinstance(arg, (str, int, float, bool)):
            return str(arg)
        elif isinstance(arg, list):
            return f"[{', '.join(self.generate_example_arg_value(a, type(a)) for a in arg)}]"
        else:
            raise NotImplementedError(f"Unsupported type {type(arg)}")

    def _grid_dim_str(self, grid_per_dim):
        if isinstance(grid_per_dim, list):
            return (
                "[" + ", ".join(self._grid_dim_str(item) for item in grid_per_dim) + "]"
            )
        else:
            return pexpr(grid_per_dim)

    def generate_kernel_call(
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
        original_fxnode_name=None,
    ):
        """
        Generates kernel call code.

        triton: Defines whether the backend uses Triton for codegen. Otherwise it uses the CUDA language when gpu=True,
                and C++ when gpu=False.
        """

        # Store buffers corresponding to each call arg.
        # This is used to generate example args for autotuning later on.
        self.args_to_buffers.update(
            {
                arg: V.graph.try_get_buffer(arg)
                for arg in call_args
                if isinstance(arg, str)
            }
        )

        device = device or V.graph.get_current_device_or_throw()
        self.writeline(
            KernelCallLine(
                self,
                kernel_name=kernel_name,
                call_args=call_args,
                raw_keys=raw_keys,
                raw_args=raw_args,
                arg_types=arg_types,
                triton=triton,
                triton_meta=triton_meta,
                device=device,
                graph_name=V.graph.name,
                original_fxnode_name=original_fxnode_name,
            )
        )

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
        device = device or V.graph.get_current_device_or_throw()
        if not (triton or device.type != "cpu"):
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            return

        call_args_str = self.prepare_triton_kernel_call(call_args)
        call_args_str = ", ".join(call_args_str)
        stream_name = PythonWrapperCodegen.write_get_raw_stream(
            self, device.index, graph_name
        )
        if not triton:
            stream_ptr = f"c_void_p({stream_name})"
            self.writeline(
                f"{kernel_name}.{kernel_name}({call_args_str}, {stream_ptr})"
            )
            return

        self.write_triton_header_once()

        if (
            config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            # Create example args for autotune in a separate epilogue
            assert arg_types is not None and len(call_args) == len(arg_types), (
                "call_args and arg_types do not match"
            )

            def get_autotune_deletion_call() -> str:
                """After all the autotune kernel calls have been written (i.e.
                self.kernel_autotune_example_args is complete), returns a deletion call
                for all autotune example tensors that are unnecessary after kernel_name
                is called."""
                tensors_to_delete = [
                    tensor
                    for tensor, kn in self.kernel_autotune_example_args.values()
                    if kn == kernel_name
                ]
                if tensors_to_delete:
                    return f"del {', '.join(tensors_to_delete)}\n"
                return ""

            all_args = []
            if raw_args is None:
                # create a dummy raw_args for uniform behavior in the following loop
                assert raw_keys is None, "keys are not None but args are"
                raw_keys = [None] * len(call_args)
                raw_args = [None] * len(call_args)
            else:
                assert len(raw_args) == len(call_args), (
                    "call_args and raw_args do not match"
                )

            autotune_args = None
            if original_fxnode_name and V.graph.autotuning_mapping:
                autotune_args = V.graph.autotuning_mapping.get(
                    original_fxnode_name, None
                )
            for i, (arg, arg_type, raw_key, raw_arg) in enumerate(
                zip(call_args, arg_types, raw_keys, raw_args)
            ):
                key = None
                if isinstance(arg, str) and "=" in str(arg):
                    # arg may be passed in a kwarg style, and then we need to extract its value
                    key, arg = arg.split("=")

                triton_input: Optional[str] = None
                if autotune_args and raw_key in autotune_args:
                    triton_input = self.get_autotuning_input_name(  # type: ignore[attr-defined]
                        autotune_args[raw_key]
                    )

                if triton_input:
                    arg_str = triton_input
                elif isinstance(arg_type, torch_dtype):
                    # workspace allocation is already generated by `generate_workspace_allocation()`
                    # in `TritonKernel.call_kernel()`.
                    if re.match(r"^(workspace|semaphore)", arg):
                        arg_str = arg
                    elif arg not in self.kernel_autotune_example_args:
                        arg_str = self.generate_example_arg_value(
                            arg, arg_type, raw_arg
                        )
                    else:
                        arg_str = self.kernel_autotune_example_args[arg][0]
                    self.kernel_autotune_example_args[arg] = (arg_str, kernel_name)
                else:
                    arg_str = self.generate_example_arg_value(arg, arg_type, raw_arg)
                all_args.append(arg_str if key is None else f"{key}={arg_str}")

            self.kernel_autotune_calls.writeline(
                f"{kernel_name}.run({', '.join(all_args)}, stream={stream_name})"
            )
            self.kernel_autotune_calls.writeline(
                DelayReplaceLine("<del_call>", get_autotune_deletion_call, "<del_call>")
            )
            self.kernel_autotune_names.add(kernel_name)
            if V.graph.cpp_wrapper:
                # For cpp wrapper, no need to continue codegen for the main body
                return

        # add debug printer code for triton kernel calls at (jit) inductor level
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(call_args, kernel_name, arg_types, None)
        with debug_printer_manager:
            self.writeline(f"{kernel_name}.run({call_args_str}, stream={stream_name})")
        self.write_triton_header_once()

    def writeline(self, line):
        self.lines.append(line)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    def val_to_arg_str(self, s, type_=None):
        from torch.utils._triton import has_triton_package

        if has_triton_package():
            import triton

        if isinstance(s, SymTypes):
            return pexpr(s.node.expr)
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                ref: Any

                def __repr__(self):
                    return self.ref

            # Explicitly call the Python version of val_to_arg_str
            return repr(
                type(s)(Shim(PythonWrapperCodegen.val_to_arg_str(self, a)) for a in s)
            )
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        elif isinstance(s, (ir.Buffer, ir.MutableBox, ReinterpretView)):
            return s.codegen_reference()
        elif has_triton_package() and isinstance(s, triton.language.dtype):  # type: ignore[possibly-undefined]
            return repr(s)
        elif isinstance(s, ir.GeneratorState):
            return s.codegen_reference()
        else:
            return repr(s)

    # The following methods are for memory management
    def make_buffer_allocation(self, buffer: BufferLike):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        allocation_shape = tuple(V.graph.get_allocation_size(buffer))
        stride = tuple(buffer.get_stride())
        return self.make_allocation(
            buffer.get_name(), device, dtype, shape, stride, allocation_shape
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None
    ):
        if allocation_shape is None:
            allocation_shape = shape

        codegen_shape_tuple = self.codegen_python_shape_tuple(shape)
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            allocation_shape
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(stride)
        if device.type in ("cpu", "cuda", "xpu"):
            # optimized path for faster allocations, saving ~2us versus the stuff below
            out = (
                f"{name} = empty_strided_{device.type}("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{dtype})"
            )
        # all other devices:
        else:
            out = (
                f"{name} = empty_strided("
                f"{codegen_allocation_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"device='{device.type}', dtype={dtype})"
            )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            # need an extra as_strided call
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"
        return out

    def make_comment(self, line):
        self.writeline(CommentLine(line))

    def make_tensor_alias(self, new_name, old_name, comment=""):
        return f"{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}"

    def make_buffer_free(self, buffer: Union[BufferLike, ir.TorchBindObject]):
        return f"del {buffer.get_name()}"

    def make_free_by_names(self, names_to_del: list[str]):
        return f"del {', '.join(name for name in names_to_del)}"

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"{self.declare_maybe_reference}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse"

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        reinterpret_view = self.codegen_reinterpret_view(
            old, new.get_size(), new.get_stride(), 0, self.wrapper_call.writeline
        )
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def codegen_deferred_allocation(self, name: str, view: ir.ReinterpretView) -> None:
        self.writeline(
            DeferredLine(
                name,
                f"{self.declare}{name} = {view.codegen_reference()}{self.ending}  {self.comment} alias",
            )
        )

    def codegen_allocation(self, buffer: ir.Buffer):
        name = buffer.get_name()

        if (
            name in V.graph.removed_buffers
            or name in self.allocated
            or isinstance(buffer, (ir.DonatedBuffer, ir.SubgraphBuffer))
        ):
            return
        self.allocated.add(name)
        if (
            isinstance(
                buffer.get_defining_op(),
                (ir.ExternKernelAlloc, ir.MultiOutput),
            )
            and not buffer.should_allocate()
        ):
            return

        layout = buffer.get_output_spec()
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            return
        if isinstance(layout, ir.NoneLayout):
            return
        if isinstance(layout, ir.NonOwningLayout):
            assert isinstance(layout.view, ir.ReinterpretView), (
                f"unexpected {type(layout.view)}: {layout.view}"
            )
            box = layout.view.data
            assert isinstance(box, ir.StorageBox), type(box)
            input_buffer = box.data
            assert isinstance(input_buffer, ir.Buffer), type(box)
            self.codegen_allocation(input_buffer)
            self.writeline(ReinterpretLine(self, input_buffer, buffer, layout))
            return

        if isinstance(layout, ir.CommBufferLayout):
            self.writeline(CommBufferAllocateLine(self, buffer))
            return

        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, (ir.InputBuffer, ir.TorchBindObject)):
            self.writeline(FreeLine(self, buffer))
            return

        if isinstance(buffer.get_output_spec(), ir.CommBufferLayout):
            # Comm buffers are not eligible for in-place reuse. Their reuse is
            # achieved exclusively via buffer planning.
            self.writeline(CommBufferFreeLine(self, buffer))
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, input_buffer, output_buffer=None):
        name = input_buffer.get_name()
        return not (
            name in V.graph.removed_buffers
            or (
                name in V.graph.graph_inputs
                and not isinstance(
                    V.graph.graph_inputs_original[name], ir.DonatedBuffer
                )
            )
            or name in V.graph.constants
            or name in V.graph.torchbind_constants
            or name in V.graph.never_reuse_buffers
            or name in self.freed
        )

    def did_reuse(self, buffer, reused_buffer):
        # Check whether a given buffer was reused by a possible reuser in the wrapper codegen
        # Can be consulted from inside ir codegen, e.g. to determine whether a copy is needed
        return (
            buffer.get_name() in self.reuses
            and self.reuses[buffer.get_name()] == reused_buffer.get_name()
        )

    def codegen_inplace_reuse(self, input_buffer: ir.Buffer, output_buffer: ir.Buffer):
        assert can_match_buffer_size(input_buffer, output_buffer)
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
            # When in CppWrapperCpu, we should only generate the declaration once
            self.unbacked_symbol_decls.add(name)
            return self.declare + name

    def codegen_unbacked_symbol_defs_for_outputs(
        self,
        output_name: str,
        outputs: Any,
        unbacked_bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]],
    ) -> None:
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, unbacked_bindings
        )

        if not unbacked_bindings:
            return

        # This code is designed to generate code expressions from symbolic paths (keypaths)
        # associated with certain symbols (unbacked bindings). These keypaths describe how
        # to access the unbacked symbol in a structured way.
        # For example, we might want to generate "u0 = outs[0].stride(1)"", where s = u0, and the keypath
        # describes the structure of "outs[0].stride(1)", like [SequenceKey(0), CallMethodKey("stride"), SequenceKey[1]].
        for s, keypath in unbacked_bindings.items():
            # `go` recursively constructs a code expression by processing each element of
            # the keypath and construct the expression incrementally.
            # For example, given output name outs and keypath [SequenceKey(0), CallMethodKey("stride", 1)],
            # it generates "outs[0]" based on SequenceKey(0), then recursively go("outs[0]", [CallMethodKey("stride"), ...])
            def go(expr: str, keypath: pytree.KeyPath):
                if keypath == ():
                    return expr

                if (
                    len(keypath) >= 2
                    and isinstance(keypath[0], CallMethodKey)
                    and isinstance(keypath[1], pytree.SequenceKey)
                ):
                    return go(
                        f"{expr}.{keypath[0].name}({keypath[1].idx})", keypath[2:]
                    )
                elif isinstance(keypath[0], CallMethodKey):
                    return go(f"{expr}.{keypath[0].name}()", keypath[1:])
                elif isinstance(keypath[0], pytree.SequenceKey):
                    return (
                        go(f"std::get<{keypath[0].idx}>({expr})", keypath[1:])
                        if V.graph.cpp_wrapper
                        else go(f"{expr}[{keypath[0].idx}]", keypath[1:])
                    )
                elif isinstance(keypath[0], DivideByKey):
                    # TODO: need to assert divisibility
                    # TODO: this is invalid C++ codegen
                    return go(f"{expr}.__floordiv__({keypath[0].divisor})", keypath[1:])
                else:
                    raise AssertionError(f"unrecognized keypath {keypath}")

            # `go_outer` manages the top-level logic for generating the final expression.
            # It handles special cases for C++ code generation and adjusts
            # the keypath based on the context (e.g., single vs. multiple outputs).
            def go_outer():  # type: ignore[no-untyped-def]
                if V.graph.cpp_wrapper:
                    # Special handling for the top level buffer access,
                    # because self.get_name() is actually never bound; the
                    # individual output arguments are bound by
                    # generate_c_shim_fallback_kernel
                    if len(outputs) == 1:
                        out = outputs[0]
                        # When fallback kernel returns a list consisting of a single tensor,
                        # the output is represented as a MultiOutput with non empty indices.
                        # In this case, we strip the first key path away.
                        return go(
                            outputs[0].get_name(),
                            keypath[1:]
                            if isinstance(out, ir.MultiOutput) and len(out.indices) != 0
                            else keypath,
                        )
                    else:
                        assert isinstance(keypath[0], pytree.SequenceKey)
                        return go(outputs[keypath[0].idx].get_name(), keypath[1:])
                else:
                    return go(output_name, keypath)

            self.writeline(
                f"{self.codegen_unbacked_symbol_decl(s)} = {go_outer()}{self.ending}"
            )

    def codegen_subgraph_by_inlining(self, subgraph, outer_inputs, outer_outputs):
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # `codegen_subgraph` function.
        #
        # However this does not work with cpp wrapper. With cpp wrapper, we make
        # two passes and the kernels are shared from the first pass to the next.
        # Therefore, both the Python and CppWrapper need to share the some
        # codegen infra. For now, CppWrapperCpu has not been updated to lift the
        # subgraph as functions. Therefore for cpp_wrapper first pass with
        # PythonWrapper, we still fallback to the old way of inlining subgraphs
        # in the output code. Once we update CppWrapperCpu, we can remove this
        # function.
        def _codegen_subgraph_prefix():
            assert len(subgraph.graph.graph_inputs) == len(outer_inputs)
            for inner_input, outer_input in zip(
                subgraph.graph.graph_inputs, outer_inputs
            ):
                self.writeline(
                    f"{self.declare}{inner_input} = {outer_input}{self.ending}"
                )

        def _codegen_subgraph_suffix():
            assert len(subgraph.graph.graph_outputs) == len(outer_outputs)
            for inner_output, outer_output in zip(
                subgraph.graph.graph_outputs, outer_outputs
            ):
                self.writeline(
                    f"{outer_output} = {inner_output.codegen_reference()}{self.ending}"
                )

        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"{self.comment} subgraph: {subgraph.name}")
            _codegen_subgraph_prefix()
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            _codegen_subgraph_suffix()
        finally:
            self.pop_codegened_graph()

    def codegen_partition_call(
        self,
        partition_id: int,
        partition_signatures: ir.GraphPartitionSignature,
    ):
        """Generate code to call a graph partition"""
        input_deallocation = partition_signatures.input_deallocation
        output_nodes = partition_signatures.output_nodes

        input_names = list(input_deallocation.keys()) + [
            symbol_input.name for symbol_input in partition_signatures.symbol_inputs
        ]

        inputs = ", ".join(input_names) + ("," if len(input_names) == 1 else "")

        output_names = [node.get_name() for node in output_nodes]
        outputs = ", ".join(output_names) + ("," if len(output_nodes) == 1 else "")

        # Create a list of inputs for the subgraph call
        self.writeline(f"partition{partition_id}_args = [{inputs}]")

        names_to_del = [
            name for name, deallocate in input_deallocation.items() if deallocate
        ]
        if names_to_del:
            self.writeline(f"del {', '.join(names_to_del)}")

        # Call the subgraph launcher function
        self.writeline(
            f"({outputs}) = self.partitions[{partition_id}](partition{partition_id}_args)"
        )
        self.writeline(f"del partition{partition_id}_args")

    def set_all_partition_names(self, num_partitions: int):
        self.all_partition_names = [f"partition_{idx}" for idx in range(num_partitions)]

    def codegen_subgraph_call_with_flattened_outputs(
        self, subgraph, outer_inputs, outer_flattened_outputs
    ):
        # Get the input and output names of the subgraph
        outer_output_names = ", ".join(outer_flattened_outputs) + (
            "," if len(outer_flattened_outputs) == 1 else ""
        )
        outer_input_names = ", ".join(outer_inputs) + (
            "," if len(outer_inputs) == 1 else ""
        )

        self.writeline(f"{subgraph.graph.name}_args = [{outer_input_names}]")

        # Call the subgraph launcher function
        self.writeline(
            f"({outer_output_names}) = {subgraph.graph.name}({subgraph.graph.name}_args)"
        )

    def codegen_subgraph_call(self, subgraph, outer_inputs, outer_buffer_name):
        # Get the input and output names of the subgraph
        outer_input_names = ", ".join(outer_inputs) + (
            "," if len(outer_inputs) == 1 else ""
        )

        self.writeline(f"{subgraph.graph.name}_args = [{outer_input_names}]")

        # Since the buffers are already put into the args list, we can free the
        # buffers here.
        V.graph.scheduler.free_buffers()

        # Call the subgraph launcher function
        self.writeline(
            f"{outer_buffer_name} = {subgraph.graph.name}({subgraph.graph.name}_args)"
        )

    def codegen_subgraph_common(self, subgraph):
        self.push_codegened_graph(subgraph.graph)
        self.writeline("")
        self.writeline(f"{self.comment} subgraph: {subgraph.name}")

        parent_graph = V.graph
        subgraph.graph.cpp_wrapper = parent_graph.cpp_wrapper

        if subgraph.graph.name not in self.already_codegened_subgraphs:
            # If it is already codegened, the parent wrapper already has
            # subgraph fn by name subgraph.graph.name
            with V.set_graph_handler(subgraph.graph):
                # do not graph partition for subgraph
                with config.patch("graph_partition", False):
                    # Call the codegen of subgraph recursively
                    subgraph_code, _ = subgraph.graph.codegen()
            self.already_codegened_subgraphs.add(subgraph.graph.name)
            self.define_subgraph_launcher_fn(subgraph_code.value)

    def codegen_subgraph_with_flattened_outputs(
        self, subgraph, outer_inputs, outer_flattened_outputs
    ):
        self.codegen_subgraph_common(subgraph)
        self.codegen_subgraph_call_with_flattened_outputs(
            subgraph, outer_inputs, outer_flattened_outputs
        )

    def codegen_subgraph(self, subgraph, outer_inputs, outer_buffer_name):
        # Codegen subgraph by recursively calling the codegen for the subgraph.
        # This lifts the subgraph as a function in the output code.
        self.codegen_subgraph_common(subgraph)
        self.codegen_subgraph_call(subgraph, outer_inputs, outer_buffer_name)

    def codegen_invoke_subgraph(self, invoke_subgraph):
        name = invoke_subgraph.get_name()

        self.writeline(f"{name} = [None] * {len(invoke_subgraph.outputs)}")
        outer_inputs = [buf.codegen_reference() for buf in invoke_subgraph.inputs]

        if V.graph.aot_mode:
            outer_outputs = [
                f"{name}[{i}]" for i in range(len(invoke_subgraph.outputs))
            ]
            self.codegen_subgraph_by_inlining(
                invoke_subgraph.subgraph, outer_inputs, outer_outputs
            )
        else:
            self.codegen_subgraph(invoke_subgraph.subgraph, outer_inputs, name)

    def codegen_conditional(self, conditional):
        name = conditional.get_name()

        outer_inputs = [buf.codegen_reference() for buf in conditional.operands]

        predicate = conditional.predicate.codegen_reference()
        if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
            # move the Tensor predicate to host
            predicate = f"{predicate}.item()"

        self.writeline(f"{name} = [None] * {len(conditional.outputs)}")
        self.writeline(f"if {predicate}:")
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        if V.graph.aot_mode:
            outer_outputs = [f"{name}[{i}]" for i in range(len(conditional.outputs))]
            self.codegen_subgraph_by_inlining(
                conditional.true_subgraph, outer_inputs, outer_outputs
            )
        else:
            self.codegen_subgraph(conditional.true_subgraph, outer_inputs, name)

        self.writeline(ExitSubgraphLine(self))
        self.writeline("else:")
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        if V.graph.aot_mode:
            outer_outputs = [f"{name}[{i}]" for i in range(len(conditional.outputs))]
            self.codegen_subgraph_by_inlining(
                conditional.false_subgraph, outer_inputs, outer_outputs
            )
        else:
            self.codegen_subgraph(conditional.false_subgraph, outer_inputs, name)
        self.writeline(ExitSubgraphLine(self))

    def codegen_while_loop(self, while_loop):
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]

        self.writeline(f"{name} = [None] * {len(outer_carried_inputs)}")
        for i, inp in enumerate(outer_carried_inputs):
            # set the initial state before the loop
            self.writeline(f"{name}[{i}] = {inp}")

        cond_outer_inputs = [
            *[f"{name}[{i}]" for i in range(len(outer_carried_inputs))],
            *outer_additional_inputs,
        ]
        cond_outer_outputs = [f"{name}_cond_result"]
        body_outer_inputs = list(
            cond_outer_inputs
        )  # same inputs for cond_fn and body_fn
        # Carry over the state from body_fn. Note: We only carry over
        # the carried_inputs part of the inputs, the additional ones
        # are passed in as they're before.
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        self.writeline("while True:")
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))

        if V.graph.aot_mode:
            self.codegen_subgraph_by_inlining(
                while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
            )
        else:
            self.codegen_subgraph_with_flattened_outputs(
                while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
            )
        self.writeline(
            f"if not {cond_outer_outputs[0]}: break"
        )  # condition doesn't hold
        self.writeline(ExitSubgraphLine(self))
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        if V.graph.aot_mode:
            self.codegen_subgraph_by_inlining(
                while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
            )
        else:
            self.codegen_subgraph_with_flattened_outputs(
                while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
            )
        self.writeline(ExitSubgraphLine(self))

    @staticmethod
    def statically_known_int_or_none(x):
        try:
            if getattr(x, "free_symbols", None):
                # _maybe_evaluate_static will return (s0 // (2 // s0)) as 2, but
                # the actual codegen will still generate the full expression here.
                return None
            if isinstance(x, int):
                return x
            val = V.graph._shape_env._maybe_evaluate_static(x)
            if val is None:
                return val
            return int(val)  # type: ignore[call-overload]
        except Exception:
            return None

    @staticmethod
    def statically_known_list_of_ints_or_none(lst):
        result = []
        for x in lst:
            num = PythonWrapperCodegen.statically_known_int_or_none(x)
            if num is None:
                return None
            result.append(num)
        return result

    @staticmethod
    def is_statically_known_list_of_ints(lst):
        return (
            PythonWrapperCodegen.statically_known_list_of_ints_or_none(lst) is not None
        )

    @staticmethod
    def static_shape_for_buffer_or_none(buffer):
        return PythonWrapperCodegen.statically_known_list_of_ints_or_none(
            buffer.get_size()
        )

    @staticmethod
    def can_prove_buffer_has_static_shape(buffer):
        return PythonWrapperCodegen.static_shape_for_buffer_or_none(buffer) is not None


class SubgraphPythonWrapperCodegen(PythonWrapperCodegen):
    """
    A wrapper codegen that generates code for a subgraph. For most of the
    methods, we rely on the implementation in the PythonWrapperCodegen. But we
    override a few functions to produce cleaner code (like avoiding writing
    imports twice in the output code)
    """

    def __init__(
        self,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        # It is necessary to set the subgraph_name before calling super __init__
        # because __init__ calls set_launcher_fn_name
        self.subgraph_name = subgraph_name
        self.parent_wrapper = parent_wrapper
        self.partition_signatures = partition_signatures

        super().__init__()

    def set_launcher_fn_name(self) -> None:
        # This sets up the name of the function containing the launcher code of
        # the subgraph.
        self.launcher_fn_name = self.subgraph_name

    def write_header(self) -> None:
        pass

    def add_benchmark_harness(self, output):
        pass

    def benchmark_compiled_module(self, output):
        pass

    def write_async_compile_wait(self):
        pass

    def next_kernel_suffix(self) -> str:
        # Ensures that subgraphs kernels do not clash with each other
        return self.parent_wrapper.next_kernel_suffix()

    def generate_after_suffix(self, result: IndentedBuffer) -> None:
        return

    def write_launcher_fn_call_get_indent(self) -> int:
        self.prefix.splice(
            f"""
            def {self.launcher_fn_name}(args):
            """
        )
        prefix_indent = 1
        return prefix_indent

    def get_wrapper_call_indent(self) -> int:
        return 1

    def get_graph_inputs(
        self,
    ) -> dict[str, Union[ir.TensorBox, ir.TorchBindObject, sympy.Expr]]:
        if signature := self.partition_signatures:
            inputs = signature.input_nodes
        else:
            inputs = V.graph.graph_inputs
        return inputs

    def get_graph_input_names(self) -> list[str]:
        if signature := self.partition_signatures:
            names = list(signature.input_nodes.keys()) + [
                symbol_input.name for symbol_input in signature.symbol_inputs
            ]
        else:
            names = V.graph.graph_input_names
        return names

    def get_graph_outputs(self) -> list[IRNode]:
        if signature := self.partition_signatures:
            outputs = signature.output_nodes
        else:
            outputs = V.graph.graph_outputs
        return outputs

    def codegen_allocation(self, buffer: ir.Buffer):
        name = buffer.get_name()
        if (signature := self.partition_signatures) and name in signature.input_nodes:
            # skip allocation if buffer is a subgraph input.
            # This allows reusing an input buffer in graph partition,
            # although this is not allowed in general.
            return

        super().codegen_allocation(buffer)

    @cache_on_self
    def write_triton_header_once(self) -> None:
        # TODO: Uncomment in future. This will be needed to support subgraph
        # codegen for cpp wrapper.
        # if config.triton.autotune_at_compile_time:
        #     import_str = self.triton_header_str()
        #     self.kernel_autotune_calls.splice(import_str)
        self.parent_wrapper.write_triton_header_once()

    @cache_on_self
    def write_get_raw_stream_header_once(self) -> None:
        # TODO: Uncomment in future. This will be needed to support subgraph
        # codegen for cpp wrapper.
        # if config.triton.autotune_at_compile_time:
        #     self.kernel_autotune_calls.writeline(
        #         V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        #     )
        self.parent_wrapper.write_get_raw_stream_header_once()
