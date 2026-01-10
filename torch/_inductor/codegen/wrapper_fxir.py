import dataclasses
import functools
import logging
import operator
import textwrap
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import sympy

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._higher_order_ops.triton_kernel_wrap import (
    TraceableTritonKernelWrapper,
    tracing_triton_hopifier_singleton,
    triton_kernel_wrapper_mutation,
)
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels  # noqa: F401
from torch._inductor.utils import convert_to_symint
from torch._inductor.virtualized import V
from torch._library.triton import wrap_triton
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import (
    CallMethodKey,
    ConvertIntKey,
    DivideByKey,
    free_unbacked_symbols,
)
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import OptimizedPythonReferenceAnalysis
from torch.utils._sympy.solve import try_solve
from .. import config, ir
from ..runtime.triton_compat import Config
from ..utils import cache_property_on_self, LineContext, ValueWithLineMap
from .common import (
    CodegenSymbol,
    FileBackedGraphModule,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .wrapper import (
    AllocateLine,
    BufferLike,
    CommBufferAllocateLine,
    CommBufferFreeLine,
    CommentLine,
    ConditionalLine,
    DynamicScalarLine,
    EnterDeviceContextManagerLine,
    EnterSubgraphLine,
    ExitDeviceContextManagerLine,
    ExitSubgraphLine,
    ExternKernelAllocLine,
    ExternKernelOutLine,
    FreeIfNotReusedLine,
    FreeLine,
    IndexPutFallbackLine,
    KernelCallLine,
    KernelDefinitionLine,
    Line,
    MultiOutputLine,
    NullLine,
    PythonWrapperCodegen,
    ReinterpretLine,
    ReuseLine,
    ScatterFallbackLine,
    SubgraphPythonWrapperCodegen,
    SymbolicCallArg,
    SymbolicCallArgLine,
    UnbackedSymbolDefsLine,
    WrapperLine,
)


aten = torch.ops.aten
log = logging.getLogger(__name__)


@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    """
    Represents a sympy.Symbol graph input.
    """

    symbol: sympy.Symbol

    def get_name(self) -> str:
        return str(self.symbol)

    def get_example(self) -> Union[torch.Tensor, torch.SymInt]:
        sym_int = convert_to_symint(self.symbol)
        assert isinstance(sym_int, torch.SymInt)
        return sym_int


CodegenBuffer = Union[BufferLike, SymbolBuffer]


@dataclasses.dataclass
class TritonKernel:
    """
    Stores metadata about Triton kernels for use in FX.
    """

    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper


def replace_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """
    Replace sympy.floor with FloorDiv.
    """

    def replace(expr: sympy.Expr) -> sympy.Expr:
        expr = sympy.together(expr)

        # Division is represented as a Mul with a Rational factor or a Pow with negative
        # exponent. We convert floor(Mul(...)) to FloorDiv(numerator, denominator) by
        # partitioning factors into the numerator and denominator.
        (numerator, denominator) = (sympy.S.One,) * 2
        for arg in sympy.Mul.make_args(expr):
            if isinstance(arg, sympy.Rational):
                numerator *= arg.numerator
                denominator *= arg.denominator
            elif isinstance(arg, sympy.Pow) and arg.exp.is_negative:
                denominator *= arg.base**-arg.exp
            else:
                numerator *= arg

        return FloorDiv(numerator, denominator)

    return expr.replace(sympy.floor, replace)


class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Backend to generate wrapper code as an FX IR graph.
    """

    supports_caching = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.subgms: dict[str, torch.fx.GraphModule] = {}

    def codegen_inputs(self) -> None:
        """
        This would generate code for symbolic input shapes, strides, etc.
        Since the FX converter handles this, do nothing here.
        """

    def codegen_conditional(self, conditional: ir.Conditional) -> None:
        """
        Conditional codegen normally emits a number of different wrapper lines.
        Instead, FX conversion uses a dedicated line for the whole conditional.
        """
        self.writeline(ConditionalLine(self, conditional))
        for subgraph in (conditional.true_subgraph, conditional.false_subgraph):
            self.codegen_subgraph_common(subgraph)

    def define_subgraph_launcher_fn(
        self, name: str, subgraph_code: Union[ValueWithLineMap, FileBackedGraphModule]
    ) -> None:
        """
        Record subgms as they're generated.
        """
        assert isinstance(subgraph_code, FileBackedGraphModule)
        self.subgms[name] = subgraph_code.gm

    @property
    @cache_property_on_self
    def is_subgraph(self) -> bool:
        return isinstance(self, SubgraphPythonWrapperCodegen)

    def get_fx_graph_inputs(
        self,
    ) -> dict[str, Union[ir.TensorBox, ir.TorchBindObject, sympy.Expr, None]]:
        """
        Get the input nodes corresponding to FX graph placeholders.
        """

        if V.aot_compilation and not self.is_subgraph:
            # AOT graphs must match the signature of the input module.
            return {
                node.name: V.graph.graph_inputs.get(node.name)
                for node in V.graph.module.graph.find_nodes(op="placeholder")  # type: ignore[operator, union-attr]
            }

        return self.get_graph_inputs()

    def _generate(self, is_inference: bool) -> tuple[FileBackedGraphModule, None]:
        self.run_wrapper_ir_passes(is_inference)

        prologue = "\n".join(
            [
                self.imports.getvalue(),
                self.header.getvalue(),
            ]
        )
        gm = FxConverter(
            lines=self.lines,
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,
            is_subgraph=self.is_subgraph,
        ).generate()

        compiled_fn = self.compile_graph(gm)

        return FileBackedGraphModule(gm, compiled_fn), None

    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
        return gm.forward

    def write_header(self) -> None:
        """
        Python subgraphs normally lack headers.
        Override this behavior to generate prologues for FX subgraphs.
        """
        PythonWrapperCodegen.write_header(self)

    @classmethod
    def create(
        cls: type["WrapperFxCodegen"],
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ) -> "WrapperFxCodegen":
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None

            # Subgraphs override some methods of PythonWrapperCodegen.
            # Apply these overrides to the user-provided class, with priority given to
            # user-provided methods.
            class SubgraphFxWrapperCodegen(cls, SubgraphPythonWrapperCodegen):  # type: ignore[misc,valid-type]
                def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
                    """
                    Skip graph compilation for subgraphs.
                    """

                    def crash_if_run(*args: Any) -> None:
                        raise NotImplementedError("Cannot run a subgraph in isolation!")

                    return crash_if_run

            return SubgraphFxWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )

        return cls()


@dataclasses.dataclass
class FxConverter:
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """

    lines: list[Line]
    prologue: str
    graph_inputs: dict[str, Union[ir.TensorBox, ir.TorchBindObject, sympy.Expr, None]]
    graph_outputs: list[ir.IRNode]
    subgms: dict[str, torch.fx.GraphModule]
    is_subgraph: bool

    def __post_init__(self) -> None:
        graph = torch.fx.Graph()
        self.gm = GraphModule({}, graph)  # Wrapper FX IR.
        self.buffer_to_node: dict[
            Optional[str], torch.fx.Node
        ] = {}  # Symbol table for codegen.
        self.kernels: dict[str, TritonKernel] = {}  # Table to store Triton kernels.
        self._unique_symbol_ids: Counter[str] = Counter()
        self.tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        self.expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy] = {}

    def _import_kernel(self, code: str, kernel_name: str) -> CachingAutotuner:
        """
        Imports a kernel from source, possibly autotuning block parameters.
        """
        module_code = "\n".join([self.prologue, code])
        mod = PyCodeCache.load(module_code)
        kernel = getattr(mod, kernel_name)

        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()

        if not isinstance(kernel, CachingAutotuner):
            raise NotImplementedError(
                textwrap.dedent(f"""
                Unsupported type for kernel {kernel_name}: {type(kernel)}.
                FX conversion only supports Triton kernels.
            """)
            )

        return kernel

    def _create_as_strided(
        self,
        input_node: torch.fx.Node,
        size: tuple[Any, ...],
        stride: tuple[Any, ...],
        offset: Union[int, sympy.Expr],
    ) -> torch.fx.Node:
        return self.gm.graph.call_function(
            torch.as_strided,
            args=(
                input_node,
                self._generate_sym_nodes(size),
                self._generate_sym_nodes(stride),
                self._generate_sym_node(offset),
            ),
        )

    def _record_allocation(self, buffer: CodegenBuffer, node: torch.fx.Node) -> None:
        """
        Updates the symbol table to record that an Inductor buffer maps to the result of
        an FX node.
        """
        assert node not in self.buffer_to_node
        self.buffer_to_node[buffer.get_name()] = node

    def _free(self, buffer: Union[CodegenBuffer, ir.TorchBindObject]) -> None:
        """
        Removes the buffer from the symbol table.
        """
        name = buffer.get_name()
        del self.buffer_to_node[name]

    def _lookup_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        Maps call args back to FX nodes.
        """
        return tuple(
            self.buffer_to_node[arg]
            if isinstance(arg, str)
            else arg.inner_expr
            if isinstance(arg, SymbolicCallArg)
            else arg
            for arg in args
        )

    def _get_buffer(self, node: ir.IRNode) -> CodegenBuffer:
        """
        Extract buffer data from an IR node.
        """
        if isinstance(node, (ir.Buffer, WorkspaceArg)):
            return node
        elif isinstance(node, (ir.BaseView, ir.MutableBox)):
            return self._get_buffer(node.data)
        elif isinstance(node, sympy.Symbol):
            return SymbolBuffer(node)
        else:
            raise NotImplementedError(f"Unable to extract buffer from node: {node}")

    def _generate_size_proxy(
        self, node: torch.fx.Node, expr: sympy.Expr
    ) -> torch.fx.Proxy:
        proxy = torch.fx.Proxy(node, tracer=self.tracer)
        self.expr_to_proxy[expr] = proxy
        return proxy

    def _generate_graph_inputs(self) -> None:
        """
        Converts graph inputs to FX placeholders.
        """

        for name, ir_node in self.graph_inputs.items():
            if ir_node is None:
                # Create dummy input nodes to match the input signature
                self.gm.graph.placeholder(name)
                continue

            # Introduce a new symbol for constant inputs.
            is_constant = isinstance(ir_node, (int, float, sympy.Integer, sympy.Float))
            buffer = (
                SymbolBuffer(sympy.Symbol(name, is_integer=True))
                if is_constant
                else self._get_buffer(ir_node)
            )
            placeholder_node = self.gm.graph.placeholder(buffer.get_name())
            placeholder_node.meta["val"] = (
                ir_node if is_constant else buffer.get_example()
            )
            self._record_allocation(buffer, placeholder_node)

            # Record symbol definitions for dynamic shapes.
            if isinstance(ir_node, sympy.Symbol):
                self._generate_size_proxy(placeholder_node, ir_node)

    def _generate_graph_input_shapes(self) -> None:
        """
        Generate nodes creating symints that are part of graph input
        shape/strides.
        """

        def _codegen_symbol(
            sym_or_exp: Union[sympy.Symbol, sympy.Expr],
            base_node: torch.fx.Node,
            target: torch._ops.OpOverload,
            dim: int,
        ) -> None:
            def codegen_proxy() -> torch.fx.Proxy:
                size_node = self.gm.graph.call_function(target, (base_node, dim))
                size_proxy = self._generate_size_proxy(size_node, sym_or_exp)
                return size_proxy

            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in self.expr_to_proxy:
                    return
                codegen_proxy()

            elif isinstance(sym_or_exp, sympy.Integer):
                return

            elif isinstance(sym_or_exp, sympy.Expr):
                # Check if we need to solve for an undefined symbol.
                undefined_symbols = [
                    sym
                    for sym in sym_or_exp.free_symbols
                    if sym not in self.expr_to_proxy
                ]
                if len(undefined_symbols) == 0:
                    self._sympy_interp(sym_or_exp)
                    return
                elif len(undefined_symbols) > 1:
                    raise ValueError(f"Underdetermined input expression: {sym_or_exp}")

                # Define a new symbol for the input size.
                size_proxy = codegen_proxy()
                size_symbol = sympy.Symbol(
                    size_proxy.node.name, integer=True, nonnegative=True
                )
                self.expr_to_proxy[size_symbol] = size_proxy

                # Solve for the undefined symbol.
                undefined_symbol = undefined_symbols[0]
                solution = try_solve(
                    sympy.Eq(sym_or_exp, size_symbol), undefined_symbol
                )
                if solution is None:
                    raise ValueError(f"Cannot solve input expression: {sym_or_exp}")

                # Since the symbol is a size, it must be an integer.
                # Therefore, we can convert division to FloorDiv.
                undefined_symbol_expr = solution[1]
                if undefined_symbol.is_integer:
                    undefined_symbol_expr = replace_floor_div(
                        sympy.floor(undefined_symbol_expr)
                    )

                # Generate FX for the symbol.
                self._sympy_interp(undefined_symbol_expr)
                self.expr_to_proxy[undefined_symbol] = self.expr_to_proxy[
                    undefined_symbol_expr
                ]

        for ir_node in self.graph_inputs.values():
            if isinstance(ir_node, ir.TensorBox):
                buffer = self._get_buffer(ir_node)
                placeholder_node = self.buffer_to_node[buffer.get_name()]

                for dim, size in enumerate(ir_node.get_size()):
                    _codegen_symbol(
                        size, placeholder_node, torch.ops.aten.sym_size.int, dim
                    )
                for dim, stride in enumerate(ir_node.get_stride()):
                    _codegen_symbol(
                        stride, placeholder_node, torch.ops.aten.sym_stride.int, dim
                    )

    def _generate_graph_constants(self) -> None:
        for name, value in V.graph.constants.items():
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = value
            setattr(self.gm, name, value)
            self.buffer_to_node[name] = node

    def _generate_buffer(self, node: ir.IRNode) -> Optional[torch.fx.Node]:
        """
        Generates FX IR for transformations on a buffer, such as ReinterpretView.
        Does nothing if no such transformations are present.
        """

        if isinstance(node, ir.ShapeAsConstantBuffer):
            # Generate FX nodes to compute the shape expression.
            return self._sympy_interp(node.expr).node

        def generate_to_buffer(node: ir.IRNode) -> Optional[BufferLike]:
            if isinstance(node, (ir.Buffer, WorkspaceArg)):
                return node
            elif isinstance(node, ir.NoneAsConstantBuffer):
                return None
            elif isinstance(node, ir.MutableBox):
                return generate_to_buffer(node.data)
            elif isinstance(node, ir.ReinterpretView):
                # We need to introduce a new symbol if the output is a ReinterpretView.
                # Use a WorkspaceArg for this.
                buffer = self._get_buffer(node.data)
                assert isinstance(buffer, (ir.Buffer, WorkspaceArg))
                unique_name = self.gm.graph._graph_namespace.create_name(
                    f"{buffer.get_name()}_view", None
                )
                device = buffer.get_device()
                assert device
                reused_as = WorkspaceArg(
                    count=buffer.get_size(),
                    zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                    device=device,
                    outer_name=unique_name,
                    dtype=buffer.get_dtype(),
                )

                # Generate FX IR for the view.
                self._generate_reinterpret_helper(buffer, reused_as, node.layout)

                return reused_as
            else:
                raise NotImplementedError(f"Unrecognized buffer/view node: {node}")

        buffer = generate_to_buffer(node)
        return self.buffer_to_node[buffer.get_name()] if buffer is not None else None

    def _generate_outputs(
        self,
    ) -> Union[Optional[torch.fx.Node], list[Optional[torch.fx.Node]]]:
        """
        Generate FX IR for graph outputs.
        """
        output_nodes = [
            self._generate_buffer(node) for idx, node in enumerate(self.graph_outputs)
        ]

        # Parent graphs with single return elements don't use a tuple.
        output_value = (
            output_nodes[0]
            if len(output_nodes) == 1 and not self.is_subgraph
            else output_nodes
        )

        return output_value

    def _generate_subgm_getattrs(self) -> None:
        """
        Generate getattr nodes for subgms.
        """

        def generate_getattr(name: str, subgm: torch.fx.GraphModule) -> torch.fx.Node:
            self.gm.add_submodule(name, subgm)
            node = self.gm.graph.get_attr(name)
            node.meta["val"] = subgm
            return node

        self.subgm_getattrs = {
            name: generate_getattr(name, subgm) for name, subgm in self.subgms.items()
        }

    def _get_subgm_attr(self, subgraph: ir.Subgraph) -> torch.fx.Node:
        """
        Look up the getattr node for a subgraph.
        """
        graph = subgraph.graph
        assert graph is not None
        return self.subgm_getattrs[graph.name]

    def generate(self) -> torch.fx.GraphModule:
        """
        Main entrypoint for FX codegen.
        """
        self._generate_graph_inputs()
        self._generate_graph_constants()
        self._generate_subgm_getattrs()

        with _set_node_metadata_hook(
            self.gm,
            functools.partial(_node_metadata_hook, fake_mode=V.fake_mode),
        ):
            self._generate_graph_input_shapes()

            # Generate FX IR from Wrapper IR lines.
            for line in self.lines:
                if isinstance(line, WrapperLine):
                    line.codegen_fx(self)(line)
                elif isinstance(line, LineContext):
                    # Ignore line context in FX IR.
                    pass
                else:
                    raise NotImplementedError(
                        textwrap.dedent(
                            f"""
                        Found line of unrecognized type '{type(line)}':
                            '{line}'

                        FX conversion only supports Wrapper IR lines.
                        """
                        )
                    )

            output = self._generate_outputs()

        self.gm.graph.output(output)
        self.gm.recompile()
        return self.gm

    def _sympy_interp(self, expr: sympy.Expr) -> torch.fx.Proxy:
        # hash cons
        if expr in self.expr_to_proxy:
            return self.expr_to_proxy[expr]
        # base cases, don't cache
        if isinstance(
            expr,
            (
                sympy.Integer,
                sympy.Number,
                sympy.Symbol,
                sympy.logic.boolalg.BooleanAtom,
            ),
        ):
            return sympy_interp(
                OptimizedPythonReferenceAnalysis, self.expr_to_proxy, expr
            )

        # hash cons on arguments, run expr handler
        self.expr_to_proxy[expr] = _run_sympy_handler(
            OptimizedPythonReferenceAnalysis,
            [self._sympy_interp(arg) for arg in expr.args],
            expr,
        )
        return self.expr_to_proxy[expr]

    def _generate_sym_node(
        self, s: Union[int, sympy.Expr]
    ) -> Union[int, torch.fx.Node]:
        if isinstance(s, (int, sympy.Integer)):
            return int(s)
        elif isinstance(s, sympy.Symbol):
            assert s in self.expr_to_proxy, (
                f"Could not find a node corresponding to the symbol {s}"
            )
            return self.expr_to_proxy[s].node
        elif isinstance(s, sympy.Expr):
            return self._sympy_interp(s).node

        elif isinstance(s, torch.fx.Node):
            return s

        else:
            raise ValueError(f"{s} of type {type(s)} is not a valid input")

    def _generate_sym_nodes(
        self, shape: Sequence[sympy.Expr]
    ) -> list[Union[int, torch.fx.Node]]:
        return [self._generate_sym_node(s) for s in shape]

    def _generate_allocate(self, line: WrapperLine) -> None:
        assert isinstance(line, AllocateLine)
        buffer = line.node
        name = buffer.get_name()
        assert name not in V.graph.removed_buffers

        device = buffer.get_device()
        assert device
        dtype = buffer.get_dtype()
        shape = self._generate_sym_nodes(buffer.get_size())
        stride = self._generate_sym_nodes(buffer.get_stride())

        node = self.gm.graph.call_function(
            torch.empty_strided,
            args=(shape, stride),
            kwargs={"dtype": dtype, "device": device.type},
        )
        assert name
        node.name = name
        self._record_allocation(buffer, node)

    def _generate_conditional(self, line: WrapperLine) -> None:
        assert isinstance(line, ConditionalLine)

        def get_subgm_attr(subgraph: Optional[ir.Subgraph]) -> torch.fx.Node:
            assert subgraph is not None
            return self._get_subgm_attr(subgraph)

        # Access the subgraphs as getattrs.
        ir_node = line.node
        (true_subgm, false_subgm) = [
            get_subgm_attr(subgraph)
            for subgraph in (ir_node.true_subgraph, ir_node.false_subgraph)
        ]

        def generate_buffer(node: Optional[ir.IRNode]) -> Optional[torch.fx.Node]:
            assert node is not None
            return self._generate_buffer(node)

        predicate = generate_buffer(ir_node.predicate)
        assert ir_node.operands is not None
        operands = tuple(generate_buffer(arg) for arg in ir_node.operands)
        fx_node = self.gm.graph.call_function(
            torch.ops.higher_order.cond,
            args=(predicate, true_subgm, false_subgm, operands),
        )
        self._record_allocation(ir_node, fx_node)

    def _generate_comment(self, line: WrapperLine) -> None:
        assert isinstance(line, CommentLine)
        # We ignore comments in FX IR.

    def _generate_dynamic_scalar(self, line: WrapperLine) -> None:
        assert isinstance(line, DynamicScalarLine)

        ir_node = line.node
        (input_ir_node,) = ir_node.inputs
        assert isinstance(input_ir_node, ir.IRNode)
        input_fx_node = self._generate_buffer(input_ir_node)
        keypath = ir_node.keypath
        graph = self.gm.graph

        def generate_item(x: Optional[torch.fx.Node]) -> torch.fx.Node:
            assert x is not None
            return graph.call_function(
                aten.item.default,
                args=(x,),
            )

        if len(keypath) == 0:
            result_fx_node = generate_item(input_fx_node)
        elif len(keypath) == 1 and isinstance(keypath[0], ConvertIntKey):
            where_fx_node = graph.call_function(
                aten.where.Scalar,
                args=(input_fx_node, 1, 0),
            )
            result_fx_node = generate_item(where_fx_node)
        else:
            raise NotImplementedError(f"Unsupported keypath: {keypath}")

        result_symbol = ir_node.sym
        result_buffer = SymbolBuffer(result_symbol)
        self._record_allocation(result_buffer, result_fx_node)
        self._generate_size_proxy(result_fx_node, result_symbol)

    def _generate_enter_device_context_manager(self, line: WrapperLine) -> None:
        assert isinstance(line, EnterDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_exit_device_context_manager(self, line: WrapperLine) -> None:
        assert isinstance(line, ExitDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_enter_subgraph(self, line: WrapperLine) -> None:
        assert isinstance(line, EnterSubgraphLine)
        # We ignore memory planning lines in FX IR.

    def _generate_exit_subgraph(self, line: WrapperLine) -> None:
        assert isinstance(line, ExitSubgraphLine)
        # We ignore memory planning lines in FX IR.

    def _generate_free(self, line: WrapperLine) -> None:
        assert isinstance(line, FreeLine)

        buf = line.node

        # No need to free placeholders.
        if self.buffer_to_node[buf.get_name()].op == "placeholder":
            return

        self._free(buf)

    def _generate_free_if_not_reused(self, line: WrapperLine) -> None:
        assert isinstance(line, FreeIfNotReusedLine)
        buf = line.node
        assert buf.get_name() not in V.graph.removed_buffers
        if not line.is_reused:
            self._free(buf)

    def _generate_line_context(self, line: WrapperLine) -> None:
        assert isinstance(line, LineContext)
        # We ignore line context in FX IR.

    def _generate_reinterpret(self, line: WrapperLine) -> None:
        assert isinstance(line, ReinterpretLine)
        self._generate_reinterpret_helper(line.node, line.reused_as, line.layout)

    def _generate_reinterpret_helper(
        self, input_buffer: BufferLike, result_buffer: BufferLike, layout: ir.Layout
    ) -> None:
        input_node = self.buffer_to_node[input_buffer.get_name()]

        # Look up output metadata.
        name = result_buffer.get_name()
        assert name
        size = tuple(layout.size)
        stride = tuple(layout.stride)
        if isinstance(layout, ir.NonOwningLayout):
            # Look up the view's layout.
            view = layout.view
            assert isinstance(view, ir.ReinterpretView), (
                f"unexpected type: {type(view)}"
            )
            layout = view.layout
        offset = input_buffer.get_offset() + layout.offset

        # Map ReinterpretView to as_strided.
        result_node = self._create_as_strided(input_node, size, stride, offset)
        result_node.name = name
        self._record_allocation(result_buffer, result_node)

    def _generate_reuse(self, line: WrapperLine) -> None:
        assert isinstance(line, ReuseLine)
        old = line.node
        new = line.reused_as
        assert not any(buf.get_name() in V.graph.removed_buffers for buf in (old, new))
        assert old.get_dtype() == new.get_dtype()

        old_node = self.buffer_to_node[old.get_name()]
        result_node = old_node

        # Change shape and stride.
        size = tuple(new.get_size())
        stride = tuple(new.get_stride())
        offset = new.get_offset()
        if (
            tuple(old.get_size()) != size
            or tuple(old.get_stride()) != stride
            or old.get_offset() != offset
        ):
            result_node = self._create_as_strided(old_node, size, stride, offset)

        self._record_allocation(new, result_node)

        # Free the old buffer, if we allocated a new tensor.
        if (
            old.get_name() not in V.graph.get_output_names()
            and line.delete_old
            and result_node is not old_node
        ):
            self._free(old)

    def _generate_multi_output(self, line: WrapperLine) -> None:
        assert isinstance(line, MultiOutputLine)

        arg_node = self.buffer_to_node[line.arg_name]

        # For non-tuple / non-list outputs, map the
        # output to the same node as the input.
        if len(line.indices) == 0:
            self.buffer_to_node[line.result_name] = arg_node
            return

        # Extract the index for tuple access.
        inds = line.indices[0][1:]
        assert len(inds) == 1, f"Cannot convert {inds} to an index."
        idx = inds[0]

        node = self.gm.graph.call_function(operator.getitem, args=(arg_node, idx))
        node.name = line.result_name
        self.buffer_to_node[line.result_name] = node

    def _generate_fallback_call(
        self,
        ir_node: ir.ExternKernel,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        fx_node = self.gm.graph.call_function(
            ir_node.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )
        result_buffer = ir_node.codegen_reference()
        self.buffer_to_node[result_buffer] = fx_node

    def _generate_index_put_fallback(self, line: WrapperLine) -> None:
        assert isinstance(line, IndexPutFallbackLine)
        ir_node = line.node

        def generate_buffer_or_none(
            x: Union[ir.IRNode, Sequence[ir.IRNode], None],
        ) -> Optional[torch.fx.Node]:
            """
            Handles None before calling _generate_buffer.
            """
            if x is None:
                return None

            assert isinstance(x, ir.IRNode)
            return self._generate_buffer(x)

        (x, values) = [generate_buffer_or_none(t) for t in ir_node.inputs[:2]]
        indices = tuple(generate_buffer_or_none(t) for t in line.indices)
        accumulate = ir_node.constant_args[0]
        args = (x, indices, values, accumulate)
        self._generate_fallback_call(ir_node, args)

    def _generate_scatter_fallback(self, line: WrapperLine) -> None:
        assert isinstance(line, ScatterFallbackLine)
        ir_node = line.node
        assert ir.is_node_sequence(ir_node.inputs)
        (x, index, src) = [self._generate_buffer(t) for t in ir_node.inputs] + (
            [] if ir_node.src_is_tensor else [ir_node.constant_args[1]]
        )
        args = (x, ir_node.constant_args[0], index, src)
        kwargs = {}
        if reduce := ir_node.kwargs.get("reduce"):
            kwargs["reduce"] = reduce

        self._generate_fallback_call(ir_node, args, kwargs)

    def _generate_null(self, line: WrapperLine) -> None:
        assert isinstance(line, NullLine)
        # Does nothing.

    def _generate_comm_buffer_allocate(self, line: WrapperLine) -> None:
        assert isinstance(line, CommBufferAllocateLine)
        raise NotImplementedError("Comm buffer allocation is not yet supported")

    def _generate_comm_buffer_free(self, line: WrapperLine) -> None:
        assert isinstance(line, CommBufferFreeLine)
        self._free(line.node)

    def _generate_triton_call(self, line: WrapperLine) -> None:
        assert isinstance(line, KernelCallLine)

        # Collect all kwargs, including autotuned block sizes.
        call_args = self._lookup_args(line.call_args)
        kernel = self.kernels[line.kernel_name]
        tuner = kernel.tuner

        class UnbackedSymintsError(Exception):
            pass

        def tune_kernel(tuner: CachingAutotuner, call_args: Sequence[Any]) -> None:
            from triton.runtime import driver

            log.info("Autotuning Triton kernel %s at compile time.", kernel_name)

            device = driver.active.get_current_device()

            stream = driver.active.get_current_stream(device)

            def node_to_tuning_arg(arg: Any) -> Any:
                """
                Create real tensors for autotuning arguments, substituting size hints
                for dynamic shapes.
                """

                def to_size_hint_sympy_int(arg: Union[sympy.Expr, int]) -> int:
                    if len(free_unbacked_symbols(arg)) > 0:
                        # NYI: tuning args require backed symints.
                        raise UnbackedSymintsError
                    return V.graph.sizevars.size_hint(arg)

                def to_size_hint_list(arg: list[Union[torch.SymInt, int]]) -> list[int]:
                    args_sympy = [
                        x.node.expr if isinstance(x, torch.SymInt) else x for x in arg
                    ]
                    return pytree.tree_map(to_size_hint_sympy_int, args_sympy)

                if not isinstance(arg, torch.fx.Node):
                    return to_size_hint_sympy_int(arg)

                fake = arg.meta["val"]
                return torch.empty_strided(
                    to_size_hint_list(fake.shape),
                    to_size_hint_list(fake.stride()),
                    dtype=fake.dtype,
                    device=device,
                ).zero_()

            # call args can be fx nodes or sympy expressions or integers!
            arg_values = [node_to_tuning_arg(arg) for arg in call_args]
            tuner.run(*arg_values, stream=stream)

        # Optionally autotune the kernels.
        # The FX backend currently only supports compile-time tuning.
        kernel_name = tuner.fn.__name__
        if config.triton.autotune_at_compile_time:
            try:
                tune_kernel(tuner, call_args)
            except UnbackedSymintsError:
                log.info(
                    "Detected unbacked symints. Skipping autotuning for kernel %s.",
                    kernel_name,
                )
        else:
            log.info(
                "Skipping autotuning for kernel %s. Set config.triton.autotune_at_compile_time = True to enable.",
                kernel_name,
            )

        triton_meta = tuner.triton_meta
        signature = triton_meta["signature"]

        def add_constants_to_call_args(
            call_args: Sequence[Any], cfg: Config
        ) -> tuple[Any, ...]:
            """
            Add constant kwargs to the arg list.
            """
            # Add args from the proper Triton signature.
            # Exclude constants and config kwargs, as those are tracked separately.
            new_call_args = []
            constants = triton_meta["constants"]
            call_kwargs = {
                key: val
                for key, val in zip(signature, call_args)
                # pyrefly: ignore [missing-attribute]
                if key not in constants and key not in cfg.kwargs
            }

            # Add constants stored as Triton metadata, in signature order.
            call_kwargs |= constants
            new_call_args = [
                call_kwargs[key]
                for key in signature
                # pyrefly: ignore [missing-attribute]
                if key not in cfg.kwargs
            ]

            # Add Inductor's extra launcher args to the end.
            if extra_launcher_args := tuner.inductor_meta.get("extra_launcher_args"):
                new_call_args.extend(
                    call_args[len(call_args) - len(extra_launcher_args) :]
                )

            return tuple(new_call_args)

        kernel_config = tuner.compile_results[0].config
        extra_options = getattr(kernel_config, "extra_options", None)
        call_args = add_constants_to_call_args(call_args, kernel_config)
        call_args, grid = tuner._interpret_args_grid(call_args, kernel_config)
        call_kwargs = dict(zip(signature, call_args))
        # pyrefly: ignore [missing-attribute]
        assert not any(kwarg in kernel_config.kwargs for kwarg in call_kwargs), (
            f"kwargs overlap config: {call_kwargs}"
        )
        # pyrefly: ignore [missing-attribute]
        call_kwargs.update(kernel_config.kwargs)

        # Replace sympy.floor with FloorDiv, to make the expression traceable.
        grid = [replace_floor_div(x) if isinstance(x, sympy.Expr) else x for x in grid]
        wrapper_grid = [tuple(self._generate_sym_nodes(grid))]
        call_kwargs = {
            name: self._generate_sym_node(val) for name, val in call_kwargs.items()
        }

        # Store non-graphable kwargs in the side table.
        (
            call_kwargs,
            constant_args_idx,
        ) = tracing_triton_hopifier_singleton.store_non_graphable_args(call_kwargs)

        triton_node = self.gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": kernel.wrapped.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": wrapper_grid,
                "tma_descriptor_metadata": {},
                "kwargs": call_kwargs,
            },
        )
        if extra_options:
            triton_node.meta["extra_options"] = extra_options

    def _generate_extern_kernel_alloc(self, line: WrapperLine) -> None:
        assert isinstance(line, ExternKernelAllocLine)
        node = line.node
        self._generate_extern_kernel_common(node, node)

    def _generate_extern_kernel_out(
        self,
        line: WrapperLine,
    ) -> None:
        assert isinstance(line, ExternKernelOutLine)
        node = line.node
        out_node = node.output_view if node.output_view else node
        self._generate_extern_kernel_common(node, out_node)

    def _generate_extern_kernel_common(
        self, kernel: ir.ExternKernel, out_ir_node: ir.IRNode
    ) -> None:
        """
        Generates FX IR from either ExternKernelAlloc or ExternKernelOut.
        """

        # Get FX nodes corresponding to the call args.
        assert ir.is_node_sequence(kernel.inputs)
        tensor_nodes = tuple(self._generate_buffer(arg) for arg in kernel.inputs)
        if hasattr(kernel, "unflatten_args"):
            args, _ = kernel.unflatten_args(tensor_nodes, kernel.constant_args)
        else:
            args = tensor_nodes + tuple(kernel.constant_args)

        # Get the result buffer.
        # Some kernels write to a pre-existing output tensor via the "out" kwarg.
        kwargs = kernel.kwargs.copy()

        result_buffer: Optional[str] = None
        if isinstance(kernel, ir.ExternKernelOut):
            kwargs["out"] = self.buffer_to_node[out_ir_node.codegen_reference()]
        elif isinstance(kernel.layout, (ir.Layout, ir.MultiOutputLayout)):
            result_buffer = kernel.get_name()
        elif isinstance(kernel.layout, ir.NoneLayout):
            pass
        else:
            raise NotImplementedError(f"Unrecognized output layout: {kernel.layout}")

        fx_node = self.gm.graph.call_function(
            kernel.op_overload,  # type: ignore[arg-type]
            args=args,
            kwargs=kwargs,
        )

        # Assign the result to the given name.
        if result_buffer:
            assert "out" not in kwargs, (
                f"Extern kernel '{kernel}' has both result and out kwarg. Expected only one."
            )
            fx_node.name = result_buffer
            self.buffer_to_node[result_buffer] = fx_node

    def _generate_kernel_call(self, line: WrapperLine) -> None:
        assert isinstance(line, KernelCallLine)
        if not line.triton:
            raise NotImplementedError("FX conversion only supports Triton kernels.")

        self._generate_triton_call(line)

    def _generate_kernel_definition(self, line: WrapperLine) -> None:
        assert isinstance(line, KernelDefinitionLine)

        # Generate code for the kernel.
        kernel_code = PythonWrapperCodegen._format_kernel_definition(
            line.kernel_name, line.kernel_body, metadata=line.metadata
        )

        # Import the module and store the JIT kernel.
        tuner = self._import_kernel(kernel_code, line.kernel_name)
        wrapped = wrap_triton(tuner.fn)
        self.kernels[line.kernel_name] = TritonKernel(tuner, wrapped)

    def _generate_symbolic_call_arg(self, line: WrapperLine) -> None:
        assert isinstance(line, SymbolicCallArgLine)
        # Store the arg: expr mapping for later use.
        arg = line.arg

        inner_expr_proxy = self._sympy_interp(arg.inner_expr)
        self.expr_to_proxy[arg.inner] = inner_expr_proxy

    def _generate_unbacked_symbol_defs(self, line: WrapperLine) -> None:
        assert isinstance(line, UnbackedSymbolDefsLine)
        graph = self.gm.graph

        def convert_key(node: torch.fx.Node, path: pytree.KeyPath) -> torch.fx.Node:
            """
            Generate FX IR for each key entry.
            """
            # Base case.
            if len(path) == 0:
                return node

            # Process the first entry and recurse.
            entry = path[0]
            if isinstance(entry, CallMethodKey):
                target = {
                    "size": aten.sym_size.int,
                    "stride": aten.sym_stride.int,
                    "storage_offset": aten.sym_storage_offset,
                }[entry.name]
                assert callable(target)
                node = graph.call_function(
                    target,
                    args=(
                        (node, path[1].idx)
                        if len(path) > 1 and isinstance(path[1], pytree.SequenceKey)
                        else (node,)
                    ),
                )
                return convert_key(node, path[1 + len(node.args) :])
            elif isinstance(entry, pytree.SequenceKey):
                node = graph.call_function(operator.getitem, args=(node, entry.idx))
                return convert_key(node, path[1:])
            elif isinstance(entry, DivideByKey):
                node = graph.call_function(
                    operator.floordiv, args=(node, entry.divisor)
                )
                return convert_key(node, path[1:])
            else:
                raise NotImplementedError(f"Unrecognized entry type: {type(entry)}")

        root_node = self.buffer_to_node[line.output_name]
        unbacked_bindings = line.unbacked_bindings
        assert unbacked_bindings is not None
        for s, keypath in unbacked_bindings.items():
            # Check if we already generated this symbol.
            if s.name in self.buffer_to_node:
                continue

            node = convert_key(root_node, keypath)
            out_buffer = SymbolBuffer(s)
            self._record_allocation(out_buffer, node)
            self._generate_size_proxy(node, s)
