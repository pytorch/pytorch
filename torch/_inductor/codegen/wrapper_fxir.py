import dataclasses
import functools
import logging
import operator
import textwrap
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import sympy

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._export.utils import _detect_fake_mode_from_gm
from torch._higher_order_ops.triton_kernel_wrap import (
    TraceableTritonKernelWrapper,
    tracing_triton_hopifier_singleton,
    triton_kernel_wrapper_mutation,
)
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import extern_kernels  # noqa: F401
from torch._inductor.utils import convert_shape_to_symint, sympy_product
from torch._inductor.virtualized import V
from torch._library.triton import wrap_triton
from torch.fx import GraphModule
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import OptimizedPythonReferenceAnalysis

from .. import config, ir
from ..runtime.triton_compat import Config
from ..utils import LineContext
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
    EnterDeviceContextManagerLine,
    EnterSubgraphLine,
    ExitDeviceContextManagerLine,
    ExitSubgraphLine,
    ExternKernelAllocLine,
    ExternKernelOutLine,
    FreeIfNotReusedLine,
    FreeLine,
    KernelCallLine,
    KernelDefinitionLine,
    Line,
    MultiOutputLine,
    NullLine,
    PythonWrapperCodegen,
    ReinterpretLine,
    ReuseLine,
    SymbolicCallArg,
    SymbolicCallArgLine,
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

    def get_example(self) -> Union[torch.Tensor, sympy.Symbol]:
        return self.symbol


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
    expr = sympy.together(expr)

    # Find division operations in the sympy.floor expression
    # Div is either represented as Mul with:
    # Rational denominator or Pow with negative exponent
    if not isinstance(expr, sympy.core.mul.Mul):
        return sympy.floor(expr)

    if isinstance(expr.args[0], sympy.Rational):
        frac = expr.args[0]
        numerator = sympy_product(expr.args[1:]) * frac.numerator
        denominator = frac.denominator

        return FloorDiv(numerator, denominator)
    elif isinstance(expr.args[0], sympy.Pow):
        base = expr.args[0].base
        exp = expr.args[0].exp
        numerator = sympy_product(expr.args[1:])
        if exp < 0:
            denominator = base ** (-exp)
        else:
            numerator = numerator * (base**exp)
            denominator = 1
        return FloorDiv(numerator, denominator)
    else:
        return sympy.floor(expr)


class WrapperFxCodegen(PythonWrapperCodegen):
    """
    Backend to generate wrapper code as an FX IR graph.
    """

    supports_caching = False

    def _generate(self, is_inference: bool) -> tuple[FileBackedGraphModule, None]:
        self.run_wrapper_ir_passes(is_inference)

        prologue = "\n".join(
            [
                self.imports.getvalue(),
                self.header.getvalue(),
            ]
        )
        gm = FxConverter(lines=self.lines, prologue=prologue).generate()
        compiled_fn = self.compile_graph(gm)

        return FileBackedGraphModule(gm, compiled_fn), None

    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]:
        """
        Converts the graph module into a runnable function. The default implementation
        is simply an interpreter calling kernels in eager mode. Derived backends can
        override this to do further compilation.
        """
        return gm.forward

    @classmethod
    def create(
        cls,
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ) -> "WrapperFxCodegen":
        if is_subgraph:
            raise NotImplementedError(
                "Subgraphs are not yet supported by FX conversion"
            )

        # For derived backends, this could be a subclass.
        return cls()


@dataclasses.dataclass
class FxConverter:
    """
    Generates FX IR from Wrapper IR. As each instance is only meant to be used once, the
    input and output code are stored as attributes.
    """

    lines: list[Line]
    prologue: str = ""

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

    def _fake_tensor(
        self,
        size: tuple[Any, ...],
        stride: tuple[Any, ...],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        with V.fake_mode:
            return torch.empty_strided(
                convert_shape_to_symint(size),
                convert_shape_to_symint(stride),
                dtype=dtype,
                device=device,
            )

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

    def _generate_graph_inputs(self) -> None:
        """
        Converts graph inputs to FX placeholders.
        """

        for node in V.graph.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            name = node.name
            if name in V.graph.graph_inputs:
                ir_node = V.graph.graph_inputs[name]

                # Introduce a new symbol for constant inputs.
                buffer = (
                    SymbolBuffer(sympy.Symbol(name, is_integer=True))
                    if isinstance(ir_node, (int, float, sympy.Integer, sympy.Float))
                    else self._get_buffer(ir_node)
                )
                placeholder_node = self.gm.graph.placeholder(buffer.get_name())
                placeholder_node.meta["val"] = buffer.get_example()
                self._record_allocation(buffer, placeholder_node)

            elif V.aot_compilation:
                # Create dummy input nodes to match the input signature
                self.gm.graph.placeholder(name)

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
            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in self.expr_to_proxy:
                    return

                size_node = self.gm.graph.call_function(target, (base_node, dim))
                size_proxy = torch.fx.Proxy(size_node, tracer=self.tracer)

                self.expr_to_proxy[sym_or_exp] = size_proxy

            elif isinstance(sym_or_exp, sympy.Integer):
                return

            elif isinstance(sym_or_exp, sympy.Expr):
                self._sympy_interp(sym_or_exp)

        for node in V.graph.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            name = node.name
            if name in V.graph.graph_inputs:
                ir_node = V.graph.graph_inputs[name]
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

    def _generate_output(self) -> None:
        """
        Generate FX IR for graph outputs.
        """
        output_nodes = [
            self._generate_buffer(node)
            for idx, node in enumerate(V.graph.graph_outputs)
        ]

        # Single return elements don't use a tuple.
        output_value = output_nodes[0] if len(output_nodes) == 1 else output_nodes

        self.gm.graph.output(output_value)

    def generate(self) -> torch.fx.GraphModule:
        """
        Main entrypoint for FX codegen.
        """
        self._generate_graph_inputs()
        self._generate_graph_constants()

        fake_mode = _detect_fake_mode_from_gm(self.gm)

        with _set_node_metadata_hook(
            self.gm,
            functools.partial(_node_metadata_hook, fake_mode=fake_mode),
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

        self._generate_output()
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
        dtype = buffer.get_dtype()
        shape = self._generate_sym_nodes(buffer.get_size())
        stride = self._generate_sym_nodes(buffer.get_stride())

        node = self.gm.graph.call_function(
            torch.empty_strided,
            args=(shape, stride),
            kwargs={"dtype": dtype, "device": device},
        )
        assert name
        node.name = name
        self._record_allocation(buffer, node)

    def _generate_comment(self, line: WrapperLine) -> None:
        assert isinstance(line, CommentLine)
        # We ignore comments in FX IR.

    def _generate_enter_device_context_manager(self, line: WrapperLine) -> None:
        assert isinstance(line, EnterDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_exit_device_context_manager(self, line: WrapperLine) -> None:
        assert isinstance(line, ExitDeviceContextManagerLine)
        # We ignore the device context in FX IR.

    def _generate_enter_subgraph(self, line: WrapperLine) -> None:
        assert isinstance(line, EnterSubgraphLine)
        raise NotImplementedError("Subgraphs are not yet supported by FX conversion")

    def _generate_exit_subgraph(self, line: WrapperLine) -> None:
        assert isinstance(line, ExitSubgraphLine)
        raise NotImplementedError("Subgraphs are not yet supported by FX conversion")

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
        # Use python_slow mode instead of python mode to avoid
        # the round to neginf behaviour, which is not the convention
        # in other languages.
        tuner.grid_mode = "python_slow"

        # Optionally autotune the kernels.
        # The FX backend currently only supports compile-time tuning.
        kernel_name = tuner.fn.__name__
        if config.triton.autotune_at_compile_time:
            from triton.runtime import driver

            log.info("Autotuning Triton kernel %s at compile time.", kernel_name)
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)

            def node_to_tuning_arg(arg: Any) -> Any:
                """
                Create real tensors for autotuning arguments, substituting size hints
                for dynamic shapes.
                """
                to_size_hint = functools.partial(
                    pytree.tree_map, V.graph.sizevars.size_hint
                )
                if not isinstance(arg, torch.fx.Node):
                    return to_size_hint(arg)

                fake = arg.meta["val"]
                return torch.empty_strided(
                    to_size_hint(fake.shape),
                    to_size_hint(fake.stride()),
                    device=device,
                ).zero_()

            arg_values = [node_to_tuning_arg(arg) for arg in call_args]
            tuner.run(*arg_values, stream=stream)
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
            new_call_args = []
            call_arg_idx = 0
            constants = triton_meta["constants"]
            for arg_name in signature:
                # Config kwargs are tracked separately.
                if arg_name in cfg.kwargs:
                    continue

                try:
                    new_arg = constants[arg_name]
                except KeyError:
                    new_arg = call_args[call_arg_idx]
                    call_arg_idx += 1
                new_call_args.append(new_arg)

            # Add Inductor's extra call args to the end.
            new_call_args.extend(call_args[call_arg_idx:])

            return tuple(new_call_args)

        kernel_config = tuner.compile_results[0].config
        call_args = add_constants_to_call_args(call_args, kernel_config)
        call_args, grid = tuner._interpret_args_grid(call_args, kernel_config)
        call_kwargs = dict(zip(signature, call_args))
        call_kwargs.update(kernel_config.kwargs)

        # Replace all sympy.floor with FloorDiv
        # _generate_sym_node does not support sympy.floor
        grid = [
            x.replace(sympy.floor, replace_floor_div)
            if isinstance(x, sympy.Expr)
            else x
            for x in grid
        ]
        wrapper_grid = [tuple(self._generate_sym_nodes(grid))]
        call_kwargs = {
            name: self._generate_sym_node(val) for name, val in call_kwargs.items()
        }

        # Store non-graphable kwargs in the side table.
        (
            call_kwargs,
            constant_args_idx,
        ) = tracing_triton_hopifier_singleton.store_non_graphable_args(call_kwargs)

        self.gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": kernel.wrapped.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": wrapper_grid,
                "tma_descriptor_metadata": {},
                "kwargs": call_kwargs,
            },
        )

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
