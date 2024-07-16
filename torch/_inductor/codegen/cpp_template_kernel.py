# mypy: allow-untyped-defs
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
from torch.utils._sympy.symbol import SymT
from .. import config, cpp_builder, ir, lowering as L

from ..autotune_process import CppBenchmarkRequest
from ..select_algorithm import PartialRender
from ..utils import sympy_index_symbol, sympy_index_symbol_with_prefix
from ..virtualized import V
from .cpp import CppKernel, CppKernelProxy, KernelGroup
from .cpp_utils import cexpr_index, DTYPE_TO_CPP, LocalBufferContext


def parse_expr_with_index_symbols(expr):
    if isinstance(expr, sympy.Expr):
        return expr
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    else:
        expr = parse_expr(str(expr))
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)


def wrap_with_tensorbox(node) -> ir.TensorBox:
    return (
        ir.TensorBox.create(node) if isinstance(node, ir.Buffer) else ir.TensorBox(node)
    )


class CppTemplateKernel(CppKernel):
    def __init__(self, kernel_name, num_threads):
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name
        self.render_hooks = {}
        self.local_buffers = {}

    def render(self, template, **kwargs):
        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: Dict[str, ir.Buffer],
        outputs: Dict[str, ir.Buffer],
        aliases: Optional[List[Tuple[ir.Buffer, ir.Buffer]]] = None,
    ) -> str:
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        if aliases is not None:
            for alias, orig in aliases:
                orig_name = orig.get_name()
                alias_name = alias.get_name()
                if orig_name in self.args.input_buffers:
                    self.args.input_buffers[alias_name] = self.args.input_buffers[
                        orig_name
                    ]
                if orig_name in self.args.output_buffers:
                    self.args.output_buffers[alias_name] = self.args.output_buffers[
                        orig_name
                    ]

        unique_sizevars = {
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        }
        unique_sizevars |= {
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        }
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook():
            # remove all aliases before generate function definition
            if aliases is not None:
                for alias, _ in aliases:
                    alias_name = alias.get_name()
                    if alias_name in self.args.input_buffers:
                        self.args.input_buffers[alias_name] = "REMOVED"
                    if alias_name in self.args.output_buffers:
                        self.args.output_buffers[alias_name] = "REMOVED"
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {self.kernel_name}({', '.join(cpp_argdefs)})"

        placeholder = "<DEF_KERNEL>"
        assert placeholder not in self.render_hooks
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node: ir.CppTemplateBuffer):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, cuda=False, arg_types=arg_types)

    def dtype(self, node: ir.Buffer) -> str:
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: ir.Buffer) -> str:
        if node.get_dtype() in [torch.float32, torch.bfloat16, torch.half]:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: ir.Buffer, indices: List[Any]) -> str:
        indexer = node.layout.as_fixed().make_indexer()
        index = indexer(parse_expr_with_index_symbols(indices))
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: List[Tuple[Any, Any]]) -> ir.ReinterpretView:
        """
        Slice the given node with a list of ranges (start and end) corresponding to its dims.
        The dim is not sliced if the corresponding range is empty.
        """
        assert len(ranges) == len(node.get_size())
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            assert len(_range) == 2
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        assert isinstance(sliced.data, ir.ReinterpretView), sliced.data
        return sliced.data

    def view(self, node, sizes: List[Any]) -> ir.View:
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        return L.view(node, sizes).data

    def permute(self, node, dims):
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        assert isinstance(permuted, ir.ReinterpretView)
        return permuted

    def maybe_codegen_profile(self) -> str:
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            return f'RECORD_FUNCTION("{prefix}{self.kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
        else:
            return ""

    def unroll_pragma(self, unroll):
        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name, sizes: List[Any], dtype=torch.float) -> str:
        """Define kernel local buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(name, ir.FixedLayout(torch.device("cpu"), dtype, sizes))
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def store_pointwise_nodes(
        self,
        dst: ir.Buffer,
        nodes: List[ir.IRNode],
        offsets: Optional[List[sympy.Expr]] = None,
        reindexers: Optional[List[Optional[Callable[[List[Any]], List[Any]]]]] = None,
    ) -> str:
        var_sizes = (tuple(dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            offsets = [sympy.Integer(0)] * len(var_sizes[0])
        if not reindexers:
            reindexers = [None] * len(nodes)
        assert len(offsets) == len(var_sizes[0])
        output_index = dst.get_layout().make_indexer()(var_ranges.keys())
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            assert isinstance(node, ir.Pointwise), node

            def fn(*args):
                assert len(args) == 2
                assert len(args[0]) == len(var_sizes[0])
                assert len(args[1]) == 0
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = ir.LoopBody(fn, (list(var_ranges.keys()), ()), var_ranges)
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)
        kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_output(
        self,
        dst: ir.Buffer,
        src: ir.Buffer,
        orig_src: Optional[ir.Buffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        offsets: Optional[List[Any]] = None,
        reindexers: Optional[List[Optional[Callable[[List[Any]], List[Any]]]]] = None,
    ):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come form the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
        assert dst.get_size() == src.get_size()
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        if epilogue_nodes:
            with LocalBufferContext(self.args) as scope:
                assert orig_src is not None
                if orig_src.get_name() != src.get_name():
                    scope.add_local_buffer(
                        src,
                        [
                            orig_src,
                        ],
                    )
                    epilogue_nodes = scope.localize_nodes(epilogue_nodes)
                return self.store_pointwise_nodes(
                    dst, epilogue_nodes, offsets, reindexers  # type: ignore[arg-type]
                )
        else:
            if dst.get_name() != src.get_name():
                # src is local
                copy = L.copy(dst, src).data.data
                with LocalBufferContext(self.args) as scope:
                    scope.add_local_buffer(src)
                    return self.store_pointwise_nodes(dst, [copy])
            else:
                assert dst.layout == src.layout
                return ""


class CppTemplateCaller(ir.ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[ir.Buffer],
        layout: ir.Layout,
        make_kernel_render: Callable[
            [ir.CppTemplateBuffer, Optional[List[ir.IRNode]]], str
        ],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: Optional[
            Dict[str, Union[ir.PrimitiveInfoType, List[ir.PrimitiveInfoType]]]
        ] = None,
    ):
        super().__init__(name, input_nodes, layout)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(
        self,
    ) -> Dict[str, Union[ir.PrimitiveInfoType, List[ir.PrimitiveInfoType]]]:
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> ir.TensorBox:
        return ir.TensorBox.create(
            ir.CppTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
                choice=self,
            )
        )
