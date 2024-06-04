# mypy: allow-untyped-defs
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
from .. import codecache, config, ir, lowering as L

from ..autotune_process import CppBenchmarkRequest
from ..select_algorithm import PartialRender
from ..utils import sympy_index_symbol
from ..virtualized import V
from .common import Kernel, OpOverrides
from .cpp import CppKernelProxy, KernelGroup
from .cpp_utils import cexpr_index, DTYPE_TO_CPP


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


class CppTemplateKernel(Kernel):
    overrides = OpOverrides

    def __init__(self, kernel_name):
        super().__init__()
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
    ) -> str:
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            if out.get_name() not in self.args.inplace_buffers:
                self.args.output_buffers[out.get_name()] = name
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
        return f"{self.args.input(node.get_name())}[{cexpr_index(index)}]"

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
        assert isinstance(sliced.data, ir.ReinterpretView)
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

    @property
    def assert_function(self) -> str:
        if V.graph.aot_mode:
            return "AOTI_TORCH_CHECK"
        else:
            return "TORCH_CHECK"

    def maybe_codegen_profile(self) -> str:
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            return f'RECORD_FUNCTION("{prefix}{self.kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
        else:
            return ""

    def unroll_pragma(self, unroll):
        if codecache.is_gcc():
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

    def store_output(
        self,
        dst: ir.Buffer,
        src: ir.Buffer,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        offsets: Optional[List[Any]] = None,
        reindexer: Optional[Callable[[List[Any]], List[Any]]] = None,
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
        """
        assert dst.get_size() == src.get_size()
        if epilogue_nodes:
            var_sizes = (tuple(dst.get_size()), ())
            var_ranges = {
                sympy.Symbol(f"z{i}"): sz for i, sz in enumerate(var_sizes[0])
            }

            # epilogues are all pointwises, hence all indexed the same way as dst
            output_index = dst.get_layout().make_indexer()(var_ranges.keys())

            if not offsets:
                offsets = [0] * len(var_sizes[0])
            assert len(offsets) == len(var_sizes[0])
            offsets = parse_expr_with_index_symbols(offsets)

            kernel_group = KernelGroup()
            kernel_group.args = self.args
            cpp_kernel_proxy = CppKernelProxy(kernel_group)
            bodies = []
            var_sizes_list = []
            for i, node in enumerate(epilogue_nodes):
                assert isinstance(node, ir.ComputedBuffer)
                output_name = (
                    node.get_name() if i < len(epilogue_nodes) - 1 else dst.get_name()
                )

                def fn(*args):
                    assert len(args) == 2
                    assert len(args[0]) == len(var_sizes[0])
                    assert len(args[1]) == 0
                    new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                    if reindexer is not None:
                        new_args = reindexer(new_args)
                    V.ops.store(
                        output_name,
                        output_index,
                        node.data.make_loader()(new_args).value,
                    )

                body = ir.LoopBody(fn, (list(var_ranges.keys()), ()), var_ranges)
                bodies.append(body)
                var_sizes_list.append(var_sizes)

            cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
            return kernel_group.loops_code.getvalue()
        else:
            # TODO(jgong5): support local acc buffer to avoid assertion below
            assert dst.get_name() == src.get_name() and dst.layout == src.layout
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
