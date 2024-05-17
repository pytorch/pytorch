import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch

from torch._inductor.autotune_process import CppBenchmarkRequest
from torch._inductor.utils import sympy_index_symbol
from .. import codecache, config, ir, lowering as L
from ..virtualized import V
from .common import Kernel, OpOverrides
from .cpp_utils import cexpr_index, DTYPE_TO_CPP


def parse_expr_with_index_symbols(expr_str: str) -> sympy.Expr:
    expr = parse_expr(expr_str)
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

    def def_kernel(
        self,
        inputs: Dict[str, ir.Buffer],
        outputs: Dict[str, ir.Buffer],
    ) -> str:
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
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
        cpp_argdefs, _, _ = self.args.cpp_argdefs()
        return f"void {self.kernel_name}({', '.join(cpp_argdefs)})"

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
        indexer = node.make_indexer()
        index = indexer([parse_expr_with_index_symbols(str(idx)) for idx in indices])
        index = self.rename_indexing(index)
        return f"{self.args.input(node.get_name())}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: List[Tuple[Any]]) -> ir.ReinterpretView:
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
            start, end = (parse_expr_with_index_symbols(str(r)) for r in _range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        assert isinstance(sliced.data, ir.ReinterpretView)
        return sliced.data

    def view(self, node, sizes: List[Any]) -> ir.View:
        node = wrap_with_tensorbox(node)
        sizes = [parse_expr_with_index_symbols(str(s)) for s in sizes]
        return L.view(node, sizes).data

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
