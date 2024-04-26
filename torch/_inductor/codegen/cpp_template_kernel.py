import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch

from torch._inductor.autotune_process import CppBenchmarkRequest
from torch._inductor.utils import sympy_index_symbol
from .. import lowering as L
from ..ir import (
    Buffer,
    ChoiceCaller,
    CppTemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    ReinterpretView,
    TensorBox,
    View,
)
from ..virtualized import V
from .common import Kernel, OpOverrides
from .cpp import cexpr_index


def parse_expr_with_index_symbols(expr_str: str) -> sympy.Expr:
    expr = parse_expr(expr_str)
    int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
    return expr.subs(int_symbols)


def wrap_with_tensorbox(node) -> TensorBox:
    return TensorBox.create(node) if isinstance(node, Buffer) else TensorBox(node)


class CppTemplateKernel(Kernel):
    overrides = OpOverrides

    def __init__(self, kernel_name):
        super().__init__()
        self.kernel_name = kernel_name

    def def_kernel(
        self,
        inputs: List[Buffer],
        outputs: List[Buffer],
        names_str: str = "",
    ) -> str:
        input_names = [inp.get_name() if inp is not None else None for inp in inputs]
        output_names = [out.get_name() for out in outputs]
        all_names = input_names + output_names
        assert len(all_names) == len(names_str.split(",")), (
            all_names,
            names_str,
        )
        names = names_str.split(",")
        for i, input_name in enumerate(input_names):
            if input_name is not None:
                self.args.input_buffers[input_name] = names[i].strip()
        for i, output_name in enumerate(output_names):
            self.args.output_buffers[output_name] = names[i + len(input_names)].strip()
        inputs_not_none = [inp for inp in inputs if inp is not None]
        unique_sizevars = {
            s
            for input in inputs_not_none
            for sym in itertools.chain(input.get_size(), input.get_stride())
            for s in sym.free_symbols
        }
        unique_sizevars |= {
            s
            for output in outputs
            for sym in itertools.chain(output.get_size(), output.get_stride())
            for s in sym.free_symbols
        }
        sizevars = sorted(unique_sizevars)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"
        cpp_argdefs, _, _ = self.args.cpp_argdefs()
        return f"void {self.kernel_name}({', '.join(cpp_argdefs)})"

    def call_kernel(self, name: str, node: CppTemplateBuffer):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, cuda=False, arg_types=arg_types)

    def dtype(self, node: Buffer) -> str:
        if node.get_dtype() == torch.float32:
            return "float"
        elif node.get_dtype() == torch.bfloat16:
            return "float"
        elif node.get_dtype() == torch.half:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def acc_dtype(self, node: Buffer) -> str:
        if node.get_dtype() == torch.float32:
            return "float"
        elif node.get_dtype() == torch.bfloat16:
            return "float"
        elif node.get_dtype() == torch.half:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: Buffer, dim: int) -> str:
        return str(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: Buffer, dim: int) -> str:
        return str(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: Buffer, indices: List[Any]) -> str:
        indexer = node.make_indexer()
        index = indexer([sympy.Symbol(str(idx), integer=True) for idx in indices])
        index = self.rename_indexing(index)
        return f"{self.args.input(node.get_name())}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: List[Tuple[Any]]) -> ReinterpretView:
        assert len(ranges) == len(node.get_size())
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            assert len(_range) == 2
            start, end = (parse_expr_with_index_symbols(str(r)) for r in _range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        assert isinstance(sliced.data, ReinterpretView)
        return sliced.data

    def view(self, node, sizes: List[Any]) -> View:
        node = wrap_with_tensorbox(node)
        sizes = [parse_expr_with_index_symbols(str(s)) for s in sizes]
        return L.view(node, sizes).data


class CppTemplateCaller(ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (CppTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[CppTemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: Optional[
            Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
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

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> TensorBox:
        return TensorBox.create(
            CppTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
                choice=self,
            )
        )
