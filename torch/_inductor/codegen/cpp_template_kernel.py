from typing import Callable, Dict, List, Optional, Union

import sympy

import torch

from torch._inductor.autotune_process import CppBenchmarkRequest
from ..ir import (
    Buffer,
    ChoiceCaller,
    CppTemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    TensorBox,
)
from ..virtualized import V
from .common import Kernel, OpOverrides
from .cpp import cexpr_index


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
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        input_names = [inp.get_name() for inp in inputs]
        output_names = [out.get_name() for out in outputs]
        all_names = input_names + output_names
        assert len(all_names) == len(names_str.split(",")), (
            all_names,
            names_str,
        )
        names = names_str.split(",")
        for i, input_name in enumerate(input_names):
            self.args.input_buffers[input_name] = names[i].strip()
        for i, output_name in enumerate(output_names):
            self.args.output_buffers[output_name] = names[i + len(input_names)].strip()
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
        return str(node.get_size()[dim])

    def index(self, node: Buffer, indices: List[str]) -> str:
        indexer = node.make_indexer()
        index = indexer([sympy.Symbol(idx) for idx in indices])
        return f"{self.args.input(node.get_name())}[{cexpr_index(index)}]"


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

    # def __str__(self):
    #     return f"CppTemplateCaller(source_file={self.bmreq.source_file})"

    # def call_name(self) -> str:
    #     return f"cpp_template_kernels.{self.name}"

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
