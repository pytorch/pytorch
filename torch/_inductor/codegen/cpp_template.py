# mypy: allow-untyped-defs
import functools
import itertools
import logging

import sys
from typing import List, Optional
from unittest.mock import patch

import sympy

from .. import codecache, config, ir
from ..autotune_process import CppBenchmarkRequest, TensorMeta
from ..utils import IndentedBuffer, Placeholder, unique
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel

log = logging.getLogger(__name__)


class CppTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes,
        layout: ir.Layout,
    ):
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: ir.Buffer = ir.Buffer("buf_out", layout)
        self.layout = layout

    def generate(self, **kwargs):
        kernel_name = f"cpp_{self.name}"
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CppTemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            code = kernel.render(self, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )

        kernel_hash_name = f"cpp_{self.name}_{next(self.index_counter)}"

        # Create the BenchmarkRequest for CPP
        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ir.CppTemplateBuffer,
            epilogue_nodes: Optional[List[ir.IRNode]] = None,
        ):
            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
            )
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.writeline(codecache.cpp_prefix())
        res.splice(
            """
                #include "c10/util/Unroll.h"
            """
        )
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        if enable_kernel_profile:
            res.writelines(["#include <ATen/record_function.h>"])
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError
