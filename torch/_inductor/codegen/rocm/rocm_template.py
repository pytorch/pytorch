# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

from ...autotune_process import TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_kernel import ROCmTemplateCaller, ROCmTemplateKernel
from .rocm_template_buffer import ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE


log = logging.getLogger(__name__)


# FIXME: unify with the CUDA version
@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str


class ROCmTemplate(KernelTemplate):
    """Base class for ROCm C++ kernel templates - templates are stateless and input dependent types come in for generation."""

    index_counter = itertools.count()
    gfx9_threads_per_warp = 64

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: Optional[list[int]] = None,
        **kwargs,
    ) -> ROCmTemplateCaller:
        """Generate ROCm template caller for autotuning."""
        output_node = Buffer(name="buf_out", layout=layout)

        kernel_name = f"rocm_{self.name}"
        kernel_hash_name = f"rocm_{self.name}_{next(self.index_counter)}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(output_node)),
            ROCmTemplateKernel(
                kernel_name=kernel_name,
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            ) as kernel,
        ):
            code = self.render(
                kernel=kernel,
                input_nodes=input_nodes,
                output_node=output_node,
                input_reorder=input_reorder,
                **kwargs,
            )
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Autotune key: %s, Generated Code:\n%s", kernel_hash_name, code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder_list = (
            input_reorder
            if input_reorder is not None
            else list(range(len(input_nodes)))
        )
        expected_args = list(
            unique(input_nodes[idx].get_name() for idx in input_reorder_list)
        )
        expected_args.extend([output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )

        size_args = (
            self.size_args(input_nodes, layout, **kwargs)
            if hasattr(self, "size_args")
            else ()
        )
        size_args_ints = [V.graph.sizevars.size_hint(arg) for arg in size_args]
        # The runtime args come right after the size args
        runtime_args = self.get_runtime_arg_values(**kwargs)
        extra_args = size_args_ints + runtime_args
        bmreq = ROCmBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ROCmTemplateBuffer,
            epilogue_nodes: Optional[Sequence[IRNode]] = None,
        ):
            kernel = ROCmTemplateKernel(
                kernel_name="KERNEL_NAME",
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                input_nodes=input_nodes,
                output_node=template_node if template_node is not None else output_node,
                input_reorder=input_reorder,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return ROCmTemplateCaller(
            kernel_hash_name,
            self.name,
            input_nodes,
            output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )

    def _fake_get_dtype(self, buf):
        return lambda name: buf.get_dtype() if name == buf.get_name() else None

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
            """
        )
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return []

    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        return []
