# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from ...autotune_process import TensorMeta
from ...ir import Buffer, FlexibleLayout, IRNode, Layout
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
    index_counter = itertools.count()
    gfx9_threads_per_warp = 64

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: list[int] | None = None,
    ) -> None:
        """

        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the ROCmTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.original_input_nodes = input_nodes
        self.input_nodes = []
        # Operation choices retain these layouts without constraining other choices.
        for node in input_nodes:
            input_layout = node.get_layout()
            if isinstance(input_layout, FlexibleLayout):
                input_layout = input_layout.get_fixed_layout_without_freezing()
            self.input_nodes.append(Buffer(name=node.get_name(), layout=input_layout))
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> ROCmTemplateCaller:
        """
        Generates the ROCm template caller object for the given GEMM template and operation. This ROCmTemplateCaller
        may be used to call and benchmark the generated ROCm kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A ROCmTemplateCaller object representing the generated ROCm template caller.
        """
        kernel_name = f"rocm_{self.name}"
        kernel_hash_name = f"rocm_{self.name}_{next(self.index_counter)}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            ROCmTemplateKernel(
                kernel_name=kernel_name,
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            ) as kernel,
        ):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Autotune key: %s, Generated Code:\n%s", kernel_hash_name, code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )

        size_args = (
            self.size_args() if hasattr(self, "size_args") else ()
        )  # subclass should define def size_args()
        # Resolve symbolic sizes to concrete ints for benchmarking only.
        size_args_ints = list(V.graph.sizevars.optimization_hints(size_args))
        # The runtime args come right after the size args
        runtime_args = self.get_runtime_arg_values(**kwargs)
        extra_args = size_args_ints + runtime_args
        nodes_by_name = {node.get_name(): node for node in self.input_nodes}
        benchmark_nodes = [nodes_by_name[name] for name in expected_args[:-1]]
        sizevars = V.graph.sizevars
        input_tensor_meta = [
            TensorMeta(
                device=node.get_device_or_error(),
                dtype=node.get_dtype(),
                sizes=sizevars.optimization_hints(node.get_size()),
                strides=sizevars.optimization_hints(node.get_stride()),
                offset=sizevars.optimization_hint(node.get_layout().offset),
                name=node.get_name(),
            )
            for node in benchmark_nodes
        ]
        bmreq = ROCmBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=input_tensor_meta,
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ROCmTemplateBuffer,
            epilogue_nodes: Sequence[IRNode] | None = None,
        ):
            kernel = ROCmTemplateKernel(
                kernel_name="KERNEL_NAME",
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return ROCmTemplateCaller(
            kernel_hash_name,
            self.name,
            self.original_input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )

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
