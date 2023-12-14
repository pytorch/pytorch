import copy
import functools
import itertools
import logging
from typing import cast, Generator, Iterable, List, Optional
from unittest.mock import patch

import sympy

import torch
from ... import ir
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import (
    Buffer,
    CUDATemplateBuffer,
    FixedLayout,
    FlexibleLayout,
    IRNode,
    Layout,
)

from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel

log = logging.getLogger(__name__)


class CUDATemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: List[Buffer],
        layout: Layout,
        input_reorder: Optional[List[int]] = None,
    ):
        """

        Baseclass for CUDA C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> Generator[CUDATemplateCaller, None, None]:
        """
        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller
        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        """

        # Generate Row-Major and Column-Major variants of all flexible input tensor layouts
        input_layout_alternatives: List[List[TensorMeta]] = []
        for input_node in self.input_nodes:
            unchanged_variant = cast(TensorMeta, TensorMeta.from_irnodes(input_node))
            input_tensor_meta_variants = [unchanged_variant]
            if (
                hasattr(input_node, "layout")
                and isinstance(input_node.layout, FlexibleLayout)
                and len(input_node.layout.stride) >= 2
                and (
                    input_node.layout.stride[-1] == 1
                    and input_node.layout.stride[-2] == input_node.layout.size[-1]
                )
                or (
                    input_node.layout.stride[-2] == 1
                    and input_node.layout.stride[-1] == input_node.layout.size[-2]
                )
            ):
                layout_variant = copy.deepcopy(unchanged_variant)
                new_strides = list(layout_variant.strides)
                # switch between row-major and column-major
                if input_node.layout.stride[-1] == 1:
                    # row to col major
                    new_strides[-1] = input_node.layout.size[-2]
                    new_strides[-2] = 1
                else:
                    # col to row major
                    new_strides[-1] = 1
                    new_strides[-2] = input_node.layout.size[-1]
                layout_variant.strides = tuple(new_strides)
                input_tensor_meta_variants.append(layout_variant)
            input_layout_alternatives.append(input_tensor_meta_variants)
        all_variant_combinations: List[List[TensorMeta]] = list(
            cast(
                Iterable[List[TensorMeta]],
                itertools.product(*input_layout_alternatives),
            )
        )
        if len(all_variant_combinations) != 1:
            log.debug(
                "Generating %d input layout variants of %s",
                len(all_variant_combinations),
                str(self),
            )
        for kernel_idx, input_tensor_meta in enumerate(all_variant_combinations):
            kernel_name = f"cuda_{self.name}_{kernel_idx}"
            with patch.object(
                V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
            ), CUDATemplateKernel(
                kernel_name=kernel_name,
            ) as kernel:
                original_layouts = [
                    getattr(input_node, "layout", None)
                    for input_node in self.input_nodes
                ]
                try:
                    # temporarily set the strides of input nodes with FlexibleLayouts
                    # to the strides of the input_tensor_meta
                    for input_node, input_tensor_meta_variant in zip(
                        self.input_nodes, input_tensor_meta
                    ):
                        if isinstance(input_node.layout, FlexibleLayout):
                            lo = input_node.layout
                            new_layout = FixedLayout(
                                lo.device,
                                lo.dtype,
                                lo.size,
                                input_tensor_meta_variant.strides,
                                lo.offset,
                            )
                            if isinstance(input_node, ir.MutableBox):
                                input_node.data.layout = new_layout  # type: ignore[attr-defined]
                            else:
                                input_node.layout = new_layout  # type: ignore[attr-defined]
                    code = self.render(kernel=kernel, **kwargs)
                finally:
                    # restore the original (still flexible until Autotuning has been resolved) strides
                    for input_node, original_layout in zip(
                        self.input_nodes, original_layouts
                    ):
                        if isinstance(original_layout, FlexibleLayout):
                            if isinstance(input_node, ir.MutableBox):
                                input_node.data.layout = original_layout  # type: ignore[attr-defined]
                            else:
                                input_node.layout = original_layout  # type: ignore[attr-defined]

                _, call_args, _ = kernel.args.python_argdefs()
                log.debug("Generated Code:\n%s", code)
                log.debug(
                    "Args: cpp_argdefs: %s, python_argdefs: %s",
                    kernel.args.cpp_argdefs(),
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
            assert list(call_args)[: len(expected_args)] == expected_args, (
                call_args,
                expected_args,
            )
            extra_args = V.graph.sizevars.size_hints(
                map(sympy.expand, call_args[len(expected_args) :])
            )

            kernel_hash_name = f"cuda_{self.name}_{next(self.index_counter)}"

            def make_kernel_render(
                template_node: CUDATemplateBuffer,
                epilogue_nodes: Optional[List[IRNode]] = None,
            ):
                kernel = CUDATemplateKernel(
                    kernel_name="KERNEL_NAME",
                )
                render = functools.partial(
                    self.render,
                    kernel=kernel,
                    template_buffer_node=template_node,
                    epilogue_nodes=epilogue_nodes,
                    **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
                )
                return kernel, render

            # create the BenchmarkRequest
            bmreq = CUDABenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=input_tensor_meta,
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=extra_args,
                source_code=code,
            )

            yield CUDATemplateCaller(
                kernel_hash_name,
                self.name,
                self.input_nodes,
                self.output_node.get_layout(),
                make_kernel_render,
                bmreq,
                self,
                kwargs,
            )

    def are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        raise NotImplementedError()

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
                using bfloat16 = nv_bfloat16;
            """
        )
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError


class CUTLASSTemplate(CUDATemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                #include "cute/tensor.hpp"
                #include "cutlass/cutlass.h"
                #include "cutlass/numeric_types.h"
                #include "cutlass/tensor_ref.h"
                #include "cutlass/util/host_tensor.h"
                #include "cutlass/util/reference/host/tensor_fill.h"
                #include "cutlass/util/reference/device/tensor_fill.h"
                #include "cutlass/util/device_memory.h"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                using namespace cute;
                #define CUTLASS_CHECK(status)                                                      \\
                {                                                                                  \\
                  cutlass::Status error = status;                                                  \\
                  if (error != cutlass::Status::kSuccess) {                                        \\
                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\
                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\
                    throw std::runtime_error(msg);                                                 \\
                  }                                                                                \\
                }

                // Used as pass-through functor in EVT just for type casting / rounding
                template <typename T>
                struct identity_op {
                  CUTLASS_HOST_DEVICE
                  T operator()(T val) const { return val; }
                };

            """
        )
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        res = ""
        if int_str in {"1", "1L"}:
            res = "cute::Int<1>{}"
        else:
            res = int_str

        return f"{res} /* {var_name} */"

    _DTYPE_TO_CUTLASS = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "cutlass::half_t",
        torch.int32: "int",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
        torch.bfloat16: "cutlass::bfloat16_t",
    }

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})"
