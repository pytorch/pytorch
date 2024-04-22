import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch

import sympy

import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout

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
    ) -> CUDATemplateCaller:
        """
        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller
        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        """
        kernel_name = f"cuda_{self.name}"
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CUDATemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            code = self.render(kernel=kernel, **kwargs)
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

        # create the BenchmarkRequest
        bmreq = CUDABenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

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

        return CUDATemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
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
        if torch.version.cuda:
            res.splice("using bfloat16 = nv_bfloat16;")
        elif torch.version.hip:
            res.splice("// using bfloat16 = hip_bfloat16;")
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


class CKTemplate(CUDATemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",
        torch.float8_e5m2fnuz: "BF8",
    }


    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK headers

                #ifdef DEBUG_LOG
                #define DEBUG_LOG_TMP DEBUG_LOG                
                #undef DEBUG_LOG
                #else
                #define DEBUG_LOG_TMP 0
                #endif
                #include "ck/ck.hpp"
                #undef DEBUG_LOG
                #define DEBUG_LOG DEBUG_LOG_TMP

                #include "ck/utility/data_type.hpp"
                #include "ck/library/utility/check_err.hpp"
                #include "ck/library/utility/device_memory.hpp"
                #include "ck/library/utility/fill.hpp"
                #include "ck/library/utility/host_tensor.hpp"
                #include "ck/library/utility/host_tensor_generator.hpp"
                #include "ck/library/utility/literals.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK globals

                template <ck::index_t... Is>
                using S = ck::Sequence<Is...>;

                using PassThrough = ck::tensor_operation::element_wise::PassThrough;

                // see "composable_kernel/include/ck/utility/data_type.hpp"
                using F8  = ck::f8_t;
                using BF8 = ck::bf8_t;
                using F16 = ck::half_t;
                using F32 = float;
                // using F64 = double;
                // using BF16 = ck::bhalf_t;
                // using I32 = int32_t;
                // using I8 = int8_t;
                // using I4 = ck::int4_t;

                #if DEBUG_LOG
                static constexpr auto kDEBUG_LOG = 1;
                #else
                static constexpr auto kDEBUG_LOG = 0;
                #endif
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"
