# mypy: allow-untyped-defs
import functools
import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import override
from unittest.mock import patch

import sympy

import torch
from torch._inductor import config
from torch._inductor.utils import clear_on_fresh_cache, Placeholder
from torch._logging import getArtifactLogger

from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel
from .cutlass_utils import DTYPE_TO_CUTLASS_TYPE


if TYPE_CHECKING:
    from ...scheduler import BaseSchedulerNode  # noqa: TC004
else:
    BaseSchedulerNode = Any

GemmOperation = Any

autotuning_log = getArtifactLogger(__name__, "autotuning")


@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str


@clear_on_fresh_cache
class CUDATemplate(KernelTemplate):
    index_counter = itertools.count()
    # dict of cache key to (code, size_args)
    code_cache: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    cache_clear = staticmethod(code_cache.clear)

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: Optional[list[int]] = None,
    ) -> None:
        """
        Baseclass for CUDA C++ Templates, derived from KernelTemplate.
        Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies
                the order of the input nodes.
        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    @classmethod
    @functools.lru_cache(None)
    # pyrefly: ignore [bad-override]
    def _template_from_string(cls, source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation) -> bool:
        return False

    def make_key(self, name: str, input_key: str, layout_repr: str) -> str:
        """
        Make a key for the code cache. The idea of the method is to cache
        everything that matters but doesn't include runtime param values, i.e.,
        self.get_runtime_arg_values().

        Args:
            kwargs: Additional keyword arguments. Including op (GemmOperation).
        """
        return hashlib.sha256(
            str(
                (
                    input_key,
                    self.input_reorder,
                    # output layout, same as self.output_node.get_layout()
                    layout_repr,
                    self.get_runtime_arg_info(),
                    name,
                )
            ).encode("utf-8")
        ).hexdigest()

    def generate_code_and_args(
        self, name: str, input_key: str, layout_repr: str, **kwargs
    ) -> tuple[str, tuple[int, ...]]:
        """
        Generate code and args with caching. We cache the code even if runtime
        args are different.
        """
        key: Optional[str] = None
        if config.cuda.enable_caching_codegen:
            key = self.make_key(name=name, input_key=input_key, layout_repr=layout_repr)

        if key is not None and key in self.code_cache:
            code, size_args, offset_args = self.code_cache[key]
            extra_args = tuple(
                list(size_args)
                + list(offset_args)
                + list(self.get_runtime_arg_values(**kwargs))
            )
            return code, extra_args

        kernel_name = str(Placeholder.KERNEL_NAME)
        kernel = CUDATemplateKernel(
            kernel_name=kernel_name,
            runtime_arg_info=self.get_runtime_arg_info(),
            runtime_arg_values=self.get_runtime_arg_values(**kwargs),
        )
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            autotuning_log.debug("Generated Code:\n%s", code)
            autotuning_log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE),
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
        V.graph.sizevars.size_hints(map(sympy.expand, call_args[len(expected_args) :]))
        size_args = V.graph.sizevars.size_hints(kernel.get_dynamic_shape_args())
        offset_args = V.graph.sizevars.size_hints(kernel.get_offset_args())

        if key is not None:
            self.code_cache[key] = code, size_args, offset_args

        # extra args has runtime params, which shouldn't be cached
        extra_args = tuple(
            list(size_args) + list(offset_args) + self.get_runtime_arg_values(**kwargs)
        )

        return code, extra_args

    def generate(  # type: ignore[override]
        self,
        name: str,
        description: str,
        input_key: str,
        layout_repr: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        **kwargs,
    ) -> CUDATemplateCaller:
        """
        Generates the CUDA template caller object for the given GEMM template and operation.
        This CUDATemplateCaller may be used to call and benchmark the generated CUDA kernel
        in a standalone manner to enable Autotuning.

        Args:
            description: op name followed by swizzle.
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        """
        code, extra_args = self.generate_code_and_args(
            name=name,
            input_key=input_key,
            layout_repr=layout_repr,
            **kwargs,
        )

        # not caching since kernel name is needed below
        kernel_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
        kernel_name = f"cutlass_{kernel_hash}"
        code = code.replace(self.name, kernel_name)

        # create the BenchmarkRequest
        bmreq = CUDABenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=input_tensor_meta,
            output_tensor_meta=output_tensor_meta,
            extra_args=extra_args,
            source_code=code,
        )

        # kwargs has "op" argument in case of CUTLASSGemmTemplate
        op = kwargs["op"]
        if not op:
            supports_epilogue_fusion = False
        else:
            # epilogue fusion is only supported for TMA kernels
            supports_epilogue_fusion = self.supports_epilogue_fusion(op)

        def make_kernel_render(
            template_node: CUDATemplateBuffer,
            epilogue_nodes: Optional[list[BaseSchedulerNode]] = None,
        ) -> tuple[CUDATemplateKernel, functools.partial[str]]:
            assert supports_epilogue_fusion or not epilogue_nodes, (
                "epilogue fusion is not supported for this kernel"
            )
            kernel = CUDATemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
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

        return CUDATemplateCaller(
            kernel_name,
            "cutlass_gemm",
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            supports_epilogue_fusion,
            self,
            kwargs,
            description,
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
        if int_str in ("1", "1L"):
            res = "cute::Int<1>{}"
        else:
            res = int_str

        return f"{res} /* {var_name} */"

    _DTYPE_TO_CUTLASS = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "cutlass::half_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
        torch.bfloat16: "cutlass::bfloat16_t",
        torch.float8_e4m3fn: "cutlass::float_e4m3_t",
        torch.float8_e5m2: "cutlass::float_e5m2_t",
    }

    _DTYPE_TO_CUTLASS_SPARSE_META = {
        torch.int32: "uint32_t",
        torch.int16: "uint16_t",
    }

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})"

    def cutlass_sparse_meta_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return (
                f"({self._DTYPE_TO_CUTLASS_SPARSE_META.get(node.get_dtype())}*)({ptr})"
            )

    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("swizzle", "const uint8_t")]

    @override
    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
        return [kwargs[arg.name] for arg in self.get_runtime_arg_info()]
