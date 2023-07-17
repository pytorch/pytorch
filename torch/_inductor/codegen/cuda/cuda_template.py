import functools
import itertools
from copy import copy
from typing import List
from unittest.mock import patch

import sympy
import torch

# import cutlass libs
import scripts as cutlass_lib

from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import jinja2_env

from . import cutlass_utils
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel


class CUDATemplate:
    index_counter = itertools.count()

    def __init__(
        self, name: str, input_nodes: List[IRNode], layout: Layout,
        input_reorder: List[int]=None,
    ):
        super().__init__()
        self.name = name
        self.input_nodes = input_nodes
        self.output_node = Buffer("buf_out", layout)
        self.input_reorder = input_reorder

    @staticmethod
    def _template_from_string(source):
        env = jinja2_env()
        if env is not None:
            return env.from_string(source)
        return None

    def maybe_append_choice(self, choices, **kwargs):
        choices.append(self.generate(**kwargs))

    @staticmethod
    def _fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype

        def get_dtype(name):
            if name == fake_out.get_name():
                return fake_out.get_dtype()
            return _get_dtype_real(name)

        return get_dtype

    def generate(self, **kwargs) -> CUDATemplateCaller:
        kernel_name = f"cuda_{self.name}"
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CUDATemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            # need to do call render twice to get all the needed args right
            # self.render(kernel=kernel, **kwargs)
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _ = kernel.args.python_argdefs()
            print("Generated Code:\n", code)
            print(f"args: {kernel.args.cpp_argdefs()}, {kernel.args.python_argdefs()}")

        input_reorder = self.input_reorder if self.input_reorder is not None else list(range(len(self.input_nodes)))
        expected_args = list(unique(self.input_nodes[idx].get_name() for idx in input_reorder))
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

        def make_kernel_render(output_node):
            kernel = CUDATemplateKernel(
                kernel_name="KERNEL_NAME",
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                output_node=output_node,
                **kwargs,
            )
            return kernel, render

        return CUDATemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
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

                using bfloat16 = nv_bfloat16;
            """
        )
        return res


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


    def render(self, **kwargs) -> str:
        raise NotImplementedError


class CutlassTemplate(CUDATemplate):
    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                #include "cutlass/cutlass.h"
                #include "cutlass/numeric_types.h"
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
                #define CUTLASS_CHECK(status)                                                      \\
                {                                                                                  \\
                  cutlass::Status error = status;                                                  \\
                  if (error != cutlass::Status::kSuccess) {                                        \\
                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\
                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\
                    throw std::runtime_error(msg);                                                 \\
                  }                                                                                \\
                }
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

