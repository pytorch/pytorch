import torch
from torch import Tensor
from typing import Callable, List

import re

__all__ : List[str] = []

class _CodeParser:
    def __init__(self, code_string: str):
        optional_ws = r"\s*"
        required_ws = r"\s+"
        template_params = r"(?P<template_params>\<.+\>)"
        return_type = r"(?P<return_type>\w+)"
        function_name = r"(?P<function_name>\w+)"
        function_params = r"(?P<function_params>\(.+\))"
        function_body = r"(?P<function_body>\{.+\})"

        pattern = \
            optional_ws \
            + "template" \
            + optional_ws + template_params \
            + optional_ws + return_type \
            + required_ws + function_name \
            + optional_ws + function_params \
            + optional_ws + function_body \
            + optional_ws

        result = re.match(pattern, code_string, re.DOTALL)  # DOTALL for matching multiline

        if result is None:
            raise Exception(f"Couldn't parse code, please check correctness:\n {code_string}")

        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


def _create_jit_fn(code_string: str, **kwargs) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op.

    The code string has to be a valid CUDA function that describes the computation for a single element. The code
    string has to follow the c++ template pattern, as shown in the example below. This function will be inlined
    into elementwise kernel template, and compiled on the fly. Compiled kernel will be cached in memory, as well as
    local temp dir.

    Jiterator-generated kernels accepts noncontiguous tensors, and supports boardcasting and type promotion.

    Args:
        code_string (string): CUDA code string to be compiled by jiterator.
        kwargs (Dict, optional): Keyword arguments for generated function

    Example:
        >>> code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return  -x + alpha * y; }"
        >>> jitted_fn = create_jit_fn(code_string, alpha=1.0)
        >>> a = torch.rand(3, device='cuda')
        >>> b = torch.rand(3, device='cuda')
        >>> # invoke jitted function like a regular python function
        >>> result = jitted_fn(a, b, alpha=3.14)


    Jiterator can be used together with python registration to override an operator's cuda kernel

    Following example is overriding gelu's cuda kernel with relu:
        >>> code_string = "template <typename T> T my_gelu(T a) { return a > 0 ? a : 0; }"
        >>> my_gelu = create_jit_fn(code_string)
        >>> my_lib = torch.library.Library("aten", "IMPL")
        >>> my_lib.impl('aten::gelu', my_gelu, "CUDA")
        >>> # torch.nn.GELU and torch.nn.function.gelu are now overridden
        >>> a = torch.rand(3, device='cuda')
        >>> torch.allclose(torch.nn.functional.gelu(a), torch.nn.functional.relu(a))


    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        Jiterator only supports up to 8 tensor inputs

    .. warning::
        All input tensors must live in CUDA device

    """
    class JittedFunction:
        def __init__(self, code_string: str, **kwargs):
            self.code_string = code_string

            parsed_code = _CodeParser(code_string)
            self.kernel_name = parsed_code.function_name

            self.kwargs_dict = kwargs
            self.is_cuda_available = torch.cuda.is_available()

        def __call__(self, *tensors: Tensor, **kwargs):
            # Jiterator follow torch.cuda's lazy initialization behavior
            # Defer checking cuda's availability at the function invocation time
            assert self.is_cuda_available, "Jiterator is only supported on CUDA GPUs, no CUDA GPUs are available."

            assert len(tensors) <= 8, "jiterator only supports up to 8 tensor inputs."

            expanded_kwargs = self.kwargs_dict.copy()
            for key, value in kwargs.items():
                if key in self.kwargs_dict:
                    expanded_kwargs[key] = value
                else:
                    raise KeyError(f"{key} is not declared in function definition")

            return torch._C._cuda_jiterator_compile_and_launch_kernel(
                self.code_string,
                self.kernel_name,
                tensors,
                expanded_kwargs)

    return JittedFunction(code_string, **kwargs)
