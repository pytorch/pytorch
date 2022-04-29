import torch
from torch import Tensor
from typing import Callable

import re

__all__ = ["create_jit_fn"]

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

        result = re.match(pattern, code_string)

        if result is None:
            raise Exception("Couldn't parse code, please check correctness: ", code_string)

        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


def create_jit_fn(code_string: str, **kwargs) -> Callable:
    """
    Create a jiterator-generated cuda kernel.

    Args:
        code_string (string): CUDA code string to be compiled by jiterator
        kwargs (Dict, optional): Keyword arguments for generated function

    Examples:
        >>> code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return  -x + alpha * y; }"
        >>> jitted_fn = create_jit_fn(code_string, alpha=1.0)
        >>> a = torch.rand(3, device='cuda')
        >>> b = torch.rand(3, device='cuda')
        >>> # invoke jitted function like a regular python function
        >>> result = jitted_fn(a, b, alpha=3.14)

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        Jiterator only supports elementwise kernels with up to 8 tensor inputs
    """
    class JittedFunction:
        def __init__(self, code_string: str, **kwargs):
            self.code_string = code_string

            parsed_code = _CodeParser(code_string)
            self.kernel_name = parsed_code.function_name

            self.kwargs_dict = kwargs

        def __call__(self, *tensors: Tensor, **kwargs):
            assert len(tensors) <= 8, "jiterator only supports up to 8 tensor inputs."

            expanded_kwargs = self.kwargs_dict.copy()
            for key, value in kwargs.items():
                if key in self.kwargs_dict:
                    expanded_kwargs[key] = value
                else:
                    raise KeyError(f"{key} is not declared in function definition")

            return torch._C._cuda_compile_kernel(
                self.code_string,
                self.kernel_name,
                tensors,
                expanded_kwargs)

    return JittedFunction(code_string, **kwargs)
