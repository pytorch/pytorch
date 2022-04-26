import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Union


def create_jit_fn(op_string: str, optional_name: str, **kwargs) -> Callable:
    class JittedFunction:
        def __init__(self, op_string: str, optional_name: str, **kwargs):
            self.op_string = op_string
            self.optional_name = optional_name
            self.kwargs_dict = kwargs

        def compile(self, *tensors: Tensor, **kwargs):
            output = torch._C._cuda_compile_kernel(
                self.op_string,
                self.optional_name,
                tensors,
                kwargs)
            return output

        def __call__(self, *tensors: Tensor, **kwargs):
            expanded_kwargs = self.kwargs_dict.copy()
            for key, value in kwargs.items():
                if key in self.kwargs_dict:
                    expanded_kwargs[key] = value
                else:
                    raise KeyError(f"{key} is not declared in function definition")

            return self.compile(*tensors, **expanded_kwargs)

    return JittedFunction(op_string, optional_name, **kwargs)


