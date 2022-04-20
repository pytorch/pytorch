import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union


def create_jit_fn(op_string: str, optional_name: str, optional_fusion_class="elementwise", **kwargs) -> Callable:

    class JittedFunction:
        def __init__(self, op_string: str, optional_name: str, optional_fusion_class, **kwargs):
            self.op_string = op_string
            self.optional_name = optional_name
            self.optional_fusion_class = optional_fusion_class
            self.kwargs_dict = kwargs
            self.jitted_fn = None

        def compile(self, *tensors: Tensor, **kwargs):
            print("calling into C++...")

            self.jitted_fn = torch._C._cuda_compile_kernel(
                self.op_string,
                self.optional_name,
                self.optional_fusion_class,
                tensors)
            # TODO: ignoring kwargs for now
            return self.jitted_fn

        def __call__(self, *tensors: Tensor, **kwargs):

            expanded_kwargs = self.kwargs_dict.copy()
            for key, value in kwargs.items():
                if key in self.kwargs_dict:
                    expanded_kwargs[key] = value
                else:
                    raise KeyError(f"{key} is not declared in function definition")

            tensor_types = [t.dtype for t in tensors]

            if self.jitted_fn is None:
                self.jitted_fn = self.compile(*tensors, **kwargs)

            return None
            # return self.jitted_fn(*tensors, **expanded_kwargs)

    return JittedFunction(op_string, optional_name, optional_fusion_class, **kwargs)


