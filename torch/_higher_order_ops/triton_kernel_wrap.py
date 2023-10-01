from torch import Tensor
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_mutation")

    def __call__(self, *, kernel, grid, kwargs):
        kernel[grid](**kwargs)


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_functional")

    def __call__(self, *, kernel, grid, kwargs):
        kwargs = {
            key: (clone_preserve_strides(val) if isinstance(val, Tensor) else val)
            for key, val in kwargs.items()
        }
        triton_kernel_wrapper_mutation(kernel=kernel, grid=grid, kwargs=kwargs)
        return kwargs


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()
