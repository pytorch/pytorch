from torch._ops import HigherOrderOperator


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_mutation")

    def __call__(self, *, kernel, grid, kwargs):
        kernel[grid](**kwargs)


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()
