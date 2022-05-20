import torch

# Ideally, tensor subclasses would would inherit directly from Tensor.
# This is just our staging ground for applying behavior that hasn't yet made it
# into the core Tensor class but that we would like to apply by default.
class BaseTensor(torch.Tensor):
    # See https://github.com/pytorch/pytorch/pull/73727 ; this is necessary
    # to ensure that super().__new__ can cooperate with each other
    @staticmethod
    def __new__(cls, elem, *, requires_grad=None, **kwargs):
        if requires_grad is None:
            return super().__new__(cls, elem, **kwargs)  # type: ignore
        else:
            return cls._make_subclass(cls, elem, requires_grad, **kwargs)

    # To ensure constructors can cooperate with one another, must accept and
    # ignore element tensor (TODO: is this right???)
    def __init__(self, elem):
        super().__init__()

    # If __torch_dispatch__ is defined (which it will be for all our examples)
    # the default torch function implementation (which preserves subclasses)
    # typically must be disabled
    __torch_function__ = torch._C._disabled_torch_function_impl

__all__ = ["BaseTensor"]
