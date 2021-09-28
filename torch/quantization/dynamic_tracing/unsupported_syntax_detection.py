from typing import Tuple, Any

import torch
from .quantization_state import (
    AutoQuantizationState,
)

class QuantizationNotImplementedError(NotImplementedError):
    pass

class ExceptionRaiser(torch.nn.Module):
    def forward(self, x):
        raise QuantizationNotImplementedError

def mark_unsupported_syntax(
    m: torch.nn.Module,
    example_inputs: Tuple[Any],
) -> None:
    """
    Traverses through each descendant of `m`.  For any descendant
    that calls forward on each item in `self._modules`, sets
    the `_calls_forward_on_each_child_module` flag.  For now,
    auto quantization is not supported for such modules.  Further down
    the quantization stack, we disable quantization for any module which
    has this flag set.

    TODO no_grad, preserve training, etc
    """

    # annotate each child with ExceptionRaiser objects
    def _annotate_with_exception_raiser(module: torch.nn.Module) -> None:
        for name, child in module.named_children():
            _annotate_with_exception_raiser(child)
        assert not hasattr(module, '_exception_raiser')
        module._exception_raiser = ExceptionRaiser()

    def _remove_exception_raiser(module: torch.nn.Module) -> None:
        for name, child in module.named_children():
            _remove_exception_raiser(child)
        if hasattr(module, '_exception_raiser'):
            del module._exception_raiser

    _annotate_with_exception_raiser(m)

    class ExceptionRaiserInterceptionModule(type(m)):

        def __call__(self, *args, **kwargs):
            orig_module_call = torch.nn.Module.__call__
            orig_nn_sequential_forward = torch.nn.Sequential.forward

            def record_module(self, *args, **kwargs):
                try:
                    output = orig_module_call(self, *args, **kwargs)

                # Catching QuantizationNotImplementedError guards against
                # calling the forward.
                # Catching IndexError guards against accessing the values
                # with `self.values()` in an unsupported way.
                # Catching RuntimeError guards against other issues encountered
                # in vovnet.
                # TODO: clean this up.
                except (QuantizationNotImplementedError, IndexError, RuntimeError):
                    self._calls_forward_on_each_child_module = True
                    _remove_exception_raiser(self)
                    output = orig_module_call(self, *args, **kwargs)
                return output

            torch.nn.Module.__call__ = record_module
            torch.nn.Sequential.forward = _nn_sequential_patched_forward
            try:
                output = super().__call__(*args, **kwargs)
                return output
            finally:
                torch.nn.Module.__call__ = orig_module_call
                torch.nn.Sequential.forward = orig_nn_sequential_forward

    old_class = m.__class__
    m.__class__ = ExceptionRaiserInterceptionModule

    with torch.no_grad():
        old_training = m.training
        m.eval()
        m(*example_inputs)
        if old_training:
            m.train()

    m.__class__ = old_class
    _remove_exception_raiser(m)

def _nn_sequential_patched_forward(cls, input):
    for module in cls:
        if not isinstance(module, ExceptionRaiser):
            input = module(input)
    return input
