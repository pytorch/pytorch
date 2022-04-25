import torch
import contextlib
from typing import Iterator

from torch.utils._mode_utils import _enable_mode, _ModeInfo
from torch._C import _get_python_mode, _set_python_mode
from dataclasses import dataclass

@dataclass
class PythonModeInfo(_ModeInfo):
    def __init__(self):
        # hacky because torch_function mode and python_mode don't yet have parity
        super().__init__(mode_name="python", mode_class=type(None),
                         base_mode_class=type(None),
                         required_fn="__torch_dispatch__")

    def is_allowed_type(self, mode) -> bool:
        return mode is None or isinstance(mode, type) and issubclass(mode, (torch.Tensor,))

    def allowed_types_for_error_message(self) -> str:
        return "Tensor-like class or None"

    def get_mode(self):
        return _get_python_mode()

    def help_text(self, mode) -> str:
        return (
            'If you intended to completely override the preexisting mode, '
            'pass ignore_preexisting=True.  This can result in unexpected '
            'behavior; please consider rewriting your mode to be a subclass '
            f'of {self.mode_class.__name__} to make it compositional!'
        )

    def set_mode(self, mode):
        return _set_python_mode(mode)

#
# TODO: Limitations and things about enable_python_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)
@contextlib.contextmanager
def enable_python_mode(mode, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    """
    Context manager that causes all pytorch operators to dispatch to the passed-in
    type's __torch_dispatch__ function, including operations that accept no tensors
    but return a tensor.

    This function is non-compositional; if there is already an existing mode,
    it will raise an error

    This function is safe to use inside a ``__torch_dispatch__`` mode handler,
    as the mode is guaranteed to be disabled in this context.  You can use
    this context manager to reinstate the mode so that calls to overridable
    APIs recursively call back into your mode handler (this can easily cause
    infinite loops, so use with care!)

    enable_python_mode is affected by _DisableTorchDispatch.

    Args:
        mode (:class:`PythonMode`, Tensor-like class or None): the
            mode to set as current mode.  If you pass a Tensor-like class,
            it will be treated as a non-compositional mode with no state,
            which is convenient if you have an existing tensor subclass
            that you'd like to apply globally in a quick and dirty way.
            Passing None will disable the current mode.
        replace (:class:`PythonMode` or Tensor-like class): the
            mode to replace.  You can use this argument to change the mode in
            a situation where you know what the current mode is (and you are
            intentionally overwriting it.)  If you don't know what the current
            mode is, use ``ignore_preexisting`` instead.
        ignore_preexisting (bool): if True, ignore any preexisting mode
            and overwrite it with the passed mode.
    """

    return _enable_mode(mode, mode_info=PythonModeInfo(), replace=replace, ignore_preexisting=ignore_preexisting)
