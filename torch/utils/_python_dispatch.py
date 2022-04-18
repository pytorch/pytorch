import contextlib
from typing import Iterator

from torch.utils._mode_utils import _enable_mode, ModeInfo

# NB: Calling an operator inside __torch_dispatch__ does go through
# __torch_dispatch__ again. Please use _DisableTorchDispatch inside
# __torch_dispatch__ to prevent infinite recursion.
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
    type's __torch_dispatch__ function, including operations that accepts no tensors
    but returns a tensor.

    This function is non-compositional; if there is already an existing mode,
    it will raise an error; prefer using :func:`push_python_mode` if your
    ``__torch_dispatch__`` implementation can defer to an inner mode.

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
    # hacky because torch_function mode and python_mode don't yet have parity
    mode_info = ModeInfo(mode_type="python", mode_class=type(None), base_mode_class=type(None), mode_class_name="")
    return _enable_mode(mode, mode_info=mode_info, replace=replace, ignore_preexisting=ignore_preexisting)
