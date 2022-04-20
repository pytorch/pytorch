import contextlib
from typing import Iterator
import functools

from torch.utils._mode_utils import _enable_mode, _push_mode, _ModeInfo, _wrap_init

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
    mode_info = _ModeInfo(mode_type="python", mode_class=PythonMode,
                          base_mode_class=BasePythonMode, mode_class_name="PythonMode")
    return _enable_mode(mode, mode_info=mode_info, replace=replace, ignore_preexisting=ignore_preexisting)

def _wrap_torch_dispatch(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        with enable_python_mode(self.inner):
            return f(self, *args, **kwargs)
    return wrapped


# Implementation note, since this is based on TorchFunctionMode, this had the
# same dilemma: I had a choice about how much of mode stacks
# to implement in Python versus in C++.  At time of writing, I did not care
# too much about implementation efficiency; however, I do care about making it
# hard for users to implement modes in the wrong way.  In the end, it turned
# out to be possible to implement mode stacks entirely from userland, with the
# C++ API providing only _get_python_mode() and
# _set_python_mode(), so I opted to provide some unsafe C++ bindings and
# have the bulk of the logic for managing the stack in Python, which helped
# simplify the C++ API surface.  It would also have been valid to build in the
# notion of mode stack directly into C++ but in this design it's substantially
# more difficult to interact with PythonModeMeta.

class PythonModeMeta(type):
    """
    Metaclass for :class:`PythonMode`; it does two things:

        * Adds an implicit ``inner`` kwarg to ``__init__``, to
          allow the modes to be chained together to form a stack.

        * Reenables the inner mode, so that by default PyTorch API calls
          will compositionally proceed to the next mode on the stack.

    The default behavior for the second bullet is important, as it is easy to
    accidentally write ``_wrap_torch_dispatch`` implementations that are not
    compositional, and the wrapping here makes the obvious code do the
    right thing (aka, this is why there is a metaclass).
    """
    def __new__(metacls, name, bases, dct):
        if '__init__' in dct:
            dct['__init__'] = _wrap_init(dct['__init__'], "PythonMode", "python")
        if '__torch_dispatch__' in dct:
            dct['__torch_dispatch__'] = _wrap_torch_dispatch(dct['__torch_dispatch__'])
        return super().__new__(metacls, name, bases, dct)


class PythonMode(metaclass=PythonModeMeta):
    """
    A ``PythonMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`PythonMode` are compositional:
    modes can be pushed onto a stack with :func:`push_python_mode`.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self):
        pass

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()


class BasePythonMode(PythonMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

@contextlib.contextmanager
def push_python_mode(ctor) -> Iterator[PythonMode]:
    mode_info = _ModeInfo(mode_type="python", mode_class=PythonMode,
                          base_mode_class=BasePythonMode, mode_class_name="PythonMode")
    return _push_mode(ctor, mode_info=mode_info)
