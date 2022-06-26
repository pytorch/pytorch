import functools
import torch
from typing import Iterator
from dataclasses import dataclass
from contextlib import contextmanager

# This file has all the logic to dedupe logic between torch dispatch and
# torch function modes
#
# Specifically, it has the helper functions for enable_ and push_X_mode and the
# ModeInfo class, which is extended by each where they are different

# used by both TorchFunctionMode and TorchDispatchMode, this will wrap the init
# function to require an "inner" kwarg
def _wrap_init(f):
    undef = object()

    @functools.wraps(f)
    def wrapped(self, *args, inner=undef, **kwargs):
        if inner is not undef:
            self.inner = inner
        return f(self, *args, **kwargs)
    return wrapped


# in order to dedupe the logic between python mode and torch_function mode, this
# is a container to hold all the differences between the modes. Then functions like
# _enable_mode are able to use this container to call functions or get correctly
# formatted names
@dataclass
class _ModeInfo:
    mode_name: str
    mode_class: type  # the class related to the mode that's allowed to be passed in
    base_mode_class: type  # the base class of mode_class that dispatches to the original function

    def mode_class_name(self):
        return self.mode_class.__name__

    def get_mode(self):
        """gets the current mode for this type of mode"""
        raise NotImplementedError()

    def set_mode(self, mode):
        """
        set mode to for this type of mode. Note that no checks are done on this, it's the unsafe
        version where checks are assumed to have been already done by the helper function
        """
        raise NotImplementedError()


# shared version of enable_torch_function/enable_torch_dispatch_mode in order to deduplicate the code.
# The differences between the modes are captured by `mode_info` and then queried when they're
# needed during the function's invocation
def _enable_mode(mode, mode_info: _ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    if not (
        mode is None or
        isinstance(mode, mode_info.mode_class) or
        (isinstance(mode, type) and not issubclass(mode, mode_info.mode_class))
    ):
        raise ValueError(f'expected to get {mode_info.mode_class_name()}, Tensor-like class, '
                         f'or None as an argument got {type(mode)} instead')
    old = mode_info.get_mode()
    if old is mode:
        yield
        return
    if old is not None and not ignore_preexisting and old is not replace:
        if isinstance(mode, mode_info.mode_class):
            help_text = f'Use push_{mode_info.mode_name}_mode instead.'
        else:
            help_text = (
                'If you intended to completely override the preexisting mode, '
                'pass ignore_preexisting=True.  This can result in unexpected '
                'behavior; please consider rewriting your mode to be a subclass '
                f'of {mode_info.mode_class_name()} to make it compositional!'
            )
        raise ValueError(
            f'Attempted to enable_{mode_info.mode_name}_mode, but there is already an '
            f'active mode {old}.  {help_text}'
        )
    # NB: we don't require TorchFunctionMode/PythonMode since this is intended to also
    # let you directly pass a Tensor subclass type to "mode-ify" it.
    required_fn = "__" + mode_info.mode_name + "__"
    if not hasattr(mode, required_fn):
        raise ValueError(
            f'The argument passed to enable_{mode_info.mode_name}_mode must implement {required_fn}'
        )
    mode_info.set_mode(mode)
    try:
        yield
    finally:
        mode_info.set_mode(old)


def _restore_mode(mode, mode_info: _ModeInfo):
    if not hasattr(mode, "ancestors"):
        raise RuntimeError(f"{mode} does not have any ancestors. Use the standard version instead of restore")
    old = mode_info.get_mode()
    if old is not None and old not in mode.ancestors:
        raise RuntimeError(f"{mode} is not valid in the current state because the current mode is not its ancestor")
    mode_info.set_mode(mode)
    try:
        yield mode
    finally:
        mode_info.set_mode(old)


# shared version of push_torch_function/push_torch_dispatch_mode in order to deduplicate the code.
# The differences between the modes are captured by `mode_info` and then queried when they're
# needed during the function's invocation
def _push_mode(ctor, mode_info: _ModeInfo) -> Iterator[object]:
    # Helper function for pushing a mode onto the stack
    if isinstance(ctor, mode_info.mode_class):
        raise ValueError(
            f'Expected a {mode_info.mode_class_name()} constructor function, but got an '
            f'instance of {mode_info.mode_class_name()} {ctor}.  Consider using '
            f'enable_{mode_info.mode_name}_mode instead.'
        )
    old = mode_info.get_mode()
    if old is None:
        inner = mode_info.base_mode_class(inner=None)
    else:
        inner = old

    mode = ctor(inner=inner)
    if not isinstance(mode, mode_info.mode_class):
        raise ValueError(
            f'The callable passed to push_{mode_info.mode_name}_mode'
            f'must return a {mode_info.mode_class_name()}'
        )
    mode_info.set_mode(mode)
    try:
        yield mode
    finally:
        mode_info.set_mode(old)


# To help with non-lexical scoping, it will error if all the modes are from different scopes or haven't been used
def find_outermost_mode(modes):
    outermost = None
    for mode in modes:
        if mode is not None:
            if not hasattr(mode, "ancestors"):
                raise RuntimeError(f"{mode}, doesn't have ancestors set so the ordering with other modes is unclear")
            if outermost is None:
                outermost = mode
            elif mode not in outermost.ancestors and outermost not in mode.ancestors:
                raise RuntimeError(f"modes {mode} and {outermost} are not compatible because they "
                                   "don't come from the same scope")
            elif outermost in mode.ancestors:
                outermost = mode
    return outermost


# returns if all are the same mode
def all_same_mode(modes):
    return all(tuple(mode == modes[0] for mode in modes))

# returns if all modes are from the current scope, ``cur_mode``
def all_same_mode_scope(modes, cur_mode):
    if not hasattr(cur_mode, "ancestors"):
        return False
    return all(tuple(mode == cur_mode or mode in cur_mode.ancestors for mode in modes))

@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard
