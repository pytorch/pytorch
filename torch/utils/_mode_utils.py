import torch
from typing import Iterator, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager

T = TypeVar('T')

# This file has all the logic to dedupe logic between torch dispatch and
# torch function modes
#
# Specifically, it has the helper functions for enable_ and push_X_mode and the
# ModeInfo class, which is extended by each where they are different


# in order to dedupe the logic between TorchDispatchMode and TorchFunctionMode, this
# is a container to hold all the differences between the modes. Then functions like
# _enable_mode are able to use this container to call functions or get correctly
# formatted names
@dataclass
class _ModeInfo:
    mode_name: str
    mode_class: type  # the class related to the mode that's allowed to be passed in

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


# shared version of _enable_inner_torch_function_mode/_enable_inner_torch_dispatch_mode in order to deduplicate the
# code. The differences between the modes are captured by `mode_info` and then queried when they're
# needed during the function's invocation
# TODO: remove when we move to the new mode stack
def _enable_mode(mode: T, mode_info: _ModeInfo) -> Iterator[T]:
    if not (mode is None or isinstance(mode, mode_info.mode_class)):
        raise ValueError(f'expected to get {mode_info.mode_class_name()}, '
                         f'or None as an argument got {type(mode)} instead')
    old = mode_info.get_mode()
    if old is mode:  # this should be unnecessary but might be backwards breaking if user did something funky
        yield mode  # type: ignore[misc]
        return

    mode_info.set_mode(mode)
    try:
        yield mode  # type: ignore[misc]
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
