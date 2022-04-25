from typing import Iterator
from dataclasses import dataclass


# in order to dedupe the logic between python mode and torch_function mode, this
# is a container to hold all the differences between the modes. Then functions like
# _enable_mode are able to use this container to call functions or get correctly
# formatted names
@dataclass
class _ModeInfo:
    mode_name: str
    mode_class: type  # the class related to the mode that's allowed to be passed in
    base_mode_class: type  # the base class of mode_class that dispatches to the original function
    required_fn: str  # string version of the function required, either torch_function or torch_dispatch

    def is_allowed_type(self, mode) -> bool:
        """determines if attr:`mode` is an allowed type for this mode"""
        raise NotImplementedError()

    def allowed_types_for_error_message(self) -> str:
        """returns a nicely formatted string version of the allowed types for the error message"""
        raise NotImplementedError()

    def help_text(self, mode) -> str:
        """
        returns help text for when the user tries to enable a mode when another mode
        of the same type is already active
        """
        raise NotImplementedError()

    def get_mode(self):
        """gets the current mode for this type of mode"""
        raise NotImplementedError()

    def set_mode(self, mode):
        """
        set mode to for this type of mode. Note that no checks are done on this, it's the unsafe
        version where checks are assumed to have been already done by the helper function
        """
        raise NotImplementedError()


# shared version of enable_torch_function/enable_python_mode in order to deduplicate the code.
# The differences between the modes are captured by `mode_info` and then queried when they're
# needed during the function's invocation
def _enable_mode(mode, mode_info: _ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    if not mode_info.is_allowed_type(mode):
        raise ValueError(f'expected to get {mode_info.allowed_types_for_error_message()} '
                         f'as an argument got {type(mode)} instead')
    old = mode_info.get_mode()
    if old is mode:
        yield
        return
    if old is not None and not ignore_preexisting and old is not replace:
        raise ValueError(
            f'Attempted to enable_{mode_info.mode_name}_mode, but there is already an '
            f'active mode {old}.  {mode_info.help_text(mode)}'
        )
    # NB: we don't require TorchFunctionMode/PythonMode since this is intended to also
    # let you directly pass a Tensor subclass type to "mode-ify" it.
    if not hasattr(mode, mode_info.required_fn):
        raise ValueError(
            f'The argument passed to enable_{mode_info.mode_name}_mode must implement {mode_info.required_fn}'
        )
    mode_info.set_mode(mode)
    try:
        yield
    finally:
        mode_info.set_mode(old)
