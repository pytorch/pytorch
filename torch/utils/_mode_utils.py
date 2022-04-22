import torch
from typing import Iterator


# a class for the helper function to package all the info about the Mode class
# so the helper function can access it without needing a circular import
class _ModeInfo:
    def __init__(self, mode_type: str, mode_class: type, base_mode_class: type, mode_class_name: str, required_fn: str):
        self.mode_type = mode_type
        self.mode_class = mode_class
        self.base_mode_class = base_mode_class
        self.mode_class_name = mode_class_name
        self.required_fn = required_fn

    def is_allowed_type(self, mode) -> bool:
        raise NotImplementedError()

    def allowed_types_for_error_message(self) -> str:
        raise NotImplementedError()

    def get_mode(self):
        raise NotImplementedError()

    def help_text(self, mode) -> str:
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()


def _enable_mode(mode, mode_info: _ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    if not mode_info.is_allowed_type(mode):
        raise ValueError(f'expected to get {mode_info.allowed_types_for_error_message()} as argument, got {type(mode)}',
                         "instead")
    old = mode_info.get_mode()
    if old is mode:
        yield
        return
    if old is not None and not ignore_preexisting and old is not replace:
        raise ValueError(
            f'Attempted to enable_{mode_info.mode_type}_mode, but there is already an '
            f'active mode {old}.  {mode_info.help_text(mode)}'
        )
    # NB: we don't require TorchFunctionMode/PythonMode since this is intended to also
    # let you directly pass a Tensor subclass type to "mode-ify" it.
    if not hasattr(mode, mode_info.required_fn):
        raise ValueError(
            f'The argument passed to enable_{mode_info.mode_type}_mode must implement {mode_info.required_fn}'
        )
    mode_info.set_mode(mode)
    try:
        yield
    finally:
        mode_info.set_mode(old)
