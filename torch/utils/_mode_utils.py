import torch
import functools
from typing import Iterator, Union
from torch._C import (
    _get_torch_function_mode, _set_torch_function_mode, _get_python_mode, _set_python_mode)

def _wrap_init(f, metaclass, mode_type):
    undef = object()

    @functools.wraps(f)
    def wrapped(self, *args, inner=undef, **kwargs):
        if inner is undef:
            assert metaclass.__name__[-4:] == "Meta"
            non_meta_class = metaclass.__name__[:-4]
            raise TypeError(
                f"missing inner keyword argument; instead of constructing a {non_meta_class} directly, "
                f"pass the constructor to push_{mode_type}_mode"
            )
        self.inner = inner
        return f(self, *args, **kwargs)
    return wrapped

class ModeMeta(type):
    def __new__(metacls, name, bases, dct, mode_type=None):
        if mode_type not in ['torch_function', 'python']:
            raise RuntimeError(f"only support torch_function or python modes, got mode_type of {mode_type}")
        if '__init__' in dct:
            dct['__init__'] = _wrap_init(dct['__init__'], metacls, mode_type)
        return super().__new__(metacls, name, bases, dct)

# a class for the helper function to package all the info about the Mode class
# so the helper function can access it without needing a circular import
class ModeInfo:
    def __init__(self, mode_type: str, mode_class: type, base_mode_class: type, mode_class_name: str):
        if mode_type not in ['torch_function', 'python']:
            raise RuntimeError(f"only support torch_function or python modes, got mode_type of {mode_type}")
        self.is_torch_function_mode = mode_type == 'torch_function'
        self.mode_type = mode_type
        self.mode_class = mode_class
        self.base_mode_class = base_mode_class
        self.mode_class_name = mode_class_name

def _enable_mode(mode, mode_info: ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    if mode_info.is_torch_function_mode:
        allowed_types = (
            isinstance(mode, mode_info.mode_class) or
            (isinstance(mode, type) and not issubclass(mode, mode_info.mode_class)))
    else:
        allowed_types = isinstance(mode, type) and issubclass(mode, (torch.Tensor,))
    if not (mode is None or allowed_types):
        if mode_info.is_torch_function_mode:
            allowed_options = "TorchFunctionMode, Tensor-like class, or None"
        else:
            allowed_options = "Tensor-like class or None"
        raise ValueError(f'expected to get {allowed_options} as argument, got {type(mode)} instead')
    old = _get_torch_function_mode() if mode_info.is_torch_function_mode else _get_python_mode()
    if old is mode:
        yield
        return
    if old is not None and not ignore_preexisting and old is not replace:
        if mode_info.is_torch_function_mode and isinstance(mode, mode_info.mode_class):
            help_text = (
                f'Use push_{mode_info.mode_type}_mode instead.'
            )
        else:
            help_text = (
                'If you intended to completely override the preexisting mode, '
                'pass ignore_preexisting=True.  This can result in unexpected '
                'behavior; please consider rewriting your mode to be a subclass '
                f'of {mode_info.mode_class_name} to make it compositional!'
            )
        raise ValueError(
            f'Attempted to enable_{mode_info.mode_type}_mode, but there is already an '
            f'active mode {old}.  {help_text}'
        )
    # NB: we don't require TorchFunctionMode/PythonMode since this is intended to also
    # let you directly pass a Tensor subclass type to "mode-ify" it.
    required_fn = '__torch_function__' if mode_info.is_torch_function_mode else '__torch_dispatch__'
    if not hasattr(mode, required_fn):
        raise ValueError(
            f'The argument passed to enable_{mode_info.mode_type}_mode must implement {required_fn}'
        )
    _set_torch_function_mode(mode) if mode_info.is_torch_function_mode else _set_python_mode(mode)
    try:
        yield
    finally:
        _set_torch_function_mode(old) if mode_info.is_torch_function_mode else _set_python_mode(old)
