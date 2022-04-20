import functools
from typing import Iterator
from torch._C import (
    _get_torch_function_mode, _set_torch_function_mode, _get_python_mode, _set_python_mode)

def _wrap_init(f, class_name, mode_type):
    undef = object()

    @functools.wraps(f)
    def wrapped(self, *args, inner=undef, **kwargs):
        if inner is undef:
            raise TypeError(
                f"missing inner keyword argument; instead of constructing a {class_name} directly, "
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
class _ModeInfo:
    def __init__(self, mode_type: str, mode_class: type, base_mode_class: type, mode_class_name: str):
        if mode_type not in ['torch_function', 'python']:
            raise RuntimeError(f"only support torch_function or python modes, got mode_type of {mode_type}")
        self.is_torch_function_mode = mode_type == 'torch_function'
        self.mode_type = mode_type
        self.mode_class = mode_class
        self.base_mode_class = base_mode_class
        self.mode_class_name = mode_class_name

def _enable_mode(mode, mode_info: _ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    if not (
        mode is None or
        isinstance(mode, mode_info.mode_class) or
        (isinstance(mode, type) and not issubclass(mode, mode_info.mode_class))
    ):
        raise ValueError(f'The argument passed to enable_{mode_info.mode_type}_mode '
                         'must be None or the type of a Tensor subclass')
    old = _get_torch_function_mode() if mode_info.is_torch_function_mode else _get_python_mode()
    if old is mode:
        yield
        return
    if old is not None and not ignore_preexisting and old is not replace:
        if isinstance(mode, mode_info.mode_class):
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

def _push_mode(ctor, mode_info: _ModeInfo) -> Iterator[type]:
    # Helper function for pushing a mode onto the stack
    if isinstance(ctor, mode_info.mode_class):
        raise ValueError(
            f'Expected a {mode_info.mode_class_name} constructor function, but got an '
            f'instance of {mode_info.mode_class_name} {ctor}.  Consider using '
            f'enable_{mode_info.mode_type}_mode instead.'
        )
    old = _get_torch_function_mode() if mode_info.is_torch_function_mode else _get_python_mode()
    if old is None:
        inner = mode_info.base_mode_class(inner=None)
    else:
        inner = old

    mode = ctor(inner=inner)
    if not isinstance(mode, mode_info.mode_class):
        raise ValueError(
            f'The callable passed to push_{mode_info.mode_type}_mode must return a {mode_info.mode_class_name}'
        )
    _set_torch_function_mode(mode) if mode_info.is_torch_function_mode else _set_python_mode(mode)
    try:
        yield mode
    finally:
        _set_torch_function_mode(old) if mode_info.is_torch_function_mode else _set_python_mode(old)
