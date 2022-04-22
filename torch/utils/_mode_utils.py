import functools
from typing import Iterator


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


def _push_mode(ctor, mode_info: _ModeInfo) -> Iterator[type]:
    # Helper function for pushing a mode onto the stack
    if isinstance(ctor, mode_info.mode_class):
        raise ValueError(
            f'Expected a {mode_info.mode_class_name} constructor function, but got an '
            f'instance of {mode_info.mode_class_name} {ctor}.  Consider using '
            f'enable_{mode_info.mode_type}_mode instead.'
        )
    old = mode_info.get_mode()
    if old is None:
        inner = mode_info.base_mode_class(inner=None)
    else:
        inner = old

    mode = ctor(inner=inner)
    if not isinstance(mode, mode_info.mode_class):
        raise ValueError(
            f'The callable passed to push_{mode_info.mode_type}_mode must return a {mode_info.mode_class_name}'
        )
    mode_info.set_mode(mode)
    try:
        yield mode
    finally:
        mode_info.set_mode(old)
