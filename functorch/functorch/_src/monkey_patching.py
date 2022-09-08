import torch
import functorch._C as _C
import functools

# Monkeypatch tensor printing in pytorch
_old_str = torch._tensor_str._str


def prep_value(text, indent=4):
    first_line_txt = ''
    lines = text.split('\n')
    lines[0] = lines[0]
    lines[0] = ' ' * indent + first_line_txt + lines[0]
    for i in range(1, len(lines)):
        lines[i] = ' ' * (indent + len(first_line_txt)) + lines[i]
    return '\n'.join(lines)


@functools.wraps(_old_str)
def _functorch_str(tensor, *, tensor_contents=None):
    level = _C.maybe_get_level(tensor)
    if level == -1:
        return _old_str(tensor)

    if _C.is_functionaltensor(tensor):
        # Since we're unwrapping the FunctionalTensorWrapper, we need to make sure
        # that it's up to date first
        torch._sync(tensor)

    value = _C.get_unwrapped(tensor)
    dl_enabled = _C.tls_set_is_included()
    try:
        # Disable temporarily kDynamicLayerFrontModeKey/kDynamicLayerBackModeKey as included dispatch keys
        if (dl_enabled):
            _C._set_dynamic_layer_keys_included(False)
        value_repr = repr(value)
    finally:
        # Reenable kDynamicLayerFrontModeKey/kDynamicLayerBackModeKey as included dispatch keys
        if (dl_enabled):
            _C._set_dynamic_layer_keys_included(True)

    if _C.is_batchedtensor(tensor):
        bdim = _C.maybe_get_bdim(tensor)
        assert bdim != -1
        return (
            f'BatchedTensor(lvl={level}, bdim={bdim}, value=\n'
            f'{prep_value(value_repr)}\n'
            f')'
        )
    if _C.is_gradtrackingtensor(tensor):
        return (
            f'GradTrackingTensor(lvl={level}, value=\n'
            f'{prep_value(value_repr)}\n'
            f')'
        )
    if _C.is_functionaltensor(tensor):
        return f'FunctionalTensor(lvl={level}, value=\\\n{value_repr})'

    raise ValueError("We don't know how to print this, please file us an issue")


torch._tensor_str._str = _functorch_str


# Monkeypatch .backward() to error out if any transforms are active.
# TODO: remove the monkeypatching and add an extension point into PyTorch core
_old_backward = torch.Tensor.backward


@functools.wraps(_old_backward)
def _backward(*args, **kwargs):
    if _C.are_transforms_active():
        raise RuntimeError(
            "backward() called inside a functorch transform. This is not "
            "supported, please use functorch.grad or functorch.vjp instead "
            "or call backward() outside of functorch transforms.")
    return _old_backward(*args, **kwargs)


torch.Tensor.backward = _backward
