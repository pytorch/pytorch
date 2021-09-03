import contextlib

import torch


def _get_module_parameters_and_buffers(module):
    """Helper function that returns a dictionary
    with only parameters and buffers, we don't use state_dict as it might
    be polluted with module attributes.
    """
    parameters_and_buffers = dict(module.named_parameters())
    parameters_and_buffers.update(dict(module.named_buffers))
    # TODO: clean entries that are already parameterized?
    # params with weight norm/spectral norm applied
    return parameters_and_buffers


@contextlib.contextmanager
def reparametrize_module(module, parameters_and_buffers):
    # Parametrization does not support to change submodules directly
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            torch.nn.utils.parametrize.register_parametrization,
            module, name.split("."), (_ReparametrizedTensor(tensor),))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            torch.nn.utils.parametrize.remove_parametrizations,
            module, name.split("."), (False,))


class _ReparametrizedTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self._tensor = tensor

    def forward(self, original):
        return self._tensor


def _apply_func_submodules(func, module, path, args):
    if len(path) == 1:
        # We should be careful as the current API does not allow to reparametrize
        # already reparametrized parameters
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(module, parameters_and_buffers, args, kwargs=None):
    # TODO allow kwargs such as unsafe and others for parametrization
    if kwargs is None:
        kwargs = {}
    with reparametrize_module(module, parameters_and_buffers):
        out = module(*args, **kwargs)
    return out
