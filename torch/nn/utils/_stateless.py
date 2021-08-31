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
        _reparametrize_in_submodule(module, name.split("."), tensor)
    yield
    for name in parameters_and_buffers:
        _remove_reparametrize_in_submodule(module, name.split("."))

class _ReparametrizedTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self._tensor = tensor

    def forward(self, original):
        return self._tensor

def _reparametrize_in_submodule(module, path, tensor):
    if len(path) == 1:
        # We should be careful as the current API does not allow to reparametrize
        # already reparametrized parameters
        torch.nn.utils.parametrize.register_parametrization(
            module, path[0], _ReparametrizedTensor(tensor))
    else:
        _reparametrize_in_submodule(module._modules[path[0]], path[1:], tensor)

def _remove_reparametrize_in_submodule(module, path):
    if len(path) == 1:
        torch.nn.utils.parametrize.remove_parametrizations(
            module, path[0], False)
    else:
        _remove_reparametrize_in_submodule(
            module._modules[path[0]], path[1:])


def functional_call(module, parameters_and_buffers, *inputs, **kwargs):
    # TODO allow kwargs such as unsafe and others for parametrization
    with reparametrize_module(module, parameters_and_buffers):
        out = module(*inputs, **kwargs)
    return out
