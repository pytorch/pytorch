"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.parameter import Parameter


def l2normalize(v, eps=1e-12):
    """Scale to inputs norm to 1."""
    denom = v.norm(p=2) + eps
    return v / denom


class SpectralNorm(object):

    def __init__(self, name='weight', n_power_iterations=1, eps=1e-12):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = module._parameters[self.name + '_org']
        u = module._buffers[self.name + '_u']
        height, _cuda = weight.size(0), weight.is_cuda
        weight_mat = weight.view(height, -1)
        for _ in range(self.n_power_iterations):
            v = l2normalize(torch.matmul(weight_mat.t(), u), self.eps)
            u = l2normalize(torch.matmul(weight_mat, v), self.eps)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight.data /= sigma
        return weight, u

    def remove(self, module):
        weight = module._parameters[self.name + '_org']
        del module._parameters[self.name]
        del module._buffers[self.name + '_u']
        del module._parameters[self.name + '_org']
        module.register_parameter(self.name, weight)

    def __call__(self, module, inputs):
        weight, u = self.compute_weight(module)
        setattr(module, self.name, weight)
        setattr(module, self.name + '_u', u)

    @staticmethod
    def apply(module, name, n_power_iterations, eps):
        fn = SpectralNorm(name, n_power_iterations, eps)
        weight = module._parameters[name]
        height = weight.size(0)

        u = l2normalize(weight.data.new(height).normal_(0, 1), fn.eps)
        module.register_parameter(fn.name + "_org", weight)
        module.register_buffer(fn.name + "_u", u)

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilize the training of discriminators(critics)
    in GANs by rescaling the weight tensor by spectral norm "sigma" of the
    weight matrix calculated by power iteration method. If the dimension of the
    weight tensor is greater than 2, reahaped to 2D in power iteration method
    to get spectral norm. This is implemented via a hook that calculates
    spectral norm and rescales weight before every :meth:`~Module.forward`
    call.

    See https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability

    Returns:
        The original module with the spectal norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])

    """
    SpectralNorm.apply(module, name, n_power_iterations, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Modue): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))
