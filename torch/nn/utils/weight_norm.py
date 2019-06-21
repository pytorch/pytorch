r"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import math
from torch.nn.parameter import Parameter
from torch import _weight_norm, norm_except_dim, ones_like
from torch.nn.init import _calculate_fan_in_and_fan_out


class WeightNorm(object):
    def __init__(self, name, dim):
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name, dim, init, gamma):
        assert init in ['default', 'norm_preserving'], \
            "Invalid init for WeightNorm ({}). It must be one of ['default', 'norm_preserving']".format(init)

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # initial value for g
        if init == 'default':
            g_init = norm_except_dim(weight, 2, dim)
        elif init == 'norm_preserving':
            fan_in, fan_out = _calculate_fan_in_and_fan_out(weight)
            g_init = ones_like(norm_except_dim(weight, 2, dim)) * math.sqrt(gamma * fan_in / fan_out)

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(g_init.data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0, init='default', gamma=2.):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
        init (str, optional): initialization scheme; one in ['default', 'norm_preserving']
        gamma (float, optional): gamma parameter for the init scheme in Arpit et al. 2019
                $\mathbf{g} = \sqrt{\gamma \cdot \text{fan-in} / \text{fan-out}} \cdot \mathbf{1}$
            Gamma needs to be set as follows:
                Layer without non-linearity:    gamma=1
                Layer followed by ReLU:         gamma=2
                Last layer in a resblock:       gamma=1/B_k, where B_k is the number of resblocks in the k-th stage
            See https://arxiv.org/abs/1906.02341

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim, init, gamma)
    return module


def remove_weight_norm(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
