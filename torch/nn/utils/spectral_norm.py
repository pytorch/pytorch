"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter


class SpectralNorm(object):

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight_and_update_u(self, module):
        # NB: This updates the _u vector **in-place**. This is very important
        #     because in DataParallel forward, the _u vector (being a buffer) is
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object on the fly. And each replica runs its
        #     own spectral norm power iteration. So simply assigning the updated
        #     _u vector to module this runs on will cause the update to be lost
        #     forever. And the next time the parallelized module is replicated,
        #     the same randomly initialized _u vector is broadcast!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important bahaviors (also enforced via tests):
        #       1. DataParallel doesn't clone storage if the broadcast tensor is
        #          alreay on correct device; and it makes sure that the
        #          parallelized module is already on device[0].
        #       2. If the out tensor in out= kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the _u tensor in-place will make sure
        #     that the module replica on device[0] will update the _u vector on
        #     the parallized module (by shared storage).
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps, out=u)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight = self.compute_weight_and_update_u(module)
            setattr(module, self.name, weight)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            weight = getattr(module, self.name).detach()
            # NB: Cannot detach weight in-place here because if this is used
            #     DataParallel, the buffers are broadcast using
            #     `broadacast_coalesced` and `weight` here is actually a view,
            #     and you can't detach views in-place.
            setattr(module, self.name, weight.requires_grad_(r_g))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1

    Returns:
        The original module with the spectal norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])

    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
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
