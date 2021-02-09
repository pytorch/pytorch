import torch

from torch.expanded_weights.expanded_weights_impl import ExpandedWeight
from ._stateless import functional_call

def per_sample_call(module: torch.nn.Module, batch_size, *args, **kwargs):
    r"""Invoked just like a forward pass, per_sample_call will produce the same
    forward result. Then, when backward is invoked, the parameters of ``module``
    will have a ``grad_sample`` field populated with the per sample gradients
    instead of the batched gradients
    """
    def maybe_build_expanded_weight(og_tensor):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size)
        else:
            return og_tensor

    params = {name: maybe_build_expanded_weight(value) for (name, value) in module.named_parameters()}
    return functional_call(module, params, args, kwargs)
