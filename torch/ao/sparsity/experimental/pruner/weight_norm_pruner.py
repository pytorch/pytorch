import torch

from .base_pruner import BasePruner

class WeightNormPruner(BasePruner):
    r"""
    Rank a tensor's rows by the weight norm, and remove the lower norm ones.
    """
    def __init__(self, sparsity_level: float = 0.5, axis=0):
        defaults = {
            'sparsity_level': sparsity_level,
            'axis': axis
        }
        super().__init__(defaults=defaults, also_prune_bias=False)

    def update_mask(self, layer, sparsity_level, axis, **kwargs):
        # Get the norms of the axis
        w = layer.weight
        dims = set(range(w.ndim)) - set([axis])
        for d in dims:
            w = w.norm(dim=d, keepdim=True)
        w_norms = w.squeeze()
        _, sorted_idx = torch.sort(w_norms)
        threshold_idx = int(round(sparsity_level * len(sorted_idx)))
        sorted_idx = sorted_idx[:threshold_idx].detach().cpu().numpy()
        sorted_idx = set(sorted_idx)
        # Reattach the pruned outputs
        layer.parametrizations.weight[0].pruned_outputs = sorted_idx
