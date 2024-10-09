# mypy: allow-untyped-defs
from .base_structured_sparsifier import BaseStructuredSparsifier


class SaliencyPruner(BaseStructuredSparsifier):
    """
    Prune rows based on the saliency (L1 norm) of each row.

    This pruner works on N-Dimensional weight tensors.
    For each row, we will calculate the saliency, whic is the sum the L1 norm of all weights in that row.
    We expect that the resulting saliency vector has the same shape as our mask.
    We then pick elements to remove until we reach the target sparsity_level.
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # tensor_name will give you the FQN, all other entries in sparse config is present in kwargs
        weights = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        # use negative weights so we can use topk (we prune out the smallest)
        if weights.dim() <= 1:
            raise Exception(  # noqa: TRY002
                "Structured pruning can only be applied to a 2+dim weight tensor!"
            )
        saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
        assert saliency.shape == mask.shape

        num_to_pick = int(len(mask) * kwargs["sparsity_level"])
        prune = saliency.topk(num_to_pick).indices

        # Set the mask to be false for the rows we want to prune
        mask.data[prune] = False
