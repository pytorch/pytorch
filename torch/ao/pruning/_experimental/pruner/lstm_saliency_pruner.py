# mypy: allow-untyped-defs
from typing import cast

import torch

from .base_structured_sparsifier import BaseStructuredSparsifier, FakeStructuredSparsity


class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency.
    For each layer {k} inside a LSTM, we have two packed weight matrices
    - weight_ih_l{k}
    - weight_hh_l{k}

    These tensors pack the weights for the 4 linear layers together for efficiency.

    [W_ii | W_if | W_ig | W_io]

    Pruning this tensor directly will lead to weights being misassigned when unpacked.
    To ensure that each packed linear layer is pruned the same amount:
        1. We split the packed weight into the 4 constituent linear parts
        2. Update the mask for each individual piece using saliency individually

    This applies to both weight_ih_l{k} and weight_hh_l{k}.
    """

    def update_mask(self, module, tensor_name, **kwargs):
        weights = getattr(module, tensor_name)

        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = cast(torch.Tensor, p.mask)

                # select weights based on magnitude
                if weights.dim() <= 1:
                    raise Exception(  # noqa: TRY002
                        "Structured pruning can only be applied to a 2+dim weight tensor!"
                    )
                # take norm over all but first dim
                dims = tuple(range(1, weights.dim()))
                saliency = weights.norm(dim=dims, p=1)

                # handle weights in 4 groups
                split_size = len(mask) // 4
                masks = torch.split(mask, split_size)
                saliencies = torch.split(saliency, split_size)

                for keep_mask, sal in zip(masks, saliencies):
                    # mask smallest k values to be removed
                    k = int(len(keep_mask) * kwargs["sparsity_level"])
                    prune = sal.topk(k, largest=False, sorted=False).indices
                    keep_mask.data[prune] = False  # modifies underlying p.mask directly
