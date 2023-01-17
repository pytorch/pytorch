from typing import cast

import torch
from .base_structured_sparsifier import BaseStructuredSparsifier, FakeStructuredSparsity

class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency
    """

    def update_mask(self, module, tensor_name, **kwargs):
        weights = getattr(module, tensor_name)

        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = cast(torch.Tensor, p.mask)

                # select weights based on magnitude
                if weights.dim() > 1:
                    # take norm over all but first dim
                    dims = tuple(range(1, weights.dim()))
                    saliency = weights.abs().norm(dim=dims, p=1)
                else:
                    # 1d param: use weights directly
                    saliency = weights.abs()

                # handle weights in 4 groups
                split_size = len(mask) // 4
                masks = torch.split(mask, split_size)
                saliencies = torch.split(saliency, split_size)

                for keep_mask, sal in zip(masks, saliencies):
                    # mask smallest k values to be removed
                    k = int(len(keep_mask) * kwargs["sparsity_level"])
                    prune = sal.topk(k, largest=False, sorted=False).indices
                    keep_mask.data[prune] = False  # modifies underlying p.mask directly
