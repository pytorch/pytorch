from .base_structured_sparsifier import BaseStructuredSparsifier

class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency
    """

    def update_mask(self, module, tensor_name, **kwargs):
        weights = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = p.mask
                masks = torch.split(mask, len(mask) // 4)

                pruned_small = []
                for small in masks:
                    # use negative weights so we can use topk (we prune out the smallest)
                    saliency = -small.norm(dim=tuple(range(1, small.dim())), p=1)
                    num_to_pick = int(len(small) * kwargs["sparsity_level"])
                    prune = saliency.topk(num_to_pick).indices
                    small.data[prune] = False

                new_mask = torch.cat(masks)
                mask.data = new_mask.data
