from .base_structured_sparsifier import BaseStructuredSparsifier

class SaliencyPruner(BaseStructuredSparsifier):
     """
     Prune filters based on the saliency
     The saliency for a filter is given by the sum of the L1 norms of all of its weights
     """

     def update_mask(self, module, tensor_name, **kwargs):
        # tensor_name will give you the FQN, all other entries in sparse config is present in kwargs
         weights = getattr(module, tensor_name)
         mask = getattr(module.parametrizations, tensor_name)[0].mask

         # use negative weights so we can use topk (we prune out the smallest)
         saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
         num_to_pick = int(len(mask) * kwargs["sparsity_level"])
         prune = saliency.topk(num_to_pick).indices

         # Set the mask to be false for the rows we want to prune
         mask.data[prune] = False
