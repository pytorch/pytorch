import torch
from torch import nn
# from torch.ao.pruning import BaseSparsifier
from sparsifier.base_sparsifier import BaseSparsifier

from pdb import set_trace as st 

class WandaSparsifier(BaseSparsifier):
    def __init__(self, sparsity_level):
        defaults = {
            'sparsity_level': sparsity_level
        }
        super().__init__(defaults=defaults)

    def update_mask(self, module, tensor_name, sparsity_level, **kwargs):
        # Step 1: get the tensor and the mask from the parametrizations
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        tensor = getattr(module.parametrizations, tensor_name).original
        # Step 2: implement the mask update logic
        ## compute the pruning metric
        tensor_flat = tensor.flatten()
        tensor_pruning_metric = tensor_flat.abs()
        ## Step 2b: Rank the elements in the tensor
        _, sorted_idx = torch.sort(tensor_pruning_metric)
        threshold_idx = int(round(sparsity_level * len(sorted_idx)))
        sorted_idx = sorted_idx[:threshold_idx]
        ## Step 2c: Create a mask with the known zero elements
        new_mask = torch.ones_like(mask)
        new_mask = new_mask.flatten()
        new_mask[sorted_idx] = 0
        new_mask = new_mask.reshape(mask.shape)
        # Step 3: Reassign back to the mask
        mask.data = new_mask

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(128, 200))
    X = torch.randn(100,128)

    sparsifier = WandaSparsifier(sparsity_level=0.5)
    sparsifier.prepare(model, config=None)
    #####################################
    ### run the model with calibration data
    y = model(X)
    #####################################
    sparsifier.step()

    sparsity_level = (model[0].weight == 0).float().mean()
    print(f"Level of sparsity: {sparsity_level.item():.2%}")

    sparsifier.squash_mask()
