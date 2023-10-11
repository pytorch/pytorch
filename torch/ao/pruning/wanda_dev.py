import torch
import os 
from torch import nn
# from torch.ao.pruning import BaseSparsifier
from pruning.sparsifier.base_sparsifier import BaseSparsifier
import quantization 
from quantization.observer import PerChannelMinMaxObserver
from quantization.utils import is_per_channel
from quantization.qconfig import QConfig, default_weight_observer

from pdb import set_trace as st 


class MovingAveragePerChannelNormObserver(PerChannelMinMaxObserver):
    r"""
    Adapted from MovingAveragePerChannelMinMaxObserver

    The activation statistics per input channel are stored in `min_val`.
    """
    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        **kwargs
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            **kwargs
        )
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        ### I use min_val to store the activation norm;
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0:
            min_val = torch.norm(y, dim=1) ** 2
        else:
            min_val_cur = torch.norm(y, dim=1) ** 2
            min_val = min_val + min_val_cur
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        return x_orig

class WandaSparsifier(BaseSparsifier):
    r"""Wanda sparsifier

    Wanda (Pruning by Weights and activations), proposed in https://arxiv.org/abs/2306.11695 
    is an activation aware pruning method. The sparsifier removes weights based on the product
    of the input activation norm and the weight magnitude.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out;
    2. `model` defines the model to be sparsified;

    Args:
        sparsity_level: The target level of sparsity;
        model: The model to be sparsified;
    """
    def __init__(self, sparsity_level, model):
        r""" Initialization function for WandaSparsifier class

        In this function, forward hooks (observer class from quantization API) 
        will be registered on the model. This hook will store average activation
        statistics per input channels.

        averaging_constant is hard set to 1.0 in order to register a forward_pre_hook
        https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantize.py#L201.
        """
        defaults = {
            'sparsity_level': sparsity_level
        }
        super().__init__(defaults=defaults)

        ### custom initialization code to set up the observer
        activation_observer = MovingAveragePerChannelNormObserver.with_args(averaging_constant=1.0, ch_axis=1)
        my_qconfig = QConfig(activation=activation_observer, weight=default_weight_observer)
        model.qconfig = my_qconfig 
        quantization.prepare(model, inplace=True)

    def update_mask(self, module, tensor_name, sparsity_level, **kwargs):
        r""" Pruning function for WandaSparsifier

        The activation statistics is retrieved first in the `act_per_input` variable. 
        Then the Wanda pruning metric is computed. The weight matrix is then pruned 
        by comparing this metric across the whole current layer.
        """

        # Step 1: get the tensor and the mask from the parametrizations
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        tensor = getattr(module.parametrizations, tensor_name).original

        act_per_input = getattr(module, "activation_post_process").min_val.sqrt()    ## get out the cumulated activation norm per channel
        # Step 2: implement the mask update logic
        ## compute the pruning metric
        pruning_metric = torch.abs(tensor) * act_per_input.reshape((1,-1))
        pruning_metric = torch.flatten(pruning_metric)
        ## Step 2b: Rank the elements in the tensor
        _, sorted_idx = torch.sort(pruning_metric)
        threshold_idx = int(round(sparsity_level * len(sorted_idx)))
        sorted_idx = sorted_idx[:threshold_idx]
        ## Step 2c: Create a mask with the known zero elements
        new_mask = torch.ones_like(mask)
        new_mask = new_mask.flatten()
        new_mask[sorted_idx] = 0
        new_mask = new_mask.reshape(mask.shape)
        # Step 3: Reassign back to the mask
        mask.data = new_mask

def test_one_layer_mlp_1():
    model = nn.Sequential(nn.Linear(4,1))
    weights = torch.tensor([[1,2,3,4]])
    model[0].weight.data.copy_(weights.data)
    X = torch.ones(1,4)

    sparsifier = WandaSparsifier(sparsity_level=0.5, model=model)
    sparsifier.prepare(model, config=None)

    model(X)

    sparsifier.step()
    sparsifier.squash_mask()

    sparsity = (model[0].weight == 0).float().mean()
    assert sparsity == 0.5, "sparsity should be 0.5"
    expected_fc = torch.tensor([[0,0,3,4]], dtype=torch.float32)
    assert torch.isclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07).all()

def test_one_layer_mlp_2():
    model = nn.Sequential(nn.Linear(4,1))
    weights = torch.tensor([[1,2,3,4]], dtype=torch.float32)
    model[0].weight.data.copy_(weights.data)
    X = torch.tensor([[100, 10, 1, 0.1]], dtype=torch.float32)

    sparsifier = WandaSparsifier(sparsity_level=0.5, model=model)
    sparsifier.prepare(model, config=None)

    model(X)

    sparsifier.step()
    sparsifier.squash_mask()

    sparsity = (model[0].weight == 0).float().mean()
    assert sparsity == 0.5, "sparsity should be 0.5"
    expected_fc = torch.tensor([[1,2,0,0]], dtype=torch.float32)
    assert torch.isclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07).all()


def test_two_layer_mlp_1():
    model = nn.Sequential(nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10))      ## C_in by C_out
    X1 = torch.randn(100,128)           ## B1 by C_in 
    X2 = torch.randn(50, 128)           ## B2 by C_in 

    sparsifier = WandaSparsifier(sparsity_level=0.5, model=model)
    sparsifier.prepare(model, config=None)

    model(X1)
    model(X2)
    sparsifier.step()

    cnt = 0 
    for m in model.modules():
        if isinstance(m, nn.Linear):
            cnt += 1
            sparsity_level = (m.weight == 0).float().mean()
            # print(f"Level of sparsity for Linear layer {cnt}: {sparsity_level.item():.2%}")
            assert sparsity_level == 0.5, f"sparsity for linear layer {cnt} should be 0.5"

    #### TODO: how to remove the hook and also parametrizations?
    sparsifier.squash_mask()

if __name__ == "__main__":
    ## test 1: check for one layer mlp and if the output mask is correct
    test_one_layer_mlp_1()

    ## test 2: check for one layer mlp and if the output mask is correct
    test_one_layer_mlp_2()

    ## test 3: check for two layer mlp and also two batches of calibration data
    test_two_layer_mlp_1()
