import torch
import os
from torch import nn
from torch.ao.pruning import BaseSparsifier
import torch.ao.quantization as quantization
from torch.ao.quantization.observer import PerChannelMinMaxObserver, UniformQuantizationObserverBase
from torch.ao.quantization.utils import is_per_channel
from torch.ao.quantization.qconfig import QConfig, default_placeholder_observer, default_weight_observer
from torch.ao.quantization.quantize import _remove_qconfig

class PerChannelNormObserver(UniformQuantizationObserverBase):
    r"""
    """
    def __init__(
        self,
        **kwargs
    ) -> None:
        from pprint import pprint
        pprint(kwargs)
        super().__init__(
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            reduce_range=False,
            quant_min=None,
            quant_max=None,
            eps=torch.finfo(torch.float32).eps,
            **kwargs
        )
        self.averaging_constant = 1.0
        self.register_buffer("norm", torch.tensor([]))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape

        new_axis_list = [i for i in range(x.dim())]  # noqa: C416
        new_axis_list[0], new_axis_list[-1] = new_axis_list[-1], new_axis_list[0]
        print(new_axis_list)
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        norm = torch.norm(y, dim=1) ** 2
        if self.norm.numel() == 0:
            self.norm.resize_(norm.shape)
            self.norm.copy_(norm)
        else:
            self.norm += norm

        return x_orig

    def calculate_qparams(self):
        pass

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
        activation_observer = PerChannelNormObserver
        model.qconfig = QConfig(activation=activation_observer, weight=default_placeholder_observer)
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

        act_per_input = getattr(module, "activation_post_process").norm.sqrt()    ## get out the cumulated activation norm per channel
        print(act_per_input)
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

    def squash_mask(self, params_to_keep=None, params_to_keep_per_layer=None):
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            _remove_qconfig(module)

        super().squash_mask(params_to_keep=params_to_keep, params_to_keep_per_layer=params_to_keep_per_layer)
