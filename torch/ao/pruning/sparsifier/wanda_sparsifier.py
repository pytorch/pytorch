import torch
import os
from torch import nn
from torch.ao.pruning import BaseSparsifier
from torch.ao.pruning.sparsifier.utils import PerChannelNormObserver
import torch.ao.quantization as quantization
from torch.ao.quantization.qconfig import QConfig, default_placeholder_observer, default_weight_observer
from torch.ao.quantization.quantize import _remove_qconfig
from typing import Callable, Dict, List, Optional, Tuple, Union

from functools import reduce

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
    def __init__(self,
                 sparsity_level: float = 0.5,
                 semi_structured_block_shape: Tuple[int, int] = None):
        defaults = {
            "sparsity_level": sparsity_level,
            "semi_structured_block_shape": semi_structured_block_shape,
        }
        super().__init__(defaults=defaults)


    def prepare(self, model, config):
        ### custom initialization code to set up the observer
        model.qconfig = QConfig(activation=PerChannelNormObserver, weight=default_placeholder_observer)
        quantization.prepare(model, inplace=True)
        super().prepare(model, config)

    def update_mask(self, module, tensor_name, sparsity_level, **kwargs):
        r""" Pruning function for WandaSparsifier
        The activation statistics is retrieved first in the `act_per_input` variable.
        Then the Wanda pruning metric is computed. The weight matrix is then pruned
        by comparing this metric across the whole current layer.
        """

        # Step 1: get the tensor and the mask from the parametrizations
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        tensor = getattr(module.parametrizations, tensor_name).original

        act_norm_per_channel= getattr(module, "activation_post_process").norm.sqrt()    ## get out the cumulated activation norm per channel
        # Step 2: implement the mask update logic
        ## compute the pruning metric
        print(act_norm_per_channel.shape)
        print(tensor.shape)
        pruning_metric = torch.abs(tensor) * act_norm_per_channel
        print(pruning_metric.shape)

        # set defaults
        block_shape = pruning_metric.numel()
        num_specified = int(block_shape * sparsity_level)
        if kwargs.get("semi_structured_block_shape", None) is not None:
            num_specified, block_shape = kwargs["semi_structured_block_shape"]
        # update mask
        pruning_inds = pruning_metric.view(-1, block_shape).argsort(dim=1)[:, :num_specified]
        mask.data.view(-1, block_shape).scatter_(1, pruning_inds, torch.zeros_like(pruning_inds, dtype=mask.dtype))



    def squash_mask(self, params_to_keep=None, params_to_keep_per_layer=None):
        # remove quantization config
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            _remove_qconfig(module)

        # remove parameterizations
        super().squash_mask(params_to_keep=params_to_keep,
                            params_to_keep_per_layer=params_to_keep_per_layer)
