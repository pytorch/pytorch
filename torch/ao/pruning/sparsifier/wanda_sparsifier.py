import torch
import os
from torch import nn
from torch.ao.pruning import BaseSparsifier
from torch.ao.pruning.sparsifier.utils import PerChannelNormObserver
from torch.ao.quantization import QConfig, default_placeholder_observer
from torch.ao.quantization.quantize import _remove_qconfig

from typing import Callable, Dict, List, Optional, Tuple, Union


class WandaSparsifier(BaseSparsifier):
    r"""Wanda sparsifier

    Wanda (Pruning by Weights and activations), proposed in https://arxiv.org/abs/2306.11695
    is an activation aware pruning method. The sparsifier removes weights based on the product
    of the input activation norm and the weight magnitude.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out;

    Args:
        sparsity_level: The target level of sparsity;
        model: The model to be sparsified;
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        semi_structured_block_size: Optional[int] = None,
    ):
        defaults = {
            "sparsity_level": sparsity_level,
            "semi_structured_block_size": semi_structured_block_size,
        }
        if semi_structured_shape is not None:
            warnings.warn(
                "Wanda pruning in semi-structured mode, sparsity_level fixed to 50%"
            )

        super().__init__(defaults=defaults)

    def prepare(self, model: nn.Module, config: List[Dict]) -> None:
        """
        This function will attach observers to the model, using the QuantizationAPI.
        Prepare the model for sparsification.
        """
        # activation: use PerChannelNormObserver
        # use no-op placeholder weight observer
        model.qconfig = QConfig(
            activation=PerChannelNormObserver, weight=default_placeholder_observer
        )
        torch.ao.quantization.prepare(model, inplace=True)

        # call superclass prepare
        super().prepare(model, config)

    def update_mask(self, module, tensor_name, sparsity_level, **kwargs) -> None:
        r"""Pruning function for WandaSparsifier
        The activation statistics is retrieved first in the `act_per_input` variable.
        Then the Wanda pruning metric is computed. The weight matrix is then pruned
        by comparing this metric across the whole current layer.
        """

        # Step 1: get the tensor and the mask from the parametrizations
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        tensor = getattr(module.parametrizations, tensor_name).original
        activation_norm_per_channel = getattr(module, "activation_post_process").norm

        # Step 2: Calculate Wx
        pruning_metric = torch.abs(tensor) * activation_norm_per_channel

        # defaults for unstructured sparsity
        block_size = pruning_metric.numel()
        num_specified = int(block_size * sparsity_level)
        # if set to use semi-structured, ignore sparsity_level
        if kwargs.get("semi_structured_shape", None) is not None:
            block_shape = kwargs["semi_structured_block_size"]
            num_specified = block_shape // 2

        # get indicies to prune
        pruning_inds = pruning_metric.view(-1, block_shape).argsort(dim=1)[
            :, :num_specified
        ]
        # update mask
        mask.data.view(-1, block_shape).scatter_(
            1, pruning_inds, torch.zeros_like(pruning_inds, dtype=mask.dtype)
        )

    def squash_mask(
        self,
        params_to_keep: Optional[Tuple[str, ...]] = None,
        params_to_keep_per_layer: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> None:
        # remove quantization config
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            _remove_qconfig(module)

        # remove parameterizations
        super().squash_mask(
            params_to_keep=params_to_keep,
            params_to_keep_per_layer=params_to_keep_per_layer,
        )
