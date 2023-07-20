from functools import reduce
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch.ao.pruning._experimental.pruner import BaseStructuredSparsifier

__all__ = ["FPGMSparsifier"]

class FPGMSparsifier(BaseStructuredSparsifier):
    r"""Filter Pruning via Geometric Median (FPGM) Structured Sparsifier
    This sparsifier prune fliter (row) in a tensor according to distances among filters according to 
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration"
    https://arxiv.org/abs/1811.00250.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dim` defines the dimension of the tensor to prune. Default: 0 (filter pruning).
    Currently, only `dim=0` is supported.
    3. `dist` defines the distance measurement type. Default: 2 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].
    
    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """
    def __init__(self,
                 sparsity_level: float = 0.5,
                 dim: Optional[int] = 0,
                 dist: Optional[Union[Callable, int]] = None):
        defaults = {
            "sparsity_level": sparsity_level,
            "dim": dim,
        }

        # TODO: support other dimensions
        if dim != 0:
            raise NotImplementedError("Only dim=0 (filter pruning) is currently supported.")

        if dist is None:
            dist = 2
        self.dist = dist
        if callable(dist):
            self.dist_fn = self.dist
        elif dist == 1:
            self.dist_fn = lambda x: torch.cdist(x, x, p=1)
        elif dist == 2:
            self.dist_fn = lambda x: torch.cdist(x, x, p=2)
        else:
            raise NotImplementedError(f"Distance function \"{self.dist}\" is not yet implemented.")
        super().__init__(defaults=defaults)

    def _compute_distance(self, t, dim):
        r"""Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            dim (int): dim identifying the filters to prune
        Returns:
            distance (torch.Tensor): distance computed across all dimensions except
                for `dim`. By construction, `distance.shape = t.shape[dim]`.
        """
        # dims = all axes, except for the one identified by `dim`
        dims = list(range(t.dim()))
        # convert negative indexing
        if dim < 0:
            dim = dims[dim]
        dims.remove(dim)

        size = t.size(dim)
        slc = [slice(None)] * t.dim()

        # flatten the tensor along the dimension
        t_flatten = [t[tuple(slc[:dim] + [i] + slc[dim+1:])].reshape(-1) for i in range(size)]
        t_flatten = torch.stack(t_flatten)

        # distance measurement
        dist_matrix = self.dist_fn(t_flatten)

        # more similar with other filter indicates large in the sum of row
        distance = torch.sum(torch.abs(dist_matrix), 1)

        return distance

    def update_mask(self, module, tensor_name, sparsity_level, dim, **kwargs):
        tensor_weight = getattr(module, tensor_name)

        # check pruning dimension
        if dim < 0:
            raise ValueError(
                "Dimension to prune must be non-negative."
            )
        elif dim >= tensor_weight.dim():
            raise ValueError(f"Invalide pruning dimension: {dim}.")

        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if mask.shape != tensor_weight.shape[dim]:      # initialize mask with correct dimension
            mask.data = torch.ones(tensor_weight.shape[dim]).bool()

        if sparsity_level <= 0:
            mask.data = torch.ones_like(mask).bool()
        elif sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask).bool()
        else:
            distance = self._compute_distance(tensor_weight, dim)

            tensor_size = tensor_weight.shape[dim]
            nparams_toprune = round(sparsity_level * tensor_size)
            if nparams_toprune == 0:
                Warning("Number of parameters to prune is 0. Skipping.")
                return 
            topk = torch.topk(distance, k=nparams_toprune, largest=False)
            mask.data[topk.indices] = False