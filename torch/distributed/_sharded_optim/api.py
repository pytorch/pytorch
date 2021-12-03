from typing import List, Union, Mapping, Dict, Any

import torch.optim as optim
from torch import Tensor
from torch.distributed._sharded_tensor import ShardedTensor


class ShardedOptimizer(optim.Optimizer):
    def __init__(
        self,
        params: Mapping[str, Union[Tensor, ShardedTensor]],
        optimizer_class,
        *optimizer_args,
        **optimizer_kwargs
    ):
        """
        Collects all Tensors and local shards of ShardedTensor, uses these
        Tensors as ``params`` for the optimizer while building the optimizer
        using ``optimizer_class``, ``*optimizer_args`` and
        ``*optimizer_kwargs``.
        """
        tensors: List[Tensor] = []
        for value in params.values():
            if isinstance(value, ShardedTensor):
                for local_shard in value.local_shards():
                    tensors.append(local_shard.tensor)
            else:
                tensors.append(value)

        self.params = params
        self._optim = optimizer_class(tensors, *optimizer_args, **optimizer_kwargs)
        self.param_groups = self._optim.param_groups
        self.state = self._optim.state

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        self._optim.zero_grad(set_to_none)

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        self._optim.step(closure)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices in torch.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.
        """
        # TODO: implement state_dict
        raise NotImplementedError("ShardedOptimizer state_dict not implemented yet!")


    def load_state_dict(self, state_dict: Mapping[str, Any]):
        r"""Loads the ShardedOptimizer state.

        Args:
            state_dict (dict): ShardedOptimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # TODO: implement load_state_dict
        raise NotImplementedError("ShardedOptimizer load_state_dict not implemented yet!")

    def add_param_group(self, param_group: Any):
        r"""Add a new param group
        """
        # TODO: implement load_state_dict
        raise NotImplementedError("ShardedOptimizer add_param_group not implemented yet!")
