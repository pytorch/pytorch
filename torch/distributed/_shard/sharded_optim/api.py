# mypy: allow-untyped-defs
from collections.abc import Mapping
from typing import Any

import torch.optim as optim
from torch import Tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor


class ShardedOptimizer(optim.Optimizer):
    def __init__(
        self,
        named_params: Mapping[str, Tensor | ShardedTensor],
        optimizer_class,
        *optimizer_args,
        **optimizer_kwargs,
    ):
        """
        ShardedOptimizer collects all tensors and local shard tensors of
        ShardedTensor, then use these tensors as ``params`` for optimizers

        Args:
            named_params (Dict[str, Union[Tensor, ShardedTensor]]) : a Dict
                of parameters, where key is the parameter key, value is either
                Tensor or ShardedTensor parameter.
            optimizer_class (torch.optim.Optimizer): the Optimizer to use
                locally, i.e. torch.optim.SGD, torch.optim.Adagrad, etc.
            *optimizer_args: the arguments to initialize the optimizer.
            **optimizer_kwargs: the key-word arguments to initialize the optimizer.

        """
        tensors: list[Tensor] = []
        for value in named_params.values():
            if isinstance(value, ShardedTensor):
                tensors.extend(
                    local_shard.tensor for local_shard in value.local_shards()
                )
            else:
                tensors.append(value)

        self.named_params = named_params
        self._optim = optimizer_class(tensors, *optimizer_args, **optimizer_kwargs)
        self.param_groups = self._optim.param_groups
        self.state = self._optim.state

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

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
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        self._optim.step(closure)

    def state_dict(self) -> dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices like torch.optim.Optimizer.
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
        raise NotImplementedError(
            "ShardedOptimizer load_state_dict not implemented yet!"
        )

    def add_param_group(self, param_group: Any):
        r"""Add a new param group"""
        # TODO: implement add_param_group
        raise NotImplementedError(
            "ShardedOptimizer add_param_group not implemented yet!"
        )
