from typing import List, Union, Mapping, Dict, Any

from torch import Tensor
from torch.distributed._sharded_tensor import ShardedTensor


class ShardedOptimizer(object):
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
        self._optim.zero_grad(set_to_none)

    def step(self, closure=None):
        """
        Regular optimizer step
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
