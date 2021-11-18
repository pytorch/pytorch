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
        state = self.state
        param_groups = self.param_groups
        params = self.params

        # actual param to key mapping, to be used for states mapping
        param_to_key = {param: key for key, param in params.item()}

        ret_state = {
            param_to_key[param]: state_val
            for param, state_val in state.items()
        }

        ret_groups = []
        for param_group in param_groups:
            ret_group = {k: v for k, v in param_group.items() if k != "params"}
            keys = [param_to_key[param] for param in param_group["params"]]
            ret_group['params'] = sorted(keys)
            ret_groups.append(ret_group)

        return {
            'state': ret_state,
            'param_groups': ret_groups
        }


    def load_state_dict(self, state_dict: Mapping[str, Any]):
        r"""Loads the ShardedOptimizer state.

        Args:
            state_dict (dict): ShardedOptimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # Validate the state_dict
        groups = self.param_groups
        state = self.state
        saved_groups = state_dict['param_groups']
        saved_state = state_dict['state']

        # load state values
        if len(state) != len(saved_state):
            raise ValueError("loaded state_dict has a different number of parameters")

        for param_key, param in self.params.items():
            if param not in state:
                continue

            if param_key not in state:
                raise ValueError(f"Parameter {param_key} not found!")

            for state_key, state_val in state[param].items():
                if state_key not in saved_state[param_key]:
                    pass
