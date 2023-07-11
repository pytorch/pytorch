import logging
import warnings

from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Union, overload

import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


__all__: List[str] = []

logger = logging.getLogger(__name__)


class _NamedOptimizer(optim.Optimizer):
    """
    ``_NamedOptimizer`` takes a dict of parameters and exposes ``state_dict`` by
    parameter key. We replace the original key (number) in an optim to the
    fully qualified name (FQN) string. User can initialize the optim as they
    initialize a PyTorch optim, the only difference is that they also need to
    pass in the FQN of each parameters.

    Args:
        named_parameters (Mapping[str, Union[torch.Tensor, ShardedTensor]]):
            Mapping from FQN to parameter.
        optimizer_class (optim.Optimizer):
            The class of optimizer to instantiate.
        param_groups (Collection[Mapping[str, Any]]):
            `param_groups` to pass to optimizer if specified.
            The key of the inner map needs to be FQNs.
            Default: None
        module (nn.Module): the module whose parameters to updated
            by the optimizer.
        args: arguments to pass to the optimizer constructor.
        kwargs: arguments to pass to the optimizer constructor.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch import optim
        >>> from torch.distributed.optim import _NamedOptimizer
        >>>
        >>> # Define the named optimizer.
        >>> m = Model(...)
        >>> named_optim = _NamedOptimizer(m.named_parameters(), optim.SGD)
        >>> # Forward pass + backward pass.
        >>> named_optim.step()
        >>> ...
        >>> # Call state_dict for the named optimizer returns a FQN state_dict.
        >>> named_optim.state_dict()

    Warning: This API is still in development and subject to change.

    TODO: Add tutorial for _NamedOptimizer.
    TODO: Add documentation in the docstring for the public attributes
          like self.param_groups and self.named_parameters.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optimizer_class: optim.Optimizer,
        param_groups: Collection[Mapping[str, Any]] = None,
        module: nn.Module = None,
        *args,
        **kwargs,
    ) -> None:
        torch._C._log_api_usage_once("torch.distributed.optim._NamedOptimizer")
        self.param_groups: Collection[Mapping[str, Any]] = param_groups  # type: ignore[assignment]
        self._param_groups_check()
        self.named_parameters = dict(named_parameters)
        params_for_optimizer = (
            self.named_parameters.values() if param_groups is None else param_groups
        )
        self._optimizer = optimizer_class(  # type: ignore[operator]
            params_for_optimizer,
            *args,
            **kwargs,
        )
        self.module = module
        if param_groups is None:
            self.ordered_param_keys = list(self.named_parameters.keys())
        else:
            warnings.warn(
                "Since we pass in param_groups, we will use param_groups to "
                "initialize the optimizer, not all parameters of the module."
            )
            param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
            ordered_param_keys = []
            for group in param_groups:
                for param in group["params"]:
                    if param not in param_to_key:
                        raise ValueError(
                            f"Expect param name {param} found in param group but is missing."
                        )
                    ordered_param_keys.append(param_to_key[param])
            self.ordered_param_keys = ordered_param_keys
        # Update param_groups from optimizer.
        self.param_groups = self._optimizer.param_groups

    def _param_groups_check(self):
        if self.param_groups is not None:
            for param_group in self.param_groups:
                assert isinstance(param_group, dict), "param group must be a dict"
                assert "params" in param_group, "param group must contain key params"
                params = param_group["params"]
                if isinstance(params, torch.Tensor):
                    params = [params]
                params = list(params)
                for param in params:
                    if not isinstance(param, torch.Tensor):
                        raise TypeError(
                            "optimizer can only optimize Tensors, "
                            "but one of the params is " + torch.typename(param)
                        )
                param_group["params"] = params

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the ``state_dict`` of the optimizer. Instead of using number to index
        parameters, we will use module fully qualified name (FQN) as the key.
        """
        state_dict = self._optimizer.state_dict()
        param_groups = state_dict["param_groups"]

        ret_state = {
            self.ordered_param_keys[st_key]: state_val
            for st_key, state_val in state_dict["state"].items()
        }

        ret_groups = []
        for group in param_groups:
            param_keys = []
            for param in group["params"]:
                param_keys.append(self.ordered_param_keys[param])
            ret_group = {"params": sorted(param_keys)}
            for k, v in group.items():
                if k != "params":
                    ret_group[k] = deepcopy(v)
            ret_groups.append(ret_group)

        return self._post_state_dict({"state": ret_state, "param_groups": ret_groups})

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on the wrapped
        optimizer.
        """
        return self._optimizer.step(closure=closure)

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:
        return self._optimizer.state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        This public function defines the default behavior to load a state_dict
        for ``_NamedOptimizer``.

        Sample Code
        ```
            my_model = MyModule()
            optimizer = _NamedOptimizer(my_model.named_parameters(), Adagrad)
            ...

            optim_state_dict = optimizer.state_dict()
            ...
            ...

            optimizer.load_state_dict(optim_state_dict)
            ...
        ```
        Args:
            state_dict (Dict[str, Any]) : A ``state_dict`` to load into the optimizer.
                Note that this state dict update is performed in place.

        .. note:: PyTorch is using lazy init to initialize the optim states.
            So it is possible that there is no optim state when user call
            ``load_state_dict`` and for ``_NamedOptimizer`` we make it stricter
            that users can only call ``load_state_dict`` after the state is initialized.
            By doing this, we can validate the optim ``state_dict`` to be loaded.
        """
        new_state_dict = self._optimizer.state_dict()
        state_dict = self._pre_load_state_dict(state_dict)
        state = state_dict["state"]
        new_state = new_state_dict["state"]
        if len(new_state) == 0:
            raise ValueError(
                "Expects the optim to be initialized before load but found not initialized."
            )

        for idx, param_key in enumerate(self.ordered_param_keys):
            # When the conditional training is performed, not all parameters are updated in the optim.
            if param_key not in state.keys():
                continue
            if len(state[param_key]) != len(new_state[idx]):
                raise ValueError(
                    f"Expects equal length as {len(new_state[idx])} for parameter {param_key} but found: {len(state[param_key])}"
                )
            # Iterate through all optimizer states.
            for state_key, state_val in new_state[idx].items():
                if state_key not in state[param_key]:
                    raise ValueError(
                        f"Expects state {state_key} for parameter {param_key} but not found."
                    )

                src_state_val = state[param_key][state_key]
                if isinstance(state_val, ShardedTensor):
                    assert isinstance(src_state_val, ShardedTensor)
                    num_shards = len(state_val.local_shards())
                    num_new_shards = len(src_state_val.local_shards())
                    if num_shards != num_new_shards:
                        raise ValueError(
                            f"Expects equal number of shards as {num_new_shards} but found {num_shards} for {param_key}/{state_key}"
                        )
                    for shard, src_shard in zip(
                        state_val.local_shards(), src_state_val.local_shards()
                    ):
                        shard.tensor.detach().copy_(src_shard.tensor)
                elif isinstance(state_val, torch.Tensor):
                    assert isinstance(src_state_val, torch.Tensor)
                    state_val.detach().copy_(src_state_val)
                else:
                    new_state[idx][state_key] = deepcopy(src_state_val)

        # Load param_groups of state_dict
        src_param_groups = state_dict["param_groups"]
        new_param_groups = new_state_dict["param_groups"]

        src_group_map = {}
        for group in src_param_groups:
            param_keys = []
            for param_key in group["params"]:
                param_keys.append(param_key)
            src_group_map[_gen_param_group_key(param_keys)] = group
        new_group_map = {}
        for new_group in new_param_groups:
            param_keys = []
            for param_key in new_group["params"]:
                param_keys.append(self.ordered_param_keys[param_key])  # type: ignore[call-overload]
            new_group_map[_gen_param_group_key(param_keys)] = new_group
        for group_key, new_group in new_group_map.items():
            # When not all parameters are used in training or receive gradient, aka., not all parameters
            # would be in the param_group. Thus we skip the group_key here.
            if group_key not in src_group_map:
                continue
            src_group = src_group_map[group_key]
            if len(src_group) != len(new_group):
                raise ValueError(
                    f"Expects equal param_group size as {len(new_group)} for group {group_key} but found {len(src_group)}."
                )
            for k in src_group:
                if k not in new_group:
                    raise ValueError(
                        f"Expects group key {k} to be in group {group_key} in `state_dict` but is missing."
                    )
                if k != "params":
                    new_group[k] = deepcopy(src_group[k])

        self._optimizer.load_state_dict(new_state_dict)

    def add_param_group(self, param_group: Mapping[str, Any]) -> None:
        """
        Add a param group to the :class:`_NamedOptimizer` s `param_groups`.

        Warning: This API is still in development and subject to change.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
        for param in param_group["params"]:
            if param not in param_to_key:
                raise ValueError("some parameters are not in the module")
            self.ordered_param_keys.append(param_to_key[param])

        self._optimizer.add_param_group(param_group)
        # Update param_groups from optimizer.
        self.param_groups = self._optimizer.param_groups

    def init_state(self) -> None:
        """
        Runs a dummy optimizer step, which allows to initialize optimizer state
        because we do lazy init for most optimizers.

        This allows doing in-place loading of optimizer state from a checkpoint.
        """
        for param in self.named_parameters.values():
            if param.requires_grad:
                t = torch.zeros_like(param)
                param.grad = torch.autograd.Variable(t)
        # Calling ``step`` will load the initial state for optimizer states.
        self.step(closure=None)

    def _pre_load_state_dict(self, state_dict) -> Dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        if isinstance(self.module, FSDP):
            return FSDP.optim_state_dict_to_load(
                self.module, self._optimizer, state_dict, is_named_optimizer=True
            )
        return state_dict

    def _post_state_dict(self, state_dict) -> Dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        if isinstance(self.module, FSDP):
            FSDP.optim_state_dict(self.module, self._optimizer, state_dict)
        return state_dict


def _gen_param_group_key(param_keys: List[str]) -> str:
    """
    Concatenate all param keys as a unique indentifier for one param group.
    """
    return "/".join(sorted(param_keys))
