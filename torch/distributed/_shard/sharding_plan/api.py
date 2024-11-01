import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch.nn as nn
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_spec import ShardingSpec


@dataclass
class ShardingPlan:
    """
    Representation of a sharding plan, describes how to shard a module
    across hosts. `plan` is used to shard module parameters according to the spec provided,
    `output_plan` and `return_local_tensor` are optional, they are used to specify the output
    layout of a module with a spec, and when to convert back to data parallel fashion.

    Args:
        plan (Dict[str, Union[:class:`torch.distributed._shard.sharding_spec.ShardingSpec`,
              :class:`torch.distributed._shard.sharder.Sharder`]):
            a dict describes how to shard a module, there're currently two ways to shard a module:
                1. directly shard a module parameter by a `ShardingSpec`, keyed by the name of
                   a parameter to a `ShardingSpec`.
                2. shard a submodule by applying a `Sharder` on it, keyed by the name of a module
                   to a `Sharder` object.
        output_plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`), optional):
            a dict specifies the layout of a module's output which produces a ShardedTensor,
            keyed by the name of module to ShardingSpec("" in key means the root module).
            Default: `None`
        return_local_tensor (List[str], optional): a list of string, each element enables
            a module's sharded output to be returned as a Tensor from its local shards to
            ensure further processing in a data parallel fashion. ("" in list means the
            root module).
            Default: None
    Example:
      Suppose we want to shard a module with two linear layers and then run it with DDP, we also
      want to convert the output of the second linear layer back to DDP, we can do it as follows:

        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> class MyModule(nn.Module):
        >>>     def __init__(self) -> None:
        >>>        super().__init__()
        >>>        self.fc1 = nn.Linear()
        >>>        self.gelu = nn.GELU()
        >>>        self.fc2 = nn.Linear()
        >>>        self.relu = nn.Linear()
        >>>
        >>>     def forward(self, input):
        >>>         return self.relu(self.fc2(self.gelu(self.fc1(input))))


        >>> # xdoctest: +SKIP("Undefined spec1, spec2)
        >>> sharding_plan = ShardingPlan(
        >>>    plan={
        >>>        "fc1.weight": spec1,
        >>>        "fc2.weight": spec2
        >>>    },
        >>>    output_plan={
        >>>        "fc2": output_spec
        >>>    },
        >>>    return_local_tensor=["fc2"]
        >>> )
    """

    plan: Dict[str, Union[ShardingSpec, Sharder]]
    output_plan: Optional[Dict[str, ShardingSpec]] = None
    return_local_tensor: Optional[List[str]] = None


class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """

    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        """
        Given a nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """
