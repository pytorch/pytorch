import abc
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict

from torch.distributed._shard.sharding_spec import ShardingSpec

@dataclass
class ShardingPlan(object):
    """
    Representation of a sharding plan.
    Dict keyed by FQN of parameter to PlacementSpec.
    """
    plan: Dict[str, ShardingSpec]


class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """
    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        pass
