from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.distributed._shard.sharding_spec import ShardingSpec

@dataclass
class ShardingPlan(object):
    """
    Representation of a sharding plan, describes how to shard a module
    across hosts.

    Args:
        plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`]):
            a dict describes how to shard the parameters of a module, keyed by the name
            of parameter to ShardingSpec.
        output_plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`), optional):
            a dict specifies the layout of a module's output which produces a ShardedTensor,
            keyed by the name of module to ShardingSpec("" in key means the root module).
            If specified, outputs are resharded according to the provided sharding specs.
            Default: `None`
        collect_local_shards (List[str], optional): a list of string, each element enables
            a module's sharded output to be collected as a Tensor from its local shards to
            ensure further processsing in a data parallel fashion.
            Default: None
    Example::

        >>> sharding_plan = ShardingPlan(
        >>>    plan={
        >>>        "fc1.weight": spec1,
        >>>        "fc2.weight": spec2
        >>>    },
        >>>    output_plan={
        >>>        "": reshard_spec
        >>>    },
        >>>    collect_local_shards=[""]
        >>> )
    """
    plan: Dict[str, ShardingSpec]
    output_plan: Optional[Dict[str, ShardingSpec]] = None
    collect_local_shards: Optional[List[str]] = None
