from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.distributed._spmd.distribute import _convert_to_distributed, Schema
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard

from torch.fx import GraphModule


class ParallelMode(ABC):
    """
    Basic Parallel Mode interface. Each parallelism pattern should implement
    this interface to describe how to partition and compile the graph in the
    spmd compiler.
    """

    @abstractmethod
    def partition(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        """
        Partition a single device graph to a distributed graph.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform a distributed graph with a set of graph transformation
        and optimization passes for each parallel mode. The returned result
        should be a parallel executable graph.
        """
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError()


class DTensorFallbackMode(ParallelMode):
    """
    The DTensor fallback parallel mode. It's replicating the parameters
    and shard the inputs to represent DDP like behavior, it's currently
    a transitent mode before we move to the new data parallel expansion.
    """

    def __init__(self, custom_passes: Callable[[GraphModule], GraphModule] = None):
        self._placements_override: Dict[int, List[Placement]] = {}
        self._gm_passes: Callable = custom_passes

    def partition(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        flat_args, _ = pytree.tree_flatten(list(args) + list(kwargs.values()))

        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()).cuda())
        shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
        # FIXME: allow other sharding schemas
        replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])

        inps, schemas = [], []

        for p in pytree.tree_flatten(params_and_buffers)[0]:
            assert isinstance(p, torch.Tensor), f"expecting Tensor but got {type(p)}"
            inps.append(p)
            schemas.append(replicate_schema)

        for o in pytree.tree_flatten(named_states)[0]:
            if isinstance(o, torch.Tensor):
                inps.append(o)
                schemas.append(replicate_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(replicate_schema)

        for a in flat_args:
            if isinstance(a, torch.Tensor):
                inps.append(a)
                if id(a) in self._placements_override:
                    schemas.append(
                        Schema(mesh=mesh, placements=self._placements_override[id(a)])
                    )
                else:
                    schemas.append(shard_schema)
            else:
                # Create dummy tensor and schema for non-tensor inputs for
                # the purpose of dtensor expansion. Non-tensor inputs are
                # guaranteed unused in dispatcher graphs produced by make_fx.
                # However, we still need to respect them so that tensor inputs
                # match wtih their placeholders.
                inps.append(torch.empty(0))
                schemas.append(shard_schema)

        return _convert_to_distributed(gm, inps, schemas, _allow_partial=False)

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform a distributed graph with a set of graph transformation
        and optimization passes for the dtensor fallback parallel mode.
        """
        raise NotImplementedError()
