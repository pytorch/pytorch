from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._spmd.data_parallel import (
    DataParallelStyle,
    partition_data_parallel,
)
from torch.distributed._tensor import DeviceMesh
from torch.fx import GraphModule


class ParallelMode(ABC):
    """Basic Parallel Mode interface."""

    @abstractmethod
    def partition(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        """expand a single device graph to a distributed graph."""
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, gm: GraphModule) -> GraphModule:
        """optimize a distributed graph with a set of optimization passes"""
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError()

    def configure_optimization_passes(self, _: List[Callable]) -> None:
        """a way to configure optimization passes per parallel mode"""
        raise NotImplementedError()


class DataParallel(ParallelMode):
    """Data Parallelism mode."""

    def __init__(self, parallel_style: str = "default"):
        if parallel_style == "replicate":
            self.parallel_style = DataParallelStyle.REPLICATE
        elif parallel_style == "fully_shard":
            self.parallel_style = DataParallelStyle.FULLY_SHARD
        elif parallel_style == "default":
            self.parallel_style = DataParallelStyle.DEFAULT
        else:
            raise RuntimeError(f"Unknown parallel style: {parallel_style}")

    def partition(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))
        gm = partition_data_parallel(
            gm,
            params_and_buffers,
            named_states,
            args,
            kwargs,
            mesh,
            self.parallel_style,
        )
        return gm

    def optimize(self, gm: GraphModule) -> GraphModule:
        """optimize a distributed graph with a set of optimization passes"""
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError()
