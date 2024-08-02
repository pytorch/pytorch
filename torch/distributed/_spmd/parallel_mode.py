from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.distributed._spmd.data_parallel import (
    DataParallelStyle,
    partition_data_parallel,
)
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
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        """
        Partition a single device graph to a distributed graph.

        TODO(@wanchaol): some of these arguments are not necessary for
        partitioning, remove the unnecessary ones later.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph
        transformation and optimization passes for each parallel mode.

        The returned result should be a compiled executable graph in
        the distributed environment.
        """
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError


class DataParallel(ParallelMode):
    """Data Parallelism mode."""

    def __init__(
        self,
        parallel_style: str = "replicate",
        *,
        input_batch_dim: int = 0,
        custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None,
    ):
        """
        DataParallel Mode that partition the model and graph to data parallel style
        parallelism (i.e. DDP/FSDP/ZERO-3). It currently supports three different
        parallel styles: "replicate", "fully_shard", and "default". See
        :class:`DataParallelStyle` for more details.

        Args:
            parallel_style (str): parallel style to use. Currently supports
                "replicate", "fully_shard", and "default".

        Keyword args:
            input_batch_dim (int): the batch dimension of the input tensor.
                 default: 0
            custom_passes (Callable[[GraphModule], GraphModule], optional):
                A custom callable that overrides the default graph transformation
                and optimization passes.
        """
        if parallel_style == "replicate":
            self.parallel_style = DataParallelStyle.REPLICATE
        elif parallel_style == "fully_shard":
            self.parallel_style = DataParallelStyle.FULLY_SHARD
        elif parallel_style == "default":
            self.parallel_style = DataParallelStyle.DEFAULT
        else:
            raise RuntimeError(f"Unknown parallel style: {parallel_style}")

        # TODO: what if user passes in a incorrect `input_batch_dim`, how should we
        # detect that and do proper error handling?
        self.input_batch_dim = input_batch_dim

        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # TODO: add a few default passes here.
            self._gm_passes = lambda gm: gm

    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        # TODO: figure out a way to avoid explicit "cuda" mesh.
        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))

        gm = partition_data_parallel(
            gm,
            model,
            optimizer,
            params_and_buffers,
            named_states,
            args,
            kwargs,
            mesh,
            self.parallel_style,
            self.input_batch_dim,
        )
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """optimize a distributed graph with a set of optimization passes"""
        # TODO: add more necessary arguments to this interface.
        return self._gm_passes(gm)


class DTensorExpandMode(ParallelMode):
    """
    The DTensor Expand mode. It's replicating the parameters and
    shard the inputs to represent DDP like behavior, it's currently
    a transitent mode before we move to the new data parallel expansion.
    """

    def __init__(
        self, custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None
    ):
        self._placements_override: Dict[int, List[Placement]] = {}
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # TODO: add a few default passes here.
            self._gm_passes = lambda gm: gm

    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()).cuda())
        shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
        # FIXME: allow other sharding schemas
        replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])

        inps, schemas = [], []

        for p in pytree.tree_leaves(params_and_buffers):
            assert isinstance(p, torch.Tensor), f"expecting Tensor but got {type(p)}"
            inps.append(p)
            schemas.append(replicate_schema)

        for o in pytree.tree_leaves(named_states):
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

        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inps = [torch.empty_like(inp) for inp in inps]

        return _convert_to_distributed(
            gm, fake_inps, schemas, default_mesh=mesh, _allow_partial=False
        )[0]

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph transformation
        and optimization passes for the dtensor fallback parallel mode.
        """
        # TODO: move the trasnformation passed to this function
        return self._gm_passes(gm)
