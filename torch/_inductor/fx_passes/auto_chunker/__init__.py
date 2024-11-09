import itertools
import functools
from typing import Sequence, Set, Optional
import os

import torch
from torch._inductor import config, metrics
from torch._inductor.utils import cache_on_self
from torch.utils import _pytree
import operator
from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
import logging
import dataclasses
from torch.utils._ordered_set import OrderedSet

from .collector import get_args_of_node_type
from .collector import Collector, CantChunk
from .partitioner import Partitioner
from .propagator import Propagator, format_node_with_chunking_meta
from .collector import get_fake_tensor_from_node, maybe_permuted, get_tangent_nodes
from .applier import ChunkingApplier

aten = torch.ops.aten
prims = torch.ops.prims
log = logging.getLogger(__name__)

class AutoChunker:
    @staticmethod
    def chunk(gm):
        """
        Chunk input tensor for operations that amplifies the tensor size significantly.
        Only chunk across the batch dimension of the tensor.
        """
        graph = gm.graph

        if gm.meta.get("produced_by_chunker", False):
            # Don't chunk a graph produced by the chunker
            return gm

        if len(get_tangent_nodes(gm.graph)) == 0:
            # no tangents. Can be the optimizer graph
            # skip
            return gm

        log.debug("Joint graph before chunking:\n%s", gm.print_readable(False))

        chunking_subgraph_nodes = Collector.collect_chunking_subgraph_nodes(graph)
        print("Chunking subgraph nodes:")
        for node in chunking_subgraph_nodes:
            print(f"  {node.format_node()}")

        chunking_subgraph = Partitioner.reorder_nodes(graph, chunking_subgraph_nodes)

        Propagator.add_chunking_meta(chunking_subgraph)

        newgm = ChunkingApplier(gm, chunking_subgraph, config.AutoChunker.num_chunk or 8).apply()

        metrics.num_auto_chunking += 1
        return newgm
