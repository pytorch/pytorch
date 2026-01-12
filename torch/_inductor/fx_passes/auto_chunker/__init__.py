import dataclasses
import functools
import itertools
import logging
import operator
import os
from collections.abc import Sequence
from typing import Optional

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config, metrics
from torch._inductor.utils import cache_on_self
from torch.fx import GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree
from torch.utils._ordered_set import OrderedSet
from .applier import ChunkingApplier
from .common import CantChunk
from .core import find_amplifier_node
from .propagator import propagate
from .utils import get_tangent_nodes, tangent_has_chunking_meta


aten = torch.ops.aten
prims = torch.ops.prims
log = torch._logging.getArtifactLogger(__name__, "auto_chunker")


def decide_num_chunks(gm: GraphModule) -> int:
    # TODO: this is just a placeholder for now.
    # We should either create heuristics or do autotuning to
    # decide the number of chunks.
    return 8


def chunk(gm: GraphModule) -> GraphModule:
    """
    Chunk input tensors for operations that amplify the tensor size significantly.
    The chunking operation is propagated thru the fx graph until a point we should
    re-generate non-chunked tensors.

    Only chunk across the batch dimension of the tensor for now.
    """
    graph = gm.graph

    if torch._inductor.config.cpp_wrapper:
        raise CantChunk("cpp wrapper does not support codegening invoke_subgraph")

    if gm.meta.get("produced_by_chunker", False):
        # Don't chunk a graph produced by the chunker
        return gm

    if len(get_tangent_nodes(gm.graph)) == 0:
        # no tangents. Can be the optimizer graph. Skip chunking
        return gm

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Joint graph before chunking:\n%s", gm.print_readable(False))

    amplifier_node = find_amplifier_node(graph)
    if amplifier_node is None:
        raise CantChunk("Skip chunking due to no amplifier node found")

    if not all(isinstance(s, int) for s in amplifier_node.meta["val"].shape):
        raise CantChunk("Can't chunk due to dynamic shape")

    propagate(amplifier_node)
    if not tangent_has_chunking_meta(gm):
        raise CantChunk(
            "Skip chunking either because the graph is for inference only or "
            "because the chunking metadata does not propagate to the backward "
            "(e.g. due to too trivial loss function)"
        )

    num_chunks = config.auto_chunker.num_chunk or decide_num_chunks(gm)
    out_gm = ChunkingApplier(gm, num_chunks).apply()
    metrics.num_auto_chunking += 1
    log.debug("AutoChunker being applied with %s chunks", num_chunks)
    return out_gm
