import itertools
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import fx

from .graph_utils import get_node_tensor_metadata


class DistributedGraph:
    def __init__(
        self,
        orig_module: Optional[nn.Module] = None,
    ) -> None:
        self.orig_module: Optional[nn.Module] = orig_module
        self.fwd_graph_modules: List[fx.GraphModule] = []
        self.bwd_graph_modules: List[fx.GraphModule] = []

        # Indicate `update()` must be called before applying any optimization.
        self._dirty = True

    def validate(self) -> None:
        return

    def update(self) -> "DistributedGraph":
        """
        Utility to put graph module into a node map for easier adjustments.
        """
        if not self._dirty:
            return self

        self.validate()
        return self
