# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict
from typing import Dict, List, Set

import torch
from torch.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _outline_submodules(orig_graph: torch.fx.Graph):
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, List[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: Dict[str, Set[str]] = defaultdict(set)
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        None,
        [("", 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
