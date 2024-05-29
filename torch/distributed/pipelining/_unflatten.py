# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict

import torch
from torch.export.unflatten import _ModuleFrame


def _outline_submodules(orig_graph: torch.fx.Graph):
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        None,
        [""],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
