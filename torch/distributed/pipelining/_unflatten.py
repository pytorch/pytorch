# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from collections import defaultdict

import torch
from torch.export.unflatten import _copy_graph_attrs, _ModuleFrame, _SubmoduleEntry


logger = logging.getLogger(__name__)


def _outline_submodules(submodule: torch.fx.GraphModule) -> torch.fx.GraphModule:
    orig_graph = submodule.graph
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()

    # move attributes that correspond to graph arguments for HOPs
    # from original submodule to unflattened submodule
    try:
        _copy_graph_attrs(submodule, new_module, seen_attrs)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"{seen_attrs=}")  # noqa: G004
        raise e

    new_module.graph.lint()
    new_module.recompile()
    return new_module
