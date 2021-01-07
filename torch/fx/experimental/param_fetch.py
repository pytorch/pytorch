from torch.fx.graph_module import GraphModule
from typing import Any, Callable, Dict, List, Tuple, Type
import torch
import torch.nn as nn


# Matching method matches the attribute name of current version to the attribute name of `target_version`
def default_matching(name: str, target_version: int) -> str:
    """Default matching method
    """
    return name

# This dict maps the nn.Module class name to the attribute name list that we want to fetch for lowering.
# The first integer in the tuple is the version number of the nn.Module class when we create the parameter list.
# If there's a version mismatch then it means the parameter names in the book might be mismatched with nn.Module.
module_fetch_book: Dict[Type, Tuple[int, List[str], Callable[[str, int], str]]] = {
    torch.nn.modules.linear.Linear: (1, ["weight", "bias"], default_matching),
    torch.nn.modules.conv.Conv2d: (
        1, ["weight", "bias", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"], default_matching
    ),
    torch.nn.modules.batchnorm.BatchNorm2d: (2, ["weight", "bias", "running_mean", "running_var", "eps"], default_matching),
    torch.nn.modules.pooling.AdaptiveAvgPool2d: (1, [], default_matching),
    torch.nn.modules.pooling.MaxPool2d: (
        1, ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"], default_matching
    ),
    torch.nn.modules.activation.ReLU: (1, ["inplace"], default_matching),
}

def extract_attrs_for_lowering(mod: nn.Module) -> Dict[str, Any]:
    """If `mod` is in `module_fetch_book`, fetch the mod's attributes that in the `module_fetch_book`
    after checking module's version is compatible with the `module_fetch_book`.
    """
    attrs_for_lowering: Dict[str, Any] = {}
    attrs_for_lowering["name"] = torch.typename(mod)

    if type(mod) in module_fetch_book:
        version, param_to_fetch, matching_method = module_fetch_book[type(mod)]
        if version < mod._version:
            raise RuntimeError(f"Fetcher version {version} try to fetch {torch.typename(mod)} version {mod._version}, "
                               "please upgrade the module_fetch_book, open an issue and @842974287 "
                               "or report a bug to AIACC team directly.")
        for attr in param_to_fetch:
            attrs_for_lowering[attr] = getattr(mod, matching_method(attr, mod._version))
    else:
        raise RuntimeError(f"{torch.typename(mod)} is not in the module_fetch_book yet, "
                           "please add it to the module_fetch_book, open an issue and @842974287 "
                           "or report a bug to AIACC team directly.")
    return attrs_for_lowering

def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None:
    """Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module.
    """
    submodules = dict(fx_module.named_modules())

    for node in fx_module.graph.nodes:
        if node.op == "call_module":
            if isinstance(submodules[node.target], GraphModule):
                lift_lowering_attrs_to_nodes(submodules[node.target])
            else:
                node.attrs_for_lowering = extract_attrs_for_lowering(submodules[node.target])
