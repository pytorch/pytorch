import inspect
import json
import os
from typing import Any, Tuple, Callable, Union, Dict, List, Optional
import re

import torch
import torch.fx
from torch.fx.passes.graph_manipulation import (
    serialize_module,
)
from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name
from torch.fx.passes import graph_drawer
from torch.fx.passes.shape_prop import TensorMetadata


def get_target_from_module(mod: torch.nn.Module, target: str):
    """
    Gets `target` from `mod` and returns it. If `target` is empty then returns `mod.`
    """
    if target == "":
        return mod

    target_atoms = target.split(".")
    curr_obj = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(curr_obj, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target '{'.'.join(target_atoms[:i])}'; "
                f" original whole target: '{target}'"
            )
        curr_obj = getattr(curr_obj, atom)
    return curr_obj


def get_attr(node: torch.fx.Node) -> Any:
    """
    Returns the underlying attr for a given node which
    must be of type get_attr.
    """
    assert node.op == "get_attr", "Expected a get_attr node"
    return get_target_from_module(node.graph.owning_module, str(node.target))


def is_acc_op(node_or_target: Union[Callable, torch.fx.Node]) -> bool:
    """
    Returns whether `node_or_target` is an acc_op. If it's a node, then checks whether
    it's a call_function target is from the acc_ops module. Otherwise it's already
    the target, which is similarly checked to see if it's from the acc_ops module.
    """
    if isinstance(node_or_target, torch.fx.Node):
        # All acc_ops are call_functions.
        if node_or_target.op != "call_function":
            return False
        target = node_or_target.target
    else:
        target = node_or_target
    return "acc_ops" in target.__module__


def is_acc_op_with_kwarg(
    node_or_target: Union[Callable, torch.fx.Node], kwarg: str
) -> bool:
    """
    Helper that inspects `node_or_target` and returns whether it is an acc_op node
    (or a target for an acc_op) that has an arg signature that includes `kwarg`.
    """
    if not is_acc_op(node_or_target):
        return False

    target = (
        node_or_target.target
        if isinstance(node_or_target, torch.fx.Node)
        else node_or_target
    )
    assert not isinstance(target, str)
    return kwarg in inspect.signature(inspect.unwrap(target)).parameters


def serialize_module_json_to_file(fx_module: GraphModule, fname: str):
    weights: Dict = {}
    serialized_json = json.dumps(serialize_module(fx_module, weights), indent=2)
    with open(fname, "w") as ofile:
        ofile.write(serialized_json)


def build_raw_tensor_meta(
    shape=None,
    dtype=None,
    requires_grad=None,
    stride=None,
    memory_format=None,
    is_quantized=None,
    qparams=None,
):
    return TensorMetadata(**locals())


def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = "fx_graph"):
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
    g = graph_drawer.FxGraphDrawer(traced, figname)
    x = g.get_main_dot_graph()
    try:
        getattr(x, "write_" + ext.lstrip("."))(fname)
    except OSError as e:
        print(f"Failed to write the FX graph due to: {e}")


def get_model_info_str(gm: torch.fx.GraphModule, header: Optional[str] = None):
    """
    Print out info of the provided `gm`.
    If `header` is provided then it's included in the printed string.
    """
    ops_and_counts: Dict[Callable, int] = dict()
    placeholder_count = get_attr_count = call_method_count = call_module_count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            ops_and_counts[node.target] = ops_and_counts.get(node.target, 0) + 1
        elif node.op == "placeholder":
            placeholder_count += 1
        elif node.op == "get_attr":
            get_attr_count += 1
        elif node.op == "call_method":
            call_method_count += 1
        elif node.op == "call_module":
            call_module_count += 1
        elif node.op == "output":
            output_count = len(node.args[0]) if isinstance(node.args[0], tuple) else 1
        else:
            raise RuntimeError(f"Unknown node found: {node.format_node()}")

    header = "" if header is None else f" [{header}]"
    model_info_str = f"Model Info{header}:\n"
    model_info_str += f"> placeholder: {placeholder_count}\n"
    model_info_str += f"> get_attr: {get_attr_count}\n"
    model_info_str += f"> output: {output_count}\n"
    if call_module_count != 0:
        model_info_str += f"> WARNING: call_module: {call_module_count}"
    if call_method_count != 0:
        model_info_str += f"> WARNING: call_method: {call_method_count}"

    # Sort and print all the other ops. Sort so it's deterministic between runs and
    # easier to parse.
    pretty_ops_and_counts: List[Tuple[str, int]] = []
    for op, count in ops_and_counts.items():
        pretty_ops_and_counts.append((_get_qualified_name(op), count))
    pretty_ops_and_counts.sort()
    for op_str, count in pretty_ops_and_counts:
        model_info_str += f"> {op_str}: {count}\n"

    return model_info_str


def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    """
    Make sure the name is unique (in a module) and can represents an attr.
    """
    # Delete all characters that are illegal in a Python identifier.
    name = re.sub("[^0-9a-zA-Z_]+", "_", name)
    if name[0].isdigit():
        name = f"_{name}"
    # Now make sure it is in fact unique to the module by incrementing suffix value.
    while hasattr(mod_traced, name):
        match = re.match(r"(.*)_(\d+)$", name)
        if match is None:
            name = name + "_1"
        else:
            base, num = match.group(1, 2)
            name = f"{base}_{int(num) + 1}"

    return name
