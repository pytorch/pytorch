import inspect
import json
import os
from typing import Any, Tuple, Callable, Union, Dict

import torch
import torch.fx
from torch.fx.experimental.graph_manipulation import (
    serialize_module,
)
from torch.fx.graph_module import GraphModule
from torch.fx.passes import graph_drawer
from torch.fx.passes.shape_prop import TensorMetadata


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


def get_field_from_acc_out_ty(
    acc_out_ty_or_dict: Union[Tuple, Dict[str, Any]], field: str
):
    """
    After tracing NamedTuple inputs are converted to standard tuples, so we cannot
    access them by name directly. Use this helper instead.
    """
    if isinstance(acc_out_ty_or_dict, dict):
        acc_out_ty = acc_out_ty_or_dict["acc_out_ty"]
    else:
        acc_out_ty = acc_out_ty_or_dict
    return acc_out_ty[TensorMetadata._fields.index(field)]


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
    getattr(x, "write_" + ext.lstrip("."))(fname)
