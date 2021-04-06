import enum

import torch
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.fx.quantize import is_activation_post_process

from typing import Optional, Any

# TODO(future PR): delete this after FX has a util for it
def print_node(node: Optional[Node]) -> None:
    if node is None:
        print(None)
    else:
        print(
            node, ', target:', node.target, ', op:', node.op,
            ', args:', node.args, ', kwargs:', node.kwargs)

def getattr_from_fqn(gm: GraphModule, fqn: str) -> Any:
    """
    Given a gm and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    fqn_parts = fqn.split(".")
    cur_val = gm
    for part in fqn_parts:
        cur_val = getattr(cur_val, part)
    return cur_val

class NodeInputType(enum.Enum):
    FP32 = enum.auto()  # first input fp32
    INT8 = enum.auto()  # first input int8
    # TODO(future PRs): dynamic quant, fake quant, etc


def get_node_input_type(node: Node, gm: GraphModule) -> NodeInputType:
    if node.op == 'call_function':
        fp32_fun_target_names = ('torch.nn.functional', 'torch.nn')
        # hack alert: this is not ready for production
        # TODO(future PR): use a real mapping
        fp32_funs = (torch.cat,)
        int8_fun_target_names = ('torch._ops.quantized',)
        # For now, hacky check to see which op is in which namespace
        # TODO(future PR): use a real mapping
        if node.target.__module__ in fp32_fun_target_names or node.target in fp32_funs:
            return NodeInputType.FP32
        else:
            assert node.target.__module__ in int8_fun_target_names, \
                'unknown node target %s with module %s' % (node.target, node.target.__module__)
            return NodeInputType.INT8
    else:
        assert node.op == 'call_module'
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        # For now, hacky check to see which mod is in which namespace
        # TODO(future PR): use a real mapping
        if mod.__module__.startswith('torch.nn.modules'):
            return NodeInputType.FP32
        else:
            assert mod.__module__.startswith('torch.nn.q'), \
                'unknown node target %s' % mod
            return NodeInputType.INT8

def return_first_non_observer_node(
    node: Node,
    gm: GraphModule,
) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    node_obj = getattr_from_fqn(gm, node.target)  # type: ignore
    if is_activation_post_process(node_obj):
        assert len(node.args) == 1
        assert isinstance(node.args[0], Node)
        node = node.args[0]
        # code duplication intended, not worth refactoring
        assert isinstance(node.target, str)
        node_obj = getattr_from_fqn(gm, node.target)
        if is_activation_post_process(node_obj):
            assert len(node.args) == 1
            assert isinstance(node.args[0], Node)
            node = node.args[0]
    return node
