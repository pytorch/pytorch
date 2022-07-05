import torch
import torch.fx
from torch.fx import Node

from torch.ao.ns.fx.utils import (
    get_target_type_str,
)
from torch.ao.ns.fx.ns_types import (
    NSSingleResultValuesType,
)
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import MatchResult

import copy
from typing import List, Dict, Tuple
import operator

SHADOW_NODE_NAME_PREFIX = 'shadow'
SHADOW_WRAPPER_NODE_NAME_PREFIX = 'shadow_wrapper'

# TODO: reuse existing mapping instead of creating a new one
BINARY_FUNCTIONS = {
    torch.add,
    torch.Tensor.add,
    operator.add,
    torch.mul,
    torch.Tensor.mul,
    operator.mul,
}

def _get_attr_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"

def _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_WRAPPER_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"


class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the input flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            if isinstance(result, torch.Tensor):
                node.traced_result = result

            env[node.name] = result

        return None

def _get_dedup_subgraphs(
    matches: Dict[str, MatchResult]
) -> Dict[str, List[Node]]:
    # the original matches variable is unique by node, make it unique by subgraph
    # instead
    seen_nodes = set()
    subgraphs_dedup = {}
    for name, cur_match in reversed(matches.items()):
        was_seen = False
        for node_or_tuple in cur_match[1]:

            # Cur_match[1] has an unusual type. It says that it's a `List[Node]`,
            # but it is really not. Furthermore, the contents of this field
            # can change from match results of multiple nodes of the same pattern
            #
            # For example, for conv -> bn -> relu, we see
            # match_results = {
            #   'conv': (relu, [(bn, conv), relu], ...),
            #   'bn': (relu, [(bn, conv), relu], ...),
            #   'relu': (relu, [(bn, conv), relu], ...),
            # }
            #
            # Ideally we should clean up the `find_matches` function to make
            # this more intuitive. For the purposes of this prototype, we hack
            # around it.

            if isinstance(node_or_tuple, Node):
                if node_or_tuple in seen_nodes:
                    was_seen = True
                seen_nodes.add(node_or_tuple)

            else:
                assert isinstance(node_or_tuple, tuple)
                assert isinstance(node_or_tuple[0], Node) and \
                    isinstance(node_or_tuple[1], Node)
                for node in node_or_tuple:
                    if node in seen_nodes:
                        was_seen = True
                    seen_nodes.add(node)

        if was_seen:
            continue

        # Start with the unusual type, convert it to [op_0, ..., op_n]
        list_of_nodes = None

        if len(cur_match[1]) == 1:
            list_of_nodes = cur_match[1]
        else:
            assert len(cur_match[1]) == 2
            # either (a, b), or ((a, b), c) or (c, (a, b))
            # cannot make any assumptions on order, not clear what the
            # find_matches function is doing to populate this

            def _order_nodes(node_a, node_b, node_c) -> Tuple[Node, Node, Node]:
                nodes = [node_a, node_b, node_c]
                first_node = None
                mid_node = None
                last_node = None
                for n in nodes:
                    prev_n = n.args[0]
                    next_n = list(n.users)[0]
                    if prev_n not in nodes:
                        first_node = n
                    elif next_n not in nodes:
                        last_node = n
                    else:
                        mid_node = n
                assert first_node is not None and mid_node is not None and \
                    last_node is not None
                assert mid_node.args[0] is first_node
                assert last_node.args[0] is mid_node
                return [last_node, mid_node, first_node]

            if isinstance(cur_match[1][0], Node) and isinstance(cur_match[1][1], Node):
                # (a, b)
                list_of_nodes = cur_match[1]
            elif isinstance(cur_match[1][0], tuple):
                # ((a, b), c)
                node_a, node_b = cur_match[1][0]
                node_c = cur_match[1][1]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
            elif isinstance(cur_match[1][1], tuple):
                # (a, (b, c))
                node_a, node_b = cur_match[1][1]
                node_c = cur_match[1][0]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)

        subgraphs_dedup[name] = list_of_nodes
        # [node_n, ..., node_0], note that the order is reversed
        # to make it chronological for simple subgraphs
        subgraphs_dedup[name].reverse()

    return subgraphs_dedup

def _get_logger_for_subgraph(
    model: torch.nn.Module,
    first_node: Node,
    last_node: Node,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    qconfig_str,
) -> torch.nn.Module:
    """
    Given a model and a linear subgraph starting from `first_node` and
    ending with `last_node`, creates a logger for the end of this
    subgraph.
    """
    logger_mod_orig = torch.ao.ns._numeric_suite_fx.OutputLogger(
        first_node.name,  # ref_node_name
        last_node.name,  # prev_node_name
        f'subgraph_{subgraph_idx}_{subgraph_candidate_idx}',  # model_name
        'model',  # ref_name
        get_target_type_str(last_node, model),  # prev_node_target_type
        get_target_type_str(first_node, model),  # ref_node_target_type
        NSSingleResultValuesType.NODE_OUTPUT.value,  # results_type
        0,  # index_within_arg
        0,  # index_of_arg
        '',  # fqn (not supported for now)
        qconfig_str,
    )
    logger_mod_orig.enabled = False
    return logger_mod_orig

def _add_logger_to_subgraph_wrapper(
    model: torch.nn.Module,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    qconfig_str: str,
) -> None:
    """
    Given a model which consists of a subgraph and nothing else, adds a logger
    to the end of this model.
    """
    first_node, last_node = None, None
    for idx, node in enumerate(model.graph.nodes):
        if idx == 0:
            first_node = node
        elif idx == len(model.graph.nodes) - 1:
            # last node is the output, so we want the first
            # arg of the output
            last_node = node.args[0]
    logger_mod = _get_logger_for_subgraph(
        model, first_node, last_node, subgraph_idx,
        subgraph_candidate_idx, qconfig_str)
    attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
    assert not hasattr(model, attr_name)
    setattr(model, attr_name, logger_mod)
    with model.graph.inserting_after(last_node):
        new_node = model.graph.call_module(
            attr_name, args=(last_node,), kwargs={})
    model.recompile()

def create_submodule_from_subgraph(
    model: torch.nn.Module,
    first_node: Node,
    last_node: Node,
) -> torch.nn.Module:
    """
    Input: a model, and a linear subgraph within the model from first_node to
      last_node.

    Output: a new submodule containing a copy of the subgraph, with the inputs
      to the first node becoming the inputs to the submodule, and all other
      nodes in the subgraph being copied.

    Example input - imagine module with graph:

      x0 -> op1 -> x1 -> op2 -> x2
             |            |
            arg1         arg2

    And first_node is op1 and last_node is op2.  Then, this function will create

      x0 -> subgraph_copy_module -> x2

    With the inside of subgraph_copy_module being

      input1 -> op1_copy -> x1 -> op2_copy -> output1
                   |                 |
                arg1_copy         arg2_copy

    """
    # TODO: handle kwargs

    #
    # create a blank GraphModule with an empty graph
    #

    class M(torch.nn.Module):
        def forward(self, x):
            pass

    m = M()
    gm = torch.fx.symbolic_trace(m)
    g = gm.graph
    for node in reversed(gm.graph.nodes):
        g.erase_node(node)

    #
    # modify the graph to have a copy of our subgraph
    #

    cur_node_orig = first_node
    cur_args_orig = cur_node_orig.args
    cur_kwargs_orig = cur_node_orig.kwargs

    cur_name_idx = 0

    iteration_limit = 100
    cur_iteration = 0

    while True:
        if cur_node_orig is first_node:
            # we are at the first node, we need to set up graph inputs
            # TODO(future): some graphs could have placeholders which are unrelated
            # to the first node, need to handle this
            cur_args_copy = []
            cur_kwargs_copy = {}
            seen_names = set()
            # args up to num_passthrough_args are passed in from module
            # input, other args are copied
            if cur_node_orig.target in BINARY_FUNCTIONS:
                num_passthrough = 2
            else:
                num_passthrough = 1

            arg_kwarg_idx = 0

            for arg in cur_node_orig.args:
                # note: for graphs starting with patterns such as `y = x + x`, we
                # need to ensure we do not add multiple placeholders with the
                # same name
                if isinstance(arg, Node) and arg_kwarg_idx < num_passthrough:
                    counter = 0
                    while arg.name + '_' + str(counter) in seen_names:
                        counter += 1
                    cur_name = arg.name + '_' + str(counter)
                    seen_names.add(cur_name)
                    cur_args_copy.append(g.placeholder(cur_name))
                elif isinstance(arg, Node):
                    # arg_kwarg_idx >= num_passthrough args, we need to copy
                    assert arg.op == 'get_attr', f'{arg.op} not handled yet'
                    new_attr_name = arg.name
                    obj = getattr_from_fqn(model, arg.target)
                    # wrap in Parameter to silence a warning in torch/fx/graph.py
                    obj_copy = torch.nn.Parameter(obj.clone().detach())
                    setattr(gm, new_attr_name, obj_copy)
                    cur_args_copy.append(g.get_attr(new_attr_name))
                else:
                    cur_args_copy.append(arg)

                arg_kwarg_idx += 1

            for kwarg_name, kwarg in cur_node_orig.kwargs.items():
                # TODO: dedup code with above
                # note: for graphs starting with patterns such as `y = x + x`, we
                # need to ensure we do not add multiple placeholders with the
                # same name
                if isinstance(kwarg, Node) and arg_kwarg_idx < num_passthrough:
                    counter = 0
                    while kwarg.name + '_' + str(counter) in seen_names:
                        counter += 1
                    cur_name = kwarg.name + '_' + str(counter)
                    seen_names.add(cur_name)
                    cur_kwargs_copy[kwarg_name] = g.placeholder(cur_name)
                elif isinstance(kwarg, Node):
                    # arg_kwarg_idx >= num_passthrough args, we need to copy
                    assert kwarg.op == 'get_attr', f'{kwarg.op} not handled yet'
                    new_attr_name = kwarg.name
                    obj = getattr_from_fqn(model, kwarg.target)
                    # wrap in Parameter to silence a warning in torch/fx/graph.py
                    obj_copy = torch.nn.Parameter(obj.clone().detach())
                    setattr(gm, new_attr_name, obj_copy)
                    cur_kwargs_copy[kwarg_name] = g.get_attr(new_attr_name)
                else:
                    cur_kwargs_copy[kwarg_name] = kwarg

                arg_kwarg_idx += 1

            cur_args_copy = tuple(cur_args_copy)
        else:
            # we are not at first node, first arg is from the previous node,
            # and all other args are copied

            # the current implementation is simplistic and cannot handle
            # ops with two or more arguments which need to be passed from
            # the previous op, so we assert them out
            assert cur_node_orig.target not in BINARY_FUNCTIONS

            # at this point in the code, cur_node_copy is pointing to the copy
            # of the previous node
            # TODO(future): this is not handling complicated graphs correctly, need to
            # look at actual relationships instead of assuming sequential graph
            # print(cur_args_orig, cur_kwargs_orig)
            cur_args_copy = [cur_node_copy]

            if len(cur_node_orig.args) > 1:
                for arg in cur_node_orig.args[1:]:
                    if isinstance(arg, nn.Parameter):
                        new_arg = arg.clone().detach()
                        mod_name = f"mod_{cur_name_idx}"
                        cur_name_idx += 1
                        setattr(gm, mod_name, new_arg)
                        new_arg_placeholder = gm.placeholder(mod_name)
                        cur_args_copy.append(new_arg_placeholder)
                    elif isinstance(arg, (float, int, torch.dtype)):
                        cur_args_copy.append(arg)
                    else:
                        raise AssertionError(f'arg of type {type(arg)} not handled yet')
            cur_args_copy = tuple(cur_args_copy)

        # copy the node
        if cur_node_orig.op == 'call_module':
            orig_mod = getattr_from_fqn(model, cur_node_orig.target)
            orig_mod_copy = copy.deepcopy(orig_mod)
            mod_name = f"mod_{cur_name_idx}"
            setattr(gm, mod_name, orig_mod_copy)
            cur_name_idx += 1
            cur_node_copy = g.call_module(mod_name, cur_args_copy, cur_kwargs_copy)

        elif cur_node_orig.op == 'call_function':
            cur_node_copy = g.call_function(
                cur_node_orig.target, cur_args_copy, cur_kwargs_copy)

        elif cur_node_orig.op == 'call_method':
            cur_node_copy = g.call_method(
                cur_node_orig.target, cur_args_copy, cur_kwargs_copy)

        else:
            raise AssertionError(f'{cur_node_orig.op} not supported yet')

        if cur_node_orig is last_node:
            break

        # go to next node
        assert len(cur_node_orig.users.keys()) == 1, \
            f'{cur_node} has more than 1 users, not supported yet'
        cur_node_orig = list(cur_node_orig.users.keys())[0]
        cur_args_orig = cur_node_orig.args
        cur_kwargs_orig = cur_node_orig.kwargs

        cur_iteration += 1
        if cur_iteration > iteration_limit:
            raise AssertionError('iteration limit exceeded')

    # set up outputs
    g.output(cur_node_copy)

    gm.recompile()
    return gm
