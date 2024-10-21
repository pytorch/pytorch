# mypy: allow-untyped-defs
import torch

from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator

from typing import Dict, List, Tuple


def split_result_tensors(
    result: torch.Tensor, inputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """
    A free function for use in the merge_matmul graph transformation below that
    splits the output from a merged matmul into the individual results for each
    input tensor.

    Arguments:
        result: The merged matmul result tensor.
        inputs: The list of inputs that were merged into one for the matmul.

    Returns:
        List of matmul results for each input tensor.
    """
    # When fx tracer is running, x.shape[0] will be torch.fx.Attribute but we
    # need an int even when tracing
    if isinstance(result, torch.fx.Proxy):
        splits = [0] * len(inputs)
    else:
        splits = [x.shape[0] for x in inputs]

    return torch.split(result, splits)


def may_depend_on(a: Node, b: Node, search_depth: int = 6):
    """
    Determine if one node depends on another in a torch.fx.Graph.

    Arguments:
        a: The node that may have a dependency on b.
        b: The node that a may have a dependency on.
        search_depth: In the case of an indirect dependency, this function
                        searches upto this many nodes away in search of a
                        data dependency. If none is found, the function
                        makes the conservative assumption that there is a
                        dependency.

    Returns:
        True if a may depend on b, False if it definitely does not.
    """
    # Equivalence is defined as dependence.
    if a == b:
        return True

    # If a has no inputs, it cannot depend on b.
    if len(a.all_input_nodes) == 0:
        return False

    # If the search depth has been exhausted and no conclusion has been
    # reached, assume that there is a data dependency.
    if search_depth == 0:
        return True

    # Recursively check all inputs of a.
    for inp in a.all_input_nodes:
        if may_depend_on(inp, b, search_depth - 1):
            return True

    return False


def are_nodes_independent(nodes: List[Node]):
    """
    Check if all of the given nodes are pairwise-data independent.

    Arguments:
        nodes: The nodes to check for data dependencies.

    Returns:
        True if any pair in nodes has a data dependency.
    """
    # For each pair in nodes:
    for i, j in itertools.combinations(nodes, 2):
        if may_depend_on(i, j) or may_depend_on(j, i):
            return False

    return True


def merge_matmul(in_mod: torch.nn.Module):
    """
    A graph transformation that merges matrix multiplication operations that share the same right-hand
    side operand into one large matrix multiplication.
               ____      _________        _________
      ----    |    |    |         |     M|  A * C  |
    M| A  |  T| B  | * K|    C    | =    |---------|
      ---- ,  |    |    |         |     T|  B * C  |
       K       ----      ---------        ---------
                K            R                R
    """
    gm = symbolic_trace(in_mod)

    rhs_users: Dict[Node, List[Node]] = {}
    lhs_users: Dict[Node, List[Node]] = {}

    # Populate rhs_users and lhs_users - maps from LHS/RHS matrix multiply operands to
    # the matmul of which they are the LHS/RHS.
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not torch.matmul:
            continue

        lhs, rhs = node.args

        # TODO: Properly handle aliasing caused by get_attr. For now,
        # use the attribute name as the operand if the node is a
        # get_attr.
        lhs = lhs.target if lhs.op == "get_attr" else lhs
        rhs = rhs.target if rhs.op == "get_attr" else rhs

        lhs_users.setdefault(lhs, []).append(node)
        rhs_users.setdefault(rhs, []).append(node)

    for rhs, mms in rhs_users.items():
        # There must be at least matmuls for a merge to make sense.
        if len(mms) < 2:
            continue

        # All matmuls must not depend on each other directly or indirectly
        # in order for the merge to be possible.
        if not are_nodes_independent(mms):
            continue

        lhs_vals = [mm.args[0] for mm in mms]

        # Merge the matmul.
        # Collect a list of LHS operands and the single RHS operand.
        lhs = [gm.graph.get_attr(l) if isinstance(l, str) else l for l in lhs_vals]
        rhs = gm.graph.get_attr(rhs) if isinstance(rhs, str) else rhs

        # Concatenate all the LHS operands.
        merge_mm_cat = gm.graph.call_function(torch.cat, (lhs,), {})

        # Multiply the concatenated LHS operands with the one RHS. This will produce
        # the same results as all the individual matmuls involving rhs in the original graph,
        # but they will all be concatenated together.
        merge_mm = gm.graph.call_function(torch.matmul, (merge_mm_cat, rhs,), {})

        # Split the result of the merged matmul using the shapes of the LHS operands
        # to ascertain how large each chunk should be.
        merge_mm_split = gm.graph.call_function(
            split_result_tensors, (merge_mm, lhs), {}
        )
        merge_mm_res = [
            gm.graph.call_function(operator.getitem, (merge_mm_split, out), {})
            for out in range(len(lhs))
        ]

        # Replace all uses of the original, unmerged matmuls with the equivalent split chunk from the merged matmul.
        for old, new in zip(mms, merge_mm_res):
            old.replace_all_uses_with(new)
            gm.graph.erase_node(old)

        # All of the new nodes created above were inserted at the end, so we need to sort
        # the nodes topologically to make sure all definitions precede uses.
        legalize_graph(gm)

    gm.recompile()
    gm.graph.lint()
    return gm
