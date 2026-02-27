# mypy: allow-untyped-defs
import collections
import copy
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.fx
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.ns.fx.ns_types import NSResultsType, NSSingleResultValuesType
from torch.ao.ns.fx.utils import (  # TODO(future PR): make this work correctly for methods
    get_normalized_nth_input,
    get_target_type_str,
)
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_map


SHADOW_NODE_NAME_PREFIX = "shadow"
SHADOW_WRAPPER_NODE_NAME_PREFIX = "shadow_wrapper"

# TODO(future PR): reuse existing mapping instead of creating a new one
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

    Given a GraphModule and an example input, saves the output flowing
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
        env: dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = fetch_attr(node.target)
            elif node.op == "call_function":
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs)
                )

            if isinstance(result, torch.Tensor):  # type: ignore[possibly-undefined]
                # pyrefly: ignore [unbound-name]
                node.traced_result = result

            # pyrefly: ignore [unbound-name]
            env[node.name] = result

        return None


def _get_dedup_subgraphs(matches: dict[str, _MatchResult]) -> dict[str, list[Node]]:
    # the original matches variable is unique by node, make it unique by subgraph
    # instead
    seen_nodes = set()
    subgraphs_dedup = {}

    # Dict items are not reversible until Python 3.8, so we hack it
    # to be compatible with previous Python versions
    # TODO(future PR): try reversed(list(matches.items()))
    matches_items_reversed: list[tuple[str, _MatchResult]] = list(
        reversed(matches.items())
    )

    # Note: the order is important.  `matches` currently provides the matches
    # in reverse order.  We would like to process the matches in non-reverse
    # order, so that we can create an intuitive naming scheme, such as
    # naming the first op's submodules `shadow_0_0` through `shadow_0_(n-1)`
    for name, cur_match in matches_items_reversed:  # type: ignore[call-overload]
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
                if not isinstance(node_or_tuple, tuple):
                    raise AssertionError(f"Expected tuple, got {type(node_or_tuple)}")
                for node in node_or_tuple:
                    if not isinstance(node, Node):
                        raise AssertionError(f"Expected Node, got {type(node)}")
                    if node in seen_nodes:
                        was_seen = True
                    seen_nodes.add(node)

        if was_seen:
            continue

        # Start with the unusual type, convert it to [op_0, ..., op_n]
        list_of_nodes = []

        if len(cur_match[1]) == 1:
            list_of_nodes = cur_match[1]
        else:
            if len(cur_match[1]) != 2:
                raise ValueError(
                    f"Expected cur_match[1] to have length 2, got {len(cur_match[1])}"
                )
            # either (a, b), or ((a, b), c) or (c, (a, b))
            # cannot make any assumptions on order, not clear what the
            # _find_matches function is doing to populate this
            # TODO(future PR): make this code less confusing,  see discussion
            # in https://github.com/pytorch/pytorch/pull/80521/files#r975918836

            def _order_nodes(node_a, node_b, node_c) -> list[Node]:
                nodes = [node_a, node_b, node_c]
                first_node = None
                mid_node = None
                last_node = None
                for n in nodes:
                    prev_n = n.args[0]
                    next_n = next(iter(n.users))
                    if prev_n not in nodes:
                        first_node = n
                    elif next_n not in nodes:
                        last_node = n
                    else:
                        mid_node = n
                if first_node is None or mid_node is None or last_node is None:
                    raise AssertionError("Expected all nodes to be non-None")
                if mid_node.args[0] is not first_node:
                    raise AssertionError("Expected mid_node.args[0] to be first_node")
                if last_node.args[0] is not mid_node:
                    raise AssertionError("Expected last_node.args[0] to be mid_node")
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

        # [node_n, ..., node_0], note that the order is reversed
        # to make it chronological for simple subgraphs
        list_of_nodes.reverse()
        subgraphs_dedup[name] = list_of_nodes

    return subgraphs_dedup


def _get_logger_for_subgraph(
    model: GraphModule,
    first_node: Node,
    last_node: Node,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    qconfig_str: str,
    logger_cls: Callable,
    fqn: str | None,
) -> torch.nn.Module:
    """
    Given a model and a linear subgraph starting from `first_node` and
    ending with `last_node`, creates a logger for the end of this
    subgraph.
    """
    if fqn is None:
        fqn = ""
    logger_mod_orig = logger_cls(
        first_node.name,  # ref_node_name
        last_node.name,  # prev_node_name
        f"subgraph_{subgraph_idx}_{subgraph_candidate_idx}",  # model_name
        "model",  # ref_name
        get_target_type_str(last_node, model),  # prev_node_target_type
        get_target_type_str(first_node, model),  # ref_node_target_type
        NSSingleResultValuesType.NODE_OUTPUT.value,  # results_type
        0,  # index_within_arg
        0,  # index_of_arg
        fqn,  # fqn
        qconfig_str,
    )
    # Usually we expect the user to add loggers, then calibrate, then convert,
    # and then populate loggers.  This is why the loggers start disabled.
    # TODO(future PR): reconsider the design to make this more intuitive.
    logger_mod_orig.enabled = False
    return logger_mod_orig


def create_submodule_from_subgraph(
    model: torch.nn.Module,
    first_node: Node,
    last_node: Node,
) -> GraphModule:
    """
    Input: a model, and a linear subgraph within the model from first_node to
      last_node.

    Output: a new submodule containing a copy of the subgraph, with the inputs
      to the first node becoming the inputs to the submodule, and all other
      nodes in the subgraph being copied.

    Example inputs:

    `model`: a module with graph

      x0 -> op1 -> x1 -> op2 -> x2
             |
            arg1

    `first_node`: op1
    `last_node`: op2

    Example output: a new module with graph

      input1 -> op1_copy -> x1 -> op2_copy -> output1
                   |
                  arg1
    """

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
            seen_names: set[str] = set()
            old_name_to_new_node: dict[str, Node] = {}

            def _add_placeholder(
                g: Graph, node: Node, seen_names, old_name_to_new_node
            ):
                # note: for graphs starting with patterns such as `y = x + x`, we
                # need to ensure we do not add multiple placeholders with the
                # same name
                counter = 0
                while node.name + "_" + str(counter) in seen_names:
                    counter += 1
                cur_name = node.name + "_" + str(counter)
                seen_names.add(cur_name)
                placeholder = g.placeholder(cur_name)
                old_name_to_new_node[node.name] = placeholder
                return placeholder

            for arg in cur_node_orig.args:
                if isinstance(arg, Node):
                    p = _add_placeholder(g, arg, seen_names, old_name_to_new_node)
                    cur_args_copy.append(p)
                elif isinstance(arg, (list, tuple)):
                    new_arg = []
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_arg.append(
                                _add_placeholder(
                                    g, inner_arg, seen_names, old_name_to_new_node
                                )
                            )
                        else:
                            new_arg.append(inner_arg)
                    cur_args_copy.append(new_arg)
                else:
                    cur_args_copy.append(arg)

            # TODO(future PR): handle non-normalized kwargs
            for kwarg_name, kwarg in cur_node_orig.kwargs.items():
                if isinstance(kwarg, Node):
                    cur_kwargs_copy[kwarg_name] = _add_placeholder(
                        g, kwarg, seen_names, old_name_to_new_node
                    )
                elif isinstance(kwarg, (list, tuple)):
                    new_kwarg = []
                    for inner_kwarg in kwarg:
                        p = _add_placeholder(
                            g,
                            inner_kwarg,  # type: ignore[arg-type]
                            seen_names,
                            old_name_to_new_node,
                        )
                        new_kwarg.append(p)
                    cur_kwargs_copy[kwarg_name] = new_kwarg
                else:
                    cur_kwargs_copy[kwarg_name] = kwarg

            cur_args_copy = tuple(cur_args_copy)  # type: ignore[assignment]
        else:
            # we are not at first node, first arg is from the previous node,
            # and all other args are copied

            # the current implementation is simplistic and cannot handle
            # ops with two or more arguments which need to be passed from
            # the previous op, so we assert them out
            if cur_node_orig.target in BINARY_FUNCTIONS:
                raise AssertionError(
                    f"Unexpected binary function target: {cur_node_orig.target}"
                )

            # at this point in the code, cur_node_copy is pointing to the copy
            # of the previous node
            # TODO(future PR): this is not handling complicated graphs correctly, need to
            # look at actual relationships instead of assuming sequential graph
            # TODO(future PR): this is ignoring kwargs, will need to support kwargs
            # for any fusion pattern which has them for a node that is not the
            # first node.
            cur_args_copy = [cur_node_copy]  # type: ignore[has-type, possibly-undefined]  # noqa: F821

            if len(cur_node_orig.args) > 1:
                for arg in cur_node_orig.args[1:]:
                    if isinstance(arg, torch.nn.Parameter):
                        new_arg = arg.detach().clone()  # type: ignore[assignment]
                        mod_name = f"mod_{cur_name_idx}"
                        cur_name_idx += 1
                        setattr(gm, mod_name, new_arg)
                        new_arg_placeholder = gm.placeholder(mod_name)  # type: ignore[operator]
                        cur_args_copy.append(new_arg_placeholder)
                    elif isinstance(arg, (float, int, torch.dtype)):
                        cur_args_copy.append(arg)
                    else:
                        raise AssertionError(f"arg of type {type(arg)} not handled yet")
            cur_args_copy = tuple(cur_args_copy)  # type: ignore[assignment]

        # copy the node
        if cur_node_orig.op == "call_module":
            orig_mod = getattr_from_fqn(model, cur_node_orig.target)  # type: ignore[arg-type]
            orig_mod_copy = copy.deepcopy(orig_mod)
            mod_name = f"mod_{cur_name_idx}"
            setattr(gm, mod_name, orig_mod_copy)
            cur_name_idx += 1
            cur_node_copy = g.call_module(mod_name, cur_args_copy, cur_kwargs_copy)  # type: ignore[possibly-undefined,arg-type]

        elif cur_node_orig.op == "call_function":
            cur_node_copy = g.call_function(
                cur_node_orig.target,  # type: ignore[arg-type]
                cur_args_copy,  # type: ignore[arg-type]
                cur_kwargs_copy,  # type: ignore[possibly-undefined]
            )

        elif cur_node_orig.op == "call_method":
            cur_node_copy = g.call_method(
                cur_node_orig.target,  # type: ignore[arg-type]
                cur_args_copy,  # type: ignore[arg-type]
                cur_kwargs_copy,  # type: ignore[possibly-undefined]
            )

        else:
            raise AssertionError(f"{cur_node_orig.op} not supported yet")

        if cur_node_orig is last_node:
            break

        # go to next node
        if len(cur_node_orig.users.keys()) != 1:
            raise AssertionError(
                f"{cur_node_orig} has more than 1 users, not supported yet"
            )
        cur_node_orig = next(iter(cur_node_orig.users.keys()))
        cur_iteration += 1
        if cur_iteration > iteration_limit:
            raise AssertionError("iteration limit exceeded")

    # set up outputs
    g.output(cur_node_copy)

    gm.recompile()
    return gm


def create_one_transformed_and_logged_copy_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    first_node: Node,
    last_node: Node,
    fqn: str | None,
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    example_inputs: Any,
    last_added_shadow_node_list: list[Node | None],
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Given a subgraph in `mt` and a subgraph candidate idx, inserts the
    subgraph candidate copy and instruments it with loggers.

    If subgraph_candidate_idx is 0, this is the baseline fp32 subgraph and we just
    add a logger to the end.

    If subgraph_candidate_idx is not 0, we create a copy of the subgraph and
    prepare it with `prepare_fx`.
    """

    # TODO(future PR): move logger classes to utils to remove circular dependency
    from torch.ao.ns._numeric_suite_fx import OutputComparisonLogger, OutputLogger

    if subgraph_candidate_idx == 0:
        # idx = 0 is the floating point (original) version of the subgraph
        # We keep the subgraph as is, and add a logger at the end

        qconfig_str = ""
        logger_mod_orig = _get_logger_for_subgraph(
            mt,
            first_node,
            last_node,
            subgraph_idx,
            subgraph_candidate_idx,
            qconfig_str,
            OutputLogger,
            fqn,
        )

        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(last_node):
            new_node = mt.graph.call_module(attr_name, args=(last_node,), kwargs={})
            last_added_shadow_node_list[0] = new_node

    else:
        # idx > 0 means we have a candidate qconfig to try, so we need
        # to make a copy of the subgraph, feed it with the right inputs,
        # and add a logger at the end

        # get the qconfig
        # subtract one because the first candidate is the floating point
        # version of the subgraph
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]

        # if no quantization is requested, skip
        # TODO(future PR): deduplicate equivalent qconfigs that come from
        #   different qconfig mapping objects
        if qconfig is None:
            return

        qconfig_mapping = QConfigMapping().set_global(qconfig)

        # create a copy of the submodule, wrapped in a separate module
        orig_mod_copy_wrapped = create_submodule_from_subgraph(
            mt, first_node, last_node
        )

        # add a call to prepare_fx on the wrapper module
        if custom_prepare_fn is None:
            orig_mod_copy_wrapped = torch.ao.quantization.quantize_fx.prepare_fx(
                orig_mod_copy_wrapped, qconfig_mapping, example_inputs=example_inputs
            )
        else:
            if custom_prepare_kwargs is None:
                custom_prepare_kwargs = {}
            for kwarg_name in [
                "example_inputs",
                "prepare_custom_config",
                "qconfig_mapping",
            ]:
                if kwarg_name in custom_prepare_kwargs:
                    raise AssertionError(
                        f"cannot specify {kwarg_name} in custom_prepare_kwargs"
                    )
            prepare_kwargs: dict[str, Any] = {
                "example_inputs": example_inputs,
                "qconfig_mapping": qconfig_mapping,
            }
            prepare_kwargs.update(custom_prepare_kwargs)
            orig_mod_copy_wrapped = custom_prepare_fn(
                orig_mod_copy_wrapped, **prepare_kwargs
            )

        # attach the wrapper to the model
        attr_name = _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, orig_mod_copy_wrapped)

        # add a call to the wrapper module from the parent graph
        insert_after_node = last_added_shadow_node_list[0]
        with mt.graph.inserting_after(insert_after_node):
            # TODO(future PR): handle fusion patterns where non-first nodes
            # need inputs

            # pass in all node args and kwargs

            new_args = []
            for arg in first_node.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
                elif (
                    isinstance(arg, (list, tuple))
                    and len(arg)
                    and isinstance(arg[0], Node)
                ):
                    new_args.extend(
                        inner_arg for inner_arg in arg if isinstance(inner_arg, Node)
                    )

            new_kwargs = {}
            for name, old_kwarg in first_node.kwargs.items():
                if isinstance(old_kwarg, Node):
                    new_kwargs[name] = old_kwarg
                elif isinstance(old_kwarg, (list, tuple)) and len(old_kwarg):
                    # TODO(future PR): clarify why we are adding kwargs to args
                    new_args.extend(old_kwarg)  # type: ignore[arg-type]

            new_args = tuple(new_args)  # type: ignore[assignment]

            new_node = mt.graph.call_module(attr_name, args=new_args, kwargs=new_kwargs)  # type: ignore[arg-type]

        # add a logger to parent graph to observe the shadow wrapper
        logger_mod_orig = _get_logger_for_subgraph(
            mt,
            first_node,
            last_node,
            subgraph_idx,
            subgraph_candidate_idx,
            str(qconfig),
            OutputComparisonLogger,
            fqn,
        )

        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(new_node):
            logger = mt.graph.call_module(
                attr_name, args=(new_node, last_node), kwargs={}
            )
            last_added_shadow_node_list[0] = logger

    mt.recompile()


def create_n_transformed_and_logged_copies_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    match_name: str,
    nodes_in_this_subgraph: list[Any],
    qconfig_mappings: list[QConfigMapping],
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Given a model `mt` and a subgraph_idx, creates the needed copies
    of the subgraph for all qconfigs, and instruments them with loggers.
    """
    # for now, assume that
    # 1. the first node has one input
    # 2. the last node has one output

    # for now, ignore all subgraphs that contain non-nodes (tuples, etc)
    # TODO(future PR): implement this
    if any(not isinstance(node, Node) for node in nodes_in_this_subgraph):
        return

    first_node = nodes_in_this_subgraph[0]
    last_node = nodes_in_this_subgraph[-1]
    # We used output propagation to populate example values on each
    # node. Use the example values from the previous node as the input
    # to the current node.
    prev_node = get_normalized_nth_input(first_node, mt, 0)
    if isinstance(prev_node, list):
        example_inputs = [x.traced_result for x in prev_node]
    elif isinstance(prev_node, tuple):
        example_inputs = (x.traced_result for x in prev_node)  # type: ignore[assignment]
    else:
        # currently some customer models do not have a traced_result in
        # every node, so we have to guard for this case since we cannot
        # quantize without an example input
        # TODO(future PR): add a test case for this once we have an easy
        # repro, see https://github.com/pytorch/pytorch/pull/80521/files#r975940489
        # for additional context
        if hasattr(prev_node, "traced_result"):
            example_inputs = (prev_node.traced_result,)  # type: ignore[attr-defined, assignment]
        else:
            print(
                "unable to get example input for node "
                + f"{first_node.format_node()}, skipping"
            )
            return

    # If there are no quantization configs for this subgraph, skip adding
    # loggers. This reduces memory usage for models where not all layers are
    # quantized.
    # TODO(future): consider making this configurable
    found_at_least_one_qconfig = False
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        if subgraph_candidate_idx == 0:
            # fp32 baseline does not need a qconfig
            continue

        # a. we have N shadows, so len(qconfig_mappings) is N
        # b. we will have the fp32 layer + N shadows, so overall number of
        #    (original_op) + (*shadows) will be N+1
        # c. since `subgraph_candidate_idx` represents (b), we need
        #    to subtract 1 to query from (a)
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is not None:
            found_at_least_one_qconfig = True
            break
    if not found_at_least_one_qconfig:
        print(
            "unable to find at least one qconfig for node "
            + f"{first_node.format_node()}, skipping"
        )
        return

    fqn = _maybe_get_fqn(first_node, mt)

    # We want the results to contain the subgraphs in natural order,
    # and the graph to also contain shadow wrappers and shadow loggers
    # in natural order.
    # If we just iterate in reverse, the graph will be in natural
    # order but the eventual results will be in reverse order.
    # So, we keep track of the last shadow logger we added and
    # always insert after it.
    last_added_shadow_node_list: list[Node | None] = [None]
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        create_one_transformed_and_logged_copy_of_subgraph(
            mt,
            subgraph_idx,
            subgraph_candidate_idx,
            first_node,
            last_node,
            fqn,
            list_of_node_name_to_qconfig,
            example_inputs,
            last_added_shadow_node_list,
            custom_prepare_fn,
            custom_prepare_kwargs,
        )


def create_add_loggers_graph(
    model: GraphModule,
    subgraphs_dedup: dict[str, list[Node]],
    qconfig_mapping: QConfigMapping,
    node_name_to_qconfig: dict[str, QConfigAny],
) -> None:
    r"""
    Given a model, a model graph partition (currently a set of matched
    subgraphs) and instructions how to transform each subgraph
    (currently quantizing it according to qconfig_mapping), modifies
    the model graph to create an alternate path through the original graph,
    with each of the subgraphs quantized.  This is useful to compare
    propagation error of a transformation such as quantization.

    For example, given layer op0 and op1, there are four cases when handling op1:
    1. op0 and op1 quantized
    2. op0 and op1 unquantized
    3. op0 quantized, op1 unquantized
    4. op0 unquantized, op1 quantized

    Example input, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \          \                 \       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog

    Example output, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \                           \        # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog

    """
    # TODO(future PR): move logger classes to utils to remove circular dependency
    from torch.ao.ns._numeric_suite_fx import OutputComparisonLogger, OutputLogger

    def _get_subgraph_containing_node(node, subgraphs_dedup):
        for subgraph in subgraphs_dedup.values():
            if node in subgraph:
                return subgraph
        return None

    # First, we need to create shadow branches, going from
    #
    #   x0 -> op0 -> x1 -> ...
    #
    #
    # to
    #
    #   x0 -> op0_0 -> x1_0 -> log -> ...
    #    \                     \
    #      -> op0_1 -> x1_1 -> clog
    #
    # Later, the outputs of each shadow will be rerouted to calculate
    # propagation error.

    # Note: we cannot iterate over matched subgraphs because some nodes
    # may not be matched. So, we iterate over nodes in the graph, and
    # associate them to matched subgraphs if possible.

    nodes_to_skip = set()
    # for each subgraph, save a mapping from first node of subgraph
    # to first and last node of the shadow of this subgraph
    orig_first_node_to_shadow_in_node = {}
    orig_first_node_to_shadow_out_node = {}
    # need to record original list because we will mutate the graph as we go
    orig_nodes = list(model.graph.nodes)  # type: ignore[union-attr, arg-type]
    cur_subgraph_idx = 0
    for n in orig_nodes:
        if n.op in ("placeholder", "get_attr", "output") or n in nodes_to_skip:
            continue

        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        insert_submodule_copy = False
        if maybe_subgraph is not None:
            first_node, last_node = maybe_subgraph[0], maybe_subgraph[-1]
            nodes_to_skip.update(maybe_subgraph)
            qconfig = node_name_to_qconfig[first_node.name]
            if qconfig is not None:
                insert_submodule_copy = True
        else:
            first_node, last_node = n, n

        if insert_submodule_copy:
            match_name = first_node.name
            create_n_transformed_and_logged_copies_of_subgraph(
                model,
                cur_subgraph_idx,
                match_name,
                maybe_subgraph,
                [qconfig_mapping],
                [node_name_to_qconfig],
                None,
                None,  # type: ignore[arg-type]
            )
            # find the created shadow module and record it so we
            # can find it easily in step 2
            expected_shadow_target = f"shadow_wrapper_{cur_subgraph_idx}_1"
            new_shadow_mod = None
            for maybe_shadow_mod in model.graph.nodes:
                if (
                    maybe_shadow_mod.op == "call_module"
                    and maybe_shadow_mod.target == expected_shadow_target
                ):
                    new_shadow_mod = maybe_shadow_mod
                    break
            if new_shadow_mod is None:
                raise AssertionError("Expected new_shadow_mod to be non-None")
            orig_first_node_to_shadow_in_node[first_node] = new_shadow_mod
            orig_first_node_to_shadow_out_node[first_node] = new_shadow_mod

        else:
            # create a copy of the subgraph by only copying FX nodes
            # but not copying any parameters, to minimize memory usage
            subgraph_to_use = (
                maybe_subgraph if maybe_subgraph is not None else [first_node]
            )

            # add a regular logger after last_node
            qconfig_str = ""
            subgraph_candidate_idx = 0
            fqn = _maybe_get_fqn(first_node, model)
            logger_mod_orig = _get_logger_for_subgraph(
                model,
                first_node,
                last_node,
                cur_subgraph_idx,
                subgraph_candidate_idx,
                qconfig_str,
                OutputLogger,
                fqn,
            )
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            if hasattr(model, attr_name):
                raise AssertionError(
                    f"Unexpected attribute '{attr_name}' found in {model}"
                )
            setattr(model, attr_name, logger_mod_orig)
            insertion_point = last_node
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(
                    attr_name, args=(last_node,), kwargs={}
                )
                insertion_point = logger

            # create a copy of the subgraph
            cur_node_orig = first_node
            cur_node_copy = None
            first_node_copy = None
            # pyrefly: ignore [bad-assignment]
            while cur_node_orig in subgraph_to_use:
                # TODO(future PR): make this support all possible args/kwargs
                if cur_node_orig is first_node:
                    new_args = cur_node_orig.args
                    new_kwargs = cur_node_orig.kwargs
                else:
                    first_arg_for_copy: Node | None = cur_node_copy
                    new_args = (first_arg_for_copy, *cur_node_orig.args[1:])
                    new_kwargs = cur_node_orig.kwargs
                # make a copy of cur_node_orig
                with model.graph.inserting_after(insertion_point):
                    cur_node_copy = model.graph.create_node(
                        cur_node_orig.op,
                        cur_node_orig.target,
                        new_args,
                        new_kwargs,
                        # cur_node_orig.name,  # TODO(future PR): set name explicitly
                    )
                    if first_node_copy is None:
                        first_node_copy = cur_node_copy
                # since now only linear subgraphs are supported, all nodes
                # except the last one must have only one user
                if cur_node_orig != last_node:
                    if len(cur_node_orig.users.keys()) != 1:
                        raise AssertionError(
                            f"Expected exactly 1, but got {len(cur_node_orig.users)}"
                        )
                cur_node_orig = next(iter(cur_node_orig.users.keys()))
                if cur_node_orig.name.startswith(SHADOW_NODE_NAME_PREFIX):
                    raise AssertionError(
                        "cur_node_orig should not start with SHADOW_NODE_NAME_PREFIX"
                    )
                insertion_point = cur_node_copy

            # add a comparison logger after last_node's copy
            subgraph_candidate_idx = 1
            logger_mod_orig = _get_logger_for_subgraph(
                model,
                first_node,
                last_node,
                cur_subgraph_idx,
                subgraph_candidate_idx,
                qconfig_str,
                OutputComparisonLogger,
                fqn,
            )
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            if hasattr(model, attr_name):
                raise AssertionError(
                    f"Unexpected attribute '{attr_name}' found in {model}"
                )
            setattr(model, attr_name, logger_mod_orig)
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(
                    attr_name, args=(cur_node_copy, last_node), kwargs={}
                )

            # save the final node so we can use it in step 2
            orig_first_node_to_shadow_in_node[first_node] = first_node_copy
            orig_first_node_to_shadow_out_node[first_node] = cur_node_copy

        cur_subgraph_idx += 1

    model.recompile()

    # Now, we go from
    #
    #   x0 -> op0_0 -> x1_0 -> log -> x1 -> op1_0 -> ...
    #    \                     \       \
    #      -> op0_1 -> x1_1 -> clog      -> op1_1 -> ...
    #
    # to
    #
    #   x0 -> op0_0 -> x1_0 -> log --> x1_0 -> op1_0 -> ...
    #    \                     \
    #      -> op0_1 -> x1_1 -> clog -> x1_1 -> op1_1 -> ...
    #
    # sample values of key internal variables for the example above:
    #
    #   orig_first_node_to_shadow_in_node = {op0_0: op0_1, op1_0: op1_1}
    #   orig_first_node_to_shadow_out_node = {op0_0: op0_1, op1_0: op1_1}
    #
    # note: for subgraphs with more than one node, in_node will be different
    # compared to out_node

    nodes_to_skip = set()
    for n in orig_nodes:
        if n.op in ("placeholder", "get_attr", "output") or n in nodes_to_skip:
            continue

        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        if maybe_subgraph is not None:
            first_node, last_node = maybe_subgraph[0], maybe_subgraph[-1]
            nodes_to_skip.update(maybe_subgraph)
        else:
            first_node, last_node = n, n

        def maybe_remap_node_to_shadow(node):
            """
            If unshadowed `node` has a shadow version, return that. If not,
            return `node`.
            """
            if not isinstance(node, Node):
                # handle scalars
                return node

            if node.op in ("placeholder", "get_attr"):
                return node

            # Find the shadowed version of this arg from the previous
            # subgraph. For this, we need to:
            # 1. navigate to the first node of the previous subgraph
            # 2. get the output of the shadow wrapper which has (1) as an input

            # For now, assume the arg is in matched subgraphs. In the
            # future we may have to handle the case where this is not true.
            prev_subgraph = _get_subgraph_containing_node(node, subgraphs_dedup)
            if prev_subgraph is None:
                prev_subgraph = [node]
            prev_first_node = prev_subgraph[0]
            prev_shadow_output = orig_first_node_to_shadow_out_node[prev_first_node]
            return prev_shadow_output

        cur_shadow_input = orig_first_node_to_shadow_in_node[first_node]
        if cur_shadow_input is None:
            raise AssertionError("Expected cur_shadow_input to be non-None")
        cur_shadow_input.args = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.args
        )
        cur_shadow_input.kwargs = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.kwargs
        )

        model.recompile()


def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module):
    # input: shadow wrapper module
    # output if shadow wrapper module has a weighted op:
    #   (quantize_fn, (quantize_fn_args))
    # output if shadow wrapper module doesn't have a weighted op:
    #   None

    # For now, assume that the weight is the second input
    # to the shadow module. If that changes, we can fix it later.
    placeholders_seen = 0
    for shadow_n in shadow_wrapper.graph.nodes:  # type: ignore[union-attr]
        if shadow_n.op != "placeholder":
            continue

        placeholders_seen += 1
        if placeholders_seen != 2:
            continue

        # the subgraph looks like
        #
        #   _input_scale_1 = self._input_scale_1
        #   _input_zero_point_1 = self._input_zero_point_1
        #   quantize_per_channel = torch.quantize_per_channel(
        #       w2_0, _input_scale_1, _input_zero_point_1,
        #       0, torch.qint8)
        #
        #  we have `w2_0`, and are navigating this subgraph
        #  to get `_input_scale_1` and `_input_zero_point_1`

        if len(shadow_n.users) != 1:
            raise AssertionError(f"Expected exactly 1, got {len(shadow_n.users)}")
        quant_node = next(iter(shadow_n.users.keys()))
        new_args: Any = None
        if quant_node.target is torch.quantize_per_channel:
            _weight, scale_node, zp_node, axis, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, axis, dtype)
        else:
            if quant_node.target != torch.quantize_per_tensor:
                raise AssertionError(
                    f"Expected torch.quantize_per_tensor, but got {quant_node.target}"
                )
            _weight, scale_node, zp_node, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, dtype)
        return (quant_node.target, new_args)

    return None


def extract_weight_comparison(m: GraphModule) -> NSResultsType:
    # example graph:
    #
    #   w1 = self.w1
    #   b1 = self.b1
    #   linear = torch._C._nn.linear(x, w1, b1)
    #   shadow_0_0 = self.shadow_0_0(linear)
    #   shadow_wrapper_0_1 = self.shadow_wrapper_0_1(x, w1, b1)
    #   shadow_0_1 = self.shadow_0_1(shadow_wrapper_0_1, linear)
    #
    # algorithm:
    # 1. for each call_function node matching our allowlist:
    # 2.   if corresponding shadow wrapper exists, extract the weight pair
    #
    # Note: this is not super robust, but that's ok because this is
    # just for legacy customers who depend on the previous two-model version
    # of this API. TBD if we need to make this robust.
    # Note: modules are not supported, since existing customers only
    # use functions.

    # TODO(future PR): move this to config
    weighted_ops = {
        torch.nn.functional.linear,
    }

    results: NSResultsType = {"model": {NSSingleResultValuesType.WEIGHT.value: {}}}

    for n in m.graph.nodes:  # type: ignore[union-attr]
        if not (n.op == "call_function" and n.target in weighted_ops):
            continue

        # Check if we have a corresponding shadow wrapper
        # TODO(future PR, if needed): support kwargs
        # TODO(future PR, if needed): support multiple shadow users
        first_arg = n.args[0]
        shadow_wrapper_node = None
        for user in first_arg.users:
            # TODO(before land): fix string match
            if user.op == "call_module" and user.target.startswith("shadow_wrapper"):
                shadow_wrapper_node = user
                break

        if shadow_wrapper_node is None:
            continue

        shadow_wrapper = getattr_from_fqn(m, shadow_wrapper_node.target)  # type: ignore[arg-type]
        weight_info = _get_weight_info_from_shadow_wrapper(shadow_wrapper)
        if weight_info is None:
            continue

        # get weight
        w_node = n.args[1]
        w_obj = getattr_from_fqn(m, w_node.target).detach()

        # get a quantized version of weight
        quant_fn, quant_fn_args_except_first = weight_info
        new_args = (w_obj, *quant_fn_args_except_first)
        w_obj_q = quant_fn(*new_args)

        # add a comparison
        ref_node_name = n.name
        prev_node_name = n.name
        ref_node_type = get_target_type_str(n, m)
        prev_node_type = ref_node_type
        fqn = None
        if hasattr(m, "_node_name_to_scope"):
            fqn = m._node_name_to_scope[n.name][0]  # type: ignore[index]
        comparison = torch.ao.ns.fx.utils.compute_sqnr(w_obj, w_obj_q)
        result_fp32 = {
            "res_type": NSSingleResultValuesType.WEIGHT.value,
            "values": [w_obj],
            "prev_node_name": prev_node_name,
            "prev_node_target_type": prev_node_type,
            "ref_node_name": ref_node_name,
            "ref_node_target_type": ref_node_type,
            "index_within_arg": 0,
            "index_of_arg": 0,
            "fqn": fqn,
            "qconfig_str": "",
            "comparisons": [comparison],
            "comparison_fn_name": "sqnr",
        }
        result_q = {
            "res_type": NSSingleResultValuesType.WEIGHT.value,
            "values": [w_obj_q],
            "prev_node_name": prev_node_name,
            "prev_node_target_type": prev_node_type,
            "ref_node_name": ref_node_name,
            "ref_node_target_type": ref_node_type,
            "index_within_arg": 0,
            "index_of_arg": 0,
            "fqn": fqn,
            "qconfig_str": "",
            "comparisons": [comparison],
            "comparison_fn_name": "sqnr",
        }

        # go from subgraph_n_1 to subgraph_n_0
        _1, _2, node_idx, _3 = shadow_wrapper_node.target.split("_")
        name_fp32 = f"subgraph_{node_idx}_0"
        name_q = f"subgraph_{node_idx}_1"

        results["model"][NSSingleResultValuesType.WEIGHT.value][name_fp32] = [
            result_fp32
        ]
        results["model"][NSSingleResultValuesType.WEIGHT.value][name_q] = [result_q]

    return results


# TODO(future PR): redesign this to make it easier to consume outputs
def group_results_by_subgraph(results: NSResultsType) -> Any:
    """
    Creates a comparison of results

    Input:

    {
      'model': {
        'node_output': {
          'subgraph_0_0': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [], ...
            'comparison_fn_name': '',
            'fqn': '...',
          ],
          'subgraph_0_1': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [torch.tensor(...), ...], ...
            'comparison_fn_name': '...',
            'fqn': '...',
          ],
          ...
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': None,
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
      },
    }

    """
    subgraph_name_to_subgraph_results: Any = collections.defaultdict(dict)

    # node_output or weight
    key_to_use = next(iter(results["model"].keys()))

    for subgraph_name_with_idx, subgraph_candidate_results in results["model"][
        key_to_use
    ].items():
        # convert from `subgraph_m_n` to `subgraph_m` and `n`
        (
            subgraph_str,
            subgraph_idx,
            subgraph_candidate_idx,
        ) = subgraph_name_with_idx.split("_")
        subgraph_name = f"{subgraph_str}_{subgraph_idx}"

        subgraph_results = {
            "ref_node_name": subgraph_candidate_results[0]["ref_node_name"],
            "ref_node_target_type": subgraph_candidate_results[0][
                "ref_node_target_type"
            ],
            "fqn": subgraph_candidate_results[0]["fqn"],
            "values": subgraph_candidate_results[0]["values"],
            "qconfig_str": subgraph_candidate_results[0]["qconfig_str"],
            "comparisons": subgraph_candidate_results[0]["comparisons"],
            "comparison_fn_name": subgraph_candidate_results[0]["comparison_fn_name"],
        }

        subgraph_name_to_subgraph_results[subgraph_name][subgraph_candidate_idx] = (
            subgraph_results
        )

    return dict(subgraph_name_to_subgraph_results)


# TODO(future PR): redesign this to make it easier to consume outputs
def create_results_comparison(
    results_grouped,
) -> Any:
    """
    Input:

    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '',
          'comparisons': [],
          'comparison_fn_name': '',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...],
          'comparison_fn_name': 'sqnr',
          'fqn': '...',
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        'ref_node_name': '...',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': 'sqnr',
            'cmp_raw': [..., ...],
            'cmp_mean': ...,
          },
          ...,
        },
      },
    }
    """

    results_comparison = {}

    for subgraph_name, subgraph_results in results_grouped.items():
        candidates = {}
        for subgraph_inner_name, subgraph_inner_result in subgraph_results.items():
            # skip comparing baseline to baseline
            if subgraph_inner_name == "0":
                continue

            # we expect the comparisons to be precalculated from
            # calibration, so we just fetch them here
            cmp_raw = subgraph_inner_result["comparisons"]
            cmp_raw_tensor = torch.stack(cmp_raw)

            candidates[subgraph_inner_name] = {
                "qconfig_str": subgraph_inner_result["qconfig_str"],
                "comparison_fn_name": subgraph_inner_result["comparison_fn_name"],
                "cmp_raw": cmp_raw_tensor,
                "cmp_mean": torch.mean(cmp_raw_tensor),
            }

        results_comparison[subgraph_name] = {
            "ref_node_name": subgraph_results["0"]["ref_node_name"],
            "ref_node_target_type": subgraph_results["0"]["ref_node_target_type"],
            "fqn": subgraph_results["0"]["fqn"],
            "candidates": candidates,
        }

    return results_comparison


# TODO(future PR): redesign this to make it easier to consume outputs
def print_n_shadows_summary(
    results_comparison,
) -> None:
    """
    Input:

    {
      'subgraph_0': {
        'ref_node_name': 'linear1',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': ...,
            'cmp_raw': [45.0, 55.0],
            'cmp_mean': 50.0,
          },
          ...,
        },
      },
    }

    Prints:

    node_name | node_type | fqn | 0    | 1    | ...
    linear1   | ...       | ... | 45.0 | 50.0 | ...
    """

    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "`print_tabular` relies on the library `tabulate`, "
            "which could not be found on this machine. Run `pip "
            "install tabulate` to install the library."
        )
        return

    results = []
    for subgraph_data in results_comparison.values():
        mean_all_candidates = [
            candidate["cmp_mean"]
            for candidate_name, candidate in subgraph_data["candidates"].items()
        ]

        data_row = [
            subgraph_data["ref_node_name"],
            subgraph_data["ref_node_target_type"],
            subgraph_data["fqn"],
            *mean_all_candidates,
        ]
        results.append(data_row)

    max_candidate_idx_len = -1
    for data_row in results:
        max_candidate_idx_len = max(max_candidate_idx_len, len(data_row[1]))
    candidate_idx_headers = [str(x) for x in range(max_candidate_idx_len)]

    headers = ["node_name", "node_type", "fqn", *candidate_idx_headers]
    print(tabulate(results, headers=headers))
