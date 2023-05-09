# Owner(s): ["oncall: distributed"]
import collections
import itertools
import logging
import operator
from dataclasses import dataclass, field
from typing import Any, cast, DefaultDict, List, Set, Tuple, Union

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import find_node, get_output, OP
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.utils._pytree import tree_flatten, tree_unflatten

from .comm_fusion import schedule_comm_wait
from .common import get_all_comm_blocks, graph_optimization_pass


aten = torch.ops.aten
logger: logging.Logger = logging.getLogger("graph_optimization")


@graph_optimization_pass(
    prerequisites=[],
    apply_after=[],
)
def remove_copy_from_optimizer(gm: IterGraphModule) -> None:
    """
    Erase the the orphant copy_ that generated when tracing optimizer.
    Two reasons why we could not simply use the DCE of fx.Graph.
    1. fx.Graph treats copy_ as a side-effect node and does not erase it.
    2. Users may want to preserve some orphan `copy_` that is not from the
       optimizer.
    If the second reason does not hold, this pass can be rewritten as using
    DCE from fx.Graph (with the overwrite to the side-effect node list).
    """
    MAX_COPY_DISTANCE = 5
    remove_candidates: Set[fx.Node] = set()
    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node.op != OP.CALL_FUNCTION or node.target != aten.copy_.default:
            continue

        copy_ancestors: Set[fx.Node] = set()
        nodes = collections.deque([node, None])
        distance = 0
        should_remove = False
        while nodes and distance < MAX_COPY_DISTANCE:
            visiting = nodes.popleft()
            if visiting is None:
                distance += 1
                if nodes:
                    nodes.append(None)
                continue
            copy_ancestors.add(visiting)
            if visiting.op == OP.CALL_FUNCTION and str(visiting.target).startswith(
                ("aten._foreach_", "aten._fused_")
            ):
                should_remove = True
            parents, _ = tree_flatten((visiting.args, visiting.kwargs))
            for parent in parents:
                if isinstance(parent, fx.Node):
                    nodes.append(parent)
        if should_remove:
            # We add all ancestors to the list and it is okay as not all of
            # them will be erased -- only those nodes with zero users will be
            # erased.
            remove_candidates.update(copy_ancestors)

    for node in reversed(gm.graph.nodes):
        if node.users:
            continue
        if node not in remove_candidates:
            continue
        gm.graph.erase_node(node)


# The args list of fused_adam function. We don't care about kwargs.
AdamArgs = collections.namedtuple(
    "AdamArgs",
    ["params", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs", "state_steps"],
)


# TODO(fegin): Have a template class for all Block class.
@dataclass(unsafe_hash=True)
class FusedAdamBlock:
    optim_node: fx.Node
    generate_output: bool
    # The output list of the copy nodes. The order follows the argument order.
    param_outputs: List[fx.Node] = field(default_factory=list)
    grad_outputs: List[fx.Node] = field(default_factory=list)
    exp_avgs_outputs: List[fx.Node] = field(default_factory=list)
    exp_avg_sqs_outputs: List[fx.Node] = field(default_factory=list)
    # TODO(fegin): populate/generate the max_exp_avg_sqs if exists
    max_exp_avg_sqs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        # Iterate all the args and generate the corresponding output lists.
        # Assuming the corrsesponding output nodes are not created yet.
        def _generate_outputs(arg_idx, output_list):
            graph = self.optim_node.graph
            with graph.inserting_after(self.optim_node):
                optim_getitem = graph.call_function(
                    operator.getitem, (self.optim_node, arg_idx)
                )
            for i, arg in enumerate(self.optim_node.args[arg_idx]):
                with graph.inserting_after(optim_getitem):
                    updated_arg = graph.call_function(
                        operator.getitem, (optim_getitem, i)
                    )
                with graph.inserting_after(updated_arg):
                    output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
                output_list.append(output_copy)

        _generate_outputs(0, self.param_outputs)
        # Do not generate gradient out list as it is not used.
        _generate_outputs(2, self.exp_avgs_outputs)
        _generate_outputs(3, self.exp_avg_sqs_outputs)

    def populate_outputs(self):
        # Populate the existing output lists from the graph.
        def _populate_outputs(args_idx, output_list):
            optim_getitem = self.optim_node
            for user in self.optim_node.users:
                assert (
                    user.target == operator.getitem
                ), f"The user of {self.optim_node} is not getitem."
                if user.args[1] == args_idx:
                    optim_getitem = user
                    break
            assert (
                optim_getitem != self.optim_node
            ), f"Cannot find the getitem node for {self.optim_node}"
            output_list.extend(
                [self.optim_node] * len(cast(List[fx.Node], self.optim_node.args[0]))
            )
            for updated_arg in optim_getitem.users:
                assert (
                    updated_arg.target == operator.getitem
                ), f"Unexpected node target {updated_arg.target}."
                idx = updated_arg.args[1]
                output_copy = next(iter(updated_arg.users))
                assert str(output_copy.target).startswith(
                    "aten.copy_"
                ), f"Unexpected node target {output_copy.target}."
                output_list[idx] = output_copy
            for i, output in enumerate(output_list):
                assert output != self.optim_node, f"{i}th output is not replaced."

            assert output_list, f"The output for {self.optim_node} is empty."

        _populate_outputs(0, self.param_outputs)
        _populate_outputs(2, self.exp_avgs_outputs)
        _populate_outputs(3, self.exp_avg_sqs_outputs)

    def __post_init__(self):
        if self.param_outputs:
            return
        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()


@dataclass(unsafe_hash=True)
class ForeachAddBlock:
    add_node: fx.Node
    generate_output: bool
    # The output list of the copy nodes. The order follows the argument order.
    outputs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        # Iterate all the args and generate the corresponding output lists
        # Assuming the corrsesponding output nodes are not created yet.
        graph = self.add_node.graph
        for i, arg in enumerate(cast(Tuple[Any, ...], self.add_node.args[0])):
            with graph.inserting_after(self.add_node):
                updated_arg = graph.call_function(operator.getitem, (self.add_node, i))
            with graph.inserting_after(updated_arg):
                output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
            self.outputs.append(output_copy)
        assert self.outputs, f"The output for {self.add_node} is empty."

    def populate_outputs(self):
        # Populate the existing output lists from the graph.
        self.outputs = [
            self.add_node for _ in cast(Tuple[Any, ...], self.add_node.args[0])
        ]
        for updated_arg in self.add_node.users:
            assert (
                updated_arg.target == operator.getitem
            ), f"Unexpected node target {updated_arg.target}"
            idx = cast(int, updated_arg.args[1])
            output_copy = next(iter(updated_arg.users))
            assert str(output_copy.target).startswith(
                "aten.copy_"
            ), f"The execpted output node is different, {str(output_copy.target)}"
            self.outputs[idx] = output_copy
        for i, output in enumerate(self.outputs):
            assert output != self.add_node, f"{i}th output is not replaced."

    def __post_init__(self):
        if self.outputs:
            return

        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()


@dataclass(unsafe_hash=True)
class FusedOptimizerBlock:
    step: ForeachAddBlock
    optim: FusedAdamBlock


def get_fused_optimizer_block(optim_node: fx.Node) -> FusedOptimizerBlock:
    """
    Given a fused optimizer node and return the FusedOptimizerBlock.
    """
    MAX_STEP_DISTANCE = 5
    # Find the step (foreach_add)
    nodes = collections.deque([optim_node, None])
    step_node = optim_node
    distance = 0
    while nodes and distance < MAX_STEP_DISTANCE:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        elif node.op == OP.CALL_FUNCTION and str(node.target).startswith(
            "aten._foreach_add"
        ):
            step_node = node
            break
        else:
            nodes.extend(
                a
                for a in tree_flatten((node.args, node.kwargs))[0]
                if isinstance(a, fx.Node)
            )
    if step_node == optim_node:
        raise RuntimeError(
            "Cannot find step node (foreach_add) for the optimizer node "
            f"{optim_node} with {MAX_STEP_DISTANCE} BFS distance. "
            "The API design does not match the tracing graph."
        )

    step = ForeachAddBlock(step_node, generate_output=False)
    optim = FusedAdamBlock(optim_node, generate_output=False)
    return FusedOptimizerBlock(step, optim)


def get_all_fused_optimizer_blocks(
    gm: IterGraphModule, optim_ops: Union[Tuple[str, ...], str]
) -> List[FusedOptimizerBlock]:
    """
    Find all the FusedOptimizerBlock that the optimizer operators are in
    `optim_ops`.
    """
    return [
        get_fused_optimizer_block(node)
        for node in gm.graph.nodes
        if node.name.startswith(optim_ops)
    ]


def _split_fused_adam(
    gm: IterGraphModule,
    orig_optim_block: FusedOptimizerBlock,
    split_gradients: Set[fx.Node],
) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    """
    Split the `orig_optim_block` into two FusedOptimizerBlock. The first one
    will be the optimizer that optimize `split_gradients`. The second one is
    used to optimize the remaining gradients.
    An assert will be raised if one of the optimizer optimize zero gradients.
    """
    orig_optim_args = AdamArgs(*orig_optim_block.optim.optim_node.args)
    optim_args = (AdamArgs([], [], [], [], [], []), AdamArgs([], [], [], [], [], []))
    # The only hint we can use to split the optimizer is the order/indices.
    orig_optim_indices: Tuple[List[int], List[int]] = ([], [])
    orig_step_indices: Tuple[List[int], List[int]] = ([], [])

    for idx, gradient in enumerate(orig_optim_args.grads):
        group_idx = 0 if gradient in split_gradients else 1
        orig_optim_indices[group_idx].append(idx)
        # Get the argument for idx-th gradient from orig_optim_args
        for orig_arg, optim_arg in zip(orig_optim_args, optim_args[group_idx]):
            # Only add the argument to the list if the original argument list
            # is not empty. If the original argument list is empty, the new
            # one must be an empty list as well.
            if orig_arg:
                optim_arg.append(orig_arg[idx])

        # If argument order of step is the same as optimizer, nothing has to be
        # done. However, it is risky to rely on this assumption so we populate
        # the orig_step_indices.
        orig_step_output = optim_args[group_idx].state_steps[-1]
        assert str(orig_step_output.target).startswith(
            "aten.copy_"
        ), f"The copy output is {orig_step_output.target}, expect aten.copy_"
        orig_step_getitem = orig_step_output.args[1]
        assert "getitem" in str(
            orig_step_getitem.target
        ), f"The copy getitem is {orig_step_getitem.target}, expect operator.getitem"
        orig_step_idx = orig_step_getitem.args[1]
        orig_step_indices[group_idx].append(orig_step_idx)

    if not all(l for l in (orig_step_indices + orig_optim_indices)):
        raise ValueError("At least one split optimizer does not have input.")

    output = get_output(gm.graph)
    results: List[FusedOptimizerBlock] = []
    flatten_output_args, spec = tree_flatten((output.args, output.kwargs))
    flatten_output_args_indices: DefaultDict[
        fx.Node, Set[int]
    ] = collections.defaultdict(set)
    for idx, output_arg in enumerate(flatten_output_args):
        if isinstance(output_arg, fx.Node):
            flatten_output_args_indices[output_arg].add(idx)

    def replace_flatten_output_args(orig_node: fx.Node, new_node: fx.Node):
        for idx in flatten_output_args_indices[orig_node]:
            flatten_output_args[idx] = new_node

    # Create the new step and optim nodes and blocks.
    for group_idx in range(2):
        step_args: List[fx.Node] = []
        orig_step_outputs: List[fx.Node] = []
        # We have to create the new step node and block first because it is used
        # for the new optim node as the input.
        with gm.graph.inserting_after(orig_optim_block.optim.optim_node):
            for idx in orig_step_indices[group_idx]:
                step_args.append(
                    cast(Tuple[fx.Node, ...], orig_optim_block.step.add_node.args[0])[
                        idx
                    ]
                )
                orig_step_outputs.append(orig_optim_block.step.outputs[idx])
            step = gm.graph.call_function(
                aten._foreach_add.Scalar,
                (step_args, 1),
            )
        step_block = ForeachAddBlock(step, generate_output=True)
        for i, step_output in enumerate(step_block.outputs):
            # Replace the original step output in the graph output node with
            # the new one.
            orig_step_output = orig_step_outputs[i]
            replace_flatten_output_args(orig_step_output, step_output)
            # Also need to replace the step output used for the new optimizer.
            assert optim_args[group_idx].state_steps[i] == orig_step_output, (
                f"The expected step output node mismatched, {orig_step_output} "
                f"{optim_args[group_idx].state_steps[i]}"
            )
            optim_args[group_idx].state_steps[i] = step_output

        # Insert the optimizer node after the first step output because its
        # topo sort order is the last.
        with gm.graph.inserting_after(step_block.outputs[0]):
            optim = gm.graph.call_function(
                aten._fused_adam.default,
                optim_args[group_idx],
                orig_optim_block.optim.optim_node.kwargs,
            )
        optim_block = FusedAdamBlock(optim, generate_output=True)
        for curr_idx, orig_idx in enumerate(orig_optim_indices[group_idx]):
            list_names = ("param_outputs", "exp_avgs_outputs", "exp_avg_sqs_outputs")
            for name in list_names:
                orig_list = getattr(orig_optim_block.optim, name)
                curr_list = getattr(optim_block, name)
                replace_flatten_output_args(orig_list[orig_idx], curr_list[curr_idx])

        results.append(FusedOptimizerBlock(step_block, optim_block))

    # Optimizer is used as the output of the train_step. Therefore, we have to
    # update the output node of the graph.
    output_args, output_kwargs = tree_unflatten(flatten_output_args, spec)
    gm.graph.node_set_args(output, output_args)
    gm.graph.node_set_kwargs(output, output_kwargs)
    # Remove the original copy_ nodes as they won't be DCE.
    for copy_output in itertools.chain(
        orig_optim_block.optim.param_outputs,
        orig_optim_block.optim.exp_avgs_outputs,
        orig_optim_block.optim.exp_avg_sqs_outputs,
    ):
        gm.graph.erase_node(copy_output)
    # Call DCE once to get rid of the old optimizer. By doing so, we will be
    # able to erase the copy_ nodes of step later.
    gm.graph.eliminate_dead_code()
    for copy_output in orig_optim_block.step.outputs:
        gm.graph.erase_node(copy_output)
    # This is not required but calling this for consistency.
    gm.graph.eliminate_dead_code()

    return results[0], results[1]


def split_fused_optimizer(
    gm: IterGraphModule,
    optim_block: FusedOptimizerBlock,
    split_gradients: Set[fx.Node],
) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    if not split_gradients:
        raise ValueError("The given split_gradients is empty.")
    if str(optim_block.optim.optim_node.target).startswith("aten._fused_adam"):
        return _split_fused_adam(gm, optim_block, split_gradients)
    else:
        raise NotImplementedError("Only fused_adam is supported now")


# TODO(fegin): The API only support fused adam now. Should extend it to support
# foreach as well.
@graph_optimization_pass(
    prerequisites=[remove_copy_from_optimizer],
    apply_after=[schedule_comm_wait],
)
def iter_move_grads_and_optimizers(
    gm: IterGraphModule,
    target_comm_node: str,
    target_dest_node: str,
) -> None:
    """Function to extract a comm block and split out a new optimizer and step for it.
    This subgraph is then moved to the forward graph.
    """
    for comm_block in get_all_comm_blocks(gm, "all_reduce"):
        if comm_block.comm_node.name == target_comm_node:
            break
    else:
        raise ValueError(f"Cannot find {target_comm_node}")

    optim_blocks = get_all_fused_optimizer_blocks(gm, "_fused_adam")
    for optim_block in optim_blocks:
        optim_args = AdamArgs(*optim_block.optim.optim_node.args)
        one_output = next(iter(comm_block.outputs))
        if one_output in optim_args.grads:
            break
    else:
        raise ValueError(f"{target_comm_node} is not used by any fused optimizer.")

    move_optim, _ = split_fused_optimizer(gm, optim_block, comm_block.outputs)

    # TODO(@fegin): Extract this logic as a generic find_all_descendants API.
    output = get_output(gm.graph)
    nodes = collections.deque([comm_block.comm_node, move_optim.step.add_node])
    move_node_set = set()
    while nodes:
        node = nodes.popleft()
        move_node_set.add(node)
        nodes += [u for u in node.users if isinstance(u, fx.Node) and u != output]
    move_nodes = [node for node in gm.graph.nodes if node in move_node_set]

    stop_node = find_node(gm.graph, lambda n: n.name == target_dest_node)[0]

    gm.graph.move_to_next_iter_before(move_nodes, stop_node)
