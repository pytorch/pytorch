# mypy: allow-untyped-defs
import copy
import functools
import heapq
import itertools
import logging
import math
import operator
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import torch
import torch._inductor.inductor_prims
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    find_symbol_binding_fx_nodes,
    free_symbols,
    hint_int,
    is_symbol_binding_fx_node,
)
from torch.fx.passes import graph_drawer
from torch.utils.checkpoint import CheckpointPolicy

from . import config
from ._activation_checkpointing.knapsack import (
    dp_knapsack,
    greedy_knapsack,
    ilp_knapsack,
)
from ._aot_autograd.logging_utils import get_aot_graph_name
from ._aot_autograd.utils import is_with_effects
from .compile_utils import fx_graph_cse, get_aten_target


if TYPE_CHECKING:
    import sympy


AOT_PARTITIONER_DEBUG: bool = config.debug_partitioner
log: logging.Logger = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""

    fusible_ops: Set[Callable]
    compute_intensive_ops: Set[Callable]
    random_ops: Set[Callable]
    view_ops: Set[Callable]
    recomputable_ops: Set[Callable]

    def is_fusible(self, node: fx.Node):
        return get_aten_target(node) in self.fusible_ops

    def is_compute_intensive(self, node: fx.Node):
        return get_aten_target(node) in self.compute_intensive_ops

    def is_random(self, node: fx.Node):
        return get_aten_target(node) in self.random_ops

    def is_view(self, node: fx.Node):
        return get_aten_target(node) in self.view_ops

    def is_recomputable(self, node: fx.Node):
        return get_aten_target(node) in self.recomputable_ops


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: List[fx.Node]
    _required_fw_nodes: Set[fx.Node]
    required_bw_nodes: Set[fx.Node]
    unclaimed_nodes: Set[fx.Node]
    fw_order: Dict[fx.Node, int]

    @functools.cached_property
    def required_fw_nodes(self) -> List[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes

    def get_fw_order(self, n: fx.Node) -> int:
        assert n in self._required_fw_nodes, f"Node {n} not in fw nodes!"
        return self.fw_order[n]


@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool
    ban_if_long_fusible_chains: bool
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool


def must_recompute(node: fx.Node) -> bool:
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def has_recomputable_ops(fx_g: fx.GraphModule) -> bool:
    found = False
    for node in fx_g.graph.nodes:
        if must_recompute(node):
            return True
    return False


def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return True
    return False


def sym_node_size(node: fx.Node) -> int:
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    assert isinstance(node.meta["val"], torch.SymFloat)
    return 4


class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph,
    inputs: List[fx.Node],
    outputs: List[fx.Node],
    subgraph: Optional[str] = None,
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        env[node] = new_node

    for node in joint_graph.nodes:
        if _must_be_in_backward(node) and subgraph != "backward":
            env[node] = InvalidNode  # type: ignore[assignment]
            continue

        if node in env:
            # Node must be one of our inputs. (Any member of env which wasn't an
            # input to start must have been created by this loop and won't be in
            # joint_graph.nodes).
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode  # type: ignore[assignment]
        elif node.op == "call_function":
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            assert not isinstance(
                env[x], InvalidNodeBase
            ), f"Node {x} was invalid, but is output"
            output_values.append(env[x])
        else:
            output_values.append(x)
    new_graph.output(tuple(output_values))

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node: fx.Node) -> bool:
    return (
        node.op == "placeholder"
        and "tangents" not in str(node.target)
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )


def _is_tangent(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" in str(node.target)


def _is_bwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "bwd_seed" in str(node.target) or "bwd_base_offset" in str(node.target)
    )


def _is_fwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "fwd_seed" in str(node.target) or "fwd_base_offset" in str(node.target)
    )


def _is_backward_state(node: fx.Node) -> bool:
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _has_tag_is_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "is_backward"


def _has_tag_must_be_in_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_backward"


def _must_be_in_backward(node: fx.Node) -> bool:
    return _has_tag_must_be_in_backward(node) or (
        _has_tag_is_backward(node) and is_with_effects(node)
    )


def _extract_fwd_bwd_outputs(
    joint_module: fx.GraphModule, *, num_fwd_outputs
) -> Tuple[List[fx.Node], List[fx.Node]]:
    outputs = pytree.arg_tree_leaves(
        *(node.args for node in joint_module.graph.find_nodes(op="output"))
    )
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs


def _remove_by_name(saved_values: List[fx.Node], name: str):
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: List[fx.Node],
    saved_sym_nodes: List[fx.Node],
    *,
    num_fwd_outputs: int,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    tangent_inputs = [*filter(_is_tangent, placeholders)]
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs,
        "backward",
    )

    for node in bwd_graph.find_nodes(op="placeholder"):
        # This is to filter out saved values that don't actually end up being used by the backwards pass
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        elif _is_backward_state(node):
            # BackwardState is saved directly
            _remove_by_name(saved_values, node.name)
            assert backward_state_inputs

    # Now that we have the finalized list of saved values, we need to ensure
    # we propagate all symbols which are referenced by backwards inputs.
    # These are not directly used in the graph but are required for downstream
    # sizevar assignment
    saved_symbols: Set[sympy.Symbol] = set()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []

    # Some symbols may already be bound in the directly saved_sym_nodes,
    # keep track of them so we don't re-bind them
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)

    # Now go through all of the prospective backward inputs and track any
    # other symbols we need to bind
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        if "val" not in node.meta:
            continue
        new_symbols = free_symbols(node.meta["val"]) - saved_symbols
        # NB: Deterministic order please!
        for s in sorted(new_symbols, key=lambda s: s.name):
            # NB: For well formed graphs, the symbol should always be present,
            # but we also have ways to produce ill-formed graphs, e.g., direct
            # make_fx usages, so don't choke in this case
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols

    # Update saved_sym_nodes that are now reordered to have all bindings at
    # front. This can also be used later on to figure out the position of saved
    # sym nodes in the output of fwd graph.
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes,
        "forward",
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes
        + saved_values
        + tangent_inputs
        + bwd_seed_offset_inputs
        + backward_state_inputs,
        bwd_outputs,
        "backward",
    )

    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    return fwd_module, bwd_module


def default_partition(
    joint_module: fx.GraphModule, _joint_inputs, *, num_fwd_outputs
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(
            joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs
        )
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, "forward"
    )
    forward_node_names = {
        node.name for node in forward_only_graph.nodes if node.op != "output"
    }
    saved_values = []
    saved_sym_nodes = []

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            # Symints must be kept separate from tensors so that PythonFunction only calls
            # save_for_backward on tensors and stashes symints in autograd .ctx
            saved_sym_nodes.append(node)
        elif "tensor_meta" not in node.meta and node.op == "call_function":
            # Since we can't save tuple of tensor values, we need to flatten out what we're saving
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in forward_node_names
            ]
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                # If we have a tensor in the forward, where only its sizes/strides are needed in the backward,
                # and not the actual tensor data,
                # then it will be a lot cheaper to save only the sizes/strides, and not the actual tensor.
                #
                # Note that saving the tensor could also cause compilation problems:
                # If the user mutated an input in the forward and uses its sizes/strides in the backward,
                # then we would be obligated to clone the input before saving it to appease autograd.
                # (This is how we originally found this bug).
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())

    return _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
    )


INT_INF = int(1e6)


def _tensor_nbytes(numel: int, dtype) -> int:
    return numel * dtype.itemsize


def _size_of(node: fx.Node) -> int:
    def object_nbytes(x) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(hint_int(x.numel(), fallback=4096), x.dtype)

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, dict):
            return sum(object_nbytes(n) for _, n in val.items())
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)

        raise RuntimeError(f"Unknown metadata type {type(val)} on node {node}")
    if node.op == "get_attr":
        return 0
    raise RuntimeError(
        f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes."
    )


# Used for some investigative purposes
def _count_ops(graph: fx.Graph):
    from collections import defaultdict

    cnt: Dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.op == "call_function":
            cnt[node.target.__name__] += 1
    log.info("%s", sorted(cnt.items(), key=lambda x: x[1], reverse=True))


@functools.lru_cache(None)
def pointwise_ops():
    ops = []
    for attr_name in dir(torch.ops.aten):
        opoverloadpacket = getattr(torch.ops.aten, attr_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            continue

        for overload in opoverloadpacket.overloads():
            op_overload = getattr(opoverloadpacket, overload)
            if torch.Tag.pointwise in op_overload.tags:
                # currently aot autograd uses packet not overload
                ops.append(opoverloadpacket)
                break

    return ops


def sort_depths(args, depth_map: Dict[fx.Node, int]) -> List[Tuple[fx.Node, int]]:
    arg_depths = {
        arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)
    }
    return sorted(arg_depths.items(), key=lambda x: x[1], reverse=True)


def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traveral, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """

    new_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx

    def insert_node_in_graph(node):
        cur_nodes = [node]
        insertable_nodes = set()
        while len(cur_nodes) > 0:
            node = cur_nodes.pop()
            if node in insertable_nodes or node in env:
                continue
            insertable_nodes.add(node)

            # Bias traversal towards the nodes that have higher depth - prioritizes
            # critical path first.
            cur_nodes += node.all_input_nodes

        insertable_nodes = sorted(insertable_nodes, key=lambda n: order[n])
        for node in insertable_nodes:
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Find first bwd node in the graph
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user

    # If gradInp does not depend upon gradOut, we may not find any nodes in the "backwards pass"
    if first_node_in_bwd is None:
        return gm

    # Build the graph op-by-op by starting from the node all the way to the end
    for node in list(gm.graph.nodes)[order[first_node_in_bwd] :]:
        insert_node_in_graph(node)

    # The output node is already built by the traversal.
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def functionalize_rng_ops(
    joint_module: fx.GraphModule,
    fw_module: fx.GraphModule,
    bw_module: fx.GraphModule,
    num_sym_nodes: int,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    # During user-driven activation checkpointing, we have to ensure that a rng
    # op in fwd yields the same output as the recomputed rng op in the bwd.  To
    # do this, we use functionalize wrappers to wrap the random ops and share
    # rng state between the fwd and bwd graphs.

    # There are 3 main steps to do this
    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    # Step 2 - Modify the fwd pass such that
    #   1) Replace rand with run_and_save_rng_state wrapper
    #   2) Replace the users of the original op with the output[1] of this op.
    #   3) Collect all the rng_state - output[0] of each op, and make them
    #   output nodes. Special care needs to be taken here because fwd outputs
    #   has symints at the very end.
    # Step 3 - Modify the bwd pass such that
    #   1) Add the input nodes just before the tangents for the stashed rng states
    #   2) Replace rand with run_with_save_rng_state wrappers
    #   3) Use the stashed states as inputs to these ops

    # Unique id to generate name
    uid = itertools.count()

    def get_rng_ops(gmod):
        random_nodes = {}
        for node in gmod.graph.nodes:
            if (
                node.op == "call_function"
                and hasattr(node.target, "tags")
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                random_nodes[node.name] = node
        return random_nodes

    def get_device(node):
        """
        Check the example value of the node outputs to find the device type.
        """
        if "val" not in node.meta:
            return None

        candidates = node.meta["val"]
        if not isinstance(candidates, tuple):
            candidates = (candidates,)

        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                if candidate.device.type == "cuda":
                    return "cuda"

        return "cpu"

    def get_sample_rng_state(device):
        if device == "cuda":
            return torch.cuda.get_rng_state()
        return torch.get_rng_state()

    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    joint_graph_rng_ops = get_rng_ops(joint_module)
    fw_graph_rng_ops = get_rng_ops(fw_module)
    bw_graph_rng_ops = get_rng_ops(bw_module)
    recomputable_rng_ops_map = {}
    for node in joint_module.graph.nodes:
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            base_node = joint_graph_rng_ops[node.name]
            fw_node = fw_graph_rng_ops[node.name]
            bw_node = bw_graph_rng_ops[node.name]
            recomputable_rng_ops_map[base_node] = {"fwd": fw_node, "bwd": bw_node}

    run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
    run_with_rng_state = torch._prims.rng_prims.run_with_rng_state
    bw_tangent_start_node = None
    for node in bw_module.graph.find_nodes(op="placeholder"):
        if "tangent" in node.name:
            bw_tangent_start_node = node
            break
    if bw_tangent_start_node is None:
        raise RuntimeError(
            "Couldn't find tangent node in graph inputs. This is unexpected, please file a bug if you see this"
        )

    fw_rng_state_outputs = []
    for base_node, node_pair in recomputable_rng_ops_map.items():
        # Step 2 - Modify the fwd pass such that
        fw_node = node_pair["fwd"]
        bw_node = node_pair["bwd"]
        fw_graph = fw_module.graph
        with fw_graph.inserting_before(fw_node):
            functional_fw_node = fw_graph.create_node(
                "call_function",
                run_and_save_rng,
                args=(fw_node.target, *fw_node.args),
                kwargs=fw_node.kwargs,
            )
            state = fw_graph.create_node(
                "call_function",
                operator.getitem,
                args=(functional_fw_node, 0),
                kwargs={},
            )
            rng_output = fw_graph.create_node(
                "call_function",
                operator.getitem,
                args=(
                    functional_fw_node,
                    1,
                ),
                kwargs={},
            )
            fw_node.replace_all_uses_with(rng_output)
            fw_graph.erase_node(fw_node)
            fw_rng_state_outputs.append(state)

        # Step 3 - Modify the bwd pass such that
        bw_graph = bw_module.graph
        with bw_graph.inserting_before(bw_tangent_start_node):
            state_name = f"rng_state_output_{next(uid)}"
            bw_rng_state_node = bw_graph.placeholder(state_name)
            bw_rng_state_node.meta["val"] = get_sample_rng_state(get_device(fw_node))

        with bw_graph.inserting_before(bw_node):
            rng_output = bw_graph.create_node(
                "call_function",
                run_with_rng_state,
                args=(bw_rng_state_node, bw_node.target, *bw_node.args),
                kwargs=bw_node.kwargs,
            )

            bw_node.replace_all_uses_with(rng_output)
            bw_graph.erase_node(bw_node)

    # Add the rng states in the output of the fwd graph. AOT Autograd assumes
    # that symints are at the end of forward graph outputs. So, insert the new
    # rng states accordingly.
    fw_output_node = next(iter(fw_module.graph.find_nodes(op="output")))
    fw_outputs = fw_output_node.args[0]
    sym_node_start_idx = len(fw_outputs) - num_sym_nodes
    outputs = (
        fw_outputs[:sym_node_start_idx]
        + tuple(fw_rng_state_outputs)
        + fw_outputs[sym_node_start_idx:]
    )
    fw_module.graph.output(outputs)
    fw_module.graph.erase_node(fw_output_node)
    fw_module.recompile()
    bw_module.recompile()
    return fw_module, bw_module


def cleanup_recompute_tags(joint_module: fx.GraphModule) -> fx.GraphModule:
    """
    If there are two consecutive checkpointed blocks with no operator in
    between, we would still want to stash the tensor at the boundary of
    checkpointed blocks. The following pass makes the last output node
    non-recomputable to allow for that.
    """
    for node in joint_module.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if (
                    must_recompute(user)
                    and user.meta["ac_graph_id"] > node.meta["ac_graph_id"]
                ):
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            if node.meta.get("has_backward_hook", False) and not any(
                must_recompute(user) for user in node.users
            ):
                # If node is AC region output and has a backward hook on it, we intentionally choose to save it.
                # This is to work around circular dependencies in Traceable FSDP2+AC.
                # Example:
                # ```
                # out = fully_shard(utils.checkpoint(module))(x)
                # norm_out = layer_norm(out)
                # ```
                # Here there is a circular dependency:
                # 1. In backward, grad_input of layer_norm aka. `out_grad` is actually dependent on `out`.
                # 2. `out` depends on `out`'s backward hook created by FSDP2 (which does all-gather for `module` weights)
                #    in order to be recomputed.
                # 3. `out`'s backward hook, as is the case for all eager backward hooks, depends on `out_grad`
                #    -> circular dependency with (1)!
                #
                # Solution: check whether `out` has a backward hook, and if so, intentionally save `out`
                # in forward graph outputs. With this, we can break the above circular dependency.
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    return joint_module


def solve_min_cut(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    min_cut_options: MinCutOptions,
    dont_ban=None,
):
    if dont_ban is None:
        dont_ban = set()
    op_types = get_default_op_list()

    if AOT_PARTITIONER_DEBUG:
        joint_module_ops = {
            str(node.target._overloadpacket)
            for node in joint_graph.nodes
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
        }
        ops_ignored = joint_module_ops - {str(i) for i in op_types.recomputable_ops}
        log.info("Ops banned from re-materialization: %s", ops_ignored)

    def can_fuse_into_auto_functionalized(a, b):
        if b.target != torch.ops.higher_order.auto_functionalized:
            return False
        mutable_op = b.args[0]
        (
            mutable_arg_names,
            _,
        ) = torch._higher_order_ops.auto_functionalize.get_mutable_args(mutable_op)
        for name in mutable_arg_names:
            arg = b.kwargs[name]
            if a is arg:
                return True
            if isinstance(arg, list):
                if a in arg:
                    return True
        return False

    def can_fuse_into_triton_kernel_wrapper_functional(a, b):
        if b.target != torch.ops.higher_order.triton_kernel_wrapper_functional:
            return False
        mutable_arg_names = b.kwargs["tensors_to_clone"]
        for name in mutable_arg_names:
            arg = b.kwargs["kwargs"][name]
            if a is arg:
                return True
        return False

    def is_fusible(a, b):
        # We can perform "memory fusion" into a cat, but cat cannot be a
        # producer to a fusion
        if get_aten_target(b) == aten.cat:
            return True
        if can_fuse_into_auto_functionalized(a, b):
            return True
        if can_fuse_into_triton_kernel_wrapper_functional(a, b):
            return True
        if (
            a.target is operator.getitem
            and a.args[0].target
            is torch.ops.higher_order.triton_kernel_wrapper_functional
        ):
            # if a is the output of a user triton kernel,
            # then (by default) we will not be able to fuse b into it
            return False
        return op_types.is_fusible(a) and op_types.is_fusible(b)

    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "Need networkx installed to perform smart recomputation " "heuristics"
        ) from e

    def is_materialized_backwards(node):
        if op_types.is_view(node):
            return False
        cur_nodes = {node}
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if not node_info.is_required_fw(user) and not is_fusible(cur, user):
                    return True
                if op_types.is_view(user):
                    cur_nodes.add(user)

        return False

    def should_ban_recomputation(node):
        if node.op != "call_function":
            return False
        if node.target == operator.getitem:
            return False
        if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
            return True
        if config.recompute_views and op_types.is_view(node):
            return False
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return False

        if min_cut_options.ban_if_not_in_allowlist:
            if not op_types.is_recomputable(node):
                return True
        else:
            if op_types.is_random(node) or op_types.is_compute_intensive(node):
                return True

        # If a node *must* be materialized in the backwards pass, then we
        # should never recompute it. This is a pretty subtle point.  In
        # general, the assumption we make is that recomputing a node in the
        # backwards pass is "free". However, if a node must be materialized
        # in the backwards pass, then recomputing it is never free.
        if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(
            node
        ):
            log.debug("materialized backwards: %s %s", node, tuple(node.users))
            return True

        # Arbitrary hack that sometimes seems to help things. The above
        # modification appears to have made this heuristic a lot less critical
        # for performance.
        # NB: As of PR #121692, this hack no longer seems necessary.
        if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:
            return True

        # If the output of an op is 4x smaller (arbitrary choice),
        # then we don't allow recomputation. The idea here is that for
        # things like reductions, saving the output of the reduction is very
        # cheap/small, and it makes sure we don't do things like recompute
        # normalizations in the backwards.
        if min_cut_options.ban_if_reduction:
            input_tensors_size = sum(
                _size_of(i) for i in node.args if isinstance(i, fx.Node)
            )
            output_size = _size_of(node)
            return output_size * 4 < input_tensors_size
        return False

    def is_materialized(node):
        if node.op == "placeholder":
            return True

        return not all(is_fusible(node, user) for user in node.users)

    def get_node_weight(node) -> float:
        mem_sz = _size_of(node)
        if config.recompute_views and op_types.is_view(node):
            # If `config.recompute_views=True`, we don't save views. This is generally
            # a good idea since views are free to recompute, and it makes it a bit simpler
            # to analyze.
            # NB: If they're not free to recompute (e.g. nested tensors)... I
            # think we should modify checks for view_ops to `is_view` and check
            # that. Basically, with nested tensors, `aten.view` is not a "view
            # op".
            return math.inf

        if isinstance(node.meta["val"], py_sym_types):
            # We never want to save symfloats
            if not isinstance(node.meta["val"], torch.SymInt):
                return INT_INF

        # Heuristic to bias towards nodes closer to the backwards pass
        # Complete guess about current value
        mem_sz = int(mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1)))
        if is_materialized(node):
            return mem_sz
        else:
            return mem_sz * 2

    nx_graph = nx.DiGraph()
    banned_nodes = set()

    def ban_recomputation_if_allowed(node):
        if op_types.is_view(node):
            return False
        if node in dont_ban:
            return False
        # This bans recomputation of the node unless we've been forced not to by
        # user annotation
        if must_recompute(node):
            return False

        if "val" in node.meta and isinstance(node.meta["val"], torch.SymFloat):
            return False

        banned_nodes.add(node)
        # A node will only ever be recomputed if there is a path from an
        # ancestor of this node to the backwards path through this node that
        # doesn't go through any saved value. If this node is saved, then that
        # condition is not possible.
        nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)
        return True

    for node in joint_graph.nodes:
        if node.op == "output":
            continue

        if node in node_info.required_bw_nodes:
            if node not in node_info.inputs:
                nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
                continue
            # If someone saves a input for backward as-is and backward
            # returns that tensor as-is as a grad input, then the node x would
            # be both a required_bw_node and an input. In this case we
            # (1) connect x_in to to the source, (2) x_out to the sink, and
            # (3) assign the proper weight to the x_in-x_out edge, so that
            # x would be part of cut nodes. A case where this happens is if
            # NestedTensor saves a offset tensor as part of the singleton int
            # in sizes.
            nx_graph.add_edge(node.name + "_out", "sink", capacity=math.inf)

        if must_recompute(node):
            # If user explicitly says they want to recompute a node, we honor it
            # by adding an inf-capacity edge from X_in to the sink.
            # This way, X_in node is guaranteed to be part of the subgraph that contains "sink"
            # after the cut, thus guaranteeing that X op will be recomputed.
            nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
            continue

        if _is_primal(node) or _is_fwd_seed_offset(node):
            ban_recomputation_if_allowed(node)

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
        if node_info.is_required_fw(node) and should_ban_recomputation(node):
            ban_recomputation_if_allowed(node)

        # Checks if a node is actually a tuple. Can be simplified to just an isinstance check if we always use faketensors.
        is_non_tensor_node = (
            "val" not in node.meta and "tensor_meta" not in node.meta
        ) or ("val" in node.meta and not isinstance(node.meta["val"], torch.Tensor))

        if is_sym_node(node):
            weight = float(sym_node_size(node))
        elif is_non_tensor_node:
            weight = (
                0.0 if isinstance(node.meta.get("val"), BackwardState) else math.inf
            )
        else:
            weight = get_node_weight(node)
        # Creates the weights on the "node" edge
        nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name + "_out", user.name + "_in", capacity=math.inf)

    # todo(chilli): This is the most questionable of the 3 heuristics for banning recompute.
    # Some example models to look at where this helps perf: poolformer_m36,
    # mixer_b16_224, cait_m36_384

    # The "rough" idea here is that if you have some node that is used by both a
    # node nearby downstream as well as a node far downstream, if we recompute
    # both of the downstream nodes, we're unlikely to be able to fuse both
    # downstream nodes together.

    # Thus, we shouldn't aim to recompute far downstream nodes that depend on
    # this node. That intuition of "far downstream" is captured by whether
    # there's an unfusible op along the chain somewhere

    # It could probably be improved by properly analyzing what's going on in the
    # backwards pass instead of only relying on whether it's unfusible in the
    # forwards.

    def find_first_unfusible(start_nodes: List[fx.Node], max_range: int) -> int:
        """
        Finds the first unfusible node in the chain of nodes starting from
        `start_nodes` and returns its position.
        """
        sorted_nodes: List[Tuple[int, fx.Node, bool]] = []
        for n in start_nodes:
            heapq.heappush(sorted_nodes, (node_info.get_fw_order(n), n, True))

        while len(sorted_nodes) > 0:
            _, node, node_is_fusible = heapq.heappop(sorted_nodes)
            if not node_is_fusible:
                return node_info.get_fw_order(node)
            for user in node.users:
                if node_info.is_required_fw(user):
                    if node_info.get_fw_order(user) > max_range:
                        continue
                    heapq.heappush(
                        sorted_nodes,
                        (node_info.get_fw_order(user), user, is_fusible(node, user)),
                    )
        return max_range

    if min_cut_options.ban_if_used_far_apart:
        for used_node in node_info.required_fw_nodes:
            orders = [
                node_info.get_fw_order(user)
                for user in used_node.users
                if node_info.is_required_fw(user)
            ]
            fw_users = [
                user for user in used_node.users if node_info.is_required_fw(user)
            ]
            if len(orders) > 0:
                first_unfusible_use = find_first_unfusible(fw_users, max(orders))
                for user in tuple(used_node.users):
                    if (
                        node_info.is_required_fw(user)
                        and node_info.get_fw_order(user) > first_unfusible_use
                        and is_fusible(used_node, user)
                    ):
                        if user in banned_nodes:
                            continue
                        log.info(
                            "used above/below fusible %s:(%s) -> %s -> %s:(%s)",
                            used_node,
                            node_info.get_fw_order(used_node),
                            first_unfusible_use,
                            user,
                            node_info.get_fw_order(user),
                        )
                        ban_recomputation_if_allowed(user)

    # This heuristic is fairly straightforward. The idea is that although it is
    # cheap to recompute bandwidth-bound ops, we don't want to end up in a situation
    # where we have a long chain of pointwise ops from the beginning to the end
    # of the model (like say, residual connections)

    # todo: I'm not totally sure why this heuristic matters. It's possible that this is
    # working around Inductor fusion decisions, or that it's a patch over
    # suboptimal partitioning decisions

    # Some models it improves perf on are cait_m36_384, mixer_b16_224, poolformer_m36

    if min_cut_options.ban_if_long_fusible_chains:
        visited = set()
        for start_node in joint_graph.nodes:
            if not node_info.is_required_fw(start_node):
                continue
            fusible = [(node_info.get_fw_order(start_node), start_node)]
            start_order = node_info.get_fw_order(start_node)
            while len(fusible) > 0:
                _, cur = heapq.heappop(fusible)
                if cur in visited:
                    continue
                visited.add(cur)
                # 100 is arbitrary choice to try and prevent degenerate cases
                if (
                    node_info.get_fw_order(cur) > start_order + 100
                    and len(fusible) == 0
                ):
                    log.info(
                        "too long %s %s %s %s",
                        cur,
                        start_node,
                        node_info.get_fw_order(cur),
                        node_info.get_fw_order(start_node),
                    )
                    ban_recomputation_if_allowed(cur)
                    break

                for user in cur.users:
                    if (
                        node_info.is_required_fw(user)
                        and is_fusible(cur, user)
                        and user not in banned_nodes
                    ):
                        heapq.heappush(fusible, (node_info.get_fw_order(user), user))

    try:
        cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    except Exception:
        log.info("Failed to compute min-cut on following graph:")
        log.info("\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph)))
        visualize_min_cut_graph(nx_graph)
        raise

    reachable, non_reachable = partition
    cutset: Set[Tuple[str, str]] = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)

    name_to_node = get_name_to_node(joint_graph)
    # To make this stuff deterministic
    node_idx = {node: idx for idx, node in enumerate(joint_graph.nodes)}
    saved_values = sorted(
        (name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x]
    )
    return saved_values, banned_nodes


def visualize_min_cut_graph(nx_graph):
    import networkx as nx
    import pydot

    dot_format = nx.nx_pydot.to_pydot(nx_graph).to_string()
    dot_graph = pydot.graph_from_dot_data(dot_format)[0]
    for edge in dot_graph.get_edges():
        weight = nx_graph[edge.get_source()][edge.get_destination()]["capacity"]
        # Set edge label to weight
        edge.set_label(str(weight))
        # Color edges with weight 'inf' as red
        if weight == float("inf"):
            edge.set_color("red")
    log.info("Visualizing the failed graph to min_cut_failed.svg")
    dot_graph.write_svg("min_cut_failed.svg")


def get_default_op_list() -> OpTypes:
    default_recomputable_ops: List[Callable] = [
        aten.add,
        aten.sub,
        aten.div,
        aten.atan2,
        aten.mul,
        aten.max,
        aten.min,
        aten.pow,
        aten.remainder,
        aten.fmod,
        aten.__and__,
        aten.__or__,
        aten.__xor__,
        aten.__lshift__,
        aten.__rshift__,
        aten.eq,
        aten.ne,
        aten.ge,
        aten.gt,
        aten.le,
        aten.lt,
        aten.abs,
        aten.bitwise_not,
        aten.ceil,
        aten.floor,
        aten.frac,
        aten.neg,
        aten.relu,
        aten.round,
        aten.silu,
        aten.trunc,
        aten.log,
        aten.log10,
        aten.log1p,
        aten.log2,
        aten.lgamma,
        aten.exp,
        aten.expm1,
        aten.erf,
        aten.erfc,
        aten.cos,
        aten.acos,
        aten.cosh,
        aten.sin,
        aten.asin,
        aten.sinh,
        aten.tan,
        aten.atan,
        aten.tanh,
        aten.atanh,
        aten.sqrt,
        aten.rsqrt,
        aten.reciprocal,
        aten.sigmoid,
        aten.softplus,
        aten.threshold,
        aten.threshold_backward,
        aten.clamp,
        aten.where,
        aten.lerp,
        aten.addcmul,
        aten.gelu,
        aten.gelu_backward,
        aten.sum,
        aten.mean,
        aten._grad_sum_to_size,
        aten.sum_to_size,
        aten.amax,
        aten.to,
        aten.type_as,
        operator.getitem,
        aten.squeeze,
        aten.unsqueeze,
        aten.rsub,
        aten._to_copy,
    ]  # noqa: E501,B950
    recomputable_view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    recomputable_view_ops += [
        aten.view,
        aten.slice,
        aten.t,
        prims.broadcast_in_dim,
        aten.expand,
        aten.as_strided,
        aten.permute,
        aten.select,
    ]
    view_ops = recomputable_view_ops
    default_recomputable_ops += [
        prims.div,
        prims.convert_element_type,
        aten.clone,
        aten._to_copy,
        aten.full_like,
        prims.var,
        prims.sum,
        aten.var,
        aten.std,
        prims.broadcast_in_dim,
        aten.select,
        aten._unsafe_view,
        aten.view,
        aten.expand,
        aten.slice,
        aten.reshape,
        aten.broadcast_tensors,
        aten.scalar_tensor,
        aten.ones,
        aten.new_zeros,
        aten.lift_fresh_copy,
        aten.arange,
        aten.triu,
        aten.var_mean,
        aten.isinf,
        aten.any,
        aten.full,
        aten.as_strided,
        aten.zeros,
        aten.empty,
        aten.empty_like,
        aten.argmax,
        aten.maximum,
        prims.iota,
        prims._low_memory_max_pool2d_offsets_to_indices,
    ]  # noqa: E501,B950
    # Natalia said that we should allow recomputing indexing :)
    default_recomputable_ops += [aten.index, aten.gather]
    default_recomputable_ops += view_ops

    default_recomputable_ops += pointwise_ops()

    default_recomputable_ops += [
        aten.zeros_like,
    ]

    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]
    recomputable_ops = set(default_recomputable_ops)

    random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]
    compute_intensive_ops = [
        aten.mm,
        aten.convolution,
        aten.convolution_backward,
        aten.bmm,
        aten.addmm,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten._scaled_mm,
    ]  # noqa: E501,B950

    fusible_ops = recomputable_ops | set(random_ops)
    return OpTypes(
        set(fusible_ops),
        set(compute_intensive_ops),
        set(random_ops),
        set(view_ops),
        set(recomputable_ops),
    )


def get_name_to_node(graph: fx.Graph):
    name_to_node = {}
    for node in graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def _optimize_runtime_with_given_memory(
    joint_graph: fx.Graph,
    memory: List[float],
    runtimes: List[float],
    max_memory: float,
    node_info: NodeInfo,
    all_recomputable_banned_nodes: List[fx.Node],
) -> Tuple[float, List[int], List[int]]:
    SOLVER = config.activation_memory_budget_solver
    if SOLVER == "greedy":
        return greedy_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "ilp":
        return ilp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp":
        return dp_knapsack(memory, runtimes, max_memory)
    elif callable(SOLVER):
        saved_node_idx, recomp_node_idx = SOLVER(
            memory, joint_graph, max_memory, node_info, all_recomputable_banned_nodes
        )
        return (0.0, saved_node_idx, recomp_node_idx)
    else:
        raise RuntimeError(f"Not aware of memory budget knapsack solver: {SOLVER}")


from torch.utils._mode_utils import no_dispatch


def estimate_runtime(node):
    RUNTIME_MODE = config.activation_memory_budget_runtime_estimator

    def materialize_arg(x):
        if isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.Tensor):
            shape = list(x.meta["val"].shape)

            def realize_symbol(d):
                return hint_int(d, fallback=4096)

            shape = [realize_symbol(s) for s in shape]
            return x.meta["val"].new_empty_strided(
                shape, stride=x.meta["tensor_meta"].stride
            )
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymInt):
            return hint_int(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymFloat):
            return 1.0
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymBool):
            return True
        else:
            return x

    if RUNTIME_MODE == "testing":
        return 1

    elif RUNTIME_MODE == "profile":
        with no_dispatch():
            from torch._inductor.runtime.benchmarking import benchmarker

            args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
            ms = benchmarker.benchmark_gpu(lambda: node.target(*args, **kwargs))
            return ms

    elif RUNTIME_MODE == "flops":
        # todo(chilli): Normalize this to also return ms
        from torch.utils.flop_counter import FlopCounterMode

        args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
        with FlopCounterMode(display=False) as mode:
            node.target(*args, **kwargs)
        counted_flops = mode.get_total_flops()
        return max(counted_flops, 1)
    else:
        raise RuntimeError(f"Not aware of runtime estimator: {RUNTIME_MODE}")


def choose_saved_values_set(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    memory_budget=1,
) -> List[fx.Node]:
    if memory_budget > 1 or memory_budget < 0:
        raise RuntimeError(
            f"The valid ranges for memory budget are 0 <= m <= 1. The provided value is {memory_budget}"
        )
    min_cut_options = MinCutOptions(
        ban_if_used_far_apart=config.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=config.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=config.ban_recompute_materialized_backward,
        ban_if_not_in_allowlist=config.ban_recompute_not_in_allowlist,
        ban_if_reduction=config.ban_recompute_reductions,
    )

    if config.aggressive_recomputation:
        min_cut_options = replace(
            min_cut_options,
            ban_if_used_far_apart=False,
            ban_if_long_fusible_chains=False,
            ban_if_materialized_backward=False,
            ban_if_not_in_allowlist=False,
        )
    if memory_budget == 0:
        return node_info.inputs

    runtime_optimized_saved_values, _ = solve_min_cut(
        joint_graph,
        node_info,
        min_cut_options,
    )
    # return runtime_optimized_saved_values
    if memory_budget == 1:
        return runtime_optimized_saved_values

    def estimate_activations_size(saved_values: List[fx.Node]) -> float:
        return sum(map(_size_of, saved_values)) / 1e9

    min_act_size = estimate_activations_size(node_info.inputs)
    max_act_size = estimate_activations_size(runtime_optimized_saved_values)
    # The optimized choice is smaller than the inputs anyways
    if max_act_size <= min_act_size:
        return runtime_optimized_saved_values

    def get_normalized_size(sz):
        return (sz / 1e9) / (max_act_size - min_act_size)

    def get_mem_ratio(activations: List[fx.Node]):
        return (estimate_activations_size(activations) - min_act_size) / (
            max_act_size - min_act_size
        )

    more_aggressive_options = replace(
        min_cut_options,
        ban_if_used_far_apart=False,
        ban_if_long_fusible_chains=False,
        ban_if_materialized_backward=False,
    )
    more_aggressive_saved_values, _ = solve_min_cut(
        joint_graph, node_info, more_aggressive_options
    )
    if get_mem_ratio(more_aggressive_saved_values) < memory_budget:
        return more_aggressive_saved_values

    aggressive_options = replace(
        more_aggressive_options,
        ban_if_not_in_allowlist=False,
    )
    aggressive_recomputation_saved_values, banned_nodes = solve_min_cut(
        joint_graph, node_info, aggressive_options
    )

    if get_mem_ratio(aggressive_recomputation_saved_values) < memory_budget:
        return aggressive_recomputation_saved_values

    from torch._inductor.fx_utils import get_node_storage

    input_storages = {get_node_storage(node) for node in node_info.inputs}

    def get_recomputable_banned_nodes(banned_nodes: Set[fx.Node]) -> List[fx.Node]:
        return [
            i
            for i in banned_nodes
            if (
                # Only allow recomputing nodes that are actually required for BW
                i.dist_from_bw < int(1e9)  # type: ignore[attr-defined]
                and get_node_storage(i) not in input_storages
            )
        ]

    recomputable_banned_nodes = get_recomputable_banned_nodes(banned_nodes)
    # sort first by name, to ensure determinism when multiple nodes have same size
    recomputable_banned_nodes = sorted(recomputable_banned_nodes, key=lambda x: x.name)

    # default: runtime_optimized_saved_values
    # more aggressive: more_aggressive_saved_values
    # full aggressive: aggressive_recomputation_saved_values

    all_recomputable_banned_nodes = sorted(
        recomputable_banned_nodes, key=_size_of, reverse=True
    )
    if len(all_recomputable_banned_nodes) == 0:
        return node_info.inputs
    memories_banned_nodes = [
        get_normalized_size(_size_of(i)) for i in all_recomputable_banned_nodes
    ]
    runtimes_banned_nodes = [
        estimate_runtime(node) for node in all_recomputable_banned_nodes
    ]
    from torch.utils._mode_utils import no_dispatch

    def get_saved_values_knapsack(memory_budget, node_info, joint_graph):
        with no_dispatch():
            (
                expected_runtime,
                saved_node_idxs,
                recomputable_node_idxs,
            ) = _optimize_runtime_with_given_memory(
                joint_graph,
                memories_banned_nodes,
                runtimes_banned_nodes,
                max(memory_budget, 0),
                node_info,
                all_recomputable_banned_nodes,
            )
            if AOT_PARTITIONER_DEBUG:
                max_runtime = max(
                    runtimes_banned_nodes
                )  # For normalizing runtimes in logs
                input_summary = [
                    f"\n\t\t\t{index}, {memory}, {runtime / max_runtime}, {node.op}, {node.target}, {node.meta}, {node.args}"
                    for index, (memory, runtime, node) in enumerate(
                        zip(
                            memories_banned_nodes,
                            runtimes_banned_nodes,
                            all_recomputable_banned_nodes,
                        )
                    )
                ]
                joint_graph_nodes = [node.name for node in joint_graph.nodes]
                joint_graph_edges = [
                    (inp.name, node.name)
                    for node in joint_graph.nodes
                    for inp in node.all_input_nodes
                ]
                knapsack_summary = f"""
Activation Checkpointing - Knapsack Problem Summary:
    Input:
        Solver: {config.activation_memory_budget_solver}
        Max Memory: {max(config.activation_memory_budget, 0)}
        Graph Nodes: {joint_graph_nodes}
        Graph Edges: {joint_graph_edges}
        (Index, Memory, Runtime, Node.Op, Node.Target, Metadata): {"".join(input_summary)}
    Output:
        Expected Runtime: {expected_runtime}
        Saved Nodes: {saved_node_idxs}
        Recomputable Nodes: {recomputable_node_idxs}
            """
                torch._logging.trace_structured(
                    name="artifact",
                    payload_fn=lambda: knapsack_summary,
                )
                log.info(knapsack_summary)
        dont_ban = set()
        for idx in recomputable_node_idxs:
            # if idx in all_recomputable_banned_nodes:
            try:
                dont_ban.add(all_recomputable_banned_nodes[idx])
            except BaseException:
                pass

        assert dont_ban.issubset(all_recomputable_banned_nodes)

        saved_values, _ = solve_min_cut(
            joint_graph,
            node_info,
            aggressive_options,
            dont_ban,
        )
        return saved_values, expected_runtime

    if config.visualize_memory_budget_pareto:
        options = []
        for sweep_memory_budget in range(100, -1, -5):
            saved_values, expected_runtime = get_saved_values_knapsack(
                sweep_memory_budget / 100, node_info=node_info, joint_graph=joint_graph
            )
            options.append(
                (
                    sweep_memory_budget,
                    sum(runtimes_banned_nodes) - expected_runtime,
                    get_mem_ratio(saved_values),
                )
            )

        import matplotlib.pyplot as plt

        x_values = [item[2] for item in options]
        y_values = [item[1] for item in options]

        # Plotting the values with updated axis labels and chart title
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker="o")

        # Adding labels for each point
        for i, txt in enumerate(x_values):
            plt.annotate(
                f"{txt:.2f}",
                (txt, y_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel("Memory Budget")
        plt.ylabel("Runtime of Recomputed Components")
        plt.title("Pareto Frontier of Memory Budget vs. Recomputation Runtime")
        plt.grid(True)
        fig = plt.gcf()
        plt.show()
        fig_name = f"memory_budget_pareto_{get_aot_graph_name()}.png"
        fig.savefig(fig_name)
        log.warning("Generated Pareto frontier curve at %s", fig_name)

    # todo(chilli): Estimated doesn't align exactly with actual - actual is
    # usually less memory than estimated. i'm guessing (actually quite
    # unsure about this) that's because estimated is just only including
    # tensors we actually banned from recompute, but there may be other
    # tensors that we choose to save.

    return get_saved_values_knapsack(
        memory_budget=memory_budget, node_info=node_info, joint_graph=joint_graph
    )[0]


def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    compiler="inductor",
    *,
    num_fwd_outputs,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """

    joint_module.graph.eliminate_dead_code()
    joint_module.recompile()

    fx_g = joint_module.graph

    #  add the CSE pass
    if config.cse:
        cse_graph = fx_graph_cse(fx_g)
        joint_module.graph = cse_graph
    joint_graph = joint_module.graph

    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        joint_module = cleanup_recompute_tags(joint_module)

    def classify_nodes(joint_module):
        name_to_node = get_name_to_node(joint_module.graph)
        required_bw_nodes = set()
        for node in joint_module.graph.nodes:
            if node.op == "placeholder" and "tangents" in node.target:
                required_bw_nodes.add(node)
            elif _must_be_in_backward(node):
                required_bw_nodes.add(node)

            if node in required_bw_nodes:
                for user in node.users:
                    required_bw_nodes.add(user)

        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        fwd_seed_offset_inputs = list(
            filter(_is_fwd_seed_offset, joint_module.graph.nodes)
        )
        inputs = primal_inputs + fwd_seed_offset_inputs
        fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
            joint_module, num_fwd_outputs=num_fwd_outputs
        )
        required_bw_nodes.update(
            o for o in bwd_outputs if o is not None and o.op != "output"
        )
        forward_only_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph, inputs, fwd_outputs, "forward"
        )
        required_fw_nodes: Set[fx.Node] = {
            name_to_node[node.name]
            for node in forward_only_graph.nodes
            if node.op != "output"
        }
        unclaimed_nodes = {
            node
            for node in joint_module.graph.nodes
            if node not in required_fw_nodes and node not in required_bw_nodes
        }
        fw_cnt = 0
        fw_order = {}
        for node in joint_module.graph.nodes:
            if node in required_fw_nodes:
                fw_order[node] = fw_cnt
                fw_cnt += 1
        return NodeInfo(
            inputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes, fw_order
        )

    node_info = classify_nodes(joint_module)

    # networkx blows up on graphs with no required backward nodes
    # Since there's nothing to partition anyway, and the default partitioner can "handle"
    # this case, send our graph over to the default partitioner.
    if len(node_info.required_bw_nodes) == 0:
        return default_partition(
            joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs
        )

    for node in reversed(joint_module.graph.nodes):
        if node.op == "output":
            node.dist_from_bw = int(1e9)
        elif not node_info.is_required_fw(node):
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = int(1e9)
            for user in node.users:
                node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)

    memory_budget = config.activation_memory_budget
    for node in joint_graph.nodes:
        if isinstance(node.meta.get("memory_budget", None), float):
            memory_budget = node.meta["memory_budget"]
            break
    saved_values = choose_saved_values_set(
        joint_graph,
        node_info,
        memory_budget=memory_budget,
    )
    # save_for_backward on tensors and stashes symints in autograd .ctx
    saved_sym_nodes = list(filter(is_sym_node, saved_values))
    saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))

    # NB: saved_sym_nodes will be mutated to reflect the actual saved symbols
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
    )

    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    if AOT_PARTITIONER_DEBUG:
        # Calculate sorted sizes of saved values
        sorted_sizes = sorted([(_size_of(i), str(i)) for i in saved_values])

        # Log total theoretical activations stored
        total_activations_size_gb = sum(_size_of(i) for i in saved_values) / 1e9
        log.debug("Theoretical Activations Stored: %.2f GB", total_activations_size_gb)

        # Log theoretical per activation storage sizes
        log.debug("Theoretical Per Activation Storage Sizes: %s", sorted_sizes)
        fw_module_nodes = {
            node.name for node in fw_module.graph.nodes if node.op == "call_function"
        }
        bw_module_nodes = {
            node.name for node in bw_module.graph.nodes if node.op == "call_function"
        }
        remat_nodes = fw_module_nodes & bw_module_nodes

        counts: Dict[str, int] = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, "_overloadpacket"):
                counts[str(node.target._overloadpacket)] += 1
        log.debug(
            "# remat/fw/bw: %d/%d/%d",
            len(remat_nodes),
            len(fw_module_nodes),
            len(bw_module_nodes),
        )
        rematerialized_ops = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        log.debug("Count of Ops Rematerialized: %s", rematerialized_ops)
    return fw_module, bw_module


def draw_graph(
    traced: torch.fx.GraphModule,
    fname: str,
    figname: str = "fx_graph",
    clear_meta: bool = True,
    prog: Optional[Union[str, List[str]]] = None,
    parse_stack_trace: bool = False,
    dot_graph_shape: Optional[str] = None,
) -> None:
    if clear_meta:
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        for node in traced.graph.nodes:
            node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = "." + config.torch_compile_graph_format
    log.info("Writing FX graph to file: %s%s", base, ext)
    g = graph_drawer.FxGraphDrawer(
        traced,
        figname,
        parse_stack_trace=parse_stack_trace,
        dot_graph_shape=dot_graph_shape,
    )
    x = g.get_main_dot_graph()
    write_method = getattr(x, "write_" + ext.lstrip("."))
    fname = f"{base}{ext}"
    if prog is None:
        write_method(fname)
    else:
        write_method(fname, prog=prog)
