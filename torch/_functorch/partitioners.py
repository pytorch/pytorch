# mypy: ignore-errors

from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    hint_int, free_symbols, is_symbol_binding_fx_node, find_symbol_binding_fx_nodes
)
from torch.fx.experimental._backward_state import BackwardState
import torch
import torch.fx as fx
import operator
import math
import heapq
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Set, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
from torch._dynamo.utils import lazy_format_graph_code
import logging
import torch.distributed as dist


AOT_PARTITIONER_DEBUG = config.debug_partitioner
log = logging.getLogger(__name__)


def must_recompute(node):
    return node.meta.get("recompute", False)

def has_recomputable_ops(fx_g):
    found = False
    for node in fx_g.graph.nodes:
        if must_recompute(node):
            return True
    return False

def has_recomputable_rng_ops(fx_g):
    for node in fx_g.graph.nodes:
        if must_recompute(node) and hasattr(node.target, "tags") and torch.Tag.nondeterministic_seeded in node.target.tags:
            return True
    return False

def sym_node_size(node):
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    assert isinstance(node.meta["val"], torch.SymFloat)
    return 4

class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(joint_graph, inputs, outputs):
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
        if node in inputs:
            continue
        elif node.op == 'placeholder':
            env[node] = InvalidNode
        elif node.op == 'call_function':
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [isinstance(env[x], InvalidNodeBase) for x in all_args if isinstance(x, fx.Node)]
            if any(all_args):
                env[node] = InvalidNode
                continue
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == 'get_attr':
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == 'output':
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            assert not isinstance(env[x], InvalidNodeBase), f"Node {x} was invalid, but is output"
            output_values.append(env[x])
        else:
            output_values.append(x)
    new_graph.output(output_values)

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node):
    return (
        node.op == "placeholder"
        and "tangents" not in node.target
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )

def _is_tangent(node):
    return node.op == "placeholder" and "tangents" in node.target

def _is_bwd_seed_offset(node):
    return node.op == "placeholder" and ("bwd_seed" in node.target or "bwd_base_offset" in node.target)

def _is_fwd_seed_offset(node):
    return node.op == "placeholder" and ("fwd_seed" in node.target or "fwd_base_offset" in node.target)

def _is_backward_state(node):
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _extract_fwd_bwd_outputs(joint_module: fx.GraphModule, *, num_fwd_outputs):
    outputs = pytree.arg_tree_leaves(*(node.args for node in joint_module.graph.find_nodes(op="output")))
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs


def _remove_by_name(saved_values, name):
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values, saved_sym_nodes, *, num_fwd_outputs):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    tangent_inputs = [*filter(_is_tangent, placeholders)]
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs
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
        fwd_outputs + saved_values + saved_sym_nodes
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs + backward_state_inputs,
        bwd_outputs
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
        return min_cut_rematerialization_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
    forward_node_names = {node.name for node in forward_only_graph.nodes if node.op != 'output'}
    saved_values = []
    saved_sym_nodes = []

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            # Symints must be kept separate from tensors so that PythonFunction only calls
            # save_for_backward on tensors and stashes symints in autograd .ctx
            saved_sym_nodes.append(node)
        elif (
            'tensor_meta' not in node.meta
            and node.op == 'call_function'
        ):
            # Since we can't save tuple of tensor values, we need to flatten out what we're saving
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [n for n in node.users if n.name not in forward_node_names]
            if 'tensor_meta' in node.meta and all(is_sym_node(n) for n in backward_usages):
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

    return _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)


def _prod(x):
    s = 1
    for i in x:
        s *= i
    return s

def _tensor_nbytes(numel, dtype):
    return numel * dtype.itemsize

def _size_of(node: fx.Node) -> int:
    if 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, py_sym_types):
            if isinstance(val, torch.SymInt):
                return 1
            else:
                return 999999
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(_tensor_nbytes(hint_int(n.numel(), fallback=4098), n.dtype) for n in val if isinstance(n, torch.Tensor))
        elif isinstance(val, torch.Tensor):
            return _tensor_nbytes(hint_int(val.numel(), fallback=4098), val.dtype)

        raise RuntimeError(f"Unknown metadata type {type(val)}")

    # Only needed since we don't always trace with fake tensors.
    if 'tensor_meta' in node.meta:
        metadata = node.meta['tensor_meta']
        # TODO: What is to_size_hint suppose to be?
        numel = _prod(map(to_size_hint, metadata.shape))  # noqa: F821
        dtype = metadata.dtype
    else:
        return 0

    return _tensor_nbytes(numel, dtype)


# Used for some investigative purposes
def _count_ops(graph):
    from collections import defaultdict
    cnt = defaultdict(int)
    for node in graph.nodes:
        if node.op == 'call_function':
            cnt[node.target.__name__] += 1
    print(sorted(cnt.items(), key=lambda x: x[1], reverse=True))


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


def sort_depths(args, depth_map):
    arg_depths = {arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)}
    return sorted(arg_depths.items(), key=lambda x: x[1], reverse=True)


def reordering_to_mimic_autograd_engine(gm):
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
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx

    # Populate depth for the nodes. Depth is the distance from the inputs.
    depths = {}
    output_node = next(iter(gm.graph.find_nodes(op="output")))
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            depths[node] = 0
        else:
            depths[node] = max([depths[arg] for arg in node.all_input_nodes], default=0)

    def insert_node_in_graph(node):
        if node in env:
            return env[node]

        # Bias traversal towards the nodes that have higher depth - prioritizes
        # critical path first.
        for arg, _ in sort_depths(node.all_input_nodes, depths):
            env[arg] = insert_node_in_graph(arg)
        env[node] = new_graph.node_copy(node, lambda x: env[x])
        return env[node]

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
    for node in list(gm.graph.nodes)[order[first_node_in_bwd]:]:
        insert_node_in_graph(node)

    # The output node is already built by the traversal.
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def functionalize_rng_ops(joint_module, fw_module, bw_module, num_sym_nodes):
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
    recomputable_rng_ops_map = dict()
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

    for node in bw_module.graph.find_nodes(op="placeholder"):
        if "tangent" in node.name:
            bw_tangent_start_node = node
            break


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
                kwargs=fw_node.kwargs
            )
            state = fw_graph.create_node("call_function", operator.getitem, args=(functional_fw_node, 0), kwargs={})
            rng_output = fw_graph.create_node("call_function", operator.getitem, args=(functional_fw_node, 1,), kwargs={})
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
                kwargs=bw_node.kwargs
            )

            bw_node.replace_all_uses_with(rng_output)
            bw_graph.erase_node(bw_node)


    # Add the rng states in the output of the fwd graph. AOT Autograd assumes
    # that symints are at the end of forward graph outputs. So, insert the new
    # rng states accordingly.
    fw_output_node = next(iter(fw_module.graph.find_nodes(op="output")))
    fw_outputs = fw_output_node.args[0]
    sym_node_start_idx = len(fw_outputs) - num_sym_nodes
    outputs = fw_outputs[:sym_node_start_idx] + fw_rng_state_outputs + fw_outputs[sym_node_start_idx:]
    fw_module.graph.output(outputs)
    fw_module.graph.erase_node(fw_output_node)
    fw_module.recompile()
    bw_module.recompile()
    return fw_module, bw_module


def cleanup_recompute_tags(joint_module):
    """
    If there are two consecutive checkpointed blocks with no operator in
    between, we would still want to stash the tensor at the boundary of
    checkpointed blocks. The following pass makes the last output node
    non-recomputable to allow for that.
    """
    for node in joint_module.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if must_recompute(user) and user.meta["recompute"] > node.meta["recompute"]:
                    node.meta["recompute"] = 0
    return joint_module


def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule, _joint_inputs, compiler="inductor", recomputable_ops=None,
    *, num_fwd_outputs
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
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("Need networkx installed to perform smart recomputation "
                           "heuristics") from e

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

    name_to_node = {}
    for node in joint_module.graph.nodes:
        name_to_node[node.name] = node

    def classify_nodes(joint_module):
        required_bw_nodes = set()
        for node in joint_module.graph.nodes:
            if node.op == 'placeholder' and "tangents" in node.target:
                required_bw_nodes.add(node)
            if node in required_bw_nodes:
                for user in node.users:
                    required_bw_nodes.add(user)

        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
        inputs = primal_inputs + fwd_seed_offset_inputs
        fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
        required_bw_nodes.update(o for o in bwd_outputs if o is not None and o.op != 'output')
        forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
        required_fw_nodes = {name_to_node[node.name] for node in forward_only_graph.nodes
                             if node.op != 'output'}
        unclaimed_nodes = {node for node in joint_module.graph.nodes
                           if node not in required_fw_nodes and node not in required_bw_nodes}
        return fwd_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes, inputs

    orig_fw_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes, inputs = classify_nodes(joint_module)

    # networkx blows up on graphs with no required backward nodes
    # Since there's nothing to partition anyway, and the default partitioner can "handle"
    # this case, send our graph over to the default partitioner.
    if len(required_bw_nodes) == 0:
        return default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)

    def is_fusible(a, b):
        # We can perform "memory fusion" into a cat, but cat cannot be a
        # producer to a fusion
        if get_aten_target(b) == aten.cat:
            return True
        return get_aten_target(a) in fusible_ops and get_aten_target(b) in fusible_ops

    fw_order = 0
    for node in joint_module.graph.nodes:
        if node in required_fw_nodes:
            node.fw_order = fw_order
            fw_order += 1

    for node in reversed(joint_module.graph.nodes):
        if node not in required_fw_nodes:
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = int(1e9)
            for user in node.users:
                node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)

    aten = torch.ops.aten
    prims = torch.ops.prims

    # compiler == "nvfuser" is the default set of recomputable ops
    default_recomputable_ops = [aten.add, aten.sub, aten.div, aten.atan2, aten.mul, aten.max, aten.min, aten.pow, aten.remainder, aten.fmod, aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__, aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor, aten.frac, aten.neg, aten.relu, aten.round, aten.silu, aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2, aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc, aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin, aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh, aten.sqrt, aten.rsqrt, aten.reciprocal, aten.sigmoid, aten.softplus, aten.threshold, aten.threshold_backward, aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu, aten.gelu_backward, aten.sum, aten.mean, aten._grad_sum_to_size, aten.sum_to_size, aten.amax, aten.to, aten.type_as, operator.getitem, aten.squeeze, aten.unsqueeze, aten.rsub, aten._to_copy]  # noqa: E501,B950
    view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    if compiler == "inductor":
        default_recomputable_ops += [prims.div, prims.convert_element_type, aten.clone, aten._to_copy, aten.full_like, prims.var, prims.sum, aten.var, aten.std, prims.broadcast_in_dim, aten.select, aten._unsafe_view, aten.view, aten.expand, aten.slice, aten.reshape, aten.broadcast_tensors, aten.scalar_tensor, aten.ones, aten.new_zeros, aten.lift_fresh_copy, aten.arange, aten.triu, aten.var_mean, aten.isinf, aten.any, aten.full, aten.as_strided, aten.zeros, aten.argmax, aten.maximum, prims.iota]  # noqa: E501,B950
        view_ops += [aten.view, aten.slice, aten.t, prims.broadcast_in_dim, aten.expand, aten.as_strided, aten.permute]
        # Natalia said that we should allow recomputing indexing :)
        default_recomputable_ops += [aten.index, aten.gather]
    default_recomputable_ops += view_ops

    default_recomputable_ops += pointwise_ops()

    default_recomputable_ops += [
        aten.zeros_like,
    ]

    default_recomputable_ops += [
        method_to_operator(m)
        for m in magic_methods
    ]
    recomputable_ops = set(recomputable_ops) if recomputable_ops is not None else set(default_recomputable_ops)

    random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]
    compute_intensive_ops = [aten.mm, aten.convolution, aten.convolution_backward, aten.bmm, aten.addmm, aten._scaled_dot_product_flash_attention, aten._scaled_dot_product_efficient_attention, aten.upsample_bilinear2d]  # noqa: E501,B950

    fusible_ops = recomputable_ops | set(random_ops)
    if AOT_PARTITIONER_DEBUG:
        joint_module_ops = {
            str(node.target._overloadpacket)
            for node in joint_module.graph.nodes
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
        }
        ops_ignored = joint_module_ops - {str(i) for i in recomputable_ops}
        print("Ops banned from rematerialization: ", ops_ignored)
        print()

    BAN_IF_USED_FAR_APART = config.ban_recompute_used_far_apart
    BAN_IF_LONG_FUSIBLE_CHAINS = config.ban_recompute_long_fusible_chains
    BAN_IF_MATERIALIZED_BACKWARDS = config.ban_recompute_materialized_backward
    BAN_IF_NOT_IN_ALLOWLIST = config.ban_recompute_not_in_allowlist
    BAN_IF_REDUCTION = config.ban_recompute_reductions

    if config.aggressive_recomputation:
        BAN_IF_MATERIALIZED_BACKWARDS = False
        BAN_IF_USED_FAR_APART = False
        BAN_IF_LONG_FUSIBLE_CHAINS = False
        BAN_IF_NOT_IN_ALLOWLIST = False

    def is_materialized_backwards(node):
        if get_aten_target(node) in view_ops:
            return False
        cur_nodes = {node}
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if user not in required_fw_nodes and not is_fusible(cur, user):
                    return True
                if get_aten_target(user) in view_ops:
                    cur_nodes.add(user)

        return False

    def should_ban_recomputation(node):
        if node.op != 'call_function':
            return False
        if node.target == operator.getitem:
            return False
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return False

        # NB: "recompute" == 0 means that must save this node.
        if node.meta.get("recompute", None) == 0:
            return True

        if BAN_IF_NOT_IN_ALLOWLIST:
            if get_aten_target(node) not in recomputable_ops:
                return True
        else:
            ignored_ops = random_ops + compute_intensive_ops
            if get_aten_target(node) in ignored_ops:
                return True


        # If a node *must* be materialized in the backwards pass, then we
        # should never recompute it. This is a pretty subtle point.  In
        # general, the assumption we make is that recomputing a node in the
        # backwards pass is "free". However, if a node must be materialized
        # in the backwards pass, then recomputing it is never free.
        if is_materialized_backwards(node) and BAN_IF_MATERIALIZED_BACKWARDS:
            log.info("materialized backwards: %s %s", node, tuple(node.users))
            return True

        # Arbitrary hack that sometimes seems to help things. The above
        # modification appears to have made this heuristic a lot less critical
        # for performance.
        # NB: As of PR #121692, this hack no longer seems necessary.
        if not graph_has_recomputable_ops:
            if compiler == "inductor" and node.dist_from_bw > config.max_dist_from_bw:
                return True

        # If the output of an op is 4x smaller (arbitrary choice),
        # then we don't allow recomputation. The idea here is that for
        # things like reductions, saving the output of the reduction is very
        # cheap/small, and it makes sure we don't do things like recompute
        # normalizations in the backwards.
        if BAN_IF_REDUCTION:
            input_tensors_size = sum(_size_of(i) for i in node.args if isinstance(i, fx.Node))
            output_size = _size_of(node)
            return (output_size * 4 < input_tensors_size)
        return False


    def is_materialized(node):
        if node.op == 'placeholder':
            return True

        return not all(is_fusible(node, user) for user in node.users)

    def get_node_weight(node) -> int:
        mem_sz = _size_of(node)

        # Heuristic to bias towards nodes closer to the backwards pass
        # Complete guess about current value
        mem_sz = int(mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1)))
        if is_materialized(node):
            return mem_sz
        else:
            return mem_sz * 2

    def min_cut():
        nx_graph = nx.DiGraph()
        banned_nodes = set()

        def ban_recomputation_if_allowed(node):
            # This bans recomputation of the node unless we've been forced not to by
            # user annotation
            # NB: "recompute" > 0 means that user annotation has asked us to
            # recompute it
            if node.meta.get("recompute", 0) > 0:
                return False

            if 'val' in node.meta and isinstance(node.meta['val'], torch.SymFloat):
                return False

            banned_nodes.add(node)
            # A node will only ever be recomputed if there is a path from an
            # ancestor of this node to the backwards path through this node that
            # doesn't go through any saved value. If this node is saved, then that
            # condition is not possible.
            nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)
            return True

        for node in joint_graph.nodes:
            if node.op == 'output':
                continue

            if node in required_bw_nodes:
                if node not in inputs:
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

            if _is_primal(node) or _is_fwd_seed_offset(node):
                ban_recomputation_if_allowed(node)

            # If a node can't be recomputed (too expensive or involves randomness),
            # we prevent it from being recomputed by adding an inf edge to the source
            # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
            if node in required_fw_nodes and should_ban_recomputation(node):
                ban_recomputation_if_allowed(node)

            # Checks if a node is actually a tuple. Can be simplified to just an isinstance check if we always use faketensors.
            is_non_tensor_node = (('val' not in node.meta and 'tensor_meta' not in node.meta) or
                                ('val' in node.meta and not isinstance(node.meta['val'], torch.Tensor)))

            if is_sym_node(node):
                weight = sym_node_size(node)
            elif is_non_tensor_node:
                weight = 0 if isinstance(node.meta.get("val"), BackwardState) else math.inf
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
            sorted_nodes = []
            for n in start_nodes:
                heapq.heappush(sorted_nodes, (n.fw_order, n, True))

            while len(sorted_nodes) > 0:
                _, node, node_is_fusible = heapq.heappop(sorted_nodes)
                if not node_is_fusible:
                    return node.fw_order
                for user in node.users:
                    if user in required_fw_nodes:
                        if user.fw_order > max_range:
                            continue
                        heapq.heappush(sorted_nodes, (user.fw_order, user, is_fusible(node, user)))
            return max_range

        if BAN_IF_USED_FAR_APART:
            for used_node in required_fw_nodes:
                orders = [user.fw_order for user in used_node.users if user in required_fw_nodes]
                fw_users = [user for user in used_node.users if user in required_fw_nodes]
                if len(orders) > 0:
                    first_unfusible_use = find_first_unfusible(fw_users, max(orders))
                    for user in tuple(used_node.users):
                        if user in required_fw_nodes and user.fw_order > first_unfusible_use and is_fusible(used_node, user):
                            if user in banned_nodes:
                                continue
                            log.info(
                                "used above/below fusible %s:(%s) -> %s -> %s:(%s)",
                                used_node, used_node.fw_order, first_unfusible_use, user, user.fw_order
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

        if BAN_IF_LONG_FUSIBLE_CHAINS:
            visited = set()
            for start_node in joint_graph.nodes:
                if start_node not in required_fw_nodes:
                    continue
                fusible = [(start_node.fw_order, start_node)]
                start_order = start_node.fw_order
                while len(fusible) > 0:
                    _, cur = heapq.heappop(fusible)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    # 100 is arbitrary choice to try and prevent degenerate cases
                    if cur.fw_order > start_order + 100 and len(fusible) == 0:
                        log.info("too long %s %s %s %s", cur, start_node, cur.fw_order, start_node.fw_order)
                        ban_recomputation_if_allowed(cur)
                        break

                    for user in cur.users:
                        if user in required_fw_nodes and is_fusible(cur, user) and user not in banned_nodes:
                            heapq.heappush(fusible, (user.fw_order, user))

        try:
            cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
        except Exception:
            print('Failed to compute min-cut on following graph:')
            print('\n'.join(nx.readwrite.edgelist.generate_edgelist(nx_graph)))
            raise

        reachable, non_reachable = partition
        cutset = set()
        for u, nbrs in ((n, nx_graph[n]) for n in reachable):
            cutset.update((u, v) for v in nbrs if v in non_reachable)

        # if dist.get_rank() == 0:
        #     print(f"reachable: {sorted(list(reachable))}")
        #     print(f"non_reachable: {sorted(list(non_reachable))}")
        #     print(f"cutset: {sorted(list(cutset))}")

        cut_nodes = set()
        for node_in, node_out in cutset:
            assert node_in[:-3] == node_out[:-4]
            node_name = node_in[:-3]
            cut_nodes.add(node_name)

        # To make this stuff deterministic
        node_idx = {node: idx for idx, node in enumerate(joint_module.graph.nodes)}
        saved_values = sorted((name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x])
        # save_for_backward on tensors and stashes symints in autograd .ctx
        saved_sym_nodes = list(filter(is_sym_node, saved_values))
        saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))

        # if dist.get_rank() == 0:
        #     # print(f"cut_value: {cut_value}")
        #     # print(f"cut_nodes: {cut_nodes}")
        #     import pickle
        #     filename = "nx_graph.pkl"
        #     try:
        #         os.remove(filename)
        #     except OSError:
        #         pass
        #     with open(filename, "wb") as file:
        #         pickle.dump(nx_graph, file)

        return saved_sym_nodes, saved_values

    saved_sym_nodes, saved_values = min_cut()

    def flatten_arg_list(args):
        flat_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                flat_args.extend(flatten_arg_list(arg))
            else:
                flat_args.append(arg)
        return flat_args

    def is_alias_of_primal_input(primal_inputs, node):
        if hasattr(node, "target") and node.target in [
            # List of view ops. TODO add more
            torch.ops.aten.t.default,
            torch.ops.aten.as_strided.default,
            torch.ops.aten.permute.default,
        ]:
            view_chain = [node]
            flattened_arg_list = flatten_arg_list(node.args)
            for arg in flattened_arg_list:
                if arg in primal_inputs:
                    return True, view_chain
                else:
                    upstream_is_alias, upstream_view_chain = is_alias_of_primal_input(primal_inputs, arg)
                    if upstream_is_alias:
                        view_chain.extend(upstream_view_chain)
                        return True, view_chain
        return False, []

    # TODO: rename
    def if_primal_input_alias_is_saved_then_move_the_view_chain_to_bwd_graph(saved_values):
        # Trace lineage of saved nodes that are aliases of primals. Move the entire view chain to BWD graph
        # so that only primals (along with non-view op output intermediates) are saved as FWD graph output.
        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        for saved_value in saved_values:
            is_alias, view_chain = is_alias_of_primal_input(primal_inputs, saved_value)
            if is_alias:
                required_bw_nodes.update(set(view_chain))

    if config.move_view_chain_to_bwd_graph:
        # TODO(yf225): if any saved nodes is an alias of primal input, save the primal input instead of the alias.
        # We do this by updating `required_bw_nodes` and then redo min-cut algorithm.
        if_primal_input_alias_is_saved_then_move_the_view_chain_to_bwd_graph(saved_values)
        saved_sym_nodes, saved_values = min_cut()

    # NB: saved_sym_nodes will be mutated to reflect the actual saved symbols
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)

    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    if AOT_PARTITIONER_DEBUG:
        from torch._inductor.fx_utils import get_node_storage
        storages = {get_node_storage(node) for node in saved_values}
        print("Theoretical Activations Stored: ", sum([_size_of(i) for i in saved_values]) / 1e9)
        sorted_sizes = sorted([(_size_of(i), str(i)) for i in saved_values])
        fw_module_nodes = {node.name for node in fw_module.graph.nodes if node.op == 'call_function'}
        bw_module_nodes = {node.name for node in bw_module.graph.nodes if node.op == 'call_function'}
        remat_nodes = fw_module_nodes & bw_module_nodes

        counts = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, '_overloadpacket'):
                counts[str(node.target._overloadpacket)] += 1
        print(f"# remat/fw/bw: {len(remat_nodes)}/{len(fw_module_nodes)}/{len(bw_module_nodes)}")
        print("Count of Ops Rematerialized: ", sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return fw_module, bw_module


def draw_graph(
    traced: torch.fx.GraphModule,
    fname: str,
    figname: str = "fx_graph",
    clear_meta: bool = True,
    prog: Union[str, List[str]] = None,
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
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
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


def draw_joint_graph(
    graph: torch.fx.GraphModule,
    joint_inputs,
    file_name: str = "full_graph.png",
    dot_graph_shape: Optional[str] = None,
):
    draw_graph(graph, file_name, dot_graph_shape=dot_graph_shape)
    return default_partition(graph, joint_inputs)
