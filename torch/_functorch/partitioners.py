from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.symbolic_shapes import (
    hint_int, magic_methods, method_to_operator, free_symbols,
    is_symbol_binding_fx_node, find_symbol_binding_fx_nodes
)
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import Tuple
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools

AOT_PARTITIONER_DEBUG = config.debug_partitioner


def is_symint_node(node):
    assert hasattr(node, 'meta'), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta['val'], torch.SymInt)


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
            all_args = pytree.tree_flatten((node.args, node.kwargs))[0]
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


def _extract_fwd_bwd_outputs(joint_module: fx.GraphModule, *, num_fwd_outputs):
    outputs = pytree.tree_flatten([node.args for node in joint_module.graph.nodes if node.op == 'output'])[0]
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs


def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values, saved_sym_nodes=(), *, num_fwd_outputs):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    tangent_inputs = list(filter(_is_tangent, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    bwd_seed_offset_inputs = list(filter(_is_bwd_seed_offset, joint_module.graph.nodes))

    # Construct the forward module
    # Keep symints separate from tensors, passed between fwd/bwd graphs, and in the right order.
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs
    )

    # This is to filter out saved values that don't actually end up being used by the backwards pass
    for node in bwd_graph.nodes:
        if node.op == 'placeholder' and not node.users:
            for saved_value in saved_values:
                if saved_value.name == node.name:
                    saved_values.remove(saved_value)
                    break

            for saved_sym in saved_sym_nodes:
                if saved_sym.name == node.name:
                    saved_sym_nodes.remove(saved_sym)
                    break

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


    # Update saved_sym_nodes that are now reordered to have all bindings
    # at front
    saved_sym_nodes = saved_sym_nodes_binding + saved_sym_nodes_derived

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs
    )

    fwd_module = fx.GraphModule(joint_module, fwd_graph)
    bwd_module = fx.GraphModule(joint_module, bwd_graph)
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
            for user in users:
                saved_values.append(user)
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
                for user in backward_usages:
                    saved_sym_nodes.append(user)
            else:
                saved_values.append(node)
    saved_values = list({k: None for k in saved_values}.keys())
    saved_sym_nodes = list({k: None for k in saved_sym_nodes}.keys())

    return _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)


def _prod(x):
    s = 1
    for i in x:
        s *= i
    return s

def _tensor_nbytes(numel, dtype):
    sizes = {
        torch.complex64: 8,
        torch.complex128: 16,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
    }
    if dtype not in sizes:
        raise NotImplementedError("Don't know the size of dtype ", dtype)

    return numel * sizes[dtype]

def _size_of(node: fx.Node) -> int:
    if 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, py_sym_types):
            if isinstance(val, torch.SymInt):
                return 1
            else:
                return 999999
        elif isinstance(val, (list, tuple)):
            return sum(_tensor_nbytes(hint_int(n.numel()), n.dtype) for n in val if isinstance(n, torch.Tensor))
        elif isinstance(val, torch.Tensor):
            return _tensor_nbytes(hint_int(val.numel()), val.dtype)

        raise RuntimeError(f"Unknown metadata type {type(val)}")

    # Only needed since we don't always trace with fake tensors.
    if 'tensor_meta' in node.meta:
        metadata = node.meta['tensor_meta']
        numel = _prod(map(to_size_hint, metadata.shape))
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


def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule, _joint_inputs, compiler="nvfuser", recomputable_ops=None,
    *, num_fwd_outputs
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimintation.

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
    full_bw_graph = joint_module.graph

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
        required_bw_nodes.update(o for o in bwd_outputs if o is not None)
        forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
        required_fw_nodes = {name_to_node[node.name] for node in forward_only_graph.nodes
                             if node.op != 'output'}
        unclaimed_nodes = {node for node in joint_module.graph.nodes
                           if node not in required_fw_nodes and node not in required_bw_nodes}
        return fwd_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes

    orig_fw_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes = classify_nodes(joint_module)

    # networkx blows up on graphs with no required backward nodes
    # Since there's nothing to partition anyway, and the default partitioner can "handle"
    # this case, send our graph over to the default partitioner.
    if len(required_bw_nodes) == 0:
        return default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)

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
        default_recomputable_ops += [prims.div, prims.convert_element_type, aten.clone, aten._to_copy, aten.full_like, prims.var, prims.sum, aten.var, aten.std, prims.broadcast_in_dim, aten.select, aten.permute, aten._unsafe_view, aten.view, aten.expand, aten.slice, aten.reshape, aten.broadcast_tensors, aten.scalar_tensor, aten.ones, aten.new_zeros, aten.lift_fresh_copy, aten.arange, aten.triu, aten.var_mean, aten.isinf, aten.any, aten.full, aten.as_strided, aten.zeros, aten.argmax, aten.maximum]  # noqa: E501,B950
        view_ops += [aten.view, aten.slice, aten.permute, aten.t, prims.broadcast_in_dim, aten.expand, aten.as_strided]
        # Natalia said that we should allow recomputing indexing :)
        default_recomputable_ops += [aten.index]
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
    compute_intensive_ops = [aten.mm, aten.convolution, aten.convolution_backward, aten.bmm, aten.addmm, aten.upsample_bilinear2d, aten._softmax, aten._softmax_backward_data, aten.native_layer_norm, aten.native_layer_norm_backward, aten.native_batch_norm, aten.native_batch_norm_backward, aten._native_batch_norm_legit]  # noqa: E501,B950

    unrecomputable_ops = random_ops + compute_intensive_ops

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

    AGGRESSIVE_RECOMPUTATION = False

    def is_materialized_backwards(node):
        cur_nodes = {node}
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if user not in required_fw_nodes and not is_fusible(cur, user):
                    return True
                if user not in required_fw_nodes and get_aten_target(user) in view_ops:
                    cur_nodes.add(user)

        return False

    def ban_recomputation(node):
        if AGGRESSIVE_RECOMPUTATION:
            return (node.op == 'call_function' and get_aten_target(node) in unrecomputable_ops)
        else:
            if node.op != 'call_function':
                return False
            if get_aten_target(node) not in recomputable_ops:
                return True
            if node.target == operator.getitem:
                return False
            if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
                return False

            # If a node *must* be materialized in the backwards pass, then we
            # should never recompute it. This is a pretty subtle point.  In
            # general, the assumption we make is that recomputing a node in the
            # backwards pass is "free". However, if a node must be materialized
            # in the backwards pass, then recomputing it is never free.
            if is_materialized_backwards(node):
                return True

            # Arbitrary hack that sometimes seems to help things. The above
            # modification appears to have made this heuristic a lot less critical
            # for performance.
            # TODO: Investigate why this hack helps.
            if compiler == "inductor" and node.dist_from_bw > config.max_dist_from_bw:
                return True
            # If the output of an op is 4x smaller (arbitrary choice),
            # then we don't allow recomputation.
            input_tensors_size = sum(_size_of(i) for i in node.args if isinstance(i, fx.Node))
            output_size = _size_of(node)
            return (output_size * 4 < input_tensors_size)

    def is_fusible(a, b):
        return get_aten_target(a) in fusible_ops and get_aten_target(b) in fusible_ops

    def is_materialized(node):
        if node.op == 'placeholder':
            return True

        return not all(is_fusible(node, user) for user in node.users)

    def get_node_weight(node) -> int:
        mem_sz = _size_of(node)

        # Heuristic to bias towards nodes closer to the backwards pass
        # Complete guess about current value
        mem_sz = int(mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1)))
        # mem_sz = int(mem_sz + node.dist_from_bw)

        if is_materialized(node):
            return mem_sz
        else:
            return mem_sz * 2

    nx_graph = nx.DiGraph()
    for node in full_bw_graph.nodes:
        if node.op == 'output':
            continue

        if node in required_bw_nodes:
            nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
            continue

        if _is_primal(node) or _is_fwd_seed_offset(node):
            nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
        if ban_recomputation(node) and node in required_fw_nodes:
            nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)

        # Checks if a node is actually a tuple. Can be simplified to just an isisinstance check if we always use faketensors.
        is_non_tensor_node = (('val' not in node.meta and 'tensor_meta' not in node.meta) or
                              ('val' in node.meta and not isinstance(node.meta['val'], torch.Tensor)))
        if is_symint_node(node):
            weight = 1
        elif is_sym_node(node):
            weight = math.inf
        elif is_non_tensor_node:
            weight = math.inf
        else:
            weight = get_node_weight(node)

        # Creates the weights on the "node" edge
        nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name + "_out", user.name + "_in", capacity=math.inf)

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

    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)

    # To make this stuff deterministic
    node_idx = {node: idx for idx, node in enumerate(joint_module.graph.nodes)}
    saved_values = sorted((name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x])
    # Symints must be kept separate from tensors so that PythonFunction only calls
    # save_for_backward on tensors and stashes symints in autograd .ctx
    saved_sym_nodes = list(filter(lambda n: is_sym_node(n), saved_values))
    saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)
    if AOT_PARTITIONER_DEBUG:
        print("Theoretical Activations Stored: ", sum([_size_of(i) for i in saved_values]) / 1e9)
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


def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = "fx_graph", clear_meta=True):
    if clear_meta:
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        for node in traced.graph.nodes:
            node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
    g = graph_drawer.FxGraphDrawer(traced, figname)
    x = g.get_main_dot_graph()
    getattr(x, "write_" + ext.lstrip("."))(f"{base}{ext}")


def draw_joint_graph(graph, joint_inputs, file_name="full_graph.png"):
    draw_graph(graph, file_name)
    return default_partition(graph, joint_inputs)
