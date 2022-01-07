import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from functorch import make_functional_with_buffers, make_fx
import torch.fx as fx
from torch.fx import immutable_collections
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch.fx.passes import graph_drawer
import copy
from functorch._C import CompileCache
from .decompositions import register_decomposition
from typing import List, Dict, Any, Tuple

pytree._register_pytree_node(immutable_collections.immutable_list, lambda x: (
    list(x), None), lambda x, c: immutable_collections.immutable_list(x))
pytree._register_pytree_node(immutable_collections.immutable_dict, lambda x: (list(x.values()), list(
    x.keys())), lambda x, c: immutable_collections.immutable_dict({key: value for key, value in zip(c, x)}))

# TODO - move this to PyTorch core. This overrides the pytree implementation for
# dict to maintain parity with Deepmind pytree.
Context = Any


def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    keys = list(sorted(d.keys()))
    values = [d[key] for key in keys]
    return values, keys


def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}


pytree._register_pytree_node(dict, _dict_flatten, _dict_unflatten)

aten = torch.ops.aten


def draw_graph(traced: torch.fx.GraphModule, fname: str, figname: str = "fx_graph", clear_meta=True):
    if clear_meta:
        traced = copy.deepcopy(traced)
    for node in traced.graph.nodes:
        node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".svg"
    print(f"Writing FX graph to file: {base}{ext}")
    g = graph_drawer.FxGraphDrawer(traced, figname)
    x = g.get_main_dot_graph()
    getattr(x, "write_" + ext.lstrip("."))(f"{base}{ext}")


class InvalidNodeBase(object):
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(joint_graph, inputs, outputs):
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as inputs and returns the specified outputs.

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
    new_graph.output([env[x] for x in outputs])

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node):
    return node.op == "placeholder" and "tangents" not in node.target


def _is_tangent(node):
    return node.op == "placeholder" and "tangents" in node.target


def _extract_fwd_bwd_outputs(joint_module: fx.GraphModule):
    num_fwd_outputs = joint_module._out_spec.children_specs[0].num_leaves
    outputs = pytree.tree_flatten([node.args for node in joint_module.graph.nodes if node.op == 'output'])[0]
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs


def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    tangent_inputs = list(filter(_is_tangent, joint_module.graph.nodes))
    # Construct the forward module
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs + saved_values)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_values + tangent_inputs, bwd_outputs)

    # This is to filter out saved values that don't actually end up being used by the backwards pass
    for node in bwd_graph.nodes:
        if node.op == 'placeholder' and not node.users:
            for saved_value in saved_values:
                if saved_value.name == node.name:
                    saved_values.remove(saved_value)
                    break

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs + saved_values)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_values + tangent_inputs, bwd_outputs)

    fwd_module = fx.GraphModule(joint_module, fwd_graph)
    bwd_module = fx.GraphModule(joint_module, bwd_graph)
    return fwd_module, bwd_module


def default_partition(joint_module: fx.GraphModule, _joint_inputs):
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module)
    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs, fwd_outputs)
    forward_node_names = set([node.name for node in forward_only_graph.nodes if node.op != 'output'])

    def node_saved(node):
        return node.name in forward_node_names and 'tensor_meta' in node.meta
    saved_values = [node for node in joint_module.graph.nodes if node_saved(node)]
    return _extract_fwd_bwd_modules(joint_module, saved_values)


def prod(x):
    s = 1
    for i in x:
        s *= i
    return s


def partition_with_recompute_fwd_in_bwd(joint_module: fx.GraphModule, _joint_inputs):
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimintation.
    """
    try:
        import networkx as nx
    except ImportError:
        raise RuntimeError("Need networkx installed to perform smart recomputation heuristics")
    # draw_graph(joint_module, "joint.svg")
    full_bw_graph = joint_module.graph

    nx_graph = nx.DiGraph()
    tangent_closure = set()
    name_to_node = {}
    for node in full_bw_graph.nodes:
        name_to_node[node.name] = node
        if node.op == 'placeholder' and "tangents" in node.target:
            tangent_closure.add(node)
        if node in tangent_closure:
            for user in node.users:
                tangent_closure.add(user)

    pointwise_ops = [aten.add, aten.sub, aten.div, aten.atan2, aten.mul, aten.max, aten.min, aten.pow, aten.remainder, aten.fmod, aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__, aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor, aten.frac, aten.neg, aten.relu, aten.round, aten.silu, aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2, aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc, aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin, aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh, aten.sqrt, aten.rsqrt,  aten.reciprocal, aten.sigmoid, aten.softplus, aten.threshold, aten.threshold_backward, aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu, aten.gelu_backward]  # noqa: E501
    reduction_ops = [aten.softmax, aten._softmax, aten._softmax_backward_data, aten.sum, aten.mean, aten._grad_sum_to_size, aten.sum_to_size, aten.amax]  # noqa: E501
    norm_ops = [aten.instance_norm, aten._batch_norm_impl_index, aten.native_batch_norm, aten.batch_norm, aten._batch_norm_impl_index_backward, aten.native_layer_norm, aten.layer_norm, aten.native_layer_norm_backward]  # noqa: E501
    misc_ops = [aten.to, aten.type_as]

    # Not used by default since NVFuser can't fuse view ops
    # view_ops = [aten.expand, aten.clone, aten.transpose, aten.t, aten.view, aten._unsafe_view, aten.permute, aten.transpose, aten.t, aten._reshape_alias, aten.squeeze, aten.unsqueeze, aten.reshape]  # noqa: E501

    recomputable_ops = set(
        pointwise_ops
        + reduction_ops
        + norm_ops
        + misc_ops
        # + view_ops
    )

    for node in full_bw_graph.nodes:
        if node in tangent_closure:
            nx_graph.add_edge(node.name+"_in", "sink", capacity=math.inf)
            continue
        is_input = False
        if node.op == 'placeholder' and "primals" in node.target:
            nx_graph.add_edge("source", node.name+"_in", capacity=math.inf)
            is_input = True

        if node.op == 'call_function' and node.target not in recomputable_ops:
            nx_graph.add_edge("source", node.name+"_in", capacity=math.inf)

        if 'tensor_meta' not in node.meta:
            weight = math.inf
        else:
            mem_sz = prod(node.meta['tensor_meta'].shape)
            if is_input:
                weight = mem_sz
            else:
                weight = mem_sz * 2

        nx_graph.add_edge(node.name+"_in", node.name+"_out", capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name+"_out", user.name+"_in", capacity=math.inf)

    cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)
    # print(len(cut_nodes), sorted(list(cut_nodes)))

    saved_values = [name_to_node[node] for node in cut_nodes]

    return _extract_fwd_bwd_modules(joint_module, saved_values)


def create_joint_forward_backward(fn):
    def joint_forward_backward(primals, tangents):
        out = fn(*primals)
        primals = [p for p in pytree.tree_flatten(primals)[0] if isinstance(p, Tensor) and p.requires_grad]
        backward_out = []
        if primals:  # todo(chilli): Make it support it if not all outputs have gradients
            backward_out = torch.autograd.grad(out, primals, grad_outputs=tangents, allow_unused=True)
        return out, backward_out
    return joint_forward_backward


def draw_joint_graph(graph, joint_inputs, file_name="full_graph.png"):
    draw_graph(graph, file_name)
    return default_partition(graph, joint_inputs)


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


aot_autograd_decompositions = {}


@register_decomposition(aten.rsub, aot_autograd_decompositions)
def rsub(a, b, alpha=1):
    return -aten.sub(a, b)


@register_decomposition(aten._reshape_alias, aot_autograd_decompositions)
def _reshape_alias(x, shape, strides):
    return aten.view(x, shape)


def create_compiled_function(flat_fn, fw_compiler, bw_compiler, partition_fn, decompositions):
    # putting these decompositions here since they shouldn't always be used
    # Kinda sketchy ... we use torch.sub here to have the correct scalar => tensor promotion logic

    joint_forward_backward = create_joint_forward_backward(flat_fn)

    compiled_fw = None
    compiled_bw = None
    num_outs = None

    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *flat_args):
            nonlocal compiled_fw, compiled_bw, num_outs
            if compiled_fw is None:
                out = flat_fn(*flat_args)
                if isinstance(out, (list, tuple)):
                    num_outs = len(out)
                else:
                    num_outs = 1

                joint_inputs = (flat_args, out)
                aot_decompositions = {**aot_autograd_decompositions, **decompositions}
                with torch.enable_grad():
                    fx_g = make_fx(joint_forward_backward, aot_decompositions)(*joint_inputs)
                fw_module, bw_module = partition_fn(fx_g, joint_inputs)
                # print(fw_module.code, bw_module.code)

                compiled_fw = fw_compiler(fw_module, flat_args)
                fw_outs = normalize_as_list(compiled_fw(*flat_args))

                bw_args = fw_outs[num_outs:] + fw_outs[0:num_outs]
                compiled_bw = bw_compiler(bw_module, bw_args)
            else:
                fw_outs = normalize_as_list(compiled_fw(*flat_args))
            ctx.save_for_backward(*fw_outs[num_outs:])
            return tuple(fw_outs[0:num_outs])

        @staticmethod
        def backward(ctx, *flat_args):
            # hmm... this doesn't feel right. todo
            # contiguous_args = [t.contiguous() for t in flat_args]
            contiguous_args = [t for t in flat_args]
            out = normalize_as_list(compiled_bw(*ctx.saved_tensors, *contiguous_args))
            out_iter = iter(out)
            grad_out = [next(out_iter) if p else None for p in ctx.needs_input_grad]
            return tuple(grad_out)

    return CompiledFunction


class _CompileCache(CompileCache):
    pass


# using a C++-based pytree reduces the overhead by about 50%
try:
    import tree
    HAS_TREE = True
except ImportError:
    HAS_TREE = False
compile_cache = None

# Inspired by autodidax (thanks!)


class PytreeThunk:
    spec = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple = None  # if the output spec is a tuple/list, we won't bother unflattening it.
    is_really_simple = None  # if the output spec is a LeafSpec

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all([isinstance(i, pytree.LeafSpec) for i in spec.children_specs]):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        return pytree.tree_unflatten(x, self.spec)


def filter_tensor_and_static_args(args, static_argnums):
    """
    Separate out the tensor and static args. Also, for the static args, store
    the hash.
    """
    tensor_args = []
    static_args = []
    static_args_hashed = []
    for idx, arg in enumerate(args):
        if idx not in static_argnums:
            tensor_args.append(arg)
        else:
            static_args.append(arg)
            static_args_hashed.append(arg.__hash__())
    return tensor_args, static_args, static_args_hashed


def rearrange(tensor_args, static_args, static_argnums):
    """
    Generate the args as per the original spec. static_argnums is sorted.
    """
    tensor_index = 0
    static_index = 0
    index = 0
    args = []
    assert len(static_args) == len(static_argnums)
    while tensor_index < len(tensor_args) and static_index < len(static_args):
        if index == static_argnums[static_index]:
            args.append(static_args[static_index])
            static_index += 1
        else:
            args.append(tensor_args[tensor_index])
            tensor_index += 1

    while tensor_index < len(tensor_args):
        args.append(tensor_args[tensor_index])
        tensor_index += 1

    while static_index < len(static_args):
        args.append(static_args[static_index])
        static_index += 1

    return args


def compiled_function(
    fn,
    fw_compiler,
    bw_compiler=None,
    partition_fn=default_partition,
    decompositions={},
    hasher_type="StaticShapeHasher",
    static_argnums=None,
):
    global compile_cache
    if compile_cache is None:
        compile_cache = CompileCache()
    if bw_compiler is None:
        bw_compiler = fw_compiler
    cached_res = None

    fn_id = id(fn)

    if isinstance(static_argnums, int):
        static_argnums = [static_argnums]
    elif static_argnums is not None and len(static_argnums) == 0:
        static_argnums = None
    elif static_argnums is not None:
        static_argnums = list(static_argnums)
        static_argnums.sort()

    def returned_function(*args, **kwargs):
        global compile_cache
        nonlocal cached_res

        # Separate out static args if static_argnums is present
        tensor_args = args
        static_args = []
        # TODO - move the hashing part of static_args to C++.
        static_args_hashed = []
        if static_argnums is not None:
            tensor_args, static_args, static_args_hashed = filter_tensor_and_static_args(args, static_argnums)

        # Now flatten the tensor args
        if HAS_TREE:
            flattened_tensor_args = tree.flatten((tensor_args, kwargs))
        else:
            flattened_tensor_args, _ = pytree.tree_flatten((tensor_args, kwargs))

        # Check if the fn is already compiled
        num_tensor_args = len(flattened_tensor_args)
        flattened_args = flattened_tensor_args + static_args
        flattened_args_for_cache = flattened_tensor_args + static_args_hashed
        cached_res = compile_cache.at(fn_id, num_tensor_args, hasher_type, *flattened_args_for_cache)

        # Compile the function and save it in the cache
        if cached_res is None:
            # Save the args_spec for flattened_tensor_args to unflatten while tracing
            _, tensor_args_spec = pytree.tree_flatten((tensor_args, kwargs))
            out_spec = PytreeThunk()

            def flat_fn(*args):
                nonlocal out_spec
                # These args are already flattened_tensor_args + static_args
                flattened_tensor_args = args[:num_tensor_args]
                static_args = args[num_tensor_args:]

                tensor_args, kwargs = pytree.tree_unflatten(flattened_tensor_args, tensor_args_spec)

                # Rearrange the args as per the original arg ordering
                if static_argnums is None:
                    args = tensor_args
                else:
                    args = rearrange(tensor_args, static_args, static_argnums)
                tree_out = fn(*args, **kwargs)
                flat_out = pytree.tree_flatten(tree_out)
                out_spec.set(flat_out[1])
                return flat_out[0]

            compiled_fn = create_compiled_function(
                flat_fn, fw_compiler, bw_compiler, partition_fn, decompositions
            ).apply
            cached_res = (compiled_fn, out_spec)

            # Save the compiled_fn in the cache
            compile_cache.insert(
                fn_id, num_tensor_args, hasher_type, cached_res, *flattened_args_for_cache
            )

        cached_fn, out_spec = cached_res
        out = cached_fn(*flattened_args)
        return out_spec.unflatten(out)

    return returned_function


def num_of_recompilations():
    global compile_cache
    if compile_cache is None:
        return 0
    return compile_cache.size()


def clear_compile_cache():
    global compile_cache
    if compile_cache is not None:
        compile_cache.clear()
        compile_cache = None


def compiled_module(mod, *args, **kwargs):
    func_mod, params, buffers = make_functional_with_buffers(mod)
    compiled_f = compiled_function(func_mod, *args, **kwargs)

    class CompiledModule(nn.Module):
        def __init__(self):
            super(CompiledModule, self).__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                tuple(self.parameters()),
                tuple(self.buffers()),
                *args,
                **kwargs
            )

    return CompiledModule()


aot_function = compiled_function
aot_module = compiled_module
