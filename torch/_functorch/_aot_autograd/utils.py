# mypy: allow-untyped-defs
"""
Contains various utils for AOTAutograd, including those for handling collections.
"""

import dataclasses
import operator
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.utils.pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import getArtifactLogger
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import py_sym_types


KNOWN_TYPES = [
    torch.Tensor,
    BackwardState,
    int,
    str,
    float,
    bool,
    type(None),
    *py_sym_types,
    FakeScriptObject,
    torch.ScriptObject,
]

original_zip = zip

aot_graphs_effects_log = getArtifactLogger(__name__, "aot_graphs_effects")


def strict_zip(*iterables, strict=True, **kwargs):
    if not strict:
        return original_zip(*iterables, **kwargs)

    length = len(iterables[0])
    for iterable in iterables[1:]:
        if len(iterable) != length:
            raise ValueError(
                "The iterables have different lengths and strict mode is enabled."
            )

    return original_zip(*iterables, **kwargs)


def _get_symint_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
    if isinstance(exprs, (list, tuple)):
        return type(exprs)(_get_symint_hints(e) for e in exprs)
    elif isinstance(exprs, torch.SymInt):
        return exprs.node.shape_env.size_hint(exprs.node.expr)
    else:
        return exprs


def partial_flatten_asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return obj.__class__([partial_flatten_asdict(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: partial_flatten_asdict(v) for k, v in obj.items()}
    else:
        return obj


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def _get_autocast_states():
    return [
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cuda"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_cache_enabled(),
    ]


def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True  # type: ignore[attr-defined]
    return g


def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_at_runtime_with_args(
    f, args: Union[Tuple[Any], List[Any]], steal_args=False, disable_amp=False
):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    context = torch._C._DisableAutocast if disable_amp else nullcontext
    with context():
        if hasattr(f, "_boxed_call"):
            out = normalize_as_list(f(args))
        else:
            # TODO: Please remove soon
            # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            warnings.warn(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
            )
            out = normalize_as_list(f(*args))
    return out


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec: Optional["pytree.PyTreeSpec"] = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple: Optional[
        bool
    ] = None  # if the output spec is a tuple/list, we won't bother unflattening it.
    is_really_simple: Optional[bool] = None  # if the output spec is a LeafSpec

    def set(self, spec: "pytree.PyTreeSpec") -> None:
        assert self.spec is None or self.spec == spec
        assert spec is not None
        self.spec = spec
        if self.spec.type in {tuple, list} and all(
            child.is_leaf() for child in spec.children()
        ):
            self.is_simple = True
        if self.spec.is_leaf():
            self.is_really_simple = True

    def unflatten(self, x: List[Any]) -> Any:
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        assert self.spec is not None
        return pytree.tree_unflatten(x, self.spec)


# Creates a function that returns flattened inputs and outputs
# Also returns the output tree spec, which is needed to recover the "unflattened"
# output tree structure later.
def create_tree_flattened_fn(fn, args, kwargs=None) -> Tuple[Callable, PytreeThunk]:
    if kwargs is None:
        kwargs = {}
    # Save the args_spec for flat_tensor_args to unflatten while tracing
    tensor_args_spec = pytree.tree_structure((args, kwargs))
    out_spec = PytreeThunk()

    def flat_fn(*flat_args):
        # The input are flattened tensor args. Prepare the args in the
        # order that original function expects. Add static args as well.
        # They will appear as tensor constants in the traced graph.
        nonlocal out_spec
        args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
        tree_out = fn(*args, **kwargs)
        flat_out, spec = pytree.tree_flatten(tree_out)
        for i in flat_out:
            is_known_type = False
            for j in KNOWN_TYPES:
                if isinstance(i, j):
                    is_known_type = True
                    break
            if not is_known_type:
                raise RuntimeError(
                    f"Found {type(i)} in output, which is not a known type. "
                    "If this type holds tensors, you need to register a pytree for it. "
                    "See https://github.com/pytorch/functorch/issues/475 for a brief "
                    "explanation why. If you don't need to register a pytree, please "
                    "leave a comment explaining your use case and we'll make this more "
                    "ergonomic to deal with"
                )
        out_spec.set(spec)
        return flat_out

    # Can't use functools.wraps here because the wrapper has different
    # calling convention
    if hasattr(fn, "_orig_mod"):
        flat_fn._orig_mod = fn._orig_mod  # type: ignore[attr-defined]

    return flat_fn, out_spec


# This function takes in a tensor t, and returns one of t, t.view(), or t.clone().
# When tracing the joint forward + backward, for any inputs in the graph that are mutated,
# we need to clone them first (and similarly for metadata-only mutations, we need to view them first).
# The idea is that when we trace the backward, we need to pass in the *original* primals
# to autograd.grad(), before they were mutated.
# Note: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
# This means that "idx" here represents the index of the (potentially) synthetic base.
# What we need to do is:
# (1) map the current (post-synthetic-base calling convention) input argument index
#     to int index pre-synthetic-base-calling-convention.
# (2) There could be multiple, if this index corresponds to a synthetic base
#     that has multiple input aliases.
# (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
def maybe_to_fresh_input(idx, t, meta):
    if not isinstance(t, torch.Tensor):
        return t
    if idx in meta.mutated_inp_runtime_indices:
        # We only need to bother cloning mutated inputs that participate in autograd.
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the mutation
            return t.clone()
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the metadata mutation
            return t.view(t.shape)
    return t


def is_with_effects(node):
    return (
        node.op == "call_function"
        and node.target == torch.ops.higher_order.with_effects
    )


def is_with_effects_op(node, op):
    return is_with_effects(node) and node.args[1] == op


def unlift_tokens(fw_module, fw_metadata, aot_config, bw_module=None):
    # Remove the tokens from the inputs/outputs of the graph since inductor does
    # not want these extra inputs/outputs, and replace them with
    # _make_token() to create a token, and _sink_tokens() to collect the
    # tokens.  See Note [Side-Effectful Tokens in AOTAutograd]
    # Logic:
    # 1. Inputs identified as input tokens:
    #    - If used as a first argument in with_effects
    #
    # 2. Outputs identified as output tokens:
    #    - If Produced by getitem(with_effects, 0)
    #
    # 3. Checks invariants of number input output tokens:
    # forward:
    # expected_num_erased_inputs == len(fw_metadata.tokens)
    # expected_num_erased_outputs == len(fw_metadata.tokens)
    # backward:
    # expected_num_erased_inputs == fw_metadata.num_backward_tokens
    # expected_num_erased_outputs == fw_metadata.num_backward_tokens
    num_forward_tokens = len(fw_metadata.tokens)
    num_backward_tokens = fw_metadata.num_backward_tokens

    def rewrite_with_effects_input_token(module, node):
        with module.graph.inserting_before(node):
            new_token_node = module.graph.call_function(
                torch.ops.prims._make_token.default, ()
            )
            new_token_node.meta["val"] = torch.tensor([])
            new_token_node.meta["tensor_meta"] = torch.tensor([])

            args = list(node.args)
            args[0] = new_token_node
            node.args = tuple(args)

    def rewrite_output(module, node, output_token_nodes, other_output_args):
        for output_token_node in output_token_nodes:
            assert (
                output_token_node.op == "call_function"
                and output_token_node.target == operator.getitem
                and output_token_node.args[1] == 0
            )
        with module.graph.inserting_before(node):
            module.graph.call_function(
                torch.ops.prims._sink_tokens.default,
                (output_token_nodes,),
            )
            node.args = (other_output_args,)

    def do(module, subgraph, expected_num_erased):
        num_erased_inputs = 0
        num_erased_outs = 0
        input_nodes = []
        input_token_nodes = set()
        with_effect_nodes = []
        output_token_nodes = []
        other_output_nodes = []
        for node in module.graph.nodes:
            if node.op == "placeholder":
                input_nodes.append(node)
            elif is_with_effects(node):
                with_effect_nodes.append(node)
                if node.args[0] in input_nodes:
                    input_token_nodes.add(node.args[0])
                    rewrite_with_effects_input_token(module, node)
            elif node.op == "output":
                outs = node.args[0]
                for out in outs:
                    if (
                        isinstance(out, torch.fx.node.Node)
                        and out.op == "call_function"
                        and out.target == operator.getitem
                        and out.args[1] == 0
                        and out.args[0] in with_effect_nodes
                    ):
                        output_token_nodes.append(out)
                    else:
                        other_output_nodes.append(out)

                rewrite_output(module, node, output_token_nodes, other_output_nodes)
                num_erased_outs = len(output_token_nodes)

        for input_token_node in input_token_nodes:
            module.graph.erase_node(input_token_node)

        num_erased_inputs = len(input_token_nodes)

        assert (
            num_erased_inputs == expected_num_erased
        ), f"{subgraph} num_erased_inputs:{num_erased_inputs} {input_token_nodes}!=expected {expected_num_erased}"
        assert (
            num_erased_outs == expected_num_erased
        ), f"{subgraph} num_erased_outs:{num_erased_outs} {output_token_nodes}!=expected {expected_num_erased}"

        module.recompile()

    if num_forward_tokens > 0:
        if aot_config.enable_log:
            from torch._dynamo.utils import lazy_format_graph_code

            aot_graphs_effects_log.debug(
                "%s",
                lazy_format_graph_code(
                    "Forward graph before unlifting tokens",
                    fw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
        do(
            fw_module,
            "forward",
            num_forward_tokens,
        )

    if bw_module is not None and num_backward_tokens > 0:
        if aot_config.enable_log:
            from torch._dynamo.utils import lazy_format_graph_code

            aot_graphs_effects_log.debug(
                "%s",
                lazy_format_graph_code(
                    "Backward graph before unlifting tokens",
                    bw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
        do(bw_module, "backward", num_backward_tokens)

    # This is sad, but we need to update the metadata to get rid of
    # the tokens.
    fw_metadata.tokens = {}
    fw_metadata.num_backward_tokens = 0


def root_module_when_exporting_non_strict(flat_fn):
    # When exporting in non-strict mode, we wrap the root module in a specific pattern.
    # See `_aot_export_non_strict` in torch.export._trace.py.
    # We look for that wrapping pattern here.
    if hasattr(flat_fn, "_orig_mod") and hasattr(flat_fn._orig_mod, "_export_root"):
        return flat_fn._orig_mod._export_root
    else:
        return None


def copy_fwd_metadata_to_bw_nodes(fx_g):
    """
    Input: `fx_g` which contains the joint fwd+bwd FX graph created by
    aot_autograd.

    This function walks the graph and copies over metadata from forward nodes
    to backward nodes, using the `seq_nr` field as a one-to-many mapping
    from forward node to backward node. This metadata is useful for performance
    profiling and debugging.
    """

    def _is_forward_node_with_seq_nr(node):
        # For now, assume that if nn_module_stack_metadata is populated, this
        # node is from the forward. Ignore nodes without `seq_nr`.
        # TODO(future): there is likely a less brittle way to do this by walking
        # the descendants of graph inputs corresponding to fwd inputs, didn't
        # seem obvious at first glance on how to partition graph inputs into
        # fwd vs bwd without relying on string names.
        return "nn_module_stack" in node.meta and "seq_nr" in node.meta

    def _is_backward_node_with_seq_nr(node):
        # For now, assume that if nn_module_stack_metadata is not populated,
        # this node is from the backward. Ignore nodes without `seq_nr`.
        # TODO(future): there is likely a less brittle way to do this, same
        # as with the forward.
        return ("nn_module_stack" not in node.meta) and "seq_nr" in node.meta

    fwd_seq_nr_to_node = {}
    for node in fx_g.graph.nodes:
        if not _is_forward_node_with_seq_nr(node):
            continue
        seq_nr = node.meta["seq_nr"]
        if seq_nr in fwd_seq_nr_to_node:
            # If we already saw an op with the current `seq_nr`, that means
            # that the current op did not create an autograd node, and there
            # is no corresponding backward node, so we skip.
            continue
        fwd_seq_nr_to_node[node.meta["seq_nr"]] = node

    for node in fx_g.graph.nodes:
        if not _is_backward_node_with_seq_nr(node):
            continue
        # fwd_node should always exist, but handle non-existence just in case
        fwd_node = fwd_seq_nr_to_node.get(node.meta["seq_nr"])
        if fwd_node is not None:
            node.meta["fwd_nn_module_stack"] = fwd_node.meta["nn_module_stack"]
            node.meta["fwd_source_fn_stack"] = fwd_node.meta.get("source_fn_stack")


def register_buffer_assignment_hook(mod, assigned_buffers):
    """
    Register a hook that intercepts buffer assignments.
    This is used to detect when a buffer is assigned to, and then we can
    map that buffer to the corresponding proxy node in the graph.
    """

    def _map_assigned_buffer_to_proxy(_mod, name, buffer):
        # We intercept buffer assignments on the root module through this hook.
        if _mod._buffers is mod._buffers:
            # either buffer is a functional tensor, which wraps a fake tensor
            if isinstance(buffer, FunctionalTensor):
                buffer = buffer.from_functional()
            # or buffer is a fake tensor
            assert isinstance(buffer, FakeTensor)
            # The fake tensor in turn is associated with a proxy node.
            proxy_mode = torch.fx.experimental.proxy_tensor.get_proxy_mode()
            assert proxy_mode is not None
            proxy = torch.fx.experimental.proxy_tensor.get_proxy_slot(
                buffer, proxy_mode.tracer
            ).proxy.node
            # We map the assigned buffer to this proxy node.
            assigned_buffers[name] = proxy.name
        return buffer

    return torch.nn.modules.module.register_module_buffer_registration_hook(
        _map_assigned_buffer_to_proxy
    )


def contain_metadata_mutation_ops(module: torch.fx.GraphModule) -> bool:
    """
    Checks if the module contains any metadata mutation ops.
    """
    for node in module.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "tags")
            and torch.Tag.inplace_view in node.target.tags
        ):
            return True
    return False
