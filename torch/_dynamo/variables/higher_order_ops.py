# mypy: ignore-errors

"""
This module contains classes and utilities for handling higher-order operators in Dynamo.
It provides functionality for tracing and transforming control flow constructs like
conditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),
and other higher-order operations.

The module includes specialized VariableTracker classes for different types of
higher-order operations, along with utilities for:
- Speculating and capturing subgraphs
- Managing control flow
- Handling autograd function applications
- Supporting function transformations
- Processing activation checkpoints

These classes work together to enable Dynamo to correctly trace and compile code
containing complex control flow patterns and higher-order functions while preserving
their semantic behavior.
"""

import contextlib
import functools
import inspect
import itertools
import logging
import types
import warnings
from typing import Optional, TYPE_CHECKING

import torch._C
import torch.fx
import torch.nn
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import get_fake_value
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch._ops import HigherOrderOperator
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree

from .. import graph_break_hints, variables
from ..exc import (
    IncorrectUsage,
    UncapturedHigherOrderOpError,
    unimplemented,
    unimplemented_v2,
    Unsupported,
)
from ..source import AttrSource, DictGetItemSource
from ..utils import proxy_args_kwargs, set_example_value
from .base import VariableTracker
from .dicts import ConstDictVariable
from .lazy import LazyVariableTracker
from .lists import ListVariable, TupleVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)


def raise_hard_error_if_graph_break(reason):
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                msg = " Scroll up to find out what causes the graph break."
                raise UncapturedHigherOrderOpError(reason + msg) from e

        return graph_break_as_hard_error

    return deco


# This function is a syntax sugar for creating a dummy new subtracer so that
# newly added nodes are added to a separate subgraph in this subtracer instead of affecting
# the main graph. This is useful for creating sample inputs for tracing the subgraph.
# For example, in FlexAttentionHigherOrderVariable, we want to create several scalars
# to trace the score_mod function but we don't want the operators that creates the scalar to
# show up in the graph, we could this function to discard the graph changes.
# Example usage:
# with discard_graph_changes():
#   sample_input= create_sample_inputs()
# speculate_subgraph(tx, f, sample_inputs, {})
@contextlib.contextmanager
def discard_graph_changes(tx):
    ctx = tx.output.subtracer("subgraph_wrapper", None)
    try:
        ctx.__enter__()
        yield
    finally:
        ctx.__exit__(None, None, None)


def check_meta_consistency_vt(
    vars1: list[VariableTracker],
    vars2: list[VariableTracker],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    from torch._higher_order_ops.utils import check_meta_consistency

    from . import TensorVariable

    def _unwrap_var(var):
        if isinstance(var, TensorVariable):
            return var.proxy.node.meta["example_value"]
        elif isinstance(var, SymNodeVariable):
            return var.sym_num
        elif isinstance(var, ConstantVariable):
            return var.as_python_constant()
        else:
            unimplemented(f"Cannot unwrap var {var}")

    unwrapped1 = [_unwrap_var(var) for var in vars1]
    unwrapped2 = [_unwrap_var(var) for var in vars2]

    return check_meta_consistency(
        unwrapped1,
        unwrapped2,
        lhs_name,
        rhs_name,
        include_contiguity=include_contiguity,
    )


@contextlib.contextmanager
def dynamo_enable_grad(tx: "InstructionTranslator", enable=True):
    from . import GradModeVariable

    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, enable, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)


@contextlib.contextmanager
def dynamo_under_activation_checkpoint(tx: "InstructionTranslator"):
    orig_val = tx.output.current_tracer.under_activation_checkpoint
    try:
        tx.output.current_tracer.under_activation_checkpoint = True
        yield
    finally:
        tx.output.current_tracer.under_activation_checkpoint = orig_val


def find_mismatched_vars(var, types, allow_none=False):
    """
    Recursively finds variables whose type is not an instance of the specified types.
    Args:
        var: The variable to check.
        types: A tuple of allowed types.
        allow_none (bool): Whether to allow None values. Defaults to False.
    Returns:
        A set of variables whose type is not an instance of the specified types.
    """
    mismatched_vars = set()
    if isinstance(var, (TupleVariable, ListVariable)):
        for item in var.items:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, ConstDictVariable):
        for value in var.items.values():
            mismatched_vars.update(find_mismatched_vars(value, types, allow_none))
    else:

        def _is_none(var):
            return var.is_python_constant() and var.as_python_constant() is None

        if not isinstance(var, types) and not (allow_none and _is_none(var)):
            mismatched_vars.add(var)
    return mismatched_vars


def only_consist_of(var, types, allow_none=False):
    mismatch_vars = find_mismatched_vars(var, types, allow_none=allow_none)
    return len(mismatch_vars) == 0


# A more read-able syntax sugar for creating a UserFunctionVariable for f
# and run call_function on it. Make it return a function to preserve the calling
# convention of the original f.
def _make_inlined(tx: "InstructionTranslator", f):
    assert callable(f), "Expect f to be a python callable."

    def inline_call(*args, **kwargs):
        return UserFunctionVariable(f).call_function(tx, args, kwargs)

    return inline_call


def _call_function_and_unflatten_output(
    tx, fn, args, kwargs, flat_example_value, ret_treespec
):
    from .builder import wrap_fx_proxy

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # Transform variable back into a list (previously made into a tuple by
    # speculate_subgraph function) so as to respect the pytree API typing.
    flat_list_variable = BuiltinVariable(list).call_function(tx, [flat_variable], {})
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_treespec)
        if ret_treespec
        else flat_variable
    )


def _assert_tensors_nonaliasing(inputs, outputs):
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    assert input_tensor_ids.isdisjoint(output_tensor_ids), (
        "inputs to function body cannot alias outputs"
    )


def _check_all_tensorvariable(args):
    from . import TensorVariable

    if not all(type(a.realize()) is TensorVariable for a in args):
        unimplemented(
            f"Expected all leaves to be of torch.Tensor type, but got {[type(a.realize()) for a in args]}."
        )


def _check_supported_callable_arg(
    tx: "InstructionTranslator", func_var: VariableTracker, arg_name
):
    is_callable = (
        BuiltinVariable(callable).call_function(tx, [func_var], {}).as_python_constant()
    )
    if not is_callable:
        unimplemented(
            f"{arg_name} should be a Callable but is of type {str(func_var)}."
        )


def are_same_graph_modules(a_mod, b_mod, fake_mode):
    from torch._subclasses._fake_tensor_utils import _CacheKeyState
    from torch._subclasses.fake_tensor import extract_tensor_metadata

    # Maps the equivalent nodes from a to b
    node_map = {}

    def check_all_args(a_nodes, b_nodes):
        for arg_a, arg_b in zip(a_nodes, b_nodes):
            if isinstance(arg_a, torch.fx.Node):
                if node_map[arg_a] != arg_b:
                    return False
            elif isinstance(arg_a, slice):
                if not isinstance(arg_b, slice):
                    return False
                if not check_all_args(
                    (arg_a.start, arg_a.stop, arg_a.step),
                    (arg_b.start, arg_b.stop, arg_b.step),
                ):
                    return False
            elif arg_a != arg_b:
                # This is a catch-all for everything else. `slice` was a
                # surprise but can there be other data structures that can
                # contain fx.Nodes in them?
                return False
        return True

    for a_node, b_node in zip(a_mod.graph.nodes, b_mod.graph.nodes):
        if a_node.op != b_node.op:
            return False

        if a_node.op == "placeholder":
            a_value = a_node.meta["example_value"]
            b_value = b_node.meta["example_value"]

            if isinstance(a_value, torch.Tensor):
                if not isinstance(b_value, torch.Tensor):
                    return False
                # Extract fake tensor metadata for a and b and then compare
                a_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                a_metadata = extract_tensor_metadata(a_value)
                a_metadata._flatten_into(a_result, fake_mode, state)

                b_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                b_metadata = extract_tensor_metadata(b_value)
                b_metadata._flatten_into(b_result, fake_mode, state)
                if a_result != b_result:
                    return False
            elif isinstance(a_value, torch.SymInt):
                if not isinstance(b_value, torch.SymInt):
                    return False
                if a_value is not b_value:
                    return False
        elif a_node.op == "call_function":
            if a_node.target is not b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                # print("call_function args failed")
                return False
        elif a_node.op == "call_method":
            if a_node.target != b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                # print("call_method args failed")
                return False
        elif a_node.op == "output":
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                # print("output args failed")
                return False
        elif a_node.op == "get_attr":
            a_attr = getattr(a_mod, a_node.target)
            b_attr = getattr(b_mod, b_node.target)
            if isinstance(a_attr, torch.fx.GraphModule):
                if not isinstance(b_attr, torch.fx.GraphModule):
                    return False
                # This is an example of a HOP inside a HOP
                if not are_same_graph_modules(a_attr, b_attr, fake_mode):
                    return False
            else:
                # TODO - write an example with tensor as a graph attribute in
                # the Fx graph
                raise NotImplementedError(f"get_attr with {type(a_attr)}")
        else:
            # TODO - call_module is not supported because Dynamo Fx graph does
            # not install a call_module
            raise NotImplementedError(f"Graph equivalence check saw a {a_node.op}")

        # Two nodes are equal - add them to them map
        node_map[a_node] = b_node

    return True


def validate_args_and_maybe_create_graph_inputs(
    sub_args,
    tracer,
    tx,
    set_subgraph_inputs,
    description,
    sub_args_names=None,
):
    from . import AutogradFunctionContextVariable
    from .builder import wrap_fx_proxy_cls

    assert tracer.parent is not None

    if set_subgraph_inputs == "flatten_manual":
        flat_args, tree_spec = _make_inlined(tx, pytree.tree_flatten)(
            ListVariable(sub_args)
        ).unpack_var_sequence(tx)

        flat_inputs = validate_args_and_maybe_create_graph_inputs(
            flat_args.unpack_var_sequence(tx),
            tracer,
            tx,
            set_subgraph_inputs="manual",
            description=description,
        )

        return _make_inlined(tx, pytree.tree_unflatten)(
            ListVariable(flat_inputs), tree_spec
        ).unpack_var_sequence(tx)
    else:
        if sub_args_names is not None:
            # Can be greater if user passes some args as kwargs
            assert len(sub_args_names) >= len(sub_args)
        args = []
        for idx, a in enumerate(sub_args):
            assert isinstance(a, VariableTracker)
            if set_subgraph_inputs == "automatic":
                args.append(a)
                continue
            elif set_subgraph_inputs == "semi_automatic":
                if isinstance(a, AutogradFunctionContextVariable):
                    example_value = a.as_proxy().node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    tracer.create_graph_input(arg_name, a.python_type(), example_value)
                elif a.maybe_fx_node() is not None:
                    node = a.maybe_fx_node()
                    example_value = node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    new_proxy = tracer.create_graph_input(
                        arg_name, a.python_type(), example_value
                    )
                    example_value = (
                        node.meta["example_value"]
                        if "example_value" in node.meta
                        else None
                    )
                    a = wrap_fx_proxy_cls(
                        target_cls=type(a),
                        tx=tx,
                        proxy=new_proxy,
                        example_value=example_value,
                    )
                args.append(a)
                continue

            if a.is_python_constant():
                # This arg is not used in the body of the higher order op.
                # Currently, this new input is added to make the calls
                # happy, which expect a fixed number of arguments. In
                # future, we can clean this up.
                arg_name = (
                    "const_unused"
                    if sub_args_names is None
                    else f"const_unused_{sub_args_names[idx]}"
                )
                tracer.create_graph_input(
                    arg_name, a.python_type(), a.as_python_constant()
                )
                new_arg = a
            # Weird special case, we probably want to delete it or fold it
            # into the next case (of `a` being placeable into a graph)
            elif isinstance(a, AutogradFunctionContextVariable):
                example_value = a.as_proxy().node.meta["example_value"]
                arg_name = (
                    a.as_proxy().node.name
                    if sub_args_names is None
                    else sub_args_names[idx]
                )
                tracer.create_graph_input(arg_name, a.python_type(), example_value)
                new_arg = a
            # If `a` can be put into a graph
            elif a.maybe_fx_node() is not None:
                node = a.maybe_fx_node()
                example_value = (
                    node.meta["example_value"] if "example_value" in node.meta else None
                )
                arg_name = node.name if sub_args_names is None else sub_args_names[idx]
                new_proxy = tracer.create_graph_input(
                    arg_name, a.python_type(), example_value
                )
                new_arg = wrap_fx_proxy_cls(
                    target_cls=type(a),
                    tx=tx,
                    proxy=new_proxy,
                    example_value=example_value,
                )
            # If `a` cannot be put into a graph
            else:
                # HOPs work much better if they use speculate_subgraph(set_subgraph_inputs="automatic").
                unimplemented(
                    f"{description} with body that accepts non-Tensors as input. "
                    f"Got: {a.python_type()}"
                )
            args.append(new_arg)
        return args


# This helper function is used to make sure two graphs share the same input signature. For example,
# in torch.cond, two branches might lift different set of tensors as inputs. This function helps to
# dedup the inputs and modify the graphs to take the same set of inputs.
def _merge_graph_inputs(
    l_graph, l_lifted_freevars, l_name, r_graph, r_lifted_freevars, r_name
):
    def dedup_and_sort_lifted_freevars(l_lifted_freevars, r_lifted_freevars):
        # The nn module attributes are guaranteed to be registered into the top-level graph module during
        # higher order op speculation. Therefore, get_attr nodes in two branches with the same
        # target refer to the same attribute and we can safely deduplicate them with their target.
        #
        # Note: ideally, dynamo should just create a single proxy for the same attribute of a nn module. But
        # true_branch and false_branch belong to two separate tracing contexts, they may register the same
        # attribute to top level seperately. This creates two get_attr proxies for the same attribute
        # that have different meta data such as stack_trace (one stack trace for the true_branch,
        # and the other for false_branch). It seems better to discard the proxy explicitly in cond
        # than make dynamo create a single proxy for the same get_attr target.
        def shared_getattrs(l_lifted_proxies, r_lifted_proxies):
            true_targets = {
                proxy.node.target: proxy
                for proxy in l_lifted_proxies
                if proxy.node.op == "get_attr"
            }
            l_shared_getattrs = {}
            r_shared_getattrs = {}

            for false_proxy in r_lifted_proxies:
                if (
                    false_proxy.node.op == "get_attr"
                    and false_proxy.node.target in true_targets
                ):
                    true_proxy = true_targets[false_proxy.node.target]
                    l_shared_getattrs[true_proxy] = true_proxy
                    r_shared_getattrs[false_proxy] = true_proxy
            return l_shared_getattrs, r_shared_getattrs

        l_shared_getattrs, r_shared_getattrs = shared_getattrs(
            l_lifted_freevars.keys(), r_lifted_freevars.keys()
        )

        l_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            l_shared_getattrs.keys()
        )
        r_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            r_shared_getattrs.keys()
        )
        unique_l_freevars = l_lifted_freevars.keys() - l_shared_freevars
        unique_r_freevars = r_lifted_freevars.keys() - r_shared_freevars

        def _sort_by_name(vars):
            return sorted(vars, key=lambda var: var.node.name)

        return (
            list(_sort_by_name(list(l_shared_freevars))),
            list(_sort_by_name(list(r_shared_freevars))),
            list(_sort_by_name(list(unique_l_freevars))),
            list(_sort_by_name(list(unique_r_freevars))),
        )

    (l_shared, r_shared, unique_l, unique_r) = dedup_and_sort_lifted_freevars(
        l_lifted_freevars, r_lifted_freevars
    )

    # Let's say we capture cond(pred, true_fn, false_fn, (x,))
    # With set_graph_input set to automatic,
    # true_fn has lifted variables x, a, b, c
    # false_fn has lifted variables x, a, b, d
    # Then fixup_branch_inps make sure both branches have the same signature, i.e.:
    # - true_fn(x, a, b, c_true_branch, d_false_branch)
    # - false_fn(x, a, b, c_true_branch, d_false_branch)
    #
    # More formally, the signature has three parts in the following order:
    # 1. used in both branches: x, a, b
    # 2. only used in true branches: c, suffixed with _true_branch
    # 3. only used in false branches: d, suffixed with _false_branch
    # Within each part, we re-order the nodes by name to have a derterministic ordering for testing.
    def fixup_branch_inps(graph, lifted_freevars, shared, unique_l, unique_r):
        def _insert_or_replace_phs(new_args, name_suffix):
            for arg in new_args:
                new_ph = graph.placeholder(arg.node.name + name_suffix)
                # Override with new_ph if there exists a old placeholder.
                if arg in lifted_freevars:
                    old_ph = lifted_freevars[arg].node
                    old_ph.replace_all_uses_with(new_ph)
                    # replace_all_uses_with doesn't clean users. Clean it mannually so that we could erase it.
                    old_ph.users = {}
                    graph.erase_node(old_ph)

        first_not_ph_node = next(
            node for node in graph.nodes if node.op != "placeholder"
        )
        with graph.inserting_before(first_not_ph_node):
            _insert_or_replace_phs(shared, "")
            _insert_or_replace_phs(unique_l, "_" + l_name)
            _insert_or_replace_phs(unique_r, "_" + r_name)

    fixup_branch_inps(l_graph, l_lifted_freevars, l_shared, unique_l, unique_r)
    fixup_branch_inps(r_graph, r_lifted_freevars, r_shared, unique_l, unique_r)
    return l_graph, r_graph, l_shared, r_shared, unique_l, unique_r


# See NOTE [HigherOrderOperator tracing design] for details of the design
def speculate_subgraph(
    tx,
    f,
    sub_args,
    sub_kwargs,
    description,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target=None,
    always_restore=False,
    enable_grad=None,
    # NOTE [argument `set_subgraph_inputs`]
    # set_subgraph_inputs controls what how to construct subgraphs' placeholders from sub_args.
    # 1. if your HOP supports arbitrary inputs, use set_subgraph_inputs="automatic" (most recommended).
    # 2. if your HOP supports only Tensor and symnode inputs, use set_subgraph_inputs="flatten_manual" (recommended).
    # If sub_args contain Pytree structure (e.g. dict/list/tuple/set), the sub_args will be flattened first.
    # Then the flattened args are manually set as subgraph's placeholders.
    # 3. if your HOP must preserve inputs that are not tensor or symnode as placeholders e.g. AutogradFunctionContextVariable
    # use set_subgraph_inputs="manual" (not recommended). We do not recommend it in general because it has the
    # restriction that user need to manually control how to create placeholders and VariableTrackers for the args.
    set_subgraph_inputs="automatic",
    restore_side_effects=True,
    should_flatten_outputs=False,
    under_activation_checkpoint=False,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation=True,
    supports_aliasing=True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer=None,
):
    if sub_kwargs is None:
        sub_kwargs = {}

    assert set_subgraph_inputs in {
        "automatic",
        "semi_automatic",
        "flatten_manual",
        "manual",
    }, "Please use one of the supported set_subgraph_inputs options."

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        unimplemented("Use `set_subgraph_inputs=automatic` when passing `sub_kwargs`.")

    try:
        # ensure guards on args get installed in parent subgraph
        f, sub_args, sub_kwargs = LazyVariableTracker.realize_all(
            (f, sub_args, sub_kwargs),
        )

        with tx.output.subtracer(source_target, tracer) as subtracer:
            sub_args_names = maybe_positional_arg_names(f)
            # User mismatch in the number of args. Will eventually lead to an error.
            if sub_args_names is not None and len(sub_args_names) < len(sub_args):
                sub_args_names = None
            args = validate_args_and_maybe_create_graph_inputs(
                sub_args,
                subtracer,
                tx,
                set_subgraph_inputs,
                description,
                sub_args_names,
            )

            validate_args_and_maybe_create_graph_inputs(
                sub_kwargs.values(),
                subtracer,
                tx,
                set_subgraph_inputs="automatic",
                description=description,
            )

            autograd_ctx = (
                dynamo_enable_grad(tx, enable_grad)
                if enable_grad is not None
                else contextlib.nullcontext()
            )
            checkpoint_ctx = (
                dynamo_under_activation_checkpoint(tx)
                if under_activation_checkpoint
                else contextlib.nullcontext()
            )

            # For handling side effects, we can make an argument that we don't
            # have to do anything here. The side effects infra does a good job
            # of graph breaking if we mutate any nonlocal or global variable
            # while subtracing. As a result if tracing succeeds, side effects
            # data structure will only contain read-only data structures that
            # are put there for tracking purposes.
            # But on the other hand, there is an argument that if we ever write
            # a new side effect in Dynamo which does not go through the side
            # effect infra, we can end up in bad state.
            # Therefore we restore the side effects after tracing. The catch is
            # that we have to special handle tensor variables. If we have seen a
            # nonlocal variable tensor during subtracing, we want to keep a
            # track of that tensor, so that later subtracing or the root tracer
            # itself does not create a new proxy for the already observed tensor
            # variable.
            if restore_side_effects:
                prev_side_effects = tx.output.side_effects.clone()

            with autograd_ctx, checkpoint_ctx:
                output = f.call_function(tx, args, sub_kwargs)

            if restore_side_effects:
                new_side_effects = tx.output.side_effects.clone()
                prev_side_effects.track_tensor_variables_from_runahead_side_effects(
                    new_side_effects
                )
                tx.output.side_effects = prev_side_effects

            treespec = None
            if should_flatten_outputs:
                # Flatten the speculated subgraph output.
                output, treespec = _make_inlined(tx, pytree.tree_flatten)(
                    output
                ).unpack_var_sequence(tx)
                # Actually, transform the list (returned by flatten) into a tuple
                # for dynamo consistency.
                output = BuiltinVariable(tuple).call_function(tx, [output], {})

            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support pytree output
            # We check always_restore because we dont use the output or side effects of always_restore code,
            # like bwd.
            if always_restore:
                # Nothing left to do here
                return (output, treespec), tx.output.graph, subtracer.lifted_freevars
            else:
                validate_subgraph_output_types(output)

                # The output proxies might not belong to this SubgraphTracer
                # (if they are free variables that were never lifted)
                # so lift them here.
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

                tx.output.create_node(
                    "output",
                    "output",
                    (subtracer.create_arg((output_proxies,))),
                    {},
                )
                graph = tx.output.graph
                graph.lint()
                lifted_freevars = subtracer.lifted_freevars

                # NOTE: [HigherOrderOperator subgraph input ordering]
                # The input ordering of the higher order ops is determined by the order of
                # the creatation of the placehoder.
                # Mannually created inputs are created in validate_args_and_maybe_create_graph_inputs before
                # speculating subgraph.
                # During subgraph speculation, we may lift closured tensors and free symbols as inputs,
                # their ordering is determined by the time they are lifted: earlier lifted ones precede later
                # lifted ones.
                #
                # Suppose the placeholders are
                # O1, O2, X1, O3, O4, X2, X3, O5 where Xs are lifted phs
                # The following code re-order the placeholders to
                # O1, O2, O3, O4, O5, X1, X2, X3
                def move_lifted_freevars_phs_to_end(
                    graph: torch.fx.Graph, lifted_freevars: tuple[torch.fx.Node]
                ):
                    lifted_ph_set = {
                        child_p.node for child_p in lifted_freevars.values()
                    }

                    prev_phs = [n for n in graph.nodes if n.op == "placeholder"]

                    # No need to reorder when graph doesn't have args or doesn't
                    # have lifted freevars or all inputs are lifted freevars.
                    if (
                        len(prev_phs) == 0
                        or len(lifted_ph_set) == 0
                        or len(prev_phs) == len(lifted_ph_set)
                    ):
                        return

                    # Step 1: find first X1
                    for x1 in prev_phs:
                        if x1 in lifted_ph_set:
                            break

                    assert x1 is not None and x1.op == "placeholder"
                    # Step 2: starting from the X1, skip Xs and prepend Os before X1.
                    cand_x = x1.next
                    while cand_x is not None and cand_x.op == "placeholder":
                        if cand_x in lifted_ph_set:
                            cand_x = cand_x.next
                        else:
                            nxt = cand_x.next
                            cand_x._remove_from_list()
                            x1.prepend(cand_x)
                            cand_x = nxt

                    # Step 3: assert that all placeholders are in the correct order as .
                    # in lifted_freevars
                    after_phs = [
                        node for node in graph.nodes if node.op == "placeholder"
                    ][-len(lifted_freevars) :]
                    assert len(after_phs) == len(lifted_freevars)
                    for child_proxy, ph in zip(lifted_freevars.values(), after_phs):
                        assert child_proxy.node is ph, (
                            "The order of placeholders is different from the order of lifted_freevars"
                        )

                    graph.lint()

                if len(lifted_freevars) > 0:
                    move_lifted_freevars_phs_to_end(graph, lifted_freevars)

                if not supports_input_mutation:
                    mutation_info = subtracer.has_input_mutation()
                    if mutation_info.has_mutation:
                        context = f"{mutation_info.msg} in\n {graph}"
                        unimplemented_v2(
                            gb_type=f"Encountered input mutation during higher order op tracing for HOP - {source_target.name()}",
                            context=context,
                            explanation="Higher order ops do not support input mutation",
                            hints=[
                                "Consider using the debug context to change user code to avoid mutation.",
                                "Please open an issue.",
                            ],
                        )

                if not supports_aliasing:
                    aliasing_info = subtracer.has_aliasing()
                    if aliasing_info.has_aliasing:
                        context = f"{aliasing_info.msg} in\n {graph}"
                        unimplemented_v2(
                            gb_type=f"Encountered aliasing during higher order op tracing for HOP - {source_target.name()}",
                            context=context,
                            explanation="Higher order ops do not support aliasing",
                            hints=[
                                "Consider using the debug context to change user code to avoid aliasing.",
                                "Please open an issue.",
                            ],
                        )

                return (
                    (output, treespec),
                    graph,
                    lifted_freevars,
                )

    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.info(msg)
        log.info(ex)
        raise ex


def make_attr(tx: "InstructionTranslator", name):
    node = tx.output.create_proxy(
        "get_attr",
        name,
        (),
        {},
    )
    return node


class TorchHigherOrderOperatorVariable(VariableTracker):
    def __init__(
        self, value: HigherOrderOperator, source: Optional[Source] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(value, source=None, **kwargs):
        from torch._higher_order_ops import BaseHOP

        if value.__name__ == "cond":
            return CondHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "while_loop":
            return WhileLoopHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ("map", "map_impl"):
            return MapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "executorch_call_delegate":
            return ExecutorchCallDelegateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "out_dtype":
            return OutDtypeHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "wrap":
            return WrapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "hints_wrapper":
            return HintsWrapperHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "flex_attention":
            return FlexAttentionHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "flex_attention_backward":
            return FlexAttentionBackwardHighOrderVariable(value, source, **kwargs)
        elif value.__name__ in (
            "wrap_activation_checkpoint",
            "tag_activation_checkpoint",
        ):
            return CheckpointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "_export_tracepoint":
            return ExportTracepointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "trace_wrapped":
            return TraceWrappedHigherOrderOperatorVariable(value, source, **kwargs)
        elif value.__name__ == "strict_mode":
            return StrictModeHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "run_with_rng_state":
            return RunWithRNGStateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "associative_scan":
            return AssociativeScanHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "scan":
            return ScanHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "call_torchbind":
            return CallTorchbindHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "wrap_with_set_grad_enabled":
            return WrapWithSetGradEnabledHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "wrap_with_autocast":
            return WrapWithAutocastHigherOrderVariable(value, source, **kwargs)
        elif (
            value.__name__ == "auto_functionalized"
            or value.__name__ == "auto_functionalized_v2"
        ):
            return AutoFunctionalizeHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "invoke_subgraph":
            return InvokeSubgraphHigherOrderVariable(value, source, **kwargs)
        elif isinstance(value, BaseHOP):
            return BaseHOPVariable(value, source, **kwargs)
        elif value.__name__ == "custom_function_call":
            return CustomFunctionHigherOrderOperatorVariable(value, source, **kwargs)
        else:
            unimplemented(f"HigherOrderOperator {value.__name__}")

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        unimplemented(f"HigherOrderOperator {self.value.__name__}")


class CustomFunctionHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Wraps torch._functorch.autograd_function.custom_function_call
    """

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return torch._dynamo.variables.UserMethodVariable(
            self.value.__call__.__func__,
            torch._dynamo.variables.UserDefinedObjectVariable(
                self.value, source=self.source
            ),
            source=AttrSource(AttrSource(self.source, "__call__"), "__func__"),
        ).call_function(tx, args, kwargs)


class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="Cond doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ListVariable, TensorVariable

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        for i, k in enumerate(["pred", "true_fn", "false_fn", "operands"]):
            if v := kwargs.pop(k, None):
                assert i == len(args), (
                    "did not provide the right number of non-keyword args"
                )
                args.append(v)

        if kwargs:
            unimplemented(f"torch.cond: Got unexpected kwargs: {list(kwargs.keys())}")

        # TODO(voz): Support fake tensor dispatch for recursive
        # ops - see torch/dispatch/_dispatcher.py
        if len(args) != 4:
            unimplemented(
                f"Expected 4 arguments but got {len(args)}.\n"
                f"Usage: cond(pred, true_fn, false_fn, operands)",
            )

        # Specialize into one of the branches since pred is constant
        pred, true_fn, false_fn, operands = args
        if type(args[0]) is ConstantVariable:
            warnings.warn(
                "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."
                " If you want torch.cond to preserve two branches, please make the predicate a boolean tensor or a SymBool.",
                UserWarning,
            )
            if pred.as_python_constant():
                return true_fn.call_function(tx, operands.unpack_var_sequence(tx), {})
            else:
                return false_fn.call_function(tx, operands.unpack_var_sequence(tx), {})

        # predicate
        if type(pred) not in (ConstantVariable, TensorVariable, SymNodeVariable):
            unimplemented(
                f"Expected pred to be bool or a boolean tensor with single "
                f"item but got {str(type(pred))} "
                f"with original python type {str(pred.python_type())}.",
            )

        # operands
        if not isinstance(operands, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected operands to be a list/tuple but got "
                f"{operands.python_type()}",
            )
        operands_seq = operands.unpack_var_sequence(tx)
        if not only_consist_of(operands, (TensorVariable, ConstantVariable)):
            unimplemented(
                "Expect operands to be a tuple of pytrees that only consists of tensor leaves."
            )

        # branches
        _check_supported_callable_arg(tx, true_fn, "true_fn")
        _check_supported_callable_arg(tx, false_fn, "false_fn")

        # Our strategy for tracing the true/false branches of cond
        # are to checkpoint our graphstate, run the true branch,
        # roll it back to the checkpoint, and run the false
        # branch, and then merge the graphstates.  Well, perhaps
        # "merge" is too strong a word: we mostly assert that
        # the resulting graphstates have to be the same.
        #
        # We only permit guards to diverge (we union the guards from
        # both branches).  In particular, this means that side
        # effects are NOT permitted inside true/false branches; this
        # would be difficult to implement, because of the path
        # explosion problem.

        def speculate_branch(branch):
            # NB: 0 is predicate
            ix = 1 if branch else 2
            # TODO: Support kwargs
            (
                (ret_val, ret_treespec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[ix],
                operands_seq,
                {},
                "cond",
                source_target=self.value,
                should_flatten_outputs=True,
            )

            if not only_consist_of(ret_val, (TensorVariable,)):
                unimplemented(
                    "Expected branches to return a possibly nested list/tuple/dict of tensors but it consists of non tensors.",
                )
            return ret_val, ret_treespec, ret_graph, ret_lifted_freevars

        (true_r, true_treespec, true_graph, true_lifted_freevars) = speculate_branch(
            True
        )
        true_nn_modules = dict(tx.output.nn_modules)

        (
            false_r,
            false_treespec,
            false_graph,
            false_lifted_freevars,
        ) = speculate_branch(False)
        false_nn_modules = dict(tx.output.nn_modules)

        same_treespec = _make_inlined(tx, pytree.TreeSpec.__eq__)(
            true_treespec, false_treespec
        )
        if not same_treespec.as_python_constant():
            unimplemented("Expected branches to return the same pytree structure.")

        (
            true_graph,
            false_graph,
            true_shared,
            _false_shared,
            unique_true,
            unique_false,
        ) = _merge_graph_inputs(
            true_graph,
            true_lifted_freevars,
            "true_branch",
            false_graph,
            false_lifted_freevars,
            "false_branch",
        )

        true_name = tx.output.install_subgraph(
            "cond_true",
            torch.fx.GraphModule(true_nn_modules, true_graph),
        )
        false_name = tx.output.install_subgraph(
            "cond_false",
            torch.fx.GraphModule(false_nn_modules, false_graph),
        )

        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)

        p_args = (
            pred.as_proxy(),
            true_node,
            false_node,
            # We pick true_shared but it shouldn't matter
            tuple(true_shared + unique_true + unique_false),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.cond,
            p_args,
            {},
            None,
            true_treespec,
        )


class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def __init__(self, hop, source, script_obj_var, method_name) -> None:
        super().__init__(hop, source)
        self.script_obj_var = script_obj_var
        self.method_name = method_name

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        args_proxy = [arg.as_proxy() for arg in args]
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(
                    [self.script_obj_var.as_proxy(), self.method_name] + args_proxy
                ),
                kwargs=kwargs_proxy,
            ),
        )


def validate_subgraph_output_types(output: VariableTracker):
    """Verify that that the output of the subgraph is a tensor,
    int, bool, SymBool, or SymInt.
    """
    from . import TensorVariable

    if non_tensor_output := find_mismatched_vars(
        output, TensorVariable, allow_none=True
    ):
        for out in non_tensor_output:
            if (
                isinstance(out, SymNodeVariable) and out.python_type() in (int, bool)
            ) or (
                isinstance(out, ConstantVariable) and out.python_type() in (int, bool)
            ):
                continue
            unimplemented(
                f"HigherOrderOperator body's output must consist of tensors or ints only but got {out.python_type()}"
            )


class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="while_loop doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.while_loop import _create_unbacked_symint

        from . import TensorVariable

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))
        cond_fn, body_fn, operands, additional_inputs = args

        # Input checks
        for i, k in enumerate(["cond_fn", "body_fn", "operands"]):
            if v := kwargs.pop(k, None):
                assert i == len(args), (
                    "did not provide the right number of non-keyword args"
                )
                args.append(v)

        if kwargs:
            unimplemented(
                f"torch.while_loop: Got unexpected kwargs: {list(kwargs.keys())}"
            )

        if len(args) != 4:
            unimplemented(
                f"Expected 4 arguments but got {len(args)}.\n"
                f"Usage: while_loop(cond_fn, body_fn, operands)",
            )

        # cond_fn and body_fn input check
        _check_supported_callable_arg(tx, cond_fn, "cond_fn")
        _check_supported_callable_arg(tx, body_fn, "body_fn")

        # operands input check
        operands_seq = operands.unpack_var_sequence(tx)

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected additional_inputs to be a list/tuple but got "
                f"{additional_inputs.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        additional_inputs_seq = additional_inputs.unpack_var_sequence(tx)

        with discard_graph_changes(tx):
            # See NOTE [unspecialize int carry with unbacked symints]
            # Note: this must be run under discard graph changes.
            def create_unbacked_sym_node_var(tx) -> SymNodeVariable:
                example_value = _create_unbacked_symint(
                    tx.output.fake_mode, ignore_fresh_unbacked_symbols=True
                )
                proxy = tx.output.current_tracer.create_graph_input(
                    "unbacked_symint", type(example_value), example_value
                )
                return SymNodeVariable.create(tx, proxy, example_value)

            new_operands_seq = [
                (
                    create_unbacked_sym_node_var(tx)
                    if (
                        isinstance(carry, ConstantVariable)
                        and carry.python_type() is int
                    )
                    or (isinstance(carry, SymNodeVariable))
                    else carry
                )
                for carry in operands_seq
            ]

        # create cond subgrpahs
        (
            (cond_r, _cond_treespec),
            cond_graph,
            cond_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            cond_fn,
            new_operands_seq + additional_inputs_seq,
            {},
            "while_loop",
            source_target=self.value,
            # NOTE [why we cannot use "automatic" for while_loop]:
            # The reason is that we want to enforce
            # the ordering of inputs and outputs to be consistent and the the ordering
            # of cond_fn and body_fn to the consistent.
            # e.g. suppose we use "automatic" and we have:
            #
            # def body_fn(ph1, ph2):
            #   new_a, new_b = ph2.cos(), ph1.sin()
            #   return new_a, new_b
            #
            # a, b = torch.randn(3), torch.randn(3)
            # new_a, new_b = body_fn(a, b)
            #
            # Using automatic, the ordering of arguments will be the order that they're
            # used. In this example, the capture graph looks like:
            #
            # def captured_body(ph1, ph2):
            #   new_a, new_b = ph1.cos(), ph2.add_(1)
            #   return new_a, new_b
            #
            # This is fine when we change the calling convention of captured_body to be
            # new_a, new_b = captured_body(b, a).
            # But for while_loop, the next iteration's input is previous iteration output
            # we'll end up feeding captured_body(new_a, new_b) instead.
            # So it's best we always enforce the ordering of carried_inputs the same as outputs
            # with "flatten_manual".
            set_subgraph_inputs="flatten_manual",
        )
        cond_nn_modules = dict(tx.output.nn_modules)
        validate_subgraph_output_types(cond_r)
        if isinstance(cond_r, TensorVariable):
            cond_r_meta = _extract_tensor_metadata(
                cond_r.proxy.node.meta["example_value"], include_contiguity=False
            )
            if (
                not cond_r_meta.dtype == torch.bool
                or not cond_r_meta.shape == torch.Size([])
            ):
                unimplemented(
                    f"Expected cond_fn to return a scalar tensor or a bool but got {cond_r_meta.shape}"
                )
        elif isinstance(cond_r, ConstantVariable):
            # short-circuiting while_loop when cond_fn returns a constant such as 0, 1 True or False
            pred = cond_r.as_python_constant()
            if pred:
                unimplemented(
                    f"Infinite loop detected because while_loop's cond_fn always returns the same value {pred}"
                )
            else:
                return operands

        # create body subgraph
        (
            (body_r, body_treespec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            body_fn,
            new_operands_seq + additional_inputs_seq,
            {},
            "while_loop",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            should_flatten_outputs=True,
        )
        validate_subgraph_output_types(body_r)

        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            body_r.unpack_var_sequence(tx),
            operands_seq,
            "body_fn_output",
            "carried_inputs",
            include_contiguity=False,
        )

        (
            cond_graph,
            body_graph,
            cond_shared,
            _body_shared,
            cond_unique,
            body_unique,
        ) = _merge_graph_inputs(
            cond_graph,
            cond_lifted_freevars,
            "cond_fn",
            body_graph,
            body_lifted_freevars,
            "body_fn",
        )

        # Note: cond_shared and body_shared refer to the same proxy in parent graph
        # so using either of them is OK. Use cond_shared as it doesnt matter.
        additional_lifted_inputs = cond_shared + cond_unique + body_unique

        body_nn_modules = dict(tx.output.nn_modules)

        cond_name = tx.output.install_subgraph(
            "cond_fn",
            torch.fx.GraphModule(cond_nn_modules, cond_graph),
        )
        body_name = tx.output.install_subgraph(
            "body_fn",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        cond_node = make_attr(tx, cond_name)
        body_node = make_attr(tx, body_name)

        p_args = (
            cond_node,
            body_node,
            tuple([operand.as_proxy() for operand in operands_seq]),
            tuple(
                [inp.as_proxy() for inp in additional_inputs_seq]
                + additional_lifted_inputs
            ),
        )

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )
        unspecialized_flat_example_value = pytree.tree_map_only(
            (int, torch.SymInt),
            lambda _: _create_unbacked_symint(
                tx.output.fake_mode, ignore_fresh_unbacked_symbols=False
            ),
            flat_example_value,
        )
        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.while_loop,
            p_args,
            {},
            unspecialized_flat_example_value,
            body_treespec,
        )


class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="associative_scan must be captured completely with torch.compile."
    )
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.utils import first_slice_copy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        def arg_extractor(combine_fn, xs, additional_inputs):
            return combine_fn, xs, additional_inputs

        combine_fn, xs, additional_inputs = arg_extractor(*args, **kwargs)

        if args[0].python_type() is functools.partial:
            # This is the standard case when the user calls the frontend
            # and the frontend invokes dynamo
            if len(args) != 2:
                unimplemented(
                    f"Expected 2 positional arguments but got {len(args)}.\n"
                    f"Usage: associative_scan(combine_fn, xs)",
                )

            xs_treespec = args[0].keywords["spec"]

            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx, combine_fn.keywords["combine_fn"], "combine_fn"
            )
        else:
            # This case is hit during re-tracing, for example in export tests
            # In this case, the combine_fn is a callable and not a functools.partial
            xs_treespec = _make_inlined(tx, pytree.tree_structure)(xs)

            _check_supported_callable_arg(tx, combine_fn, "combine_fn")

        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected xs to be a list/tuple but got "
                f"{xs.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        xs_vars = xs.unpack_var_sequence(tx)
        _check_all_tensorvariable(xs_vars)

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected additional_inputs to be a list/tuple but got "
                f"{additional_inputs.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        additional_inputs_vars = additional_inputs.unpack_var_sequence(tx)
        _check_all_tensorvariable(additional_inputs_vars)

        scan_length = get_fake_value(xs_vars[0].as_proxy().node, tx).size()[0]
        if scan_length == 0:
            unimplemented(
                "associative_scan() operator doesn't support zero-sized tensors during tracing."
            )

        # Trace the subgraph
        # The sub_args is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
        # the sub_args shape will be (4, ).
        with discard_graph_changes(tx):
            sub_args = [
                _make_inlined(tx, first_slice_copy)(leaf)
                for leaf in itertools.chain(xs_vars, xs_vars)
            ]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=(), kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args + sub_args_additional_inputs
        (
            (combine_result, _combine_treespec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description="associative_scan_combine_fn",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
        )

        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()

        # Collect the results from the combine_fn
        results, _combine_treespec = _make_inlined(tx, pytree.tree_flatten)(
            combine_result
        ).unpack_var_sequence(tx)

        # Check whether the combine_fn returns one child tree for the output.
        if _combine_treespec.as_python_constant().num_leaves < 1:
            unimplemented(
                f"combine_fn needs to produce one pytree for the output "
                f"but combine_fn produces the pytree {_combine_treespec.as_python_constant()}."
            )

        # Check whether the outs produced by combine_fn has the same treespec as xs
        # We need to have this check this way, because in case init is a TreeSpec and carry
        # but carry is only a LeafSpec, these two cannot be compared correctly.
        if (
            isinstance(xs_treespec.as_python_constant(), pytree.LeafSpec)
            != isinstance(_combine_treespec.as_python_constant(), pytree.LeafSpec)
        ) or not _make_inlined(tx, pytree.TreeSpec.__eq__)(
            xs_treespec, _combine_treespec
        ).as_python_constant():
            unimplemented(
                f"The tree structure of the xs and the outs of the combine_fn are are expected to be identical, but got "
                f"xs: {xs_treespec.as_python_constant()} vs output: {_combine_treespec.as_python_constant()}."
            )

        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            [_make_inlined(tx, first_slice_copy)(t) for t in xs_vars],
            results.items,
            "initial_xs",
            "combine_fn_output",
            include_contiguity=False,
        )

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)

        from torch._higher_order_ops.utils import (
            _has_potential_branch_input_alias,
            _has_potential_branch_input_mutation,
            _maybe_fake_tracing,
        )
        from torch._inductor.utils import is_pointwise_use
        from torch._subclasses.fake_tensor import FakeTensor

        with tx.fake_mode:
            xs_fake = [
                first_slice_copy(leaf.proxy.node.meta["example_value"].clone())
                for leaf in itertools.chain(xs_vars, xs_vars)
            ]
            additional_fake = [
                leaf.proxy.node.meta["example_value"].clone()
                for leaf in additional_inputs_vars
            ] + [
                leaf.node.meta["example_value"].clone()
                if isinstance(leaf.node.meta["example_value"], FakeTensor)
                else leaf.node.meta["example_value"]
                for leaf in combine_lifted_freevars.keys()
            ]
            sub_args_fake = xs_fake + additional_fake
            pre_dispatch = False

            fx = _maybe_fake_tracing(
                combine_gm, sub_args_fake, pre_dispatch=pre_dispatch
            )

            for node in fx.graph.nodes:
                # Check that the combine_fn is pointwise, if combine_mode='pointwise'
                if not all(
                    is_pointwise_use(use) or use.op == "output" for use in node.users
                ):
                    raise RuntimeError(
                        "For combine_mode='pointwise', the combine_fn needs to be pointwise"
                    )

            if _has_potential_branch_input_mutation(
                combine_gm, sub_args_fake, pre_dispatch=pre_dispatch
            ):
                raise RuntimeError("Combine_fn might be modifying the input!")  # noqa: F541
            if _has_potential_branch_input_alias(
                combine_gm, sub_args_fake, pre_dispatch=pre_dispatch
            ):
                raise RuntimeError("Combine_fn might be aliasing the input!")  # noqa: F541

        combine_fn_name = tx.output.install_subgraph(
            "associative_scan_combine_fn", combine_gm
        )

        # Compute the proxies
        xs_proxy = xs.as_proxy()
        combine_freevars_proxy = tuple(combine_lifted_freevars.keys())
        additional_inputs_proxy = additional_inputs.as_proxy() + combine_freevars_proxy

        p_args = (
            make_attr(tx, combine_fn_name),
            xs_proxy,
            additional_inputs_proxy,
        )

        with tx.fake_mode:
            out_meta = tuple(
                inp_proxy.node.meta["example_value"].clone() for inp_proxy in xs_proxy
            )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.associative_scan,
            p_args,
            {},
            out_meta,
            xs_treespec,
        )


class ScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="scan must be captured completely with torch.compile."
    )
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.scan import _extract_carry_and_out, stack_y
        from torch._higher_order_ops.utils import first_slice_copy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        # combine_fn input check
        def _check_combine_fn_is_normalized(combine_fn_var):
            if not isinstance(
                combine_fn_var,
                (
                    variables.nn_module.NNModuleVariable,
                    variables.FunctoolsPartialVariable,
                ),
            ):
                unimplemented(
                    f"Expected combine_fn to be wrapped as functools.partial in scan user-facing api "
                    f"or a graph module if we're re-exporting but got "
                    f"{combine_fn.python_type()}. Please report an issue to PyTorch if you're seeing this."
                )
            return isinstance(combine_fn_var, variables.nn_module.NNModuleVariable)

        def arg_extractor(combine_fn, init, xs, additional_inputs):
            return combine_fn, init, xs, additional_inputs

        combine_fn, init, xs, additional_inputs = arg_extractor(*args, **kwargs)
        init_vars = init.unpack_var_sequence(tx)
        xs_vars = xs.unpack_var_sequence(tx)
        additional_inputs_vars = additional_inputs.unpack_var_sequence(tx)

        # combine_fn input check
        combine_fn_is_normalized = _check_combine_fn_is_normalized(combine_fn)
        if combine_fn_is_normalized:
            combine_gm = combine_fn.value
            assert isinstance(combine_gm, torch.fx.GraphModule), (
                combine_fn,
                combine_gm,
            )
        else:
            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx, combine_fn.keywords["combine_fn"], "combine_fn"
            )
        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected xs to be a list/tuple but got "
                f"{xs.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        # init input check
        if not isinstance(init, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected init to be a list/tuple with at least one element but got "
                f"{init.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        if len(init_vars) == 0:
            unimplemented(
                "scan() operator requires init leaves.  It seems to be an "
                "internal error, please report an issue to PyTorch."
            )
        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected additional_inputs to be a list/tuple but got "
                f"{additional_inputs.python_type()}. It seems to be an "
                f"internal error, please report an issue to PyTorch."
            )
        # scan_length check
        scan_length = get_fake_value(xs_vars[0].as_proxy().node, tx).size()[0]
        if scan_length == 0:
            unimplemented("NYI: scan() operator doesn't support zero scan_length.")
        _check_all_tensorvariable(init_vars)
        _check_all_tensorvariable(xs_vars)
        _check_all_tensorvariable(additional_inputs_vars)

        with discard_graph_changes(tx):
            sub_args_init = [
                ini.call_method(tx, "clone", args=(), kwargs={}) for ini in init_vars
            ]
            # The sub_args_inp is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
            # the sub_args_inp shape will be (4, ).
            sub_args_inp = [_make_inlined(tx, first_slice_copy)(inp) for inp in xs_vars]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=(), kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args_init + sub_args_inp + sub_args_additional_inputs
        (
            (combine_result, _combine_treespec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description="scan_combine_fn",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
        )
        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()
        combine_freevars_proxy = list(combine_lifted_freevars.keys())
        combine_result_vars = combine_result.unpack_var_sequence(tx)

        if combine_fn_is_normalized:
            carry_vars, out_vars = _extract_carry_and_out(
                combine_result_vars, len(init_vars)
            )
        else:
            if len(combine_result_vars) != 2:
                unimplemented(
                    f"Expect combine_fn to return a tuple (next_carry, y) but got {combine_result_vars}"
                )
            carry_tree, out_vars = combine_result_vars
            carry_vars, carry_treespec = _make_inlined(tx, pytree.tree_flatten)(
                carry_tree
            ).unpack_var_sequence(tx)
            carry_vars = carry_vars.unpack_var_sequence(tx)
            out_vars = _make_inlined(tx, pytree.tree_leaves)(
                out_vars
            ).unpack_var_sequence(tx)

            # additional output checking
            _combine_treespec = _make_inlined(tx, pytree.tree_structure)(combine_result)

            check_meta_consistency_vt(
                init_vars,
                carry_vars,
                "init",
                "carry",
            )

        # Check meta data of carries and inits. If we pass this stage, we are sure that the init and carries
        # have the same tree structure.
        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            init_vars,
            carry_vars,
            "init",
            "carry",
            include_contiguity=False,
        )

        xs_proxy = xs.as_proxy()
        init_proxy = init.as_proxy()
        additional_inputs_proxy = list(additional_inputs.as_proxy()) + list(
            combine_freevars_proxy
        )
        y_proxies = [out_var.as_proxy() for out_var in out_vars]

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)
        combine_fn_name = tx.output.install_subgraph("scan_combine_fn", combine_gm)

        p_args = (
            make_attr(tx, combine_fn_name),
            init_proxy,
            xs_proxy,
            additional_inputs_proxy,
        )

        with tx.fake_mode:
            example_carry = [
                init_p.node.meta["example_value"].clone() for init_p in init_proxy
            ]
            # For the fake mode, we need to duplicate the init tensor along the dim
            # to have the same size as the xs arguments
            example_stacked_out = [
                stack_y(y.node.meta["example_value"], scan_length) for y in y_proxies
            ]
            out_meta = [*example_carry, *example_stacked_out]

        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.scan, p_args, {}, out_meta, _combine_treespec
        )


def non_single_tensor_return_unsupported(api, ret):
    from . import TensorVariable

    if not isinstance(ret, TensorVariable):
        raise Unsupported(
            f"{api} over function that returns something other than one Tensor"
        )


class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from . import TensorVariable
        from .builder import wrap_fx_proxy_cls

        if len(kwargs) > 0:
            unimplemented(
                "torch.ops.higher_order.map: kwargs are not supported in the map operator."
            )

        _check_supported_callable_arg(tx, args[0].realize(), "map_fn")

        assert type(args[1].realize()) is TensorVariable

        sample_shape = get_fake_value(args[1].as_proxy().node, tx).size()

        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented(
                "map() operator doesn't support scalar or zero-sized tensors during tracing."
            )

        # To get the example output from map() we will need to provide at least one sample to
        # the loop body. In our case we will always use xs[0], and our map() won't support zero
        # sized tensor during tracing.
        with discard_graph_changes(tx):
            first_dim = wrap_fx_proxy_cls(
                target_cls=TensorVariable, tx=tx, proxy=args[1].as_proxy()[0]
            )

        # TODO: Support kwargs
        (
            (body_r, body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            [
                first_dim,
                *args[2:],
            ],
            {},
            "torch.ops.higher_order.map",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            should_flatten_outputs=True,
        )

        subgraph_example_value = [
            proxy.node.meta["example_value"] for proxy in body_r.as_proxy()
        ]

        with tx.output.fake_mode:
            # We need to expand the example output from map() so that it has
            # the same first dimension as the mapped input.
            # We also do a clone with contiguous_format. This is to be consistent with
            # eager semantic of map, which stacks the outputs. The result is contiguous
            # as a result of the stack operation.
            map_example_out = [
                t.expand(sample_shape[0], *t.size()).clone(
                    memory_format=torch.contiguous_format
                )
                for t in subgraph_example_value
            ]

        body_nn_modules = dict(tx.output.nn_modules)

        body_name = tx.output.install_subgraph(
            "map_body",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        p_args = (
            body_node,
            [args[1].as_proxy()],
            [arg.as_proxy() for arg in args[2:]] + list(body_lifted_freevars.keys()),
        )

        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.map_impl, p_args, {}, map_example_out, body_spec
        )


class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        # This is operator for delegation within Executorch which calls a
        # specific function in the given lowered module with the given
        # operators. The actual operator is defined in the Executorch codebase.
        # This is a bad hierarchical violation since
        # executorch_call_delegate sits at a higher level than dynamo, but
        # there's no real solution to this issue yet.
        if len(kwargs) > 0:
            unimplemented(
                "executorch_call_delegate: kwargs arguments were not enabled."
            )
        lowered_module = tx.output.get_submodule(args[0].module_key)

        lowered_node = make_attr(tx, args[0].module_key)

        p_args = tuple(arg.as_proxy() for arg in args[1:])
        real_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: get_fake_value(a.node, tx), p_args
        )

        with tx.fake_mode:
            example_value = lowered_module.original_module.module()(*real_sub_args)

        # NOTE [Guaranteeing the 1-1 correspondence of FakeTensors and real tensors]:
        # executorch modules promise not to alias inputs and outputs.
        # Thus, output FakeTensors will correctly not alias input FakeTensors.
        _assert_tensors_nonaliasing(real_sub_args, example_value)

        p_args = (lowered_node,) + p_args

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class FunctorchHigherOrderVariable(UserFunctionVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return super().call_function(tx, args, kwargs)


class FunctionalCallVariable(FunctorchHigherOrderVariable):
    def call_function(
        self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker:
        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            unimplemented(
                "torch.func.functional_call capture is disabled, "
                "it can be turned on by setting "
                "`torch._dynamo.config.inline_inbuilt_nn_modules=True`"
            )
        return super().call_function(tx, args, kwargs)


class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = True
    supports_aliasing = True

    def install_subgraph_in_output_graph(
        self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name="wrap_body"
    ):
        return tx.output.install_subgraph(
            f"{attr_name}",
            body_gmod,
        )

    def create_wrapped_node(
        self,
        tx: "InstructionTranslator",
        fn_vt,
        fn_args_vt,
        kwargs,
        description,
        under_activation_checkpoint=False,
        *,
        subgraph_name="wrap_body",
    ):
        # See NOTE [HigherOrderOperator tracing design] for more details

        (
            (body_r, treespec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            description,
            source_target=self.value,
            should_flatten_outputs=True,
            under_activation_checkpoint=under_activation_checkpoint,
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = self.install_subgraph_in_output_graph(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            body_gmod,
            attr_name=subgraph_name,
        )
        body_node = make_attr(tx, body_name)

        # Since, we call `speculate_subgraph` with `set_subgraph_inputs="automatic`,
        # all the arguments are lifted.
        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())

        proxy_args = (body_node,) + lifted_args
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return proxy_args, {}, example_value, body_r, treespec, body_gmod, body_name

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This flattens the kwargs into lifted args
        (
            p_args,
            p_kwargs,
            _example_value,
            body_r,
            treespec,
            _,
            _,
        ) = self.create_wrapped_node(tx, args[0], args[1:], kwargs, "wrap")

        if len(p_kwargs) > 0:
            unimplemented("kwargs should have been flattened into lifted args")

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx, self.value, tuple(p_args), p_kwargs, flat_example_value, treespec
        )


class WrapWithSetGradEnabledHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                f"wrap_with_set_grad_enabled: Got unexpected kwargs: {list(kwargs.keys())}"
            )

        grad_enabled, fn_var, *rest_args = args

        if not isinstance(grad_enabled, ConstantVariable):
            unimplemented("grad_enabled must be a constant")

        _check_supported_callable_arg(tx, fn_var, "enable_grad_fn")

        with torch.set_grad_enabled(grad_enabled.as_python_constant()):
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                "torch.ops.higher_order.wrap_with_set_grad_enabled",
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                f"wrap_with_set_grad_enabled: Got unexpected freevars {body_lifted_freevars}"
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                grad_enabled.as_python_constant(),
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )
        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec
        )


class WrapWithAutocastHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                f"wrap_with_autocast: Got unexpected kwargs: {list(kwargs.keys())}"
            )

        device_type, dtype, enabled, cache_enabled, fn_var, *rest_args = args

        for arg in [device_type, dtype, enabled, cache_enabled]:
            if not isinstance(arg, ConstantVariable):
                unimplemented(
                    "device_type, dtype, enabled, cache_enabled must be constants"
                )

        _check_supported_callable_arg(tx, fn_var, "autocast")

        python_constants = [
            arg.as_python_constant()
            for arg in [device_type, dtype, enabled, cache_enabled]
        ]

        with torch.autocast(*python_constants):
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                "torch.ops.higher_order.wrap_with_autocast",
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                f"wrap_with_autocast: Got unexpected freevars {body_lifted_freevars}"
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                *python_constants,
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec
        )


class HintsWrapperHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="Hints_wrapper doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self, tx, args: "list[VariableTracker]", kwargs: "dict[str, VariableTracker]"
    ) -> "VariableTracker":
        _check_supported_callable_arg(tx, args[0], "body_fn")

        # inputs
        if len(args) != 3:
            unimplemented(
                f"Expected 3 arguments but got {len(args)}.\n"
                f"Usage: hints_wrapper(body_fn, args, kwargs, hints).\n"
                f"kwargs required to be provided explicitly."
            )

        if not isinstance(args[1], (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected a tuple but got {args[1].python_type()}",
            )
        operands = args[1].unpack_var_sequence(tx)

        if not isinstance(args[2], ConstDictVariable):
            unimplemented(
                f"Expected a dict but got {args[2].python_type()}",
            )

        if "hints" not in kwargs:
            raise IncorrectUsage("hints_wrapper - key hints not provided")

        (
            (body_r, treespec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],  # function
            operands,
            args[2].as_python_constant(),
            "hints_wrapper",
            source_target=self.value,
            should_flatten_outputs=True,
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "hints_wrapper_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        # Since, we call `speculate_subgraph` with `set_subgraph_inputs="automatic`,
        # all the arguments are lifted.
        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())
        p_args = (body_node, lifted_args, {})

        p_kwargs = {}
        # add hints into p_kwargs
        p_kwargs["hints"] = kwargs["hints"].as_python_constant()

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx, self.value, p_args, p_kwargs, flat_example_value, treespec
        )


class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        if len(kwargs) > 0:
            unimplemented("out_dtype does not handle kwargs")

        p_args = tuple(arg.as_proxy() for arg in args)
        op = p_args[0]
        output_dtype = p_args[1]
        fake_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: a.node.meta["example_value"], p_args[2:]
        )
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        example_value = op(*fake_sub_args).to(dtype=output_dtype)

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class StrictModeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="strict_mode HOO doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unpacked_sequence = args[1].unpack_var_sequence(tx)
        # TODO (tmanlaibaatar) support pytree here
        for arg in unpacked_sequence:
            if isinstance(arg, (ListVariable, TupleVariable, ConstDictVariable)):
                unimplemented("strict_mode HOO only works for flat inputs for now")

        if kwargs:
            unimplemented(
                f"strict_mode HOO received unexpected kwargs: {list(kwargs.keys())}"
            )

        (
            (ret_val, ret_treespec),
            ret_graph,
            ret_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            unpacked_sequence,
            {},
            "strict_mode",
            source_target=self.value,
            should_flatten_outputs=True,
        )

        strict_mode_nn_modules = dict(tx.output.nn_modules)

        strict_mode_name = tx.output.install_subgraph(
            "strict_mode_body",
            torch.fx.GraphModule(strict_mode_nn_modules, ret_graph),
        )

        strict_mode_node = make_attr(tx, strict_mode_name)
        p_args = (
            strict_mode_node,
            tuple(arg for arg in ret_lifted_freevars.keys()),
        )

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            ret_val.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.strict_mode,
            p_args,
            {},
            flat_example_value,
            ret_treespec,
        )


class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn

        from .builder import wrap_fx_proxy

        context_fn = None
        if "context_fn" in kwargs and kwargs["context_fn"] != noop_context_fn:
            ctx = kwargs.pop("context_fn")
            if isinstance(ctx, torch._dynamo.variables.UserFunctionVariable):
                context_fn = ctx.fn
            elif isinstance(
                ctx, torch._dynamo.variables.functions.FunctoolsPartialVariable
            ):
                context_fn = ctx.as_python_constant()
            else:
                raise NotImplementedError(
                    f"checkpoint not implemented for {type(ctx)} context_fn"
                )

        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)

        # Here we use checkpoint_kwargs (and not gmod kwargs). gmod_kwargs are
        # already flattened above and managed inside the fx graph.
        (
            p_args,
            _,
            example_value,
            _body_r,
            treespec,
            checkpointed_gmod,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[0],
            args[1:],
            gmod_kwargs,
            "torch.utils.checkpoint.checkpoint",
            under_activation_checkpoint=True,
        )
        if context_fn is not None:
            checkpointed_gmod.meta["_checkpoint_context_fn"] = context_fn

        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)

        # Store the invocation as a call
        variable = wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs=checkpoint_kwargs,
            ),
            example_value=example_value,
        )

        if treespec is None:
            return variable

        # Transform variable back into a list (previously made into a tuple by
        # speculate_subgraph function) so as to respect the pytree API typing.
        variable = BuiltinVariable(list).call_function(tx, [variable], {})

        return _make_inlined(tx, pytree.tree_unflatten)(variable, treespec)


class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class AutoFunctionalizeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "list[VariableTracker]", kwargs: "dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class FlexAttentionBackwardHighOrderVariable(TorchHigherOrderOperatorVariable):
    def proxy_submod(self, tx, arg):
        assert isinstance(arg.source, DictGetItemSource)
        submod_name = tx.output.install_subgraph(arg.source.index, arg.value)
        p_submod = make_attr(tx, submod_name)
        set_example_value(p_submod.node, arg.value)
        return p_submod

    def to_proxy(self, tx, arg):
        if isinstance(arg, UnspecializedNNModuleVariable):
            return self.proxy_submod(tx, arg)
        elif isinstance(arg, (ListVariable, TupleVariable)):
            return arg.python_type()(
                self.to_proxy(tx, nested_arg) for nested_arg in arg.items
            )
        else:
            return arg.as_proxy()

    def call_function(
        self, tx, args: "list[VariableTracker]", kwargs: "dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        try:
            p_args = tuple(self.to_proxy(tx, arg) for arg in args)
            p_kwargs = {key: self.to_proxy(tx, arg) for key, arg in kwargs.items()}
        except (NotImplementedError, Unsupported) as err:
            raise Unsupported(
                "Missing Dynamo support for FlexAttentionBackward HOP argument. Please file an issue."
            ) from err
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Handles torch._dynamo._trace_wrapped_higher_order_op.inner_trace
    by unwrapping the higher order op and inlining through it.  This op
    is created by dynamo to survive through AotAutograd, then unwrapped
    here in the call to dynamo from compiled autograd.
    """

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        kwargs = dict(kwargs)
        fn = kwargs.pop("fn")
        return fn.call_function(tx, args, kwargs)


class FlexAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @staticmethod
    def normalize_to_args(args, kwargs):
        # input signature is (query, key, value, score_mod, block_mask, *other_buffers),
        # block_mask is a tuple, and we don't want to flatten it.
        # only flatten kwargs into lists
        flat_kwargs = pytree.tree_flatten(kwargs)[0]

        # Combine the flattened lists
        all_args = args + flat_kwargs
        return all_args

    def create_wrapped_node(
        self,
        tx: "InstructionTranslator",
        query: "VariableTracker",
        fn: "VariableTracker",
        fn_name: str,
    ):
        from .._trace_wrapped_higher_order_op import TransformGetItemToIndex

        tx: InstructionTranslator = tx

        def create_scalar():
            return query.call_method(
                tx,
                "new_empty",
                (VariableTracker.build(tx, []),),
                {
                    "dtype": VariableTracker.build(tx, torch.int32),
                },
            )

        with discard_graph_changes(tx):
            bhmn = [create_scalar() for _ in range(4)]
            if fn_name == "score_mod":
                scores_require_grad: bool = query.requires_grad
                score = query.call_method(
                    tx,
                    "new_empty",
                    (VariableTracker.build(tx, []),),
                    {"requires_grad": VariableTracker.build(tx, scores_require_grad)},
                )
                new_args = [score, *bhmn]
            else:
                assert fn_name == "mask_fn", "Illegal function name: " + fn_name
                new_args = [*bhmn]

        with TransformGetItemToIndex():
            (
                (_body_output, _body_treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                new_args,
                {},  # expect only args no kwargs for now
                description=fn_name,
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        body_name = tx.output.install_subgraph(
            fn_name,
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        # It is possible that the score-mod function captures some free variables that are not
        # passed in as arguments. In this case, we need to lift them, which is handled by speculate_subgraph.
        # We then need to create proxies for this + the inputs.

        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())

        proxy_args = (body_node, lifted_args)

        return proxy_args

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from torch._higher_order_ops.flex_attention import flex_attention_fake_impl

        from .builder import wrap_fx_proxy

        (
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
        ) = self.normalize_to_args(args, kwargs)

        score_mod_node, score_mod_lifted_args = self.create_wrapped_node(
            tx, query, score_mod, "score_mod"
        )
        mask_fn = block_mask.items[-1]
        if isinstance(mask_fn, ConstantVariable):
            mask_fn = UserFunctionVariable(torch.nn.attention._flex_attention._no_mask)
        mask_fn_node, mask_fn_lifted_args = self.create_wrapped_node(
            tx, query, mask_fn, "mask_fn"
        )

        proxied_args = [
            query,
            key,
            value,
            TupleVariable(block_mask.items[:-1], source=block_mask.source),
            scale,
            kernel_options,
        ]

        # Store the invocation as a call
        # Norm_kwargs contains the score_function and we dont want to proxy this because
        # Proxying user defined functions is not supported.
        inp_args, _ = proxy_args_kwargs(proxied_args, {})

        query_meta = query.as_proxy().node.meta["example_value"]
        value_meta = value.as_proxy().node.meta["example_value"]
        with torch._guards.TracingContext.try_get().fake_mode:
            out_meta, lse_meta = flex_attention_fake_impl(query_meta, value_meta)
        example_value = (out_meta, lse_meta)

        # Compose the ordered HOO args:
        # - inp_args: [query, key, value, block_mask, scale, kernel_options]
        # - subgraph node: [score_mod, mask_fn_node]
        # - lifted args from tracing subgraph: [score_mod_other_buffers, mask_fn_other_buffers]
        _, _, _, inp_arg_block_mask, inp_arg_scale, inp_arg_kernel_options = inp_args
        block_mask = tuple(inp_arg_block_mask + (mask_fn_node,))
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=inp_args[:3]
                + (
                    score_mod_node,
                    block_mask,
                    inp_arg_scale,
                    inp_arg_kernel_options,
                    score_mod_lifted_args,
                    mask_fn_lifted_args,
                ),
                kwargs={},
            ),
            example_value=example_value,
        )


class AutogradFunctionApplyVariable(VariableTracker):
    def __init__(self, fwd_graph, bwd_graph, parent_source, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fwd_graph = fwd_graph
        self.bwd_graph = bwd_graph
        self.parent_source = parent_source

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            AutogradFunctionContextVariable,
            UserDefinedClassVariable,
            UserFunctionVariable,
            UserMethodVariable,
        )
        from .builder import wrap_fx_proxy

        """
        Consider the following:
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()
            @staticmethod
            def backward(ctx, grad):
                x, = ctx.saved_tensors
                return grad * x.cos()
        We want the resulting graphs to look like:
        def fwd(ctx, x):
            # (output, saved tensors / attrs)
            return (x.sin(), [x])
        # bwd(ctx, grad0, grad1, ..., gradn, *saved_tensors_or_attrs)
        def bwd(ctx, grad, x):
            return grad * x.cos()
        To accomplish this, we're going to:
        1. Construct a ctx object
        2. (fwd_out, _), fwd_graph, fwd_freevars = speculate_subgraph on MySin.forward (manually_set_inputs=True)
        3. (bwd_out, _), bwd_graph, bwd_freevars = speculate_subgraph on MySin.backward, while manually setting
        the ctx and grad inputs.
        4. Manually rewriting the fwd graph's output to be (output, stuff_that_gets_used in bwd_graph)
        Getting from 3 to 4 is pretty elegant: stuff_that_gets_used in bwd graph is
        just the bwd_freevars returned from speculate_subgraph, assuming MySin.backward
        doesn't capture any arguments.
        All these steps work if MySin.backward doesn't capture any values. This is a
        limitation in general that we should check for.
        """

        prev_side_effects = tx.output.side_effects.clone()
        fwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=tx.output.current_tracer,
            source_target="autograd.Function",
        )

        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        if isinstance(self.fwd_graph, types.FunctionType):
            fwd_fn = UserFunctionVariable(self.fwd_graph)
            fwd_args = [ctx, *args]
        elif isinstance(self.fwd_graph, types.MethodType):
            fwd_fn = UserMethodVariable(
                self.fwd_graph.__func__,
                UserDefinedClassVariable(self.fwd_graph.__class__),
            )
            fwd_args = [fwd_fn.obj, ctx, *args]
        else:
            unimplemented("non-function or method")

        # Speculate subgraph on the fwd
        (fwd_out, _), fwd_graph, fwd_freevars = speculate_subgraph(
            tx,
            fwd_fn,
            fwd_args,
            kwargs,
            "autograd.Function",
            enable_grad=False,
            set_subgraph_inputs="semi_automatic",
            restore_side_effects=False,
            tracer=fwd_tracer,
        )

        if ctx in tx.output.side_effects.store_attr_mutations:
            if (
                "_materialize_non_diff_grads"
                in tx.output.side_effects.store_attr_mutations[ctx]
            ):
                unimplemented("NYI")

        bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=fwd_tracer,
            source_target="autograd.Function",
        )

        # Speculate subgraph on the backward. We make the
        # bwd tracer a child of the fwd tracer, because backward may rely on
        # tensors/attrs created in the fwd tracer.

        if isinstance(fwd_out, variables.BaseListVariable):
            bwd_args = [ctx, *fwd_out.items]
        else:
            bwd_args = [ctx, fwd_out]

        bwd_src = AttrSource(self.parent_source, member="backward")
        if isinstance(self.bwd_graph, types.FunctionType):
            bwd_fn = UserFunctionVariable(self.bwd_graph, source=bwd_src)
        elif isinstance(self.bwd_graph, types.MethodType):
            bwd_fn = UserMethodVariable(
                self.bwd_graph.__func__,
                UserDefinedClassVariable(self.bwd_graph.__class__),
                source=bwd_src,
            )
            bwd_args = [bwd_fn.obj, *bwd_args]
        else:
            unimplemented("non-function or method")

        def is_strict_for(v: VariableTracker):
            if isinstance(v, variables.TensorVariable):
                # we can be more lax for stuff from forward
                return v.proxy.tracer is not fwd_tracer
            return True

        with (
            tx.output.subtracer(fwd_fn, fwd_tracer),
            tx.strict_translation_mode(is_strict_for),
        ):
            try:
                (bwd_out, _), bwd_graph, bwd_freevars = speculate_subgraph(
                    tx,
                    bwd_fn,
                    bwd_args,
                    kwargs,
                    "autograd.Function",
                    enable_grad=False,
                    set_subgraph_inputs="manual",
                    restore_side_effects=False,
                    tracer=bwd_tracer,
                )
            except torch._dynamo.exc.Unsupported as e:
                if isinstance(
                    e, torch._dynamo.exc.UnknownPropertiesDuringBackwardTrace
                ):
                    from unittest import mock

                    bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
                        tx.output,
                        parent=fwd_tracer,
                        source_target="autograd.Function",
                    )
                    from .._trace_wrapped_higher_order_op import (
                        autograd_function_backward_rewritten,
                    )

                    if isinstance(self.bwd_graph, types.FunctionType):
                        bwd_fn = UserFunctionVariable(
                            autograd_function_backward_rewritten(self.bwd_graph)
                        )
                    elif isinstance(self.bwd_graph, types.MethodType):
                        bwd_fn = UserMethodVariable(
                            autograd_function_backward_rewritten(
                                self.bwd_graph.__func__
                            ),
                            UserDefinedClassVariable(self.bwd_graph.__class__),
                        )
                    else:
                        unimplemented("non-function or method")

                    with mock.patch(
                        "torch._dynamo.config._autograd_backward_strict_mode_conditional_banned_ops",
                        [],
                    ):
                        (bwd_out, _), bwd_graph, bwd_freevars = speculate_subgraph(
                            tx,
                            bwd_fn,
                            bwd_args,
                            kwargs,
                            "autograd.Function",
                            enable_grad=False,
                            set_subgraph_inputs="manual",
                            restore_side_effects=False,
                            tracer=bwd_tracer,
                        )
                else:
                    raise e

        # TODO: assert that bwd_graph didn't capture values that were
        # not created inside fwd_graph.

        # TODO(oulgen): Ideally, we would not do a linear search for output
        # node but as things currently are there could be nodes after the
        # output node
        # This is bug prone as if there's code after the output node, then
        # graph.output will append the output at the very end
        # This might be a behavior difference

        # If users call ctx.mark_non_differentiable, we should capture these output tensors who
        # are marked as non-differentiable and pass them to ApplyTemplate
        # at torch._functorch.autograd_function.AutogradFunctionApply for reconstruction.
        non_differentiable_idx = []
        if ctx.non_differentiable is not None:
            non_differentiable_set = set(ctx.non_differentiable)
            assert isinstance(fwd_out, variables.BaseListVariable)
            for i, x in enumerate(fwd_out.items):
                if (
                    isinstance(x, variables.TensorVariable)
                    and x.as_proxy() in non_differentiable_set
                ):
                    non_differentiable_idx.append(i)

        # Rewrite the output of fwd_graph to (output, stuff_necessary_for_bwd)
        for node in fwd_graph.find_nodes(op="output"):
            fwd_graph.erase_node(node)
            break

        # Because we lift the bwd_freevars as inputs of the bwd_graph,
        # we have to manually add the bwd_freevars as output of fwd_graph.
        # However, the bwd_freevars got from speculate_subgraph use the Proxies in the bwd_graph,
        # we need to convert them to Proxies in the fwd_graph and then generate new fwd_graph output.
        fwd_proxy_of_bwd_freevars = []
        for k in bwd_freevars.keys():
            if k in fwd_freevars:
                fwd_proxy_of_bwd_freevars.append(fwd_freevars[k])
            else:
                fwd_proxy_of_bwd_freevars.append(k)

        def unwrap_proxy(x):
            if isinstance(x, torch.fx.Proxy):
                return x.node
            else:
                assert variables.ConstantVariable.is_literal(x), (
                    f"Only constant is allowed. Got {x}"
                )
                return x

        new_fwd_graph_outputs = (fwd_out.as_proxy(), fwd_proxy_of_bwd_freevars)
        new_fwd_graph_outputs = pytree.tree_map(unwrap_proxy, new_fwd_graph_outputs)
        fwd_graph.output(new_fwd_graph_outputs)
        fwd_graph.lint()

        # Store fwd_body
        fwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        fwd_name = tx.output.install_subgraph(
            "fwd_body",
            torch.fx.GraphModule(fwd_nn_modules.nn_modules, fwd_graph),
        )

        fwd_node = make_attr(tx, fwd_name)

        # The type of original args can be arbitrary, but we only support basic type in FX graph.
        # So the speculated subgraph input includes original tensor args and the lifted freevars.
        # We need to filter out the original tensor args and concat them with the lifted freevars
        # to generate the proxy args for the FX call_function node.
        filtered_args = []
        # A boolean list to mark if the type of corresponding argument is tensor.
        # This is used to determine if a FX node's argument should be an argument of
        # ApplyTemplate.forward and if we should skip the output from ApplyTemplate.backward
        # at torch._functorch.autograd_function.AutogradFunctionApply.
        args_tensor_mask = [False] * len(args)
        for i, arg in enumerate(args):
            if isinstance(arg, (variables.TensorVariable, variables.SymNodeVariable)):
                filtered_args.append(arg)
                args_tensor_mask[i] = True

        # Rewrite the output of bwd_graph to remove the grad output for the non-Tensor args.
        new_bwd_graph_outputs = None
        for node in bwd_graph.find_nodes(op="output"):
            bwd_graph.erase_node(node)
            break

        # The same as the above fwd proxies, we need to use the bwd proxies in the bwd_graph
        # if some of the output is from fwd_freevars.
        bwd_out_proxy = bwd_out.as_proxy()
        bwd_proxy_of_fwd_freevars = []
        if isinstance(bwd_out_proxy, (tuple, list)):
            for k in bwd_out_proxy:
                if k in bwd_freevars:
                    bwd_proxy_of_fwd_freevars.append(bwd_freevars[k])
                else:
                    bwd_proxy_of_fwd_freevars.append(k)
        else:
            if bwd_out_proxy in bwd_freevars:
                bwd_proxy_of_fwd_freevars = bwd_freevars[bwd_out_proxy]
            else:
                bwd_proxy_of_fwd_freevars = bwd_out_proxy

        # Remove bwd output for non-Tensor args.
        output_proxy = bwd_proxy_of_fwd_freevars
        if isinstance(output_proxy, (tuple, list)):
            new_bwd_graph_outputs = ()
            for x, mask in zip(output_proxy, args_tensor_mask):
                if mask:
                    new_bwd_graph_outputs = new_bwd_graph_outputs + (x,)
                else:
                    assert x is None, f"Grad of non-Tensor arg {x} is not None."
        else:
            new_bwd_graph_outputs = output_proxy

        # Update the bwd graph output.
        new_bwd_graph_outputs = pytree.tree_map(
            lambda x: None if x is None else x.node, new_bwd_graph_outputs
        )
        bwd_graph.output(new_bwd_graph_outputs)
        bwd_graph.lint()

        # Store bwd_body
        bwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        bwd_name = tx.output.install_subgraph(
            "bwd_body",
            torch.fx.GraphModule(bwd_nn_modules.nn_modules, bwd_graph),
        )

        bwd_node = make_attr(tx, bwd_name)

        tx.output.side_effects = prev_side_effects

        p_args = (
            fwd_node,
            bwd_node,
            *([arg.as_proxy() for arg in filtered_args] + list(fwd_freevars.keys())),
        )
        kwargs = {
            "args_tensor_mask": args_tensor_mask,
            "non_differentiable_idx": non_differentiable_idx,
        }

        # Store the invocation as a call
        from torch._functorch.autograd_function import autograd_function_apply

        # We use speculate_subgraph to get the fwd graph, but it's alway under no grad mode like what eager mode does.
        # The fwd outputs (tensor's example_value) need to be inferred from fake tensor prop to get the correct attributes
        # (e.g, tensor.requires_grad), which would be used by downstream Dynamo tracing.
        # Since there can be other ops like Triton kernels, which depends on python dispatcher, we have to enable it.
        with enable_python_dispatcher():
            with tx.output.fake_mode:
                fake_args = (
                    tx.output.nn_modules[fwd_node.node.name],
                    tx.output.nn_modules[bwd_node.node.name],
                    *(
                        [
                            _get_fake_value(arg)
                            for arg in filtered_args + list(fwd_freevars.keys())
                        ]
                    ),
                )
                example_value = autograd_function_apply(*fake_args, **kwargs)

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                autograd_function_apply,
                args=p_args,
                kwargs=kwargs,
            ),
            example_value=example_value,
        )


def _get_fake_value(x):
    if isinstance(x, variables.VariableTracker):
        return x.as_proxy().node.meta["example_value"]
    elif isinstance(x, torch.fx.Proxy):
        return x.node.meta["example_value"]
    else:
        return x


def maybe_positional_arg_names(func):
    result = []
    if not hasattr(func, "get_function"):
        return None
    try:
        fn = func.get_function()
    except (Unsupported, NotImplementedError):
        return None
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if (
            param.kind is inspect.Parameter.POSITIONAL_ONLY
            or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            if name == "self":
                # FX graphs can't have a placeholder named self
                result.append("self_")
            else:
                result.append(name)
    return result


class BaseHOPVariable(WrapHigherOrderVariable):
    supports_input_mutation = False
    supports_aliasing = False

    def python_type(self):
        return type(self.value)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            treespec,
            body_gmod,
            body_name,
        ) = self.create_wrapped_node(
            tx, args[0], args[1:], {}, self.value._name, subgraph_name="subgraph"
        )
        assert len(p_kwargs) == 0

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )
        p_kwargs = {key: value.as_proxy() for key, value in kwargs.items()}
        return _call_function_and_unflatten_output(
            tx, self.value, p_args, p_kwargs, flat_example_value, treespec
        )


class InvokeSubgraphHigherOrderVariable(WrapHigherOrderVariable):
    supports_input_mutation = False
    supports_aliasing = False

    def install_subgraph_in_output_graph(
        self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name
    ):
        # Check if the subgraph from speculate_subgraph (body_gmod) and the fake
        # inputs have already been seen before. If yes, the subgraph is already
        # installed in the output graph and we can just access the subgraph
        # using the saved attr name.

        if not isinstance(fn_vt, (UnspecializedNNModuleVariable, UserFunctionVariable)):
            unimplemented_v2(
                gb_type=f"Encountered non user function variable during invoke_subgraph HOP tracing : {fn_vt}",
                context="",
                explanation="invoke_subgraph does not support non user function variable",
                hints=graph_break_hints.SUPPORTABLE,
            )

        invoke_subgraph_cache = (
            tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
                torch._higher_order_ops.invoke_subgraph
            )
        )

        if isinstance(fn_vt, UserFunctionVariable):
            fn_id = id(fn_vt.get_function())
        else:
            assert isinstance(fn_vt, UnspecializedNNModuleVariable)
            fn_id = id(fn_vt.value.forward.__func__)
        previously_installed_submodules = []
        if invoke_subgraph_cache:
            previously_installed_submodules = (
                invoke_subgraph_cache.get_dynamo_installed_submodules(fn_id)
            )
            current_mod = body_gmod
            # NB - reverse is more likely to cause a hit sooner because first
            # graph can have requires_grad=False for a few inputs
            for submodule_name in reversed(previously_installed_submodules):
                assert submodule_name in tx.output.nn_modules
                previous_mod = tx.output.nn_modules[submodule_name]
                if are_same_graph_modules(previous_mod, current_mod, tx.fake_mode):
                    return submodule_name

        body_name = super().install_subgraph_in_output_graph(
            tx, fn_vt, fn_args_vt, kwargs, body_gmod, "subgraph"
        )
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_dynamo_installed_submodule(fn_id, body_name)

        return body_name

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This flattens the kwargs into lifted args
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            treespec,
            body_gmod,
            body_name,
        ) = self.create_wrapped_node(tx, args[0], args[1:], kwargs, "invoke_subgraph")

        if len(p_kwargs) > 0:
            unimplemented("kwargs should have been flattened into lifted args")

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        p_args = (
            p_args[0],
            body_name,
            *p_args[1:],
        )
        return _call_function_and_unflatten_output(
            tx,
            torch._higher_order_ops.invoke_subgraph,
            tuple(p_args),
            p_kwargs,
            flat_example_value,
            treespec,
        )
