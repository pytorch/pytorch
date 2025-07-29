# mypy: allow-untyped-defs
import contextlib
import functools
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, overload, TypeVar, Union

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dispatch.python import suspend_functionalization
from torch._guards import detect_fake_mode
from torch._higher_order_ops.schema import HopSchema
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    make_fx,
)
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.multiprocessing.reductions import StorageWeakRef


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def autograd_not_implemented_inner(
    operator: OperatorBase, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The Operator to call with the *args and **kwargs with
        op_name: The name of the Operator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the Operator
        kwargs: The keyword arguments to the Operator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the Operator
    """
    with torch._C._AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands = pytree.arg_tree_leaves(*args)
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )
            else:
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")
        return result


def autograd_not_implemented(op: OperatorBase, deferred_error: bool) -> Callable:
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


def _maybe_compile_and_run_fn(fn, *args):
    if not torch.compiler.is_dynamo_compiling():
        from torch._dynamo.backends.debugging import (
            make_eager_backend_with_torch_function_mode,
        )

        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"
                return torch.compile(fn, backend=backend, fullgraph=True)(*args)
    else:
        return fn(*args)


def reenter_make_fx(fn):
    from torch._guards import detect_fake_mode
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER
    from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

    @functools.wraps(fn)
    def wrapped(*args):
        assert _CURRENT_MAKE_FX_TRACER is not None, (
            "Cannot reenter make_fx when we're not under a make_fx tracing session"
        )
        gm = _CURRENT_MAKE_FX_TRACER.trace_subgraph(
            _maybe_run_with_interpreter(fn), *args
        )
        if (fake_mode := detect_fake_mode()) and fake_mode.shape_env is not None:
            insert_deferred_runtime_asserts(gm, fake_mode.shape_env, "reenter_make_fx")
            gm.recompile()
        return gm

    return wrapped


def _maybe_reenter_make_fx(fn):
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    if _CURRENT_MAKE_FX_TRACER is not None:
        return reenter_make_fx(fn)
    else:

        def _maybe_make_fx_with_fake_mode(fn):
            @functools.wraps(fn)
            def wrapped(*args):
                from torch._guards import detect_fake_mode

                fake_mode = detect_fake_mode(args)
                if fake_mode is None:
                    # we creaeta a fake_mode here to make sure we could
                    # trace the graph with data-dependent calls e.g. .item()
                    return make_fx(fn, tracing_mode="fake")(*args)
                # Tracing with real if all inputs have been fakfied
                return make_fx(fn)(*args)

            return wrapped

        return _maybe_make_fx_with_fake_mode(fn)


def check_meta_consistency(
    lhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
    rhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    def diff_meta_pairs(
        lhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
        rhs_list: list[Union[torch.Tensor, torch.SymInt, int]],
    ) -> list[str]:
        def diff_meta(
            lhs: Union[torch.Tensor, torch.SymInt, int],
            rhs: Union[torch.Tensor, torch.SymInt, int],
        ) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                return ", ".join(
                    diff_tensor_meta(
                        _extract_tensor_metadata(
                            lhs, include_contiguity=include_contiguity
                        ),
                        _extract_tensor_metadata(
                            rhs, include_contiguity=include_contiguity
                        ),
                        check_grad=False,
                    )
                )
            else:

                def _both_int_types(lhs, rhs):
                    return isinstance(lhs, (int, torch.SymInt)) and isinstance(
                        rhs, (int, torch.SymInt)
                    )

                def _both_tensor(lhs, rhs):
                    return isinstance(lhs, torch.Tensor) and isinstance(
                        rhs, torch.Tensor
                    )

                if not _both_int_types(lhs, rhs) and not _both_tensor(lhs, rhs):
                    return f"type: {lhs} vs {rhs}"

            return ""

        # Manually check the device of lhs and rhs as this field is currently not part of TensorMetadata
        def diff_device(
            lhs: Union[torch.Tensor, torch.SymInt, int],
            rhs: Union[torch.Tensor, torch.SymInt, int],
        ) -> str:
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                if (
                    rhs.device.type == lhs.device.type
                    and rhs.device.index == lhs.device.index
                ):
                    return ""
                else:
                    return "device"
            return ""

        if len(lhs_list) != len(rhs_list):
            raise torch._dynamo.exc.UncapturedHigherOrderOpError(
                f"Expected {lhs_name} and {rhs_name} to have same number of outputs but got lhs:{lhs_list} and rhs:{rhs_list}"
            )
        all_diffs = []
        for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
            if diff := diff_meta(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
            if diff := diff_device(lhs, rhs):
                all_diffs.append(
                    f"pair[{i}] differ in {diff}, where lhs is {lhs} and rhs is {rhs}"
                )
        return all_diffs

    if all_diffs := diff_meta_pairs(lhs_list, rhs_list):
        diff_str = "\n".join(all_diffs)
        raise torch._dynamo.exc.UncapturedHigherOrderOpError(
            f"Expected {lhs_name} and {rhs_name} to have same metadata but found:\n{diff_str}"
        )


@contextmanager
def _set_compilation_env():
    _old_is_tracing = torch.fx._symbolic_trace._is_fx_tracing_flag
    _old_allow_empty_graphs = torch._dynamo.config.allow_empty_graphs
    _old_capture_scalar_outputs = torch._dynamo.config.capture_scalar_outputs
    # The issue is tracked in https://github.com/pytorch/pytorch/issues/144360: when dynamo finds
    # the top-level frame produces no graph, the default behavior is to fallback to eager.
    # Then when it encounters an inner function, it will try to trace that function again, which is unnecessary.
    # For while_loop, during inspecting the inner call, we trace into the python dispathcer
    # logic, which is not tracable as of today. So the proper fix can be either 1. allow dispatch
    # logic to be dynamo tracable or 2. fixing https://github.com/pytorch/pytorch/issues/144360.
    # but it exposes some bugs in existing tests so we have to have a temporary flag to control
    # the behavior, which allows dynamo to store an empty graph for a frame without falling back to eager
    try:
        # We need to turn off the is_fx_tracing_flag. Remove this flag check from dyanmo
        # once we are confident fx tracing works with dynamo.
        torch.fx._symbolic_trace._is_fx_tracing_flag = False
        torch._dynamo.config.allow_empty_graphs = True
        torch._dynamo.config.capture_scalar_outputs = True
        yield
    finally:
        torch.fx._symbolic_trace._is_fx_tracing_flag = _old_is_tracing
        torch._dynamo.config.allow_empty_graphs = _old_allow_empty_graphs
        torch._dynamo.config.capture_scalar_outputs = _old_capture_scalar_outputs


# The invariant here is that we always trace the branch with fake tensor
def _maybe_fake_tracing(fn, inputs: list[Any], pre_dispatch):
    fake_mode = detect_fake_mode(inputs)
    tracing_mode = "real"
    if fake_mode is None:
        fake_mode = nullcontext()
        tracing_mode = "fake"

    # Note: we need to turn off proxy tensor mode to avoid tracing infra
    # code that happens in make_fx e.g. we now call as_strided when wrapping tensor
    # as fake tensor.
    with fake_mode, disable_proxy_modes_tracing():
        gm = make_fx(
            fn,
            tracing_mode=tracing_mode,
            pre_dispatch=pre_dispatch,
            _error_on_data_dependent_ops=False,
        )(*inputs)
        if not isinstance(fake_mode, nullcontext) and fake_mode.shape_env is not None:
            insert_deferred_runtime_asserts(
                gm, fake_mode.shape_env, "hoo_maybe_fake_tracing", export=True
            )
        return gm


def potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    try:
        gm = _maybe_fake_tracing(gm, inputs, pre_dispatch)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond_op is
        # functionalized
        return True
    except Exception as e:
        raise e

    example_inputs = [
        ph.meta.get("val", None) for ph in gm.graph.find_nodes(op="placeholder")
    ]
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        inp_mutation,
    ) = check_input_alias_and_mutation(gm, example_inputs)
    return (inp_inp_alias_map, inp_out_alias_map, out_out_alias_map), inp_mutation


def analyze_potential_input_alias_or_mutation(name, aliases, input_mutations):
    if any(len(a) > 0 for a in aliases):
        # TODO: Investigate here further which node is exactly aliasing
        raise RuntimeError(
            f"{name} where aliases appear. "
            + f"In particular, these inputs \
            {set(el for el_map in aliases if len(el_map.keys()) > 0 for el in el_map.keys())} "  # noqa: C401
            + "get aliased. Please ensure that this doesn't happen."
        )
    if len(input_mutations):
        # TODO: Investigate here further which node is exactly mutating the inputs
        raise RuntimeError(
            f"{name} where the inputs are mutated. "
            + f"In particular, these nodes are mutating the inputs \
            {set(el for el in input_mutations)}."  # noqa: C401
            + "Please ensure that this doesn't happen."
        )


def _has_potential_branch_input_mutation(gm, inputs, pre_dispatch=False):
    (
        (_, _, _),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)

    return len(inp_mutation) > 0


def has_potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    (
        (
            inp_inp_alias_map,
            inp_out_alias_map,
            out_out_alias_map,
        ),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)
    return (
        any(
            (
                len(inp_inp_alias_map) > 0,
                len(inp_out_alias_map) > 0,
                len(out_out_alias_map) > 0,
            )
        ),
        len(inp_mutation) > 0,
    )


def _collect_fake_inputs(inputs):
    from torch._subclasses.fake_tensor import FakeTensor

    # Get the example values of the inputs.
    inputs_fake: list[Union[FakeTensor, torch.Tensor, int]] = []
    for inp in inputs:
        if isinstance(inp, (torch.fx.proxy.Proxy, torch.fx.node.Node)):
            inp = inp.node if isinstance(inp, torch.fx.proxy.Proxy) else inp
            if hasattr(inp, "meta"):
                val = inp.meta["example_value"]
                if isinstance(val, torch.Tensor):
                    if torch._C._functorch.is_batchedtensor(
                        val
                    ) or torch._C._functorch.is_functionaltensor(val):
                        # This case is for batched or functional tensors
                        # Unwrap the tensors
                        while torch._C._functorch.is_batchedtensor(
                            val
                        ) or torch._C._functorch.is_functionaltensor(val):
                            val = torch._C._functorch.get_unwrapped(val)
                        assert isinstance(val, FakeTensor)
                        inputs_fake.append(val)
                    else:
                        # This is the standard case of a TensorVariable
                        assert isinstance(val, FakeTensor)
                        inputs_fake.append(val)
                else:
                    # This case is for SymInts and other non-Tensor elements
                    assert not isinstance(val, torch.Tensor)
                    inputs_fake.append(val)
        else:
            # This case is for ints
            assert isinstance(inp, int)
            inputs_fake.append(inp)

    return inputs_fake


def _check_alias_and_mutation(graph_module, inputs_fake, name, pre_dispatch):
    aliases, inp_mutation = has_potential_input_alias_or_mutation(
        graph_module, inputs_fake, pre_dispatch=pre_dispatch
    )
    if aliases:
        raise RuntimeError(f"{name} might be aliasing the input or the output!")  # noqa: F541
    if inp_mutation:
        raise RuntimeError(f"{name} might be modifying the input!")  # noqa: F541


def unique_graph_id(proxy_mode, prefix):
    """Returns a unique name and id for a graph to be added to a proxy_mode tracer"""
    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    return unique_graph_name_with_root(proxy_mode.tracer.root, prefix)


def unique_graph_name_with_root(
    root: torch.fx.GraphModule, prefix: str
) -> tuple[int, str]:
    next_name = None
    i = 0
    while not next_name:
        candidate = f"{prefix}_{i}"
        if hasattr(root, candidate):
            i += 1
        else:
            next_name = candidate
    return i, next_name


def _from_fun(t):
    from torch._functorch.aot_autograd import from_fun

    if isinstance(t, torch.Tensor):
        if t.dtype != torch.bool:
            return torch.empty_strided(
                t.size(),
                t.stride(),
                dtype=t.dtype,
                requires_grad=t.requires_grad,
                device=t.device,
            )
        else:
            # clone of a functional tensor produces a functional tensor
            # but we want to avoid it so we clone a non-functional version
            maybe_unfunc_t = t
            if isinstance(t, FunctionalTensor):
                torch._sync(t)
                maybe_unfunc_t = from_fun(t)
            elif torch._is_functional_tensor(t):
                # need to handle both types of functionalization here:
                # these are the tensors that came from the user,
                # which could be either FunctionalTensorWrapper or FunctionalTensor
                torch._sync(t)
                maybe_unfunc_t = torch._from_functional_tensor(t)
            return maybe_unfunc_t.clone()
    return t


def clone_outputs_aliasing_inputs(args):
    input_storage = {
        StorageWeakRef(arg._typed_storage())
        for arg in args
        if isinstance(arg, torch.Tensor)
    }

    def maybe_clone(t):
        if (
            isinstance(t, torch.Tensor)
            and StorageWeakRef(t._typed_storage()) in input_storage
        ):
            return t.clone()
        return t

    return maybe_clone


def prepare_fw_with_masks(fn):
    def fw_with_masks(*args):
        fw_out = fn(*args)
        return fw_out, [
            True if isinstance(ret, torch.Tensor) and ret.requires_grad else False
            for ret in fw_out
        ]

    return fw_with_masks


def prepare_fw_with_masks_all_requires_grad(fn):
    def fw_with_masks(*args):
        fw_out = fn(*args)
        # Note [force all outputs to be require grad]
        # Instead of using the original fn, we set the output of original
        # fn to all require grad. This is consistent with the behavior
        # of autograd.Function, where if any one of the inputs requires grad
        # all output will be require grad. This also makes the downstream
        # require_gradness reasoning much easier.
        if pytree.tree_any_only(torch.Tensor, lambda t: t.requires_grad, args):
            fw_out = pytree.tree_map_only(
                torch.Tensor, lambda x: x.requires_grad_(True), fw_out
            )
        return fw_out, pytree.tree_map_only(
            torch.Tensor, lambda x: x.requires_grad, fw_out
        )

    return fw_with_masks


# This function replaces None gradients with all-zero gradients.
# `None` gradients are problematic for CUDA graphs. Those gradients are
# replaced with an all-zero tensor for better optimization
def unmask_none_gradients(grads, operands):
    allowed_types = (torch.Tensor, int, torch.SymInt)
    assert all(isinstance(o, allowed_types) for o in operands), (
        f"operands can only be of {allowed_types} but got {[type(o) for o in operands]}"
    )

    unmasked_grads = []
    for g, o in zip(grads, operands):
        if g is not None:
            unmasked_grads.append(g)
        else:
            # In case the operand is an int or a torch.SymInt, return None
            # This can happen for lifted_arguments. E.g., the shapes of a dynamic tensor are lifted and passed
            # as additional arguments
            unmasked_grads.append(
                torch.zeros_like(o) if isinstance(o, torch.Tensor) else None
            )

    return unmasked_grads


def _maybe_fake_prop_ignore_unbacked(fn, args):
    with ExitStack() as ctx_stack:
        if (fake_mode := detect_fake_mode(args)) is not None:
            ctx_stack.enter_context(fake_mode)
            if fake_mode.shape_env is not None:
                ctx_stack.enter_context(
                    fake_mode.shape_env.ignore_fresh_unbacked_symbols()
                )
        return fn(*args)


def redirect_to_mode(hop: OperatorBase, mode):
    """Utility for redispatching HOP to underlying mode

    Args:
        hop: The HOP to redispatch
        mode: The mode to redispatch to

    Returns:
        A decorated function that implements the HOP for the given mode
    """

    @hop.py_impl(mode)
    def impl(mode, *args, **kwargs):
        return mode.__torch_dispatch__(hop, [], args, kwargs)

    return impl


# TODO: The parameter use_output_and_grad_bw is required because some operations
# that utilize this function, such as the while_loop, may require (grad, fwd_outputs)
def create_fw_bw_graph(fn, use_output_and_grad_bw, fw_inputs, fw_outputs):
    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # Note:[HOP create fw_bw graph] We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    example_grad = [_from_fun(out) for out in fw_outputs]
    num_grads = len(example_grad)
    fw_graph = _maybe_reenter_make_fx(fn)(*fw_inputs)

    def joint_fn(*joint_operands_grads):
        if use_output_and_grad_bw:
            grads = joint_operands_grads[0]
            inputs = joint_operands_grads[1][-1:]
        else:
            grads = joint_operands_grads[:num_grads]
            inputs = joint_operands_grads[num_grads:]

        joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
        _, grads = joint(
            list(inputs),
            [grad for grad in grads if grad is not None and grad.requires_grad],
        )

        # Unmask None gradients to all-zero gradients
        unmasked_grads = unmask_none_gradients(grads, inputs)

        # In order to keep map functional for backward graph,
        # we clone outputs that are aliasing inputs
        maybe_clone = clone_outputs_aliasing_inputs(joint_operands_grads)

        return pytree.tree_map(maybe_clone, unmasked_grads)

    if use_output_and_grad_bw:
        example_xs_out = list(fw_inputs) + list(fw_outputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            (list(example_grad), list(example_xs_out))
        )
    else:
        example_xs_out = list(fw_inputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            *(list(example_grad) + list(example_xs_out))
        )

    return fw_graph, joint_graph


def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(
            f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
        )

    a = zip(*flat_xs)

    pytrees = [pytree.tree_unflatten(tuple, inspec) for tuple in a]
    return pytrees


def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    assert out_spec is not None
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


# We cannot call save_for_backward for symints. This helper function
# can be used to save symints as direct attributes of ctx in autograd.Function.
#
# For example, if args = (x, y, s0, z, s1),
# save_tensors_and_symints_for_backward will partition the args into two lists, and a bookkeeping list pos:
#   partitioned_args[0] = (x, y, z)
#   partitioned_args[1] = (s0, s1)
#   pos = (0, 0, 1, 0, 1)
# pos list keeps track of which partition the args
# is partitioned into in order to recover it in saved_tensors_and_symints.
#
# In saved_tensors_and_symints, we can recover the original args by:
# iterating over the pos list and pop one item from the front of paritioned_args[pos[i]].
# We use t_idx and s_idx to keep track of the next index of the item we are going to pop for the two lists.
def save_tensors_and_symints_for_backward(ctx, args):
    assert all(
        isinstance(arg, (torch.Tensor, torch.SymInt, int, type(None))) for arg in args
    ), args
    partitioned_args: list[Any] = [[], []]
    pos = []
    for arg in args:
        idx = 0 if isinstance(arg, torch.Tensor) else 1
        partitioned_args[idx].append(arg)
        pos.append(idx)

    assert not hasattr(ctx, "sym_int_args"), "ctx already has sym_int_args attribute."
    assert not hasattr(ctx, "pos"), "ctx already has pos attribute."
    ctx.save_for_backward(*partitioned_args[0])
    ctx.sym_int_args = partitioned_args[1]
    ctx.pos = pos


def saved_tensors_and_symints(ctx):
    args = []
    t_idx = 0
    s_idx = 0
    saved_tensors = ctx.saved_tensors
    for p in ctx.pos:
        if p == 0:
            args.append(saved_tensors[t_idx])
            t_idx += 1
        else:
            args.append(ctx.sym_int_args[s_idx])
            s_idx += 1
    assert t_idx + s_idx == len(ctx.pos)
    return tuple(args)


def get_dummy_aot_autograd_config():
    from torch._functorch.aot_autograd import AOTConfig

    return AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )


# Slices off the first element of a given dimension
def first_slice_copy(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)


# Reports the difference between meta of two tensors in a string
def diff_tensor_meta(
    meta1: TensorMetadata, meta2: TensorMetadata, check_grad=True
) -> list[str]:
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    pair_diffs = []
    for meta_name in TensorMetadata._fields:
        if not check_grad and meta_name == "requires_grad":
            continue
        val1 = getattr(meta1, meta_name)
        val2 = getattr(meta2, meta_name)
        try:
            if val1 != val2:
                pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
        except GuardOnDataDependentSymNode as _:
            pair_diffs.append(f"'{meta_name}: {val1} vs {val2}'")
            continue
    return pair_diffs


# Note [lifted arg types in hop]
# For dynamoed hops, we automatically lift the free symbols in tensors as arguments.
# This has implications for the types of lifted args for different dispatch keys:
#   1. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd need to support torch.Symint
#      lifted args because it's on the path of torch.compile(dynamic=True).
#   2. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd, CompositeExplicitAutograd need
#      to support int arguments. In the eager run case, we re-trace the subgraph in AutogradKey, so inner
#      hops may receive int inputs from the shape of outer tensor inputs.
#      However, CompositeExplicitAutograd won't receive SymInt inputs because it only accepts real tensor inputs.
def validate_subgraph_args_types(lifted_args: Union[tuple[Any, ...], list[Any]]):
    allowed_types = (torch.Tensor, int, torch.SymInt)
    assert all(
        isinstance(arg, (torch.Tensor, int, torch.SymInt)) for arg in lifted_args
    ), (
        f"{lifted_args} can only be of {allowed_types} but got {tuple(type(arg) for arg in lifted_args)}"
    )


# TODO: Return a more detailed information as to which node
# causes a mutation or an alias. This may requires a per operator tensor version checking
def check_input_alias_and_mutation(
    gm: torch.fx.GraphModule,
    fake_args: list[FakeTensor],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
    (
        inp_inp_alias_map,
        inp_out_alias_map,
        out_out_alias_map,
        mutated_inputs,
    ) = check_input_alias_and_mutation_return_outputs(gm, fake_args)[:-1]
    return inp_inp_alias_map, inp_out_alias_map, out_out_alias_map, mutated_inputs


def _tensor_storage(t) -> StorageWeakRef:
    return StorageWeakRef(t._typed_storage())


def check_input_alias_and_mutation_return_outputs(
    gm: torch.fx.GraphModule,
    fake_args: Union[list[FakeTensor], tuple[FakeTensor, ...]],
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    list[int],
    Union[tuple[Any, ...], list[Any]],
]:
    # This function can be called under autograd, functional, proxy and fake tensor mode.
    # We need to return either a fake tensor or a real tensor depending on the mode.
    # to detect the input mutation/aliasing.
    with (
        disable_proxy_modes_tracing(),
        disable_functional_mode(),
        suspend_functionalization(),
    ):

        def _from_functional_tensor(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, FunctionalTensor) or torch._is_functional_tensor(t):
                return torch.empty_strided(
                    t.size(),
                    t.stride(),
                    dtype=t.dtype,
                    requires_grad=t.requires_grad,
                    device=t.device,
                )
            return t

        fake_args = pytree.tree_map_only(
            torch.Tensor, _from_functional_tensor, fake_args
        )
    # We want to disable active functional, proxy and fake modes if any.
    # to create a encapsulated environment for fake tensor prop
    with torch.utils._python_dispatch._disable_current_modes():
        """This function returns mutated inputs, inp-inp alias, inp-out alias, out-out alias
        in the graph module gm. It checks whether input tensor versions have
        changed after run gm once to detect mutation and checks tensor storage
        to detect alias.
        """

        def _tensor_version(t) -> Optional[int]:
            if isinstance(t, torch.Tensor):
                if not isinstance(t, FakeTensor):
                    raise RuntimeError("Only fake tensor is allowed")
                return t._version
            return None

        def _get_shape_env(
            fake_args,
        ) -> Optional[torch.fx.experimental.symbolic_shapes.ShapeEnv]:
            # detect_fake_mode requires there could be only one active fake mode. This
            # restricts the usage of this function because the global TracingContext
            # has a persistent fake mode but fake tensors can be created
            # outside of the tracing context (e.g. in testing).
            # Instead, we just look at fake_args fake tensor mode
            if len(fake_args) == 0:
                return torch.fx.experimental.symbolic_shapes.ShapeEnv()

            for arg in fake_args:
                if isinstance(arg, FakeTensor):
                    return arg.fake_mode.shape_env
            return None

        # Clone the fake args to avoid mutating the original fake args
        with ExitStack() as ctx_stack:
            # We need to reuse prev_fake_mode's shape env to resolve
            # the runtime assertions for unbacked symbols.
            new_fake_mode = torch._subclasses.FakeTensorMode(
                shape_env=_get_shape_env(fake_args),
                # In executorch, there's an scalar_to_tensor pass that turns scalar inputs into a tensor constant
                # e.g. add(a, 1) 1 is turned into a tensor, which becomes a constant tensor attribute in the graph.
                # We allow non fake inputs for this purpose. This is fine for mutation detection purpose:
                # inputs are all fake and all mutations/aliasing are still detected.
                allow_non_fake_inputs=True,
            )
            # We need to temporarily turn inference_mode off because
            # under inference mode, tensor version counter is not tracked.
            no_inference_mode_ctx = torch.inference_mode(False)
            ctx_stack.enter_context(new_fake_mode)
            ctx_stack.enter_context(no_inference_mode_ctx)
            if new_fake_mode.shape_env is not None:
                ctx_stack.enter_context(
                    new_fake_mode.shape_env.ignore_fresh_unbacked_symbols()
                )

            # create new fake tensors in new fake mode to avoid mutating original tensors
            cloned = [
                torch.empty_strided(
                    arg.size(),
                    arg.stride(),
                    dtype=arg.dtype,
                    device=arg.device,
                    requires_grad=arg.requires_grad,
                    layout=arg.layout,
                )
                if isinstance(arg, torch.Tensor)
                else arg
                for arg in fake_args
            ]
            before = [_tensor_version(arg) for arg in cloned]
            outputs = gm(*cloned)
            outputs = [outputs] if not isinstance(outputs, (list, tuple)) else outputs
            after = [_tensor_version(arg) for arg in cloned]
            mutated_inputs = [
                i for i, (v1, v2) in enumerate(zip(before, after)) if v1 != v2
            ]
        # We need to analyze the original fake_args to detect
        # inp-inp alias.
        inp_storage_map = {
            _tensor_storage(inp): i
            for i, inp in enumerate(fake_args)
            if isinstance(inp, torch.Tensor)
        }
        inp_inp_alias_map = {
            i: inp_storage_map[_tensor_storage(inp)]
            for i, inp in enumerate(fake_args)
            if isinstance(inp, torch.Tensor)
            and inp_storage_map[_tensor_storage(inp)] != i
        }
        out_storage_map = {
            _tensor_storage(out): i
            for i, out in enumerate(outputs)
            if isinstance(out, torch.Tensor)
        }
        out_out_alias_map = {
            i: out_storage_map[_tensor_storage(out)]
            for i, out in enumerate(outputs)
            if isinstance(out, torch.Tensor)
            and out_storage_map[_tensor_storage(out)] != i
        }
        inp_out_alias_map = {
            i: out_storage_map[_tensor_storage(inp)]
            for i, inp in enumerate(cloned)
            if isinstance(inp, torch.Tensor) and _tensor_storage(inp) in out_storage_map
        }
        return (
            inp_inp_alias_map,
            inp_out_alias_map,
            out_out_alias_map,
            mutated_inputs,
            outputs,
        )


registered_hop_fake_fns: dict[torch._ops.OpOverload, Callable] = {}


F = TypeVar("F", bound=Callable)


@overload
def register_fake(hop, fn: None = None) -> Callable[[F], F]: ...


@overload
def register_fake(hop, fn: F) -> F: ...


def register_fake(hop, fn=None):
    """
    Register a fake function for a HOP. This is conceptually equivalent of the
    register_fake utility for the custom ops. The registered function is called
    inside the fake_tensor _dispatch_impl.
    """
    assert hop not in registered_hop_fake_fns

    def register(func: F) -> F:
        from torch._subclasses.fake_tensor import FakeTensorMode

        redirect_to_mode(hop, FakeTensorMode)

        registered_hop_fake_fns[hop] = func
        return func

    if fn is None:
        return register
    return register(fn)


class FunctionalizeCtxWrapper:
    """
    This is a dummy wrapper to facilitate fake tensor caching.

    For AOT Dispatcher metadata collection pass, HOPs go from functionalization
    key to fake tensor key. The functionalization key wraps the subgraphs in a
    function, which changes from call to call even though the subgraph might
    still be same.

    To enable fake tensor caching, we just wrap the ctx and subgraph in this
    class and then use the subgraph as the hash.
    """

    # Prevents PYTORCH_TEST_WITH_DYNAMO=1 test failures
    @torch._disable_dynamo
    def __init__(self, ctx, subgraph):
        self.ctx = ctx
        self.subgraph = subgraph

    def __hash__(self):
        return id(self.subgraph)

    def __repr__(self):
        return f"FunctionalizeCtxWrapper on subgraph {self.subgraph})"

    def __call__(self, *args, **kwargs):
        if isinstance(self.subgraph, torch.fx.GraphModule):
            # Running graph with interpreter is needed for propagating the stack_trace
            with fx_traceback.preserve_node_meta():
                return self.ctx.functionalize(torch.fx.Interpreter(self.subgraph).run)(
                    *args, **kwargs
                )
        return self.ctx.functionalize(self.subgraph)(*args, **kwargs)


# A wrapper over HigherOrderOperator that also carries its schema
class HopInstance:
    def __init__(self, op: HigherOrderOperator, schema: HopSchema):
        assert isinstance(op, HigherOrderOperator), op
        self._op = op
        # Using "_" to be consistent with how we access _schema of OpOverload
        self._schema = schema

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    @staticmethod
    def create(hop: HigherOrderOperator, *args, **kwargs):
        return HopInstance(hop, hop.gen_schema(*args, **kwargs))


# This call_op can be used to call a HopInstance with
# flat args and kwargs. We need to make use of the hop's schema's tree_spec
# to unflatten the args and kwargs before calling the hop.
def call_op(op: Union[OpOverload, HopInstance], args, kwargs):
    if isinstance(op, OpOverload):
        return op(*args, **kwargs)

    assert isinstance(op, HopInstance), op
    schema = op._schema
    bound_args = list(args)
    bound_kwargs = {}
    for arg in schema.arguments[len(bound_args) :]:
        assert arg.name in kwargs, (arg.name, kwargs)
        val = kwargs[arg.name]
        if not arg.kwarg_only:
            bound_args.append(val)
        else:
            bound_kwargs[arg.name] = val

    if schema.tree_spec is not None:
        assert len(bound_args) == len(schema.arguments) and len(bound_kwargs) == 0
        args, kwargs = pytree.tree_unflatten(bound_args, schema.tree_spec)
        return op(*args, **kwargs)
    else:
        assert len(bound_args) + len(bound_kwargs) == len(schema.arguments)
        return op(*bound_args, **bound_kwargs)


def materialize_as_graph(
    fn: Callable,
    args: tuple[Any],
    include_key_set: Optional[torch._C.DispatchKeySet] = None,
    exclude_key_set: Optional[torch._C.DispatchKeySet] = None,
    force_enable_grad=False,
) -> torch.fx.GraphModule:
    if include_key_set is None:
        include_key_set = torch._C._dispatch_tls_local_include_set()
    if exclude_key_set is None:
        exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

    @torch._dynamo.disable(recursive=True, reason=None)
    def _materialize_as_graph_inner():
        with suspend_functionalization(), disable_functional_mode():
            with disable_proxy_modes_tracing():
                unfunc_t = [_from_fun(arg) for arg in args]
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
                )
                if force_enable_grad:
                    stack.enter_context(torch.enable_grad())
                return _maybe_reenter_make_fx(fn)(*unfunc_t)

    gm = _materialize_as_graph_inner()
    assert gm is not None
    return gm


def materialize_callable_in_args(op: HopInstance, args, kwargs):
    schema = op._schema
    hop = op._op
    flat_args, flat_spec = pytree.tree_flatten((args, kwargs))

    def wrapped_fn(*flat_args):
        return call_op(op, args, kwargs)

    # We need to trace the higher order op in order to materilaize the callable inputs that
    # are a callable (e.g. after functionalization key)
    gm = reenter_make_fx(wrapped_fn)(*flat_args)
    hop_node = gm.graph.find_nodes(op="call_function", target=hop)[0]
    arg_proxies = pytree.tree_leaves((hop_node.args, hop_node.kwargs))
    assert isinstance(schema, torch._C.FunctionSchema) and len(arg_proxies) == len(
        schema.arguments
    )

    # call_op preserves ordering of proxies via schema
    materialized_args = []
    for i, (proxy, arg) in enumerate(zip(arg_proxies, schema.arguments)):
        if (
            isinstance(proxy, torch.fx.Node)
            and proxy.op == "get_attr"
            and isinstance(getattr(gm, proxy.target), torch.fx.GraphModule)  # type: ignore[arg-type]
        ):
            assert callable(flat_args[i]), (schema, args, kwargs)
            materialized_args.append(getattr(gm, proxy.target))  # type: ignore[arg-type]
        else:
            materialized_args.append(flat_args[i])

    return pytree.tree_unflatten(materialized_args, flat_spec)


def has_user_subclass(args, allowed_subclasses):
    """Check if any tensor arguments are user subclasses.

    This is used to determine if tensor subclasses should get a chance to run
    their own implementation first before falling back to the default implementation.

    Args:
        args: Arguments to check (will be flattened with pytree)
        allowed_subclasses: Tuple of allowed subclass types

    Returns:
        True if user tensor subclasses are found, False otherwise
    """
    flat_args, _ = pytree.tree_flatten(args)

    val = any(
        isinstance(a, torch.Tensor)
        and type(a) is not torch.Tensor
        and not isinstance(a, allowed_subclasses)
        for a in flat_args
    )
    return val


def _has_gen_schema(op: HigherOrderOperator):
    # There is an InvokeQuant argument we cannot gen_schema.
    if op is torch.ops.higher_order.invoke_quant_packed:
        return False
    method = "gen_schema"
    return hasattr(type(op), method) and getattr(type(op), method) is not getattr(
        HigherOrderOperator, method
    )
