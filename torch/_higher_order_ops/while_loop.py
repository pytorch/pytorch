# mypy: allow-untyped-defs
from typing import Callable, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _maybe_run_with_interpreter,
    _set_compilation_env,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

from .utils import _from_fun, _maybe_reenter_make_fx, create_fw_bw_graph


def create_fw_bw_graph_combinefn(cond_fn, body_fn, carried_inputs, additional_inputs):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are
    # detached in order not to raise problems with the function aliasing outputs

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            num_xs = len(carried_inputs)
            num_additional_inputs = len(additional_inputs)

            fw_xs = [pytree.tree_map(_from_fun, x) for x in carried_inputs]
            fw_xs_list = [
                torch.unsqueeze(pytree.tree_map(_from_fun, x), 0)
                for x in carried_inputs
            ]
            roll_cnt = pytree.tree_map(
                _from_fun,
                torch.zeros(1, dtype=torch.int64, device=carried_inputs[0].device),
            )
            max_cnt = pytree.tree_map(
                _from_fun,
                torch.zeros(1, dtype=torch.int64, device=carried_inputs[0].device),
            )
            fw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]
            bw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]
            outs = body_fn(*fw_xs, *fw_additional_inputs)

            fw_outputs = [pytree.tree_map(_from_fun, o) for o in outs]
            fw_outputs_bwd = [pytree.tree_map(_from_fun, o) for o in outs] + [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]
            if any(not isinstance(out, torch.Tensor) for out in fw_outputs):
                raise RuntimeError(
                    "Expect outputs produced by combine_fn to only contains tensors. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            fw_graph, joint_graph = create_fw_bw_graph(
                body_fn,
                False,
                (*fw_xs, *fw_additional_inputs),
                (*fw_outputs,),
            )

            def wrapper_fwd(*args):
                # The first num_xs elements are the concatenations of all previous outputs
                # The remainder are the inputs for the current iteration and the additional inputs
                outs = args[:num_xs]
                inp = args[num_xs:]

                # Run the forward body
                fw_outs = fw_graph(*inp)

                # Append the outputs to the front of the previous outputs
                # outs = [torch.concat([o, torch.unsqueeze(fwo, 0)], 0) for o, fwo in zip(outs, fw_outs)]
                outs = [
                    torch.concat([torch.unsqueeze(fwo, 0), o], 0)
                    for o, fwo in zip(outs, fw_outs)
                ]
                return *outs, *fw_outs

            fw_graph_new = _maybe_reenter_make_fx(wrapper_fwd)(
                *fw_xs_list, *fw_xs, *fw_additional_inputs
            )

            def wrapper_bwd(*args):
                # The first num_xs of elements represent the current gradients
                # The next two elements are the current loop counter and the maximum loop counter
                # The next num_xs elements is the concatenated list of all outputs of the FWD path, starting with
                # the ouput T-1 as the 0-th element
                # The next last set of elements are the additional arguments
                grad = args[:num_xs]
                grad_additional_inputs = args[num_xs : num_xs + num_additional_inputs]
                rcnt = args[num_xs + num_additional_inputs]
                mcnt = args[num_xs + num_additional_inputs + 1]
                outs = args[
                    num_xs
                    + num_additional_inputs
                    + 2 : 2 * num_xs
                    + num_additional_inputs
                    + 2
                ]
                bw_add_args = args[2 * num_xs + num_additional_inputs + 2 :]

                last_out = [torch.squeeze(o[rcnt], 0) for o in outs]

                new_grad = joint_graph(*grad, *last_out, *bw_add_args)
                grad_additional_inputs = [
                    ng * g for ng, g in zip(new_grad[num_xs:], grad_additional_inputs)
                ]

                new_grad = new_grad[:num_xs] + grad_additional_inputs

                return *new_grad, rcnt + 1, mcnt, *outs

            new_joint_graph = _maybe_reenter_make_fx(wrapper_bwd)(
                *fw_outputs_bwd,
                roll_cnt,
                max_cnt,
                *fw_xs_list,
                *bw_additional_inputs,
            )

            def wrapper_cond_fw(*args):
                # The first num_xs elements are the concatenations of all previous outputs
                # The second num_xs elements are the inputs for the current iteration
                inp = args[num_xs:]

                fw_outs = cond_fn(*inp)

                return fw_outs

            cond_fn_graph_new = _maybe_reenter_make_fx(wrapper_cond_fw)(
                *fw_xs_list, *fw_xs, *fw_additional_inputs
            )

            def bwd_cond_fn(*args):
                # The two elements are the current loop counter and the maximum loop counter
                rcnt = args[num_xs + num_additional_inputs]
                mcnt = args[num_xs + num_additional_inputs + 1]

                # Check whether the maximum of the loop counter is reached
                return (rcnt < mcnt - 1)[0]

            bwd_cond_fn_graph_new = _maybe_reenter_make_fx(bwd_cond_fn)(
                *fw_outputs_bwd,
                roll_cnt,
                max_cnt,
                *fw_xs_list,
                *bw_additional_inputs,
            )

        return cond_fn_graph_new, fw_graph_new, bwd_cond_fn_graph_new, new_joint_graph


class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    def extract_init_xs_additional_inputs(flat_args, num_leaves_init, num_leaves_xs):
        init = flat_args[:num_leaves_init]
        xs = flat_args[num_leaves_init : num_leaves_init + num_leaves_xs]
        additional_inputs = flat_args[num_leaves_init + num_leaves_xs :]
        return init, xs, additional_inputs

    @staticmethod
    def forward(
        ctx,
        cond_fn_graph,
        fw_graph,
        bwd_cond_fn_graph,
        joint_graph,
        num_leaves_xs,
        *flat_args,
    ):
        ctx._joint_graph = joint_graph
        ctx._bwd_cond_fn_graph = bwd_cond_fn_graph
        flat_args_list = list(flat_args)
        xs, additional_inputs = (
            flat_args_list[:num_leaves_xs],
            flat_args_list[num_leaves_xs:],
        )
        ctx._num_leaves_xs = num_leaves_xs
        ctx._num_additional_inputs = len(additional_inputs)

        with torch._C._AutoDispatchBelowAutograd():
            ones = [torch.unsqueeze(x, 0) for x in xs]
            inp = *ones, *xs
            outs = while_loop_op(cond_fn_graph, fw_graph, inp, tuple(additional_inputs))
            out_list, outs = outs[:num_leaves_xs], outs[num_leaves_xs:]
            ctx.save_for_backward(*(list(out_list) + xs + additional_inputs))

            return (*outs,)

    @staticmethod
    def backward(ctx, *flat_grads):
        num_leaves_xs = ctx._num_leaves_xs
        num_additional_inputs = ctx._num_additional_inputs

        joint_graph = ctx._joint_graph
        bwd_cond_fn_graph = ctx._bwd_cond_fn_graph

        # Collect the variables from the FWD path
        fwd_vals = ctx.saved_tensors
        out_list, xs, additional_inputs = (
            fwd_vals[:num_leaves_xs],
            fwd_vals[num_leaves_xs : 2 * num_leaves_xs],
            fwd_vals[2 * num_leaves_xs :],
        )

        with torch._C._AutoDispatchBelowAutograd():
            out_list = [
                torch.concat([o[1:], torch.zeros_like(o[0:1, :])], 0) for o in out_list
            ]
            roll_cnt = torch.zeros(1, dtype=torch.int64, device=xs[0].device)
            max_cnt = (
                torch.ones(1, dtype=torch.int64, device=xs[0].device)
                * out_list[0].shape[0]
            )
            flat_grads_plus_additional_inputs = list(flat_grads) + [
                torch.ones_like(a) for a in additional_inputs
            ]
            inp = *flat_grads_plus_additional_inputs, roll_cnt, max_cnt, *out_list
            grads = while_loop_op(
                bwd_cond_fn_graph, joint_graph, inp, tuple(additional_inputs)
            )[: num_leaves_xs + num_additional_inputs]
            return *[None] * 5, *grads


class WhileLoopOp(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        additional_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        /,
    ):
        if not isinstance(carried_inputs, tuple):
            raise RuntimeError(
                f"carried_inputs must be a tuple, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, tuple):
            raise RuntimeError(
                f"additional_inputs must be a tuple, got {type(additional_inputs)}"
            )
        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in carried_inputs
        ):
            raise RuntimeError(
                "carried_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{carried_inputs}"
            )

        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in additional_inputs
        ):
            raise RuntimeError(
                "additional_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{additional_inputs}"
            )
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


while_loop_op = WhileLoopOp()


def while_loop(cond_fn, body_fn, carried_inputs):
    r"""
    Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
    initial carried_inputs.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following:

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors): A tuple of inputs to cond_fn and body_fn. It's also
            the initial value of states that are carried across iterations.

    Example:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: Tuple = ()
    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)

    def _validate_input(cond_fn, body_fn, carried_inputs):
        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callable.")

        if not isinstance(carried_inputs, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {carried_inputs}."
            )

    _validate_input(cond_fn, body_fn, carried_inputs)

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _while_loop_op_wrapper(*args, **kwargs):
        return while_loop_op(*args, **kwargs)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        with _temp_remove_metadata_torch_function_mode() as metadata_mode:
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"
                return torch.compile(
                    _while_loop_op_wrapper, backend=backend, fullgraph=True
                )(cond_fn, body_fn, carried_inputs, additional_inputs)


@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, carried_inputs, additional_inputs):
    carried_vals = carried_inputs

    def _is_boolean_scalar_tensor(pred):
        return (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        )

    # TODO: move these checks to another place?
    if not isinstance(carried_inputs, tuple):
        raise RuntimeError(
            f"carried_inputs must be a tuple but got {type(carried_inputs)}"
        )

    while pred := cond_fn(*carried_vals, *additional_inputs):
        if not _is_boolean_scalar_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        out = body_fn(*carried_vals, *additional_inputs)

        # TODO: move those checks to another place?
        assert isinstance(
            out, tuple
        ), f"body_fn should return a tuple but got {type(out)}"
        assert len(out) == len(
            carried_inputs
        ), "body_fn should return the same number of elements as carried_inputs"
        carried_vals = out
    return carried_vals


@while_loop_op.py_impl(DispatchKey.Autograd)
def while_loop_autograd(cond_fn, body_fn, carried_inputs, additional_inputs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    # TODO: Figure out how to do this in dispatcher so that we don't have to do this check here
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (carried_inputs, additional_inputs),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)

    # TODO: Support partial gradients of inputs in the future
    if pytree.tree_any(
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (carried_inputs, additional_inputs),
    ):
        raise RuntimeError(
            "while_loop currently only supports Autograd if all init, xs and lifted parameters require gradients."
        )

    # TODO: The create_fw_bw is always invoked twice:
    # Once in the forward path and
    # once in the backward path, where it should only be invoked for the grad grad case.
    # We don't support this currently
    if not torch.is_grad_enabled():
        # This clause is hit in the case of double backward.
        # Currently scan does not support this and thus we just dummy call another scan
        # The scan dim in the backward backward is always zero, because the
        # scan outputs during the forward are always collected at dim=0
        with torch._C._AutoDispatchBelowAutograd():
            return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)

    num_leaves_xs = len(carried_inputs)

    (
        cond_fn_graph,
        fw_graph,
        bwd_cond_fn_graph,
        joint_graph,
    ) = create_fw_bw_graph_combinefn(
        cond_fn, body_fn, carried_inputs, additional_inputs
    )

    flat_out = WhileLoopAutogradOp.apply(
        cond_fn_graph,
        fw_graph,
        bwd_cond_fn_graph,
        joint_graph,
        num_leaves_xs,
        *(carried_inputs + additional_inputs),
    )
    return (*flat_out,)


@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(mode, cond_fn, body_fn, carried_inputs, additional_inputs):
    def _trace_while_loop(
        proxy_mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
    ):
        cond_graph = reenter_make_fx(cond_fn)(*carried_inputs, *additional_inputs)
        body_graph = reenter_make_fx(body_fn)(*carried_inputs, *additional_inputs)

        next_name = None
        i = 0
        while not next_name:
            candidate = f"while_loop_cond_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        cond_graph_name = next_name
        body_graph_name = f"while_loop_body_graph_{i}"
        assert not hasattr(proxy_mode.tracer.root, body_graph_name)

        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        args = (cond_graph, body_graph, carried_inputs, additional_inputs)

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", while_loop_op, proxy_args, {}, name="while_loop"
        )

        # body_fn return output with the same pytree and tensor meta data as carried_inputs
        # so we could just return the output after one iteration.
        out = body_fn(*carried_inputs, *additional_inputs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )

    return _trace_while_loop(
        mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
    )


@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs
):
    with mode:
        return body_fn(*carried_inputs, *additional_inputs)


@while_loop_op.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, carried_inputs, additional_inputs):
    unwrapped_carried_inputs = ctx.unwrap_tensors(carried_inputs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    unwrapped_inputs = unwrapped_carried_inputs + unwrapped_additional_inputs
    with ctx.redispatch_to_next() as m:
        functional_cond_fn = ctx.functionalize(_maybe_run_with_interpreter(cond_fn))
        functional_body_fn = ctx.functionalize(_maybe_run_with_interpreter(body_fn))
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (functional_cond_fn, "cond_fn"),
            (functional_body_fn, "body_fn"),
        ]:
            if _has_potential_branch_input_mutation(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

            if _has_potential_branch_input_alias(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        ret = while_loop_op(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried_inputs,
            unwrapped_additional_inputs,
        )
        return ctx.wrap_tensors(ret)


def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands
