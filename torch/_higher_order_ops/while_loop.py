# mypy: allow-untyped-defs
import contextlib
import functools
from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    check_meta_consistency,
    materialize_as_graph,
    reenter_make_fx,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class WhileLoopOp(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[Union[torch.Tensor, int, float, bool]],
        additional_inputs: tuple[Union[torch.Tensor, torch.SymInt, int], ...],
        /,
    ):
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )

        validate_subgraph_args_types(carried_inputs)
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


class WhileLoopWithCheckpointOp(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("while_loop_with_checkpoint")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[Union[torch.Tensor, int, float, bool]],
        additional_inputs: tuple[Union[torch.Tensor, torch.SymInt, int], ...],
        /,
    ):
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )

        validate_subgraph_args_types(carried_inputs)
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


while_loop_op = WhileLoopOp()
while_loop_with_checkpoint_op = WhileLoopWithCheckpointOp()


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
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor or a python boolean.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors or ints

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors or ints): A tuple of inputs to cond_fn and body_fn.
            It's also the initial value of states that are carried across iterations. Note that when pass an integer as carry,
            the corresponding return of while_loop will be another int with unknown values because we don't know how many
            iterations while_loop will run.

    Example 1:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Example 2:

        def cond_fn(int_iter, x):
            return 2 * int_iter < x.shape[0]

        def body_fn(int_iter, x):
            return int_iter + 1, x + int_iter

        while_loop(cond,_fn, body_fn, (0, torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors or int with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python variables (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot alias any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: tuple = ()

    # The reason we flatten the output before calling into dynamo is that
    # we want to create a consistent input ordering for cond_fn and body_fn.
    # and we also want to the input ordering matches the output ordering.
    # Also see NOTE: [why we cannot use "automatic" for while_loop]
    # Construct flat cond_fn and flat_body_fn, which takes flattened inputs
    flat_inputs, in_spec = pytree.tree_flatten((carried_inputs, additional_inputs))

    def flat_cond_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return cond_fn(*carried, *additional)

    def flat_body_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return body_fn(*carried, *additional)

    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())

    def _validate_input(cond_fn, body_fn, carried_inputs):
        from torch._higher_order_ops.utils import validate_subgraph_args_types

        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callable.")

        validate_subgraph_args_types(flat_inputs)

        if not pytree.tree_all(
            lambda t: isinstance(t, (torch.Tensor, torch.SymInt, int)), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor or int leaves, but got {carried_inputs}."
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
                )(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())


def _while_loop_dense_impl(
    cond_fn, body_fn, carried_inputs, additional_inputs, *, with_checkpoint=False
):
    """Shared implementation for while_loop_dense and while_loop_with_checkpoint_dense."""
    carried_vals = carried_inputs

    def _validate_cond_output(pred):
        if (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        ) or isinstance(pred, bool):
            return
        else:
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor or a boolean but got {pred}"
            )

    if not isinstance(carried_inputs, (tuple, list)):
        raise RuntimeError(
            f"carried_inputs must be a tuple or list but got {type(carried_inputs)}"
        )

    checkpoints: list[list[torch.Tensor]] = [[] for _ in range(len(carried_inputs))]
    while pred := cond_fn(*carried_vals, *additional_inputs):
        _validate_cond_output(pred)

        if with_checkpoint:
            for i, carry in enumerate(carried_vals):
                checkpoints[i].append(carry)

        out = body_fn(*carried_vals, *additional_inputs)
        assert isinstance(out, tuple), (
            f"body_fn should return a tuple but got {type(out)}"
        )
        assert len(out) == len(carried_inputs), (
            "body_fn should return the same number of elements as carried_inputs"
        )
        carried_vals = out

    if with_checkpoint:
        checkpoint_tensors = []
        for ckp_list, val in zip(checkpoints, carried_vals):
            checkpoint_tensors.append(
                val.unsqueeze(0).clone()
                if len(ckp_list) == 0
                else torch.stack(ckp_list)
            )
        return tuple(carried_vals) + tuple(checkpoint_tensors)
    else:
        return tuple(carried_vals)


while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)(
    functools.partial(_while_loop_dense_impl, with_checkpoint=False)
)

while_loop_with_checkpoint_op.py_impl(DispatchKey.CompositeExplicitAutograd)(
    functools.partial(_while_loop_dense_impl, with_checkpoint=True)
)


class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cond_fn,
        body_fn,
        with_checkpoint,
        num_carried_inputs,
        num_additional_inputs,
        *carries_and_inputs,
    ):
        from torch._higher_order_ops.scan import split_into_chunks

        carries, additional_inputs = split_into_chunks(
            carries_and_inputs, [num_carried_inputs, num_additional_inputs]
        )
        with torch._C._AutoDispatchBelowAutograd():
            out, checkpoints = split_into_chunks(
                while_loop_with_checkpoint_op(
                    cond_fn, body_fn, carries, additional_inputs
                ),
                [num_carried_inputs, num_carried_inputs],
            )

        assert not hasattr(ctx, "fw_cond_fn")
        assert not hasattr(ctx, "fw_body_fn")
        assert not hasattr(ctx, "carries")
        assert not hasattr(ctx, "additional_inputs")
        assert not hasattr(ctx, "checkpoints")
        ctx.fw_cond_fn = cond_fn
        ctx.fw_body_fn = body_fn
        ctx.carries = carries
        ctx.additional_inputs = additional_inputs
        ctx.checkpoints = checkpoints

        for carry, ckp in zip(carries, ctx.checkpoints):
            ckp.requires_grad_(carry.requires_grad)

        ctx.with_checkpoint = with_checkpoint
        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        assert len(checkpoints) > 0, "checkpoints shouldn't be empty"
        if with_checkpoint:
            return tuple(out) + tuple(checkpoints)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        from torch._higher_order_ops.cond import create_bw_fn
        from torch._higher_order_ops.scan import split_into_chunks
        from torch._higher_order_ops.utils import is_differentiable_dtype

        num_carried_inputs = len(ctx.carries)
        grad_carries, _ = (
            split_into_chunks(
                grads,
                [num_carried_inputs, num_carried_inputs],
            )
            if ctx.with_checkpoint
            else grads
        )

        bw_body_fn = create_bw_fn(ctx.fw_body_fn, ctx.carries + ctx.additional_inputs)

        # For input tensors that don't require gradient, we return zeros tensors.
        # For non-tensor inputs, e.g. lifted free symbols in additional_inputs,
        # we return None as their gradients.
        additional_inputs_tensor_masks = [
            True
            if isinstance(t, torch.Tensor) and is_differentiable_dtype(t.dtype)
            else False
            for t in ctx.additional_inputs
        ]
        carries_tensor_masks = [
            True
            if isinstance(t, torch.Tensor) and is_differentiable_dtype(t.dtype)
            else False
            for t in ctx.carries
        ]

        def _filter_with_masks(
            maybe_tensors: tuple[Optional[torch.Tensor], ...], tensor_masks
        ) -> tuple[Any, ...]:
            assert len(maybe_tensors) == len(tensor_masks)
            tensors = tuple(
                t for is_tensor, t in zip(tensor_masks, maybe_tensors) if is_tensor
            )
            assert all(isinstance(t, torch.Tensor) for t in tensors)
            return tensors

        def _fill_none_with_masks(
            tensors: tuple[torch.Tensor, ...], tensor_masks
        ) -> tuple[Optional[torch.Tensor], ...]:
            assert len(tensors) <= len(tensor_masks) and all(
                isinstance(t, torch.Tensor) for t in tensors
            )
            tensors_iter = iter(tensors)
            maybe_tensors = tuple(
                next(tensors_iter) if is_tensor else None for is_tensor in tensor_masks
            )
            return maybe_tensors

        grad_additional_inputs = tuple(
            torch.zeros_like(t)
            for t in _filter_with_masks(
                ctx.additional_inputs, additional_inputs_tensor_masks
            )
        )
        grad_carries = tuple(
            t for t in _filter_with_masks(grad_carries, carries_tensor_masks)
        )
        init_idx = torch.zeros((), dtype=torch.int64)
        _, spec = pytree.tree_flatten(
            (
                init_idx,
                grad_carries,
                grad_additional_inputs,
                ctx.checkpoints,
                ctx.additional_inputs,
            )
        )

        def cond_fn(*flat_args):
            (
                idx,
                grad_carries,
                grad_additional_inputs,
                checkpoints,
                additional_inputs,
            ) = pytree.tree_unflatten(flat_args, spec)
            return idx < checkpoints[0].size(0)

        def body_fn(*flat_args):
            (
                idx,
                grad_carries,
                grad_additional_inputs,
                checkpoints,
                additional_inputs,
            ) = pytree.tree_unflatten(flat_args, spec)
            reversed_idx = checkpoints[0].size(0) - idx - 1
            selected_checkpoints = [
                ckp.select(0, reversed_idx.item()) for ckp in checkpoints
            ]
            cur_grad_carries, cur_grad_additional_inputs = split_into_chunks(
                bw_body_fn(*selected_checkpoints, *additional_inputs, *grad_carries),
                [len(ctx.carries), len(ctx.additional_inputs)],
            )
            cur_grad_additional_inputs = _filter_with_masks(
                cur_grad_additional_inputs, additional_inputs_tensor_masks
            )

            def _acc_grad(grad, acc):
                if grad is None:
                    return
                return acc + grad

            return (
                idx + 1,
                *_filter_with_masks(cur_grad_carries, carries_tensor_masks),
                *(
                    _acc_grad(cur_grad, grad)
                    for cur_grad, grad in zip(
                        cur_grad_additional_inputs, grad_additional_inputs
                    )
                ),
            )

        args_single_step_bw = (
            init_idx,
            *grad_carries,
            *grad_additional_inputs,
            *ctx.checkpoints,
            *ctx.additional_inputs,
        )
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        with disable_proxy_modes_tracing():
            cond_gm = materialize_as_graph(
                cond_fn,
                args_single_step_bw,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )
            body_gm = materialize_as_graph(
                body_fn,
                args_single_step_bw,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )
        outs = while_loop_op(
            cond_gm,
            body_gm,
            (
                init_idx,
                *grad_carries,
                *grad_additional_inputs,
            ),
            (*ctx.checkpoints, *ctx.additional_inputs),
        )
        _, final_grad_carries, final_grad_additional_inputs = split_into_chunks(
            outs, [1, len(grad_carries), len(grad_additional_inputs)]
        )
        return (
            None,
            None,
            None,
            None,
            None,
            *_fill_none_with_masks(final_grad_carries, carries_tensor_masks),
            *_fill_none_with_masks(
                final_grad_additional_inputs, additional_inputs_tensor_masks
            ),
        )


def _while_loop_autograd_impl(
    cond_fn, body_fn, operands, additional_inputs, with_checkpoint
):
    return WhileLoopAutogradOp.apply(
        cond_fn,
        body_fn,
        with_checkpoint,
        len(operands),
        len(additional_inputs),
        *operands,
        *additional_inputs,
    )


while_loop_op.py_autograd_impl(
    functools.partial(_while_loop_autograd_impl, with_checkpoint=False)
)
while_loop_with_checkpoint_op.py_autograd_impl(
    functools.partial(_while_loop_autograd_impl, with_checkpoint=True)
)


def _find_or_create_fake_mode() -> FakeTensorMode:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    fake_mode = torch._guards.detect_fake_mode()
    if fake_mode is None:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    return fake_mode


def _create_unbacked_symint(
    fake_mode: FakeTensorMode, ignore_fresh_unbacked_symbols: bool
) -> torch.SymInt:
    assert fake_mode is not None and fake_mode.shape_env is not None, (
        "Must provide a fake_mode with shape_env."
    )
    ctx = (
        contextlib.nullcontext()
        if not ignore_fresh_unbacked_symbols
        else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
    )
    with ctx:
        return fake_mode.shape_env.create_unbacked_symint()


def _trace_while_loop(
    proxy_mode,
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    with_checkpoint=False,
):
    # NOTE [unspecialize int carry with unbacked symints]
    # When we support int carry, we'll also need to support int output of body_fn because.
    # previous iteration's output is next iteration's input and they must match.
    # For carries, when we start tracing while_loop, they can be
    #   - constants e.g. (0, [1, 3])
    #   - backed symints (x.shape[0], [x.shape[1] + x.stride[1], x.shape[2]])
    #   - unbacked symints e.g. (u0, [u0 + u1, u2])
    #   We choose the most conservative design: in all cases, we create new unbacked symints to trace the
    #   subgraph. It's possible to do some analysis on initial carry and the output of first
    #   iteration to determine a better range for the output unbacked symbol e.g. when input is an unbacked
    #   symint >= 0 before the while_loop but in general this is difficult because we don't know
    #   the number of iterations. Users would have to re-constrain the unbacked symint in subgraph if needed.
    #
    # For output of fake cond_fn, it could be constant bool or SymBool (e.g. return x.shape[0] < 4,
    #   where x.shape[0] can be either static of dynamic). In the case of constant bool, we should do a
    #   specialization (NYI).

    # For output of fake body_fn, it could be all three types though from user's point of view,
    # they're all integers e.g.

    #   init_carry = (0, s0, u1, t)
    #   def body_fn(u0, s0, u1, t):
    #     ...
    #     return (t.shape[0], t.shape[1], t.shape[2], y + 1)
    #
    #   It may seem that a constant output isn't possible: users shouldn't write a while_loop
    #   that always return 0. But it could be that a shape is not set as dynamic properly (e.g.
    #   automatic dynamic hasn't been triggered).
    #
    #   For this reason, we treat int, symint outputs in the same way:
    #   - they can match against any of int, symint carry
    #   - we unspecialize them with new unbacked symints in fake while_loop
    #   Similarly, we could do some analysis to refine the output ranges but it's easier to start with
    #   fresh unbacked symints. One surprising case can be: an input unbacked symint is constrained by
    #   users to be >= 0 (either before while_loop or inside body_fn) and it increments by 1 in each
    #   iteration. Ideally, we should know that the final output is >= 0 but we didn't constrain the
    #   unbacked symint output of subgraph as of today because this requires a smart range analysis.
    fake_mode: FakeTensorMode = _find_or_create_fake_mode()

    def _unspecialize_carried_inputs(x):
        if isinstance(x, (int, torch.SymInt)):
            return _create_unbacked_symint(
                fake_mode, ignore_fresh_unbacked_symbols=True
            )
        # Note: [unspecialize constant tensor carry]
        # We need to disable constant specialization for tensor inputs that become loop carries.
        # Here's the problem: when a user creates a constant tensor e.g. torch.tensor(0), PyTorch calls aten.lift_fresh_copy
        # to create a safe copy (avoiding aliasing issues), which creates a FakeTensor with constant=True.
        # But when this FakeTensor becomes a loop carry, we have a problem:
        # - Operations like .item() will read the constant value and bake it into the traced code
        # - This is incorrect because carry variables change between loop iterations
        # - The traced code would use the wrong constant value for all iterations
        # Solution: We clone the constant tensors and mark the cloned tensor as non-constant so they won't
        # be specialized to fixed values during tracing body_fn or cond_fn.
        elif isinstance(x, torch.Tensor):
            x = x.clone()
            if hasattr(x, "constant") and x.constant is not None:
                x.constant = None
        return x

    with disable_proxy_modes_tracing():
        unspecialized_carried_inputs = pytree.tree_map_only(
            (int, torch.SymInt, torch.Tensor),
            # For temporarily created unbacked symints, we don't need to bind them to any proxy
            lambda x: _unspecialize_carried_inputs(x),
            carried_inputs,
        )

        def produce_graph(fn):
            cloned_carried_inputs = pytree.tree_map_only(
                torch.Tensor, lambda x: x.clone(), unspecialized_carried_inputs
            )
            return reenter_make_fx(fn)(*cloned_carried_inputs, *additional_inputs)

        cond_graph = produce_graph(cond_fn)
        body_graph = produce_graph(body_fn)

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

    op = while_loop_with_checkpoint_op if with_checkpoint else while_loop_op
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", op, proxy_args, {}, name="while_loop"
    )

    out = op(cond_graph, body_graph, unspecialized_carried_inputs, additional_inputs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


while_loop_op.py_impl(ProxyTorchDispatchMode)(
    functools.partial(_trace_while_loop, with_checkpoint=False)
)
while_loop_with_checkpoint_op.py_impl(ProxyTorchDispatchMode)(
    functools.partial(_trace_while_loop, with_checkpoint=True)
)


def _while_loop_fake_tensor_mode(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs, with_checkpoint=False
):
    with mode:
        # NOTE: [Handling unback symints in subgraph of while_loop]
        # The idea is that the scope of unbacked symints are limited to the subgraph.
        #
        # We're implementing the fake tensor mode of while_loop operator.
        # and we run body_fn once to get an fake output.
        # Let's first consider the case that unbacked symints are tensor shapes:
        #
        # Case 1:
        # if the unbacked symints is local to the subgraph e.g.
        #   def body_fn(it, x):
        #     nz = x.nonzero()
        #     return it+1. nz.sum()
        # we can just ignore the newly created unbacked symints because it has
        # no effect on the output of while_loop and it's tracked when we tracing.
        # the subgraph.
        #
        # Case 2:
        # if the unbacked symints are shape of output of while_loop e.g.
        #   def body_fn(it, x):
        #     nz = x.nonzero()
        #     return it+1, nz
        # This will fail the shape check because in each iteration, the carried_input's shape
        # must match the output shape as nz.shape contains newly allocated unbacked symint, this
        # won't match the carried_input's shape.
        #
        # Case 3:
        # if the unbacked symints are shape of carried_inputs e.g.
        #   nz = a.nonzero()
        #   body_fn(it, nz):
        #     return it+1. nz.sin() + 1,
        # There's no new unbacked symints allocated in subgraph, so we're safe.
        with mode.shape_env.ignore_fresh_unbacked_symbols():
            # body_fn return output with the same pytree and tensor meta data as carried_inputs
            # so we could just return the output after one iteration.
            body_outs = body_fn(*carried_inputs, *additional_inputs)
            check_meta_consistency(
                carried_inputs,
                body_outs,
                "carried_inputs",
                "body_output",
                include_contiguity=False,
            )
        fake_checkpoints: tuple[torch.Tensor, ...] = tuple()
        if with_checkpoint:
            n_iter = _create_unbacked_symint(mode, ignore_fresh_unbacked_symbols=False)
            assert all(isinstance(x, torch.Tensor) for x in carried_inputs)
            fake_checkpoints = tuple(
                out.clone()
                .unsqueeze(0)
                .repeat((n_iter,) + tuple(1 for _ in range(out.dim())))
                for out in body_outs
            )
        # See NOTE [unspecialize int carry with unbacked symints]
        return pytree.tree_map_only(
            (int, torch.SymInt),
            # For while_loop's unbacked symint output, we want them to be bound
            # to the proxy of while_loop's output.
            lambda _: _create_unbacked_symint(
                mode, ignore_fresh_unbacked_symbols=False
            ),
            body_outs + fake_checkpoints,
        )


while_loop_op.py_impl(FakeTensorMode)(
    functools.partial(_while_loop_fake_tensor_mode, with_checkpoint=False)
)
while_loop_with_checkpoint_op.py_impl(FakeTensorMode)(
    functools.partial(_while_loop_fake_tensor_mode, with_checkpoint=True)
)


def _while_loop_functionalize_impl(
    ctx, cond_fn, body_fn, carried_inputs, additional_inputs, with_checkpoint=False
):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    hop = while_loop_with_checkpoint_op if with_checkpoint else while_loop_op

    unwrapped_carried_inputs = ctx.unwrap_tensors(carried_inputs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    unwrapped_inputs = unwrapped_carried_inputs + unwrapped_additional_inputs
    with ctx.redispatch_to_next():
        functional_cond_fn = ctx.functionalize(_maybe_run_with_interpreter(cond_fn))
        functional_body_fn = ctx.functionalize(_maybe_run_with_interpreter(body_fn))
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (cond_fn, "cond_fn"),
            (body_fn, "body_fn"),
        ]:
            _check_alias_and_mutation(fn, unwrapped_inputs, fn_name, pre_dispatch)
        ret = hop(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried_inputs,
            unwrapped_additional_inputs,
        )
        return ctx.wrap_tensors(ret)


while_loop_op.py_functionalize_impl(
    functools.partial(
        _while_loop_functionalize_impl,
        with_checkpoint=False,
    )
)
while_loop_with_checkpoint_op.py_functionalize_impl(
    functools.partial(
        _while_loop_functionalize_impl,
        with_checkpoint=True,
    )
)
