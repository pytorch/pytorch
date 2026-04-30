# mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if func is torch.ops.higher_order.cond:
            # Re-enter the mode to support nested conditionals
            with self:
                return if_else_node(*args)
        if func is torch.ops.higher_order.while_loop:
            # mutated_arg_indices may be passed as a kwarg by the HOP; it's
            # informational for downstream graph passes and doesn't change
            # how we drive the on-device while node, so we drop it here.
            kwargs = {} if kwargs is None else kwargs
            kwargs.pop("mutated_arg_indices", None)
            with self:
                return while_loop_node(*args, **kwargs)
        kwargs = {} if kwargs is None else kwargs
        return func(*args, **kwargs)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    """The purpose of this TodchDispatchMode is to "warm up" both sides of a torch.cond() statement.

    For data-dependent control flow code, only one side will be
    executed. Therefore, it is not safe to stream capture a
    torch.cond() statement naively, since we don't have a guarantee
    that all ops will have been "warmed up". The clever workaround is
    to use a "relaxed" stream capture whose final cuda graph we throw
    away. This works because stream capture does not actually execute
    any GPU code, and because true_fn and false_fn are both fxgraphs,
    which do not have any CPU side effects.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.supports_higher_order_operators = True
        self.capture_stream = torch.cuda.Stream()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = {} if kwargs is None else kwargs

        # Warm up both sides of this torch.cond()
        if func is torch.ops.higher_order.cond:
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.cond() nested within another
                # torch.cond() function.
                with self:
                    # We re-enter the mode in case of nested calls to torch.cond()
                    return if_else_node(*args)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    if_else_node(*args)

                return func(*args, **kwargs)
        elif func is torch.ops.higher_order.while_loop:
            # Same idea as the cond branch: a relaxed throwaway capture so
            # the body's kernels get primed (stream capture records nodes
            # without executing them). The "real" while_loop result still
            # comes from the dense impl below so the host-side state
            # reflects the loop completing.
            kwargs.pop("mutated_arg_indices", None)
            if torch.cuda.is_current_stream_capturing():
                with self:
                    return while_loop_node(*args, **kwargs)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    while_loop_node(*args, **kwargs)

                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def while_loop_node(cond_fn, body_fn, carried_inputs, additional_inputs):
    """Capture a torch.while_loop HOP as a CUDA `WHILE` conditional node.

    CUDA evaluates the conditional handle at the *start* of every iteration
    (including iter 0). We:
      1. Compute the initial predicate from the entry-time carried state and
         set the handle to it (begin_capture_to_while_node does this).
      2. Begin capturing the body subgraph.
      3. Run body_fn — it returns fresh tensors for the new carry. Since the
         captured graph reuses the same buffers across iterations, we
         in-place ``copy_`` those new values back into the original carry
         tensors.
      4. Recompute the predicate against the updated carry and call
         ``set_conditional_handle`` from inside the body so the next
         iteration's check reads the just-written value.
    """
    if not isinstance(carried_inputs, (tuple, list)):
        raise TypeError(
            f"while_loop_node expects carried_inputs to be a tuple/list, got {type(carried_inputs)}"
        )

    initial_pred = cond_fn(*carried_inputs, *additional_inputs)
    if not (isinstance(initial_pred, torch.Tensor) and initial_pred.is_cuda):
        raise ValueError(
            "while_loop_node requires cond_fn to return a CUDA scalar tensor "
            f"(got {type(initial_pred).__name__})"
        )

    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    handle = current_cuda_graph.begin_capture_to_while_node(initial_pred)
    try:
        new_carry = body_fn(*carried_inputs, *additional_inputs)
        if not isinstance(new_carry, (tuple, list)):
            raise TypeError(
                f"body_fn must return a tuple/list, got {type(new_carry)}"
            )
        if len(new_carry) != len(carried_inputs):
            raise ValueError(
                f"body_fn returned {len(new_carry)} values but expected "
                f"{len(carried_inputs)} to match the carry"
            )
        # Write the new carry back into the original carry tensors so the
        # next iteration's cond/body reads the updated values from the same
        # storage. (The HOP contract forbids body_fn from in-place mutating
        # carries directly; this copy_ is the dispatch handler's job.)
        for old, new in zip(carried_inputs, new_carry):
            old.copy_(new)

        # Re-evaluate the predicate against the now-updated carry, then
        # update the handle so the while-conditional sees fresh state at
        # the next iteration check.
        next_pred = cond_fn(*carried_inputs, *additional_inputs)
        torch.cuda.CUDAGraph.set_conditional_handle(handle, next_pred)
    finally:
        current_cuda_graph.end_capture_to_conditional_node()
    return tuple(carried_inputs)


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    if not pred.is_cuda:
        raise ValueError(
            "Conditions must be on a cuda device to use conditional node in cuda graphs"
        )
    # if-else is not supported until CUDA 12.8. Therefore, we use two
    # if conditions, where one evaluates !pred
    outs = []

    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))

            # The output of the else branch gets copied into the
            # output of the if branch. This is done because the rest
            # of the cudagraph after the conditional node has fixed
            # inputs, so we need to merge the two outputs into a
            # single output.
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]
