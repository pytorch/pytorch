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
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.higher_order.cond:
            # Re-enter the mode to support nested conditionals
            _check_no_cond_kwargs(kwargs)
            with self:
                return if_else_node(*args)
        if func is torch.ops.higher_order.while_loop:
            # Re-enter the mode to support nested control flow
            with self:
                return while_loop_node(*args, **kwargs)
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
                # control-flow function.
                _check_no_cond_kwargs(kwargs)
                with self:
                    # We re-enter the mode in case of nested calls to torch.cond()
                    return if_else_node(*args)
            else:
                _check_no_cond_kwargs(kwargs)
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
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.while_loop() nested within another
                # control-flow function.
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


def _check_no_cond_kwargs(kwargs) -> None:
    if kwargs:
        raise RuntimeError("CUDA graph conditional torch.cond does not support kwargs")


def _is_boolean_scalar_cuda_tensor(pred: object) -> bool:
    return (
        isinstance(pred, torch.Tensor)
        and pred.size() == torch.Size([])
        and pred.dtype == torch.bool
        and pred.is_cuda
    )


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


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


@contextmanager
def _while_body(pred: torch.Tensor):
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_while_node(pred)
    try:
        yield current_cuda_graph
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def while_loop_node(
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
):
    flat_carried_inputs, carried_spec = pytree.tree_flatten(carried_inputs)
    if not all(isinstance(inp, torch.Tensor) for inp in flat_carried_inputs):
        raise RuntimeError(
            "CUDA graph while_loop conditional nodes only support tensor carried_inputs"
        )

    loop_carried = pytree.tree_map_only(
        torch.Tensor, lambda inp: inp.clone(), carried_inputs
    )
    flat_loop_carried = pytree.tree_leaves(loop_carried)

    pred = cond_fn(*loop_carried, *additional_inputs)
    if not _is_boolean_scalar_cuda_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
        )

    with _while_body(pred) as current_cuda_graph:
        body_out = body_fn(*loop_carried, *additional_inputs)
        flat_body_out, body_out_spec = pytree.tree_flatten(body_out)
        if body_out_spec != carried_spec:
            raise AssertionError(
                "body_fn should return the same pytree structure as carried_inputs"
            )
        if not all(isinstance(out, torch.Tensor) for out in flat_body_out):
            raise RuntimeError(
                "CUDA graph while_loop conditional nodes only support tensor "
                "body_fn outputs"
            )

        for carried, out in zip(flat_loop_carried, flat_body_out):
            if carried.data_ptr() != out.data_ptr():
                carried.copy_(out)

        pred = cond_fn(*loop_carried, *additional_inputs)
        if not _is_boolean_scalar_cuda_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
            )
        current_cuda_graph.set_conditional_handle_for_current_node(pred)

    return loop_carried
