# mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


# TODO: Move this into torch/cuda/graphs.py


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
            pred, true_fn, false_fn, operands = cast(tuple[Any, Any, Any, Any], args)
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.cond() nested within either
                # another torch.cond() function.
                with self:
                    # We re-enter the mode in case of nested calls
                    if_else_node(*args)
            else:
                # TODO: What is this good for? It matters only if I
                # have another stream capture on this thread, which I
                # shouldn't have
                # torch.cuda.graphs.thread_cuda_stream_capture_mode(
                #     torch.cuda.cudart().cudaStreamCaptureMode.Relaxed
                # ),
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

        # Eagerly execute original function after warmup
        return func(*args, **kwargs)


def _is_boolean_scalar_cuda_tensor(pred: Any) -> bool:
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
    # if-else is not supported yet in CUDA 12.4. Therefore, we use two if conditions, where one evaluates !pred
    outs = []

    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))
            # Copy these two outputs into a new output buffer. Well,
            # actually, what we would like is to be able to merge these two
            # tensors into the same tensor... Is there an obvious way to do
            # that?
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]
