# mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


# TODO: Move this into torch/cuda/graphs.py


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
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
        with torch.cuda.graphs.thread_cuda_stream_capture_mode(
            torch.cuda.cudart().cudaStreamCaptureMode.Relaxed
        ):
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
    assert len(outs) == 2
    return outs[0]
