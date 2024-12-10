# mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import Any, Generator, Optional
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    def __init__(self):
        self.warmed_up_control_flow_ops = set()
        self.inside_already_warmed_up_op = False

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = {} if kwargs is None else kwargs
        return func(*args, **kwargs)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    def __init__(
        self,
    ) -> None:
        self.stream = torch.cuda.graphs.create_external_stream()
        self.throw_away_graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_ctx: Optional[torch.cuda.graph] = None

    def __enter__(self) -> Self:
        self.throw_away_graph = torch.cuda.CUDAGraph()
        # relaxed stream capture can still fail if a synchronizing API
        # is called. But then this workload could not be captured in a
        # cuda graph anyway, so such a failure is fine.
        self.graph_ctx = torch.cuda.graph(
            self.throw_away_graph,
            stream=self.stream,
            capture_error_mode="relaxed",
            collect_garbage=False,
        )
        self.graph_ctx.__enter__()
        super().__enter__()
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        assert self.graph_ctx is not None
        with torch.cuda.graphs.thread_cuda_stream_capture_mode(
            torch.cuda.cudart().cudaStreamCaptureMode.Relaxed
        ):
            self.graph_ctx.__exit__(exc_type, exc_val, exc_tb)
            # The destructor of self.throw_away_graph calls
            # cudaGraphExecDestroy(), which is an unsafe call for any
            # other streams that are currently capturing to a graph. To
            # prevent invalidating other capturing streams, this thread
            # must remain in relaxed stream capture mode when the
            # destructor runs. Therefore, we manually delete
            # self.throw_away_graph (and self.graph_ctx, which has a
            # strong reference to it) now rather than letting them be
            # automatically destroyed when this
            # ControlFlowOpWarmupDispatchMode instance is deleted.
            del self.graph_ctx
            del self.throw_away_graph

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


@contextmanager
def _while_loop_body(pred: torch.Tensor) -> Generator[int, None, None]:
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    conditional_handle = current_cuda_graph.begin_capture_to_while_loop_node(pred)
    try:
        yield conditional_handle
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def while_loop_node(cond_fn, body_fn, carried_inputs, additional_inputs):
    carried_vals = carried_inputs
    pred = cond_fn(*carried_vals, *additional_inputs)
    if not _is_boolean_scalar_cuda_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar cuda tensor but got {pred}"
        )

    with _while_loop_body(pred) as conditional_handle:
        out = body_fn(*carried_vals, *additional_inputs)
        out_flat, out_spec = pytree.tree_flatten(out)
        assert len(out_flat) == len(
            carried_inputs
        ), "body_fn should return the same number of elements as carried_inputs"

        for c, o in zip(carried_vals, out_flat):
            # TODO: Consider skipping the copy_ if the data_ptr is the
            # same.
            c.copy_(o)

        # call the cond_fn again to update the pred.
        pred = cond_fn(*carried_vals, *additional_inputs)
        if not _is_boolean_scalar_cuda_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        torch.cuda.CUDAGraph.set_conditional_handle(conditional_handle, pred)

    return pytree.tree_unflatten(carried_vals, out_spec)
