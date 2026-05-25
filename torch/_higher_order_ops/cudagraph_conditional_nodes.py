# mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


def cuda_graph_conditional_nodes_supported() -> bool:
    cuda_version = torch.version.cuda
    if not torch.backends.cuda.is_built() or cuda_version is None:
        return False

    try:
        version_parts = cuda_version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    except ValueError:
        return False

    return major > 12 or (major == 12 and minor >= 4)


def _can_capture_while_loop(args, kwargs) -> bool:
    if not cuda_graph_conditional_nodes_supported():
        return False
    if kwargs.get("mutated_arg_indices"):
        return False
    if len(args) < 4:
        return False

    carried_inputs = args[2]
    additional_inputs = args[3]
    if not isinstance(carried_inputs, (tuple, list)):
        return False
    if not all(isinstance(inp, torch.Tensor) and inp.is_cuda for inp in carried_inputs):
        return False
    if not isinstance(additional_inputs, (tuple, list)):
        return False
    return all(
        not isinstance(inp, torch.Tensor) or inp.is_cuda for inp in additional_inputs
    )


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
            with self:
                return if_else_node(*args, **kwargs)
        if func is torch.ops.higher_order.while_loop and _can_capture_while_loop(
            args, kwargs
        ):
            # Re-enter the mode to support nested control flow.
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
        elif func is torch.ops.higher_order.while_loop and _can_capture_while_loop(
            args, kwargs
        ):
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


@contextmanager
def _while_body(pred: torch.Tensor) -> Generator[None, None, None]:
    if not cuda_graph_conditional_nodes_supported():
        raise RuntimeError(
            "CUDA graph conditional nodes require a CUDA 12.4+ non-ROCm build"
        )
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_while_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def _validate_cuda_bool_pred(pred: torch.Tensor) -> None:
    if (
        not isinstance(pred, torch.Tensor)
        or not pred.is_cuda
        or pred.size() != torch.Size([])
        or pred.dtype != torch.bool
    ):
        raise ValueError(
            "Conditions must be boolean scalar tensors on a cuda device "
            "to use conditional nodes in cuda graphs"
        )


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    _validate_cuda_bool_pred(pred)
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


def _clone_loop_state(carried_inputs):
    return tuple(
        inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in carried_inputs
    )


def _copy_loop_state(dst, src) -> None:
    if len(dst) != len(src):
        raise RuntimeError(
            "body_fn must return the same number of elements as carried_inputs"
        )

    for dst_t, src_t in zip(dst, src):
        if not isinstance(dst_t, torch.Tensor) or not isinstance(src_t, torch.Tensor):
            raise RuntimeError(
                "CUDA graph while_loop only supports tensor carried inputs"
            )
        dst_t.copy_(src_t)


def while_loop_node(
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    *,
    mutated_arg_indices: str = "",
):
    if mutated_arg_indices:
        raise RuntimeError("CUDA graph while_loop does not support mutated inputs yet")

    state = _clone_loop_state(carried_inputs)

    def operands():
        # _copy_loop_state mutates the cloned state tensors in-place, so each
        # condition/body call observes the latest loop-carried values.
        return (*state, *additional_inputs)

    pred = cond_fn(*operands())
    _validate_cuda_bool_pred(pred)

    with _while_body(pred):
        body_out = body_fn(*operands())
        if not isinstance(body_out, tuple):
            raise RuntimeError(
                f"body_fn should return a tuple but got {type(body_out)}"
            )
        _copy_loop_state(state, body_out)
        pred = cond_fn(*operands())
        _validate_cuda_bool_pred(pred)
        current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
        current_cuda_graph.set_current_conditional_node_condition(pred)

    return state


def inductor_while_loop_use_cuda_graph(inputs, stack_output: bool) -> bool:
    return (
        not stack_output
        and cuda_graph_conditional_nodes_supported()
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
        and all(not isinstance(inp, torch.Tensor) or inp.is_cuda for inp in inputs)
    )


def inductor_while_loop_cuda_graph(
    cond_graph,
    body_graph,
    carried_inputs,
    additional_inputs,
):
    state = _clone_loop_state(carried_inputs)

    def operands():
        return [*state, *additional_inputs]

    (pred,) = cond_graph(operands())
    _validate_cuda_bool_pred(pred)

    with _while_body(pred):
        body_out = body_graph(operands())
        _copy_loop_state(state, body_out)
        (pred,) = cond_graph(operands())
        _validate_cuda_bool_pred(pred)
        current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
        current_cuda_graph.set_current_conditional_node_condition(pred)

    return list(state)
