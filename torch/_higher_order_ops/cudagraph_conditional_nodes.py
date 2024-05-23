from contextlib import contextmanager

import torch


def _is_boolean_scalar_cuda_tensor(pred):
    return (
        isinstance(pred, torch.Tensor)
        and pred.size() == torch.Size([])
        and pred.dtype == torch.bool
        and pred.is_cuda
    )

@contextmanager
def _if_body(pred: torch.Tensor):
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()

def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    if not pred.is_cuda:
        raise ValueError("Conditions must be on a cuda device to use conditional node in cuda graphs")
    # if-else is not supported yet in CUDA 12.4. Therefore, we use two if conditions, where one evaluates !pred
    outs = []

    for lazy_pred, fn in [(lambda: pred, true_fn), (lambda: torch.logical_not(pred), false_fn)]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))
            # Copy these two outputs into a new output buffer. Well,
            # actually, what we would like is to be able to merge these two
            # tensors into the same tensor... Is there an obvious way to do
            # that?
            if len(outs) == 2:
                for if_out, else_out in zip(outs[0], outs[1]):
                    if_out.copy_(else_out)
    assert len(outs) == 2
    return outs[0]

@contextmanager
def _while_loop_body(pred: torch.Tensor):
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    conditional_handle = current_cuda_graph.begin_capture_to_while_loop_node(pred)
    try:
        yield conditional_handle
    finally:
        current_cuda_graph.end_capture_to_conditional_node()

### conditional node: while_loop
def while_loop_node(cond_fn, body_fn, carried_inputs, additional_inputs):
    if not isinstance(carried_inputs, tuple):
        raise RuntimeError(
            f"carried_inputs must be a tuple but got {type(carried_inputs)}"
        )

    carried_vals = carried_inputs
    pred = cond_fn(*carried_vals, *additional_inputs)
    if not _is_boolean_scalar_cuda_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar cuda tensor but got {pred}"
        )

    with _while_loop_body(pred) as conditional_handle:
        out = body_fn(*carried_vals, *additional_inputs)
        assert isinstance(
            out, tuple
        ), f"body_fn should return a tuple but got {type(out)}"
        assert len(out) == len(
            carried_inputs
        ), "body_fn should return the same number of elements as carried_inputs"

        for c, o in zip(carried_vals, out):
            c.copy_(o)

        # call the cond_fn again to update the pred
        pred = cond_fn(*carried_vals, *additional_inputs)
        if not _is_boolean_scalar_cuda_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        torch.cuda.CUDAGraph.set_conditional_handle(conditional_handle, pred)

    return carried_vals
