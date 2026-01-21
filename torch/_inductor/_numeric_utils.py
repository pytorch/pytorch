import random
from contextlib import contextmanager

import torch


@contextmanager
def random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def clone_args(args):
    return [
        arg.detach().clone().requires_grad_(arg.requires_grad)
        if isinstance(arg, torch.Tensor)
        else arg
        for arg in args
    ]


def assert_batch_invariance(fn, batch_ids_in, batch_ids_out, args, *, rtol=0, atol=0):
    """Assert that fn produces batch-invariant results.

    Args:
        fn: Function to test
        batch_ids_in: Tuple of batch dimension indices for input args (None for non-batched args)
        batch_ids_out: Int or tuple of batch dimension indices for outputs
        args: Input arguments to fn
        rtol: Relative tolerance for comparison (default 0 for bitwise equality)
        atol: Absolute tolerance for comparison (default 0 for bitwise equality)
    """
    batch_size = next(
        iter(
            arg.shape[batch_id]
            for arg, batch_id in zip(args, batch_ids_in)
            if batch_id is not None
        )
    )
    device = next(
        iter(
            arg.device
            for arg, batch_id in zip(args, batch_ids_in)
            if batch_id is not None
        )
    )

    def index_at(arg, idx, value):
        if idx is None:
            return arg
        assert value.max().item() < arg.shape[idx]
        return arg[tuple([slice(None) if i != idx else value for i in range(arg.ndim)])]

    def compare_tensors(a, b):
        if rtol == 0 and atol == 0:
            return torch.equal(a, b)
        else:
            return torch.allclose(a, b, rtol=rtol, atol=atol)

    ref_result = fn(*clone_args(args))
    with random_seed(0):
        for _ in range(3):
            num = random.randint(1, batch_size)
            selection = torch.tensor(
                [random.randint(0, batch_size - 1) for _ in range(num)], device=device
            )
            batch_result = fn(
                *[
                    index_at(arg, batch_id, selection)
                    for arg, batch_id in zip(args, batch_ids_in)
                ]
            )
            if type(batch_ids_out) is int:
                assert compare_tensors(
                    index_at(ref_result, batch_ids_out, selection), batch_result
                ), f"Batch invariance failed with rtol={rtol}, atol={atol}"
            else:
                for ref, batch, batch_id in zip(
                    ref_result, batch_result, batch_ids_out
                ):
                    assert compare_tensors(index_at(ref, batch_id, selection), batch), (
                        f"Batch invariance failed with rtol={rtol}, atol={atol}"
                    )
