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


def assert_batch_invariance(fn, batch_ids_in, batch_ids_out, args):
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
    print(f"{batch_size=} {batch_ids_in=} {batch_ids_out=}")
    print([(i, a.shape) for i, a in enumerate(args)])

    def index_at(arg, idx, value):
        if idx is None:
            return arg
        print(f"{arg.shape=} {value=}")
        assert value.max().item() < arg.shape[idx]
        return arg[tuple([slice(None) if i != idx else value for i in range(arg.ndim)])]

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
                assert torch.equal(
                    index_at(ref_result, batch_ids_out, selection), batch_result
                )
            for ref, batch, batch_id in zip(ref_result, batch_result, batch_ids_out):
                if batch_id is None:
                    continue
                assert torch.equal(index_at(ref, batch_id, selection), batch)
