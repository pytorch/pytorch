import contextlib
from typing import Sequence

import torch
from torch._custom_op.impl import custom_op
from torch.utils._content_store import ContentStoreReader

LOAD_TENSOR_READER = None


@contextlib.contextmanager
def load_tensor_reader(loc):
    global LOAD_TENSOR_READER
    assert LOAD_TENSOR_READER is None
    # load_tensor is an "op", and we will play merry hell on
    # Inductor's memory planning if we return a tensor that
    # aliases another tensor that we previously returned from
    # an operator.  So unlike standard ContentStoreReader use,
    # we disable the cache so that you always get fresh storages
    # (no aliasing for you!)
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        LOAD_TENSOR_READER = None


def register_debug_prims():
    @custom_op("debugprims::load_tensor")
    def load_tensor(  # type: ignore[empty-body]
        name: str,
        size: Sequence[int],
        stride: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        ...

    @load_tensor.impl_factory()
    def load_tensor_factory(name, size, stride, dtype, device):
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided

            return rand_strided(size, stride, dtype, device)
        else:
            from torch._dynamo.utils import clone_input

            # device argument here takes care of coercion
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            assert list(r.size()) == size, f"{r.size()} != {size}"
            assert list(r.stride()) == stride, f"{r.stride()} != {stride}"
            assert r.device == device, f"{r.device} != {device}"

            # Unlike the other properties, we will do coercions for dtype
            # mismatch
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)
            return r
