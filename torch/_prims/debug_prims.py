import torch
from torch._custom_op import custom_op
from torch.utils._content_store import ContentStoreReader
import contextlib

LOAD_TENSOR_READER = None

@contextlib.contextmanager
def load_tensor_reader(loc):
    global LOAD_TENSOR_READER
    assert LOAD_TENSOR_READER is None
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        LOAD_TENSOR_READER = None

def register_debug_prims():
    @custom_op('debugprims::load_tensor')
    def load_tensor(
        name: str,
        sizes: Tuple[int, ...],
        strides: Tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        ...

    @load_tensor.impl_factory()
    def load_tensor_factory(name, sizes, strides, dtype, device):
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided
            return rand_strided(sizes, strides, dtype, device)
        else:
            from torch._dynamo.utils import clone_input
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)
            return r
