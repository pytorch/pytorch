import contextlib
from collections.abc import Generator, Sequence

import torch
from torch.utils._content_store import ContentStoreReader


LOAD_TENSOR_READER: ContentStoreReader | None = None


@contextlib.contextmanager
def load_tensor_reader(loc: str) -> Generator[None, None, None]:
    global LOAD_TENSOR_READER
    if LOAD_TENSOR_READER is not None:
        raise AssertionError("LOAD_TENSOR_READER is already set")
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


def register_debug_prims() -> None:
    torch.library.define(
        "debugprims::load_tensor",
        "(str name, int[] size, int[] stride, *, ScalarType dtype, Device device) -> Tensor",
    )

    @torch.library.impl("debugprims::load_tensor", "BackendSelect")
    def load_tensor_factory(
        name: str,
        size: Sequence[int],
        stride: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided

            return rand_strided(size, stride, dtype, device)
        else:
            from torch._dynamo.utils import clone_input

            # device argument here takes care of coercion
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            if list(r.size()) != size:
                raise AssertionError(f"{r.size()} != {size}")
            if list(r.stride()) != stride:
                raise AssertionError(f"{r.stride()} != {stride}")
            if r.device != device:
                raise AssertionError(f"{r.device} != {device}")

            # Unlike the other properties, we will do coercions for dtype
            # mismatch
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)  # type: ignore[no-untyped-call]
            return r
