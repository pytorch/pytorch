import threading
from contextlib import contextmanager
from typing import Optional, Iterator

# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.
#
# WARNING WARNING: LOOKING TO EDIT THIS FILE?  Think carefully about
# why you need to add a toggle to the global behavior of code
# generation.  The parameters here should really only be used
# for "temporary" situations, where we need to temporarily change
# the codegen in some cases because we cannot conveniently update
# all call sites, and are slated to be eliminated once all call
# sites are eliminated.  If you don't have a plan for how to get there,
# DON'T add a new entry here.

class Locals(threading.local):
    use_const_ref_for_mutable_tensors: Optional[bool] = None
    structured_sparse: Optional[bool] = None

_locals = Locals()

def use_const_ref_for_mutable_tensors() -> bool:
    assert _locals.use_const_ref_for_mutable_tensors is not None, \
        "need to initialize local.use_const_ref_for_mutable_tensors with " \
        "local.parametrize"
    return _locals.use_const_ref_for_mutable_tensors

def structured_sparse() -> bool:
    assert _locals.structured_sparse is not None, \
        "need to initialize local.structured_sparse with " \
        "local.parametrize"
    return _locals.structured_sparse

def structured_sparse_allow_uninit() -> bool:
    return _locals.structured_sparse

@contextmanager
def parametrize(*, use_const_ref_for_mutable_tensors: bool, structured_sparse: bool) -> Iterator[None]:
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_structured_sparse = _locals.structured_sparse
    try:
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.structured_sparse = structured_sparse
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = old_use_const_ref_for_mutable_tensors
        _locals.structured_sparse = old_structured_sparse
