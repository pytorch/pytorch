import threading
from contextlib import contextmanager
from typing import Iterator, Optional

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
    use_ilistref_for_tensor_lists: Optional[bool] = None


_locals = Locals()


def use_const_ref_for_mutable_tensors() -> bool:
    assert _locals.use_const_ref_for_mutable_tensors is not None, (
        "need to initialize local.use_const_ref_for_mutable_tensors with "
        "local.parametrize"
    )
    return _locals.use_const_ref_for_mutable_tensors


def use_ilistref_for_tensor_lists() -> bool:
    assert _locals.use_ilistref_for_tensor_lists is not None, (
        "need to initialize local.use_ilistref_for_tensor_lists with "
        "local.parametrize"
    )
    return _locals.use_ilistref_for_tensor_lists


@contextmanager
def parametrize(
    *, use_const_ref_for_mutable_tensors: bool, use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = use_ilistref_for_tensor_lists
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists
