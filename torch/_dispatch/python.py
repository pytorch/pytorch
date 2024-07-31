# mypy: allow-untyped-defs
import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator

import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree


__all__ = ["enable_python_dispatcher", "no_python_dispatcher", "enable_pre_dispatch"]

no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch

CROSSREF_FUNCTIONALIZE = False


def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    Warning: the set of overloads this will report is very subtle.  It is precisely
    the set of torch.ops functions that have actually been accessed from Python
    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT
    from the set of registered operators, which will in general be a larger set,
    as this would include all operators which we ran C++ static initializers or
    Python operator registration on.  This does not eagerly populate the list on
    torch.ops.aten; this list is lazy!

    In other words, this is good for traversing over everything that has an
    OpOverload object allocated in Python.  We use it for cache invalidation, but
    don't rely on this list being complete.

    Note that even if we did report all C++ registered overloads, this isn't guaranteed
    to be complete either, as a subsequent lazy load of a library which triggers more
    registrations could add more things to the set.
    """
    for ns in torch.ops:
        packets = getattr(torch.ops, ns)
        for op_name in packets:
            packet = getattr(packets, op_name)
            for overload in packet:
                yield getattr(packet, overload)


@contextmanager
def suspend_functionalization():
    f_tls = torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.Functionalize
    )
    f_rv = torch._C._functionalization_reapply_views_tls()
    if f_tls:
        torch._disable_functionalization()
    try:
        yield
    finally:
        if f_tls:
            torch._enable_functionalization(reapply_views=f_rv)


def check_tensor_metadata_matches(nv, rv, desc):
    assert callable(desc)
    assert nv.size() == rv.size(), f"{desc()}: sizes {nv.size()} != {rv.size()}"
    assert nv.dtype == rv.dtype, f"{desc()}: dtype {nv.dtype} != {rv.dtype}"
    same_strides, idx = torch._prims_common.check_significant_strides(
        nv, rv, only_cuda=False
    )
    assert (
        same_strides
    ), f"{desc()}: strides {nv.stride()} != {rv.stride()} (mismatch at index {idx})"


def check_metadata_matches(n, r, desc):
    assert callable(desc)
    n_vals, n_spec = pytree.tree_flatten(n)
    r_vals, r_spec = pytree.tree_flatten(r)
    # TODO: test the specs match; empirically  sometimes we have a tuple
    # on one side and a list on the other
    assert len(n_vals) == len(r_vals), f"{len(n_vals)} != {len(r_vals)}"
    for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
        if not isinstance(rv, torch.Tensor):
            continue
        check_tensor_metadata_matches(nv, rv, lambda: f"{desc()} output {i}")


class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def _fmt(a: object) -> object:
    if isinstance(a, torch.Tensor):
        return Lit(
            f"torch.empty_strided({tuple(a.size())}, {a.stride()}, dtype={a.dtype})"
        )
    else:
        return a


def make_crossref_functionalize(op, final_key):
    from torch._subclasses.fake_tensor import FakeTensorMode

    # This case is pretty weird, suppress it for now
    if op == torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args, **kwargs):
        fake_mode = FakeTensorMode()

        def fakeify_defun(t):
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    # NB: This assumes that the inner tensor sizes/strides match
                    # the outer tensor sizes/strides.  This doesn't necessarily have to
                    # be the case, see discussion at
                    # https://github.com/pytorch/pytorch/pull/87610/files/401ddeda1d769bedc88a12de332c7357b60e51a4#r1007264456
                    assert t.size() == r.size()
                    assert t.stride() == r.stride()
                else:
                    r = t
                # TODO: suppress guards
                return fake_mode.from_tensor(r)
            return t

        def maybe_detach(t):
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t

        # TODO: This probably does the wrong thing if you're running other
        # substantive modes with the normal op outside here
        with torch.utils._python_dispatch._disable_current_modes(), suspend_functionalization():
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(
                maybe_detach, (f_args, f_kwargs)
            )
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)
        r = op._op_dk(final_key, *args, **kwargs)

        def desc():
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(pytree.tree_map(_fmt, a)) for a in orig_f_args),
                    (
                        f"{k}={pytree.tree_map(_fmt, v)}"
                        for k, v in orig_f_kwargs.items()
                    ),
                )
            )
            return f"{op}({fmt_args})"

        check_metadata_matches(f_r, r, desc)
        return r

    return handler


# NB: enabling this is slow, don't do it in a hot loop.  This is purely
# for debugging purposes.
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
    try:
        with enable_python_dispatcher(), unittest.mock.patch(
            "torch._dispatch.python.CROSSREF_FUNCTIONALIZE", True
        ):
            yield
    finally:
        for op in all_py_loaded_overloads():
            op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
