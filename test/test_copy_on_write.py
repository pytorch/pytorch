# Owner(s): ["module: viewing and reshaping", "module: internals"]

import warnings

import pytest

import torch

def test_copy_on_write_warns():
    t = torch.arange(4.0)
    # Does not warn with no aliasing.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        t.add_(torch.ones(4))

    u = t.reshape(2, 2)
    assert t.data_ptr() == u.data_ptr()

    # We now have two tensors that are logically distinct but share data.

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # This is not desired, this should warn.
        t[0] = 7

    # But the warning does work here.
    with pytest.warns(UserWarning, match='You have written through a view created by calling reshape().'):
        t.add_(torch.ones(4))


def test_copy_on_write():
    t = torch.arange(4.0)
    assert torch._C.copy_on_write_refcount(t) is None

    # Views share the same data and storage.
    tv = t.view(2, 2)
    assert t.data_ptr() == tv.data_ptr()
    assert torch._C.storage_id(t) == torch._C.storage_id(tv)
    assert torch._C.copy_on_write_refcount(t) is None

    # Copies never share the same storage.
    u = t.reshape(2, 2)  # u is a lazy copy of t
    assert t.data_ptr() == u.data_ptr()
    assert torch._C.storage_id(t) != torch._C.storage_id(u)
    assert torch._C.copy_on_write_refcount(t) == 2
    assert torch._C.copy_on_write_refcount(u) == 2

    # A view of a copy-on-write tensor still shares the same data and storage.
    uv = u.view(4)
    assert u.data_ptr() == uv.data_ptr()
    assert torch._C.storage_id(u) == torch._C.storage_id(uv)
    assert torch._C.copy_on_write_refcount(t) == 2
    assert torch._C.copy_on_write_refcount(uv) == 2
