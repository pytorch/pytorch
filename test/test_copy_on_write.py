# Owner(s): ["module: viewing and reshaping", "module: internals"]

import torch

def test_copy_on_write():
    t = torch.ones(4)
    u = t.reshape(2, 2)  # this is a lazy copy
    assert torch._C._storage_id(t) != torch._C._storage_id(u)  # different storages
    assert t.data_ptr() == u.data_ptr()                        # but same data

    v = t.view(2, 2)
    assert torch._C._storage_id(t) == torch._C._storage_id(v)  # same storages
    assert t.data_ptr() == u.data_ptr() == v.data_ptr()
    assert torch.equal(u, v)

    # Write to t: t and u alias, so they both see bumps.
    t.add_(torch.ones(4))
    # No longer aliasing.
    assert t.data_ptr() != u.data_ptr()  # u is no longer an alias
    assert t.data_ptr() == v.data_ptr()  # but v still is

    assert not torch.equal(u, v)  # u sees the write, but not v


def test_grad():
    t = torch.arange(4.0, requires_grad=True)

    # Copies never share the same storage.
    u = t.reshape(2, 2)  # u is a lazy copy of t

    u.sum().backward()
