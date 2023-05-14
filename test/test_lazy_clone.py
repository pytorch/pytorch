# Owner(s): ["module: internals"]

import torch


def test_lazy_clone():
    t = torch.tensor([[0, 1],
                      [2, 3]], dtype=torch.int8)
    clone = t._lazy_clone()
    assert torch._C._storage_address(t) != torch._C._storage_address(clone)
    assert torch._C._data_address(t) == torch._C._data_address(clone)
    view = t.view([4])
    assert torch._C._storage_address(t) == torch._C._storage_address(view)

    t += 1  # this write causes a copy to take place
    assert torch._C._data_address(t) != torch._C._data_address(clone)
