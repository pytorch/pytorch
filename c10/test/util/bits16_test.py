import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

class TensorSubclassDemo(torch.Tensor):
    def __new__(cls, elem):
        assert elem.dtype == torch.bits16
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem):
        super().__init__()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.int16)
            return t

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        with no_dispatch():
            out = func(*args, **kwargs)
        return out.view(torch.bits16)

    def __repr__(self) -> str:
        with no_dispatch():
            return f"TensorSubclassDemo{self.view(torch.int16)}"


class TestBits16(TestCase):
    def test(self):
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        _ = torch.empty(20, dtype=torch.bits16)

        s = TensorSubclassDemo(t)
        s = s + 1


if __name__ == '__main__':
    run_tests()
