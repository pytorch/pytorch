# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

class Int16Tensor(torch.Tensor):
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

        def wrap(t):
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.bits16)
            return t
        out = tree_map(wrap, out)
        return out

    def __repr__(self) -> str:
        with no_dispatch():
            t16 = self.view(torch.int16)
            return f"TensorSubclassDemo{self.view(torch.int16)}"


class TestBits(TestCase):
    def test_types(self):
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            _ = torch.zeros(20, dtype=torch.int32).view(bits_type)
            _ = torch.empty(20, dtype=bits_type)

    def test_subclass(self):
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        s = Int16Tensor(t)
        s = s + 1 - 1
        self.assertTrue(torch.allclose(s, torch.zeros(20, dtype=torch.bits16)))


if __name__ == '__main__':
    run_tests()
