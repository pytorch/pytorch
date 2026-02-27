# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

import itertools

class Int16Tensor(torch.Tensor):
    def __new__(cls, elem):
        if elem.dtype != torch.bits16:
            raise AssertionError(f"Expected dtype torch.bits16, got {elem.dtype}")
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

    # This most likely should be removed (and thus use the disabled impl)
    # but the test below fail under Dynamo in that case.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self) -> str:
        with no_dispatch():
            self.view(torch.int16)
            return f"TensorSubclassDemo{self.view(torch.int16)}"


class TestBits(TestCase):
    def test_types(self, device):
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            _ = torch.zeros(20, dtype=torch.int32, device=device).view(bits_type)
            _ = torch.empty(20, dtype=bits_type, device=device)
            x = torch.randint(100, (20, 20), dtype=torch.int8, device=device).view(bits_type)
            y = x.t().contiguous()
            view_type = torch.int8 if x.element_size() == 1 else torch.int16
            self.assertEqual(x.t().view(view_type), y.view(view_type))
            y = x.t().clone()
            self.assertEqual(x.t().view(view_type), y.view(view_type))

    def test_cat(self, device):
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            view_type = torch.int8 if bits_type.itemsize == 1 else torch.int16
            x_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            x = x_int.view(bits_type)
            y_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            y = y_int.view(bits_type)
            for dim, transpose in itertools.product(range(x_int.ndim), (True, False)):
                y_ref = y_int.t() if transpose else y_int
                y_b = y.t() if transpose else y
                z_ref = torch.cat([x_int, y_ref], dim=dim)
                z = torch.cat([x, y_b], dim=dim)
                self.assertEqual(z_ref, z.view(view_type))


    def test_subclass(self):
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        s = Int16Tensor(t)
        s = s + 1 - 1
        self.assertTrue(torch.allclose(s, torch.zeros(20, dtype=torch.bits16)))

instantiate_device_type_tests(TestBits, globals())


if __name__ == '__main__':
    run_tests()
