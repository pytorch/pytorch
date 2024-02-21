# Owner(s): ["oncall: pt2"]
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import HAS_CPU


class DistributedPatternTests(TestCase):
    @torch.no_grad()
    def test_storage_resize_zero(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            y = torch.sin(x)
            x.untyped_storage().resize_(0)
            return torch.cos(y)

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x))
        y = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)

    @torch.no_grad()
    def test_storage_resize_nonzero(self):
        @torch.compile(fullgraph=True)
        def fn(x, out):
            y = torch.sin(x)
            assert out.untyped_storage().size() == 0
            out.untyped_storage().resize_(x.untyped_storage().size())
            out.copy_(y.cos())

        x = torch.randn(10)
        out = torch.randn(10)
        expected = torch.cos(torch.sin(x))
        out.untyped_storage().resize_(0)
        fn(x, out)
        self.assertEqual(out.untyped_storage().size(), x.untyped_storage().size())
        self.assertEqual(out, expected)


if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
