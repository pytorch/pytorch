# Owner(s): ["module: dynamo"]
import unittest
from torch.testing._internal.common_utils import TestCase
from torch._dynamo import config
from torch._dynamo.exc import TorchRuntimeError
import torch


class ErrorMessageTests(TestCase):
    def setUp(self):
        super().setUp()

        torch.library.define("test_ns::split_with_sizes", "(Tensor input, int[] sizes) -> Tensor[]")

        def impl(input, sizes):
            return [t.clone() for t in torch.ops.aten.split_with_sizes.default(input, sizes)]

        torch.library.impl("test_ns::split_with_sizes", "default", impl)

        @torch.library.register_fake("test_ns::split_with_sizes")
        def abstract(input, sizes):
            rs = torch.ops.aten.split_with_sizes.default(input, sizes)
            return [input.new_empty(r.size()) for r in rs]

    def test_symint_error_message(self):
        @torch.compile
        def f(sz, x):
            s0, s1 = sz.tolist()
            r0, r1 = torch.ops.test_ns.split_with_sizes.default(x, [s0, s1])
            return torch.ops.aten.sort.default(r1)

        N = 100
        S0 = 40
        S1 = N - S0

        with self.assertRaisesRegex(
            TorchRuntimeError,
            r"(?s).*Expected a value of type 'List\[int\]' for argument 'sizes'.*"
            r".*Hint: When using torch\.compile\(\), consider using SymInt\[\].*"
        ):
            f(torch.tensor([S0, S1]), torch.randn(N))

test_classes = {}


def make_error_message_cls(cls):
    test_class = cls
    test_classes[test_class.__name__] = test_class
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    ErrorMessageTests,
]

for test in tests:
    make_error_message_cls(test)
del test


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
