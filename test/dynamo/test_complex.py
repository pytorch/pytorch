# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import instantiate_parametrized_tests


def sample_function(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.mm(a, b)
    return y.abs() + x


class ComplexTests(torch._dynamo.test_case.TestCase):
    def test_simple(self):
        fn_c = torch.compile(sample_function, fullgraph=True)
        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)

        self.assertEqual(fn_c(a, b, x), sample_function(a, b, x))

    def test_no_complex_inputs(self):
        def f(x, y):
            a = torch.complex(torch.sin(x), torch.cos(y))
            b = torch.complex(torch.cos(x), torch.sin(y))
            return sample_function(a, b, x)

        fn_c = torch.compile(f, fullgraph=True)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertEqual(fn_c(x, y), f(x, y))

    def test_list_out(self):
        def f(a, b, x):
            return [sample_function(a, b, x), sample_function(b, a, x)]

        fn_c = torch.compile(f, fullgraph=True)

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)
        self.assertEqual(fn_c(a, b, x), f(a, b, x))


instantiate_parametrized_tests(ComplexTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
