import torch
from torch._dynamo import config
from torch.testing._internal.common_utils import TestCase, run_tests


class TestIndexPropagationDtype(TestCase):
    @config.patch(capture_scalar_outputs=True)
    def test_symbolic_full_bool_cast(self):
        def fn(x):
            return torch.full((2,), x.item(), dtype=torch.bool).sum()

        x = torch.tensor(3)
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
