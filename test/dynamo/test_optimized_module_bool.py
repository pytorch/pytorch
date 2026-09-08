import torch
from torch.testing._internal.common_utils import TestCase


class TestOptimizedModuleBool(TestCase):
    def test_bool_without_len(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        compiled = torch.compile(Model(), backend="eager")
        self.assertTrue(bool(compiled))


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
