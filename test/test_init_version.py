import torch
import torch.nn as nn
from torch.nn.init import init_version, _torch_version, use_init_version

from torch.testing._internal.common_utils import TestCase, run_tests

class InitVersionTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_linear(self):
        # >= 1.7.0
        with use_init_version('1.7.0') as v:
            l = nn.Linear(2, 3)

        x = l.bias.data
        y = torch.zeros(3, dtype=torch.float32)
        self.assertEqual(x, y)

        # < 1.7.0
        torch.manual_seed(4)
        with use_init_version('1.6.1') as v:
            l = nn.Linear(2, 3)

        x = l.bias.mean().item()
        y = 0.48299
        self.assertEqual(x, y)

    def test_init_version_remains_same(self):
        # _init_version should remain same before and after using init_version
        before = init_version
        with use_init_version() as v:
            l = nn.Linear(2, 3)
        after = init_version

        x = torch.tensor(before)
        y = torch.tensor(after)
        self.assertEqual(x, y)

    def test_version_greater_than_torch(self):
        # cannot pass version greater than torch.__version__ in use_init_version
        current_torch_version = _torch_version
        new_torch_version = (current_torch_version[0], current_torch_version[1] + 1, current_torch_version[2])

        with self.assertRaises(ValueError):
            with use_init_version(new_torch_version) as version:
                l = nn.Linear(2, 3)

if __name__ == "__main__":
    run_tests()
