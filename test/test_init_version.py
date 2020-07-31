import torch
import torch.nn as nn
from torch.nn.init import parse_version, init_version, use_init_version

from torch.testing._internal.common_utils import TestCase, run_tests

class InitVersionTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_pre_1_7_0_linear(self):
        # uniform init is used
        torch.manual_seed(4)
        with use_init_version('1.6.1') as v:
            l = nn.Linear(2, 3)

        x = l.bias.mean().item()
        y = 0.48299
        self.assertEqual(x, y)

    def test_1_7_0_linear(self):
        # zero bias is used
        with use_init_version('1.7.0') as v:
            l = nn.Linear(2, 3)

        x = l.bias.data
        y = torch.zeros(3, dtype=torch.float32)
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
        # cannot pass version greater than torch.__version__ in init_version
        current_torch_version = parse_version(torch.__version__)
        new_torch_version = (current_torch_version[0], current_torch_version[1] + 1, current_torch_version[2])

        try:
            with use_init_version(new_torch_version) as v:
                l = nn.Linear(2, 3)
            raise ValueError(
                f'Cannot pass version number {new_torch_version} greater than torch.__version__ ',
                f'{current_torch_version} in nn.init.init_version'
            )
        except ValueError as e:
            error_message = e.args[0]
            expected_error_message = f'version {new_torch_version} should be less than torch version {current_torch_version}'
            if error_message != expected_error_message:
                raise ValueError(
                    f'Cannot pass version number {new_torch_version} greater than torch.__version__ ',
                    f'{current_torch_version} in nn.init.init_version'
                )

if __name__ == "__main__":
    run_tests()
