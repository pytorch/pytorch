# Owner(s): ["oncall: mobile"]

import torch
from test.jit.fixtures_srcs.generate_models import ALL_MODULES
from torch.testing._internal.common_utils import run_tests, TestCase


class TestUpgraderModelGeneration(TestCase):
    def test_all_modules(self):
        for a_module in ALL_MODULES.keys():
            module_name = type(a_module).__name__
            self.assertTrue(
                isinstance(a_module, torch.nn.Module),
                f"The module {module_name} "
                f"is not a torch.nn.module instance. "
                f"Please ensure it's a subclass of torch.nn.module in fixtures_src.py"
                f"and it's registered as an instance in ALL_MODULES in generated_models.py",
            )


if __name__ == "__main__":
    run_tests()
