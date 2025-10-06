# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRNG(TestCase):
    def test_generator(self):
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        state = torch.openreg.get_rng_state(0)
        torch.openreg.set_rng_state(state, 0)

    def test_manual_seed(self):
        torch.openreg.manual_seed_all(2024)
        self.assertEqual(torch.openreg.initial_seed(), 2024)


if __name__ == "__main__":
    run_tests()
