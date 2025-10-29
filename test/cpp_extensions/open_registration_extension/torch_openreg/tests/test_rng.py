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

        states = torch.openreg.get_rng_state_all()
        torch.openreg.set_rng_state_all(states)

    def test_manual_seed(self):
        torch.openreg.manual_seed_all(42)
        self.assertEqual(torch.openreg.initial_seed(), 42)

    def test_seed(self):
        torch.openreg.seed_all()
        seed1 = torch.openreg.initial_seed()
        torch.openreg.seed_all()
        seed2 = torch.openreg.initial_seed()
        self.assertNotEqual(seed1, seed2)

    # LITERALINCLUDE START: OPENREG GENERATOR TEST EXAMPLES
    def test_create_generator_and_seed(self):
        generator = torch.Generator(device="openreg:0")
        generator.manual_seed(42)
        self.assertEqual(generator.initial_seed(), 42)

    # LITERALINCLUDE END: OPENREG GENERATOR TEST EXAMPLES


if __name__ == "__main__":
    run_tests()
