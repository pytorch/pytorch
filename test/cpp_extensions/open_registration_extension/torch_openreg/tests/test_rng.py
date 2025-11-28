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

    def test_rand_with_explicit_generator(self):
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(1234)
        a = torch.rand((4, 5), device="openreg:0", generator=gen)
        # Reset seed and ensure determinism
        gen.manual_seed(1234)
        b = torch.rand((4, 5), device="openreg:0", generator=gen)
        self.assertEqual(a, b)

    def test_randn_with_explicit_generator(self):
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(5678)
        a = torch.randn((3, 3), device="openreg:0", generator=gen)
        gen.manual_seed(5678)
        b = torch.randn((3, 3), device="openreg:0", generator=gen)
        self.assertEqual(a, b)

    def test_generator_device_mismatch_raises(self):
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(1)
        with self.assertRaisesRegex(
            RuntimeError, "Generator device index.*does not match"
        ):
            torch.rand((2,), device="openreg:1", generator=gen)

    def test_state_save_restore_determinism(self):
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(42)
        state = gen.get_state()
        x1 = torch.rand((5,), device="openreg:0", generator=gen)
        # restore state and re-sample yields same values
        gen.set_state(state)
        x2 = torch.rand((5,), device="openreg:0", generator=gen)
        self.assertEqual(x1, x2)

    def test_empty_tensors(self):
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(123)
        r = torch.rand((0,), device="openreg:0", generator=gen)
        n = torch.randn((0,), device="openreg:0", generator=gen)
        self.assertEqual(r.numel(), 0)
        self.assertEqual(n.numel(), 0)


if __name__ == "__main__":
    run_tests()
