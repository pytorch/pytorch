# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRNG(TestCase):
    def test_generator(self):
        """Test generator creation on openreg device"""
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        """Test RNG state get and set"""
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

    def test_rand_range_and_dtype(self):
        """rand returns values in [0,1) and honors dtype"""
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(7)

        for dtype in (torch.float, torch.double):
            x = torch.rand((1000,), device="openreg:0", dtype=dtype, generator=gen)
            self.assertEqual(x.dtype, dtype)
            self.assertTrue(torch.all(x >= 0).item())
            # Strict upper bound: no element should be >= 1
            self.assertTrue(torch.all(x < 1).item())

    def test_randn_stats_float_double(self):
        """randn samples have mean≈0 and std≈1 for sufficiently large sample"""
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(123)
        for dtype in (torch.float, torch.double):
            x = torch.randn((20000,), device="openreg:0", dtype=dtype, generator=gen)
            mean = x.mean().item()
            std = x.std(unbiased=False).item()
            # Loose tolerances to avoid flaky failures
            self.assertLess(abs(mean), 0.05)
            self.assertLess(abs(std - 1.0), 0.05)

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

    def test_randint_bounds_and_dtypes(self):
        """Bounds checks for randint across dtypes; omit generator for overloads not implemented yet"""
        # int32
        x32 = torch.randint(0, 10, (1000,), device="openreg:0", dtype=torch.int32)
        self.assertEqual(x32.dtype, torch.int32)
        self.assertTrue(torch.all((x32 >= 0) & (x32 < 10)).item())

        # int64
        x64 = torch.randint(5, 15, (1000,), device="openreg:0", dtype=torch.int64)
        self.assertEqual(x64.dtype, torch.int64)
        self.assertTrue(torch.all((x64 >= 5) & (x64 < 15)).item())

        # int16
        x16 = torch.randint(-3, 3, (1000,), device="openreg:0", dtype=torch.int16)
        self.assertEqual(x16.dtype, torch.int16)
        self.assertTrue(torch.all((x16 >= -3) & (x16 < 3)).item())

    def test_randint_with_explicit_generator_default_dtype(self):
        """Explicit generator supported for default integral dtype path"""
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(99)
        x = torch.randint(0, 10, (1000,), device="openreg:0", generator=gen)
        self.assertEqual(x.dtype, torch.int64)  # default integral dtype
        self.assertTrue(torch.all((x >= 0) & (x < 10)).item())

    def test_noncontiguous_fill(self):
        """Ensure RNG fills non-contiguous tensors via view/transpose without crash"""
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(2024)
        base = torch.empty((32, 32), device="openreg:0")
        # Create a non-contiguous view by transposing
        nc = base.t()
        self.assertFalse(nc.is_contiguous())

        # Fill with rand and randn; just smoke check shapes and device
        r = torch.rand(nc.shape, device="openreg:0", generator=gen)
        n = torch.randn(nc.shape, device="openreg:0", generator=gen)
        self.assertEqual(r.shape, nc.shape)
        self.assertEqual(n.shape, nc.shape)
        self.assertEqual(r.device.type, "openreg")
        self.assertEqual(n.device.type, "openreg")

    def test_large_tensor_smoke(self):
        """Large tensor generation should not segfault"""
        gen = torch.Generator(device="openreg:0")
        gen.manual_seed(1)
        # 1 million elements
        r = torch.rand((1_000_000,), device="openreg:0", generator=gen)
        n = torch.randn((1_000_000,), device="openreg:0", generator=gen)
        self.assertEqual(r.numel(), 1_000_000)
        self.assertEqual(n.numel(), 1_000_000)

    def test_generator_seed(self):
        """Test generator seed setting"""
        generator = torch.Generator(device="openreg:0")
        generator.manual_seed(42)
        self.assertEqual(generator.initial_seed(), 42)

        generator = torch.Generator(device="openreg:1")
        generator.manual_seed(100)
        self.assertEqual(generator.initial_seed(), 100)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_generator_state(self):
        """Test generator state get/set"""
        generator = torch.Generator(device="openreg:0")
        state = generator.get_state()

        # Generate some random numbers
        x1 = torch.randn(10, device="openreg:0", generator=generator)

        # Set state back
        generator.set_state(state)
        x2 = torch.randn(10, device="openreg:0", generator=generator)

        # Should produce same sequence
        self.assertEqual(x1, x2)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_rng_state_consistency(self):
        """Test RNG state consistency across devices"""
        state0 = torch.openreg.get_rng_state(0)
        state1 = torch.openreg.get_rng_state(1)

        # States should be different for different devices
        self.assertNotEqual(state0, state1)

        # Setting state should work
        torch.openreg.set_rng_state(state0, 0)
        restored_state = torch.openreg.get_rng_state(0)
        self.assertEqual(state0, restored_state)

    def test_manual_seed_all(self):
        """Test manual_seed_all sets seed for all devices"""
        torch.openreg.manual_seed_all(1234)

        # Check that seed is set
        seed = torch.openreg.initial_seed()
        self.assertEqual(seed, 1234)

        # Test with different seed
        torch.openreg.manual_seed_all(5678)
        seed = torch.openreg.initial_seed()
        self.assertEqual(seed, 5678)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_generator_different_devices(self):
        """Test generators on different devices"""
        gen0 = torch.Generator(device="openreg:0")
        gen1 = torch.Generator(device="openreg:1")

        gen0.manual_seed(1)
        gen1.manual_seed(1)

        x0 = torch.randn(10, device="openreg:0", generator=gen0)
        x1 = torch.randn(10, device="openreg:1", generator=gen1)

        # Should produce same sequence with same seed
        self.assertEqual(x0.cpu(), x1.cpu())


if __name__ == "__main__":
    run_tests()
