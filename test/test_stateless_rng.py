# Owner(s): ["module: random"]

import torch
import torch.func._random as random
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


all_floating_dtypes = floating_types_and(torch.half, torch.bfloat16)


class TestStatelessRNGKey(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = random.key(42, device=device)
        self.assertEqual(key.shape, (2,))
        self.assertEqual(key.dtype, torch.uint64)
        self.assertEqual(key.device, torch.device(device))

    def test_different_seeds(self, device):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        self.assertNotEqual(key1, key2)

    def test_determinism(self, device):
        key1 = random.key(42, device=device)
        key2 = random.key(42, device=device)
        self.assertEqual(key1, key2)

    def test_error_unsupported_impl(self, device):
        with self.assertRaisesRegex(
            NotImplementedError, "does not support PRNG impl 'unsupported'"
        ):
            random.key(42, impl="unsupported", device=device)


class TestStatelessRNGKeySplit(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 4)
        self.assertEqual(splits.shape, (4, 2))
        self.assertEqual(splits.dtype, torch.uint64)
        self.assertEqual(splits.device, key.device)

    def test_single_split(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 1)
        self.assertEqual(splits.shape, (1, 2))

    def test_large_num_splits(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 10000)
        self.assertEqual(splits.shape, (10000, 2))

    def test_determinism(self, device):
        key = random.key(42, device=device)
        splits1 = random.split(key, 8)
        splits2 = random.split(key, 8)
        self.assertEqual(splits1, splits2)

    def test_all_keys_unique(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 100)
        unique_keys = torch.unique(splits, dim=0)
        self.assertEqual(unique_keys.shape[0], 100)

    def test_different_seeds_produce_different_outputs(self, device):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_different_offsets_produce_different_outputs(self, device):
        key1 = random.key(42, device=device)
        key2 = random.fold_in(key1, 1)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_offset_zero_vs_one_produce_different_splits(self, device):
        key1 = random.key(42, device=device)
        key2 = torch.tensor([42, 1], dtype=torch.uint64, device=device)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_batched(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        num_splits = 3
        batched = random.split(keys, num_splits)  # (3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 4, 2))
        for k in range(4):
            individual = random.split(keys[k], num_splits)
            for s in range(num_splits):
                self.assertEqual(batched[s][k], individual[s])

    def test_multi_batch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)
        num_splits = 5
        batched = random.split(keys, num_splits)  # (5, 3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.split(keys[i][j], num_splits)
                for s in range(num_splits):
                    self.assertEqual(batched[s][i][j], individual[s])

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.split(key, 4)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            random.split(key, 4)

    def test_error_invalid_num_splits(self, device):
        key = random.key(42, device=device)
        with self.assertRaisesRegex(RuntimeError, "num_splits must be positive"):
            random.split(key, 0)
        with self.assertRaisesRegex(RuntimeError, "num_splits must be positive"):
            random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.split(key, 4)

    def test_offset_overflow(self, device):
        near_max = (1 << 64) - 1
        key = torch.tensor([42, near_max], dtype=torch.uint64, device=device)
        splits = random.split(key, 3)
        # split_idx=1 wraps offset to 0, split_idx=2 wraps to 1
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        self.assertEqual(splits[1], random.fold_in(key0, 0))
        self.assertEqual(splits[2], random.fold_in(key0, 1))

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.split(key_cpu, 100),
            random.split(key_cuda, 100).cpu(),
        )


class TestStatelessRNGKeyFoldIn(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = random.key(42, device=device)
        result = random.fold_in(key, 7)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.uint64)
        self.assertEqual(result.device, key.device)

    def test_determinism(self, device):
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 7)
        result2 = random.fold_in(key, 7)
        self.assertEqual(result1, result2)

    def test_fold_in_produces_new_key_for_zero_data(self, device):
        key = random.key(42, device=device)
        folded = random.fold_in(key, 0)
        self.assertNotEqual(folded, key)

    def test_different_data_produces_different_outputs(self, device):
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 0)
        result2 = random.fold_in(key, 1)
        self.assertNotEqual(result1, result2)

    def test_consistency_with_split(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 10)
        for i in range(10):
            folded = random.fold_in(key, i)
            self.assertEqual(folded, splits[i])

    def test_batched(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (4, 2)
        self.assertEqual(batched.shape, (4, 2))
        for k in range(4):
            individual = random.fold_in(keys[k], data)
            self.assertEqual(batched[k], individual)

    def test_multi_batch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.fold_in(key, 0)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.fold_in(key, 0)

    def test_offset_overflow(self, device):
        near_max = (1 << 64) - 1
        key = torch.tensor([42, near_max], dtype=torch.uint64, device=device)
        # fold_in(data=1) wraps offset to 0, so it should match fold_in on
        # a key with offset=0 and data=0.
        result = random.fold_in(key, 1)
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        self.assertEqual(result, random.fold_in(key0, 0))

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.fold_in(key_cpu, 7),
            random.fold_in(key_cuda, 7).cpu(),
        )


class TestStatelessRNGDistribution(TestCase):
    def _gen(self, gen_fn_name, *args, **kwargs):
        return getattr(random, gen_fn_name)(*args, **kwargs)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        result = self._gen(gen_fn_name, key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        a = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        b = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype, gen_fn_name):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = self._gen(gen_fn_name, key1, (1000,), dtype=dtype)
        b = self._gen(gen_fn_name, key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_batched_keys(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_batched_keys_large(self, device, dtype, gen_fn_name):
        # Large event_numel to exercise the multi-key tiled kernel path.
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 10000), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (10000,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        keys = random.split(key, 6).view(2, 3, 1, 2)
        result = self._gen(gen_fn_name, keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = self._gen(gen_fn_name, keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_key_broadcasting_semantics(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)

        # Broadcast key dim: size-1 dims replicate, other dims index keys.
        keys = random.split(key, 3).unsqueeze(0).unsqueeze(-2)  # (1, 3, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 3, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])
        for j in range(1, 3):
            self.assertNotEqual(result[0][0], result[0][j])

        # All-broadcast key matches unbatched.
        batched = self._gen(gen_fn_name, key.view(1, 1, 2), (4, 100), dtype=dtype)
        unbatched = self._gen(gen_fn_name, key, (400,), dtype=dtype)
        self.assertEqual(batched.flatten(), unbatched)

        # Multiple trailing size-1 dims to broadcast over.
        keys = random.split(key, 4).view(4, 1, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 10, 100), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (10, 100), dtype=dtype)
            self.assertEqual(result[i], individual)
        keys_flat = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        flat = self._gen(gen_fn_name, keys_flat, (4, 1000), dtype=dtype)
        self.assertEqual(result.view(4, 1000), flat)

        # No generation dims: every element gets its own key.
        keys = random.split(key, 12).view(4, 3, 2)
        result = self._gen(gen_fn_name, keys, (4, 3), dtype=dtype)
        for i in range(4):
            for j in range(3):
                individual = self._gen(gen_fn_name, keys[i][j], (1,), dtype=dtype)
                self.assertEqual(result[i][j], individual.squeeze())

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_error_wrong_key_dtype(self, device, gen_fn_name):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            self._gen(gen_fn_name, key, (100,))

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_error_key_shape(self, device, gen_fn_name):
        key = random.key(42, device=device)
        # Last dim must be 2.
        bad_key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(2,\) or \(\*batch, 2\)"
        ):
            self._gen(gen_fn_name, bad_key, (100,))
        # Key batch ndim must equal output ndim (too few).
        with self.assertRaisesRegex(
            RuntimeError, "batched key must have ndim == output ndim \\+ 1"
        ):
            self._gen(gen_fn_name, random.split(key, 3), (3, 4, 100))
        # Key batch ndim must equal output ndim (too many).
        with self.assertRaisesRegex(
            RuntimeError, "batched key must have ndim == output ndim \\+ 1"
        ):
            self._gen(gen_fn_name, random.split(key, 3).view(3, 1, 1, 2), (3, 100))
        # Key batch dims must be broadcastable with output.
        with self.assertRaisesRegex(
            RuntimeError, "is not broadcastable with output shape"
        ):
            self._gen(gen_fn_name, random.split(key, 5).unsqueeze(-2), (3, 100))

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_offset_shift_consistency(self, device, dtype, gen_fn_name):
        seed = 42
        n = 100
        key0 = random.key(seed, device=device)
        ref = self._gen(gen_fn_name, key0, (n,), dtype=dtype)

        # as a key's offset shifts, we expect the stream to shift by
        # the number of elements per philox call (2 for double; 4 otherwise)
        for offset in range(1, 4):
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            elems_per_call = 2 if dtype == torch.float64 else 4
            expected_shift = offset * elems_per_call
            result = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(ref[expected_shift:], result[:-expected_shift])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_offset_overflow(self, device, dtype, gen_fn_name):
        seed = 42
        n = 100
        last_offset_before_wrap = (1 << 64) - 1
        key = torch.tensor(
            [seed, last_offset_before_wrap], dtype=torch.uint64, device=device
        )
        result = self._gen(gen_fn_name, key, (n,), dtype=dtype)

        # ensure offset wraps around to 0 by comparing with 0-offset key results
        key0 = random.key(seed, device=device)
        result0 = self._gen(gen_fn_name, key0, (n,), dtype=dtype)
        elems_per_call = 2 if dtype == torch.float64 else 4
        self.assertEqual(result[elems_per_call:], result0[:-elems_per_call])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_small_output_sizes(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        large = self._gen(gen_fn_name, key, (100,), dtype=dtype)
        for n in [0, 1, 2, 3, 4, 5, 7]:
            result = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(result.shape, (n,))
            # Determinism.
            result2 = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(result, result2)
            # Prefix consistency: first n elements of a larger output.
            if n > 0:
                self.assertEqual(result, large[:n])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @parametrize("layout", ["contiguous", "noncontiguous", "unaligned"])
    @dtypes(*all_floating_dtypes)
    def test_inplace(self, device, dtype, gen_fn_name, layout):
        key = random.key(42, device=device)
        if layout == "contiguous":
            result = torch.empty(1000, dtype=dtype, device=device)
        elif layout == "noncontiguous":
            result = torch.empty(2000, dtype=dtype, device=device)[::2]
        else:
            # Contiguous but data pointer is not aligned to vectorized write width.
            result = torch.empty(1001, dtype=dtype, device=device)[1:]
        inplace_fn = getattr(random, gen_fn_name + "_")
        out = inplace_fn(key, result)
        self.assertIs(out, result)
        functional = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        self.assertEqual(result, functional)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_empty_output(self, device, dtype, gen_fn_name):
        key = random.key(42, device=device)
        result = self._gen(gen_fn_name, key, (0,), dtype=dtype)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, dtype)
        result = self._gen(gen_fn_name, key, (3, 0), dtype=dtype)
        self.assertEqual(result.shape, (3, 0))
        self.assertEqual(result.dtype, dtype)

    # Distribution-specific tests

    @dtypes(*all_floating_dtypes)
    def test_standard_normal_statistics(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item()) < 0.05)
        self.assertTrue(abs(result.std().item() - 1.0) < 0.05)

    @dtypes(*all_floating_dtypes)
    def test_custom_mean_std(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), mean=5.0, std=2.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)
        self.assertTrue(abs(result.std().item() - 2.0) < 0.1)

    @dtypes(*all_floating_dtypes)
    def test_standard_uniform_statistics(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() >= 0.0)
        self.assertTrue(result.max().item() < 1.0)

    @dtypes(*all_floating_dtypes)
    def test_custom_low_high(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), low=2.0, high=5.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    @dtypes(*all_floating_dtypes)
    @onlyCUDA
    def test_cross_device_uniform_consistency(self, device, dtype):
        key_cpu = random.fold_in(random.key(42), 7)
        key_cuda = random.fold_in(random.key(42, device=device), 7)
        # Uniform generation uses no transcendentals, so results must be bitwise identical.
        self.assertEqual(
            self._gen("uniform", key_cpu, (1000,), dtype=dtype),
            self._gen("uniform", key_cuda, (1000,), dtype=dtype).cpu(),
            atol=0,
            rtol=0,
        )

    @dtypes(*all_floating_dtypes)
    @onlyCUDA
    def test_cross_device_normal_consistency(self, device, dtype):
        key_cpu = random.fold_in(random.key(42), 7)
        key_cuda = random.fold_in(random.key(42, device=device), 7)
        # Normal generation uses Box-Muller (log, sin, cos), and CUDA uses fast-math
        # intrinsics (__logf, __sincosf) that differ slightly from CPU std::log / std::sin /
        # std::cos. Results are approximately but not bitwise equal. assertEqual() by default
        # allows for some tolerance in the comparisons.
        self.assertEqual(
            self._gen("normal", key_cpu, (1000,), dtype=dtype),
            self._gen("normal", key_cuda, (1000,), dtype=dtype).cpu(),
        )

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_portable_default(self, device, gen_fn_name):
        key = random.key(42, device=device)
        a = self._gen(gen_fn_name, key, (1000,))
        b = self._gen(gen_fn_name, key, (1000,), portable=True)
        self.assertEqual(a, b)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_portable_false_deterministic(self, device, gen_fn_name):
        key = random.key(42, device=device)
        a = self._gen(gen_fn_name, key, (1000,), portable=False)
        b = self._gen(gen_fn_name, key, (1000,), portable=False)
        self.assertEqual(a, b)
        self.assertEqual(a.shape, (1000,))


class TestStatelessRNGCompile(TestCase):
    def test_split_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.split(key, 4)

        self.assertEqual(f(key), random.split(key, 4))

    def test_fold_in_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.fold_in(key, 7)

        self.assertEqual(f(key), random.fold_in(key, 7))

    def test_uniform_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.uniform(key, (100,))

        self.assertEqual(f(key), random.uniform(key, (100,)))

    def test_normal_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.normal(key, (100,))

        self.assertEqual(f(key), random.normal(key, (100,)))

    def test_batched_normal_fullgraph(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            return random.normal(keys, (4, 50))

        self.assertEqual(f(keys), random.normal(keys, (4, 50)))

    def test_split_then_normal_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            keys = random.split(key, 4).unsqueeze(-2)
            return random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), random.normal(random.split(key, 4).unsqueeze(-2), (4, 100))
        )

    def test_fold_in_then_uniform_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            k = random.fold_in(key, 3)
            return random.uniform(k, (100,))

        self.assertEqual(f(key), random.uniform(random.fold_in(key, 3), (100,)))


instantiate_device_type_tests(TestStatelessRNGKey, globals(), only_for=("cpu", "cuda"))
instantiate_device_type_tests(
    TestStatelessRNGKeySplit, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGKeyFoldIn, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGDistribution, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGCompile, globals(), only_for=("cpu", "cuda")
)


class TestUnbind(TestCase):
    def test_1d_shape(self, device):
        k = random.key(42, device=device)
        keys = random.unbind(k, (120,), (10,))
        self.assertEqual(keys.shape, (10, 2))
        self.assertEqual(keys.dtype, torch.uint64)

    def test_1d_uniform_reconstruction(self, device):
        k = random.key(42, device=device)
        num_tiles = 10
        keys = random.unbind(k, (120,), (num_tiles,))
        full = random.uniform(k, (120,))
        tile_size = 120 // num_tiles
        tiled = torch.cat(
            [random.uniform(keys[i], (tile_size,)) for i in range(num_tiles)]
        )
        self.assertEqual(full, tiled)

    def test_1d_normal_reconstruction(self, device):
        k = random.key(42, device=device)
        num_tiles = 10
        keys = random.unbind(k, (120,), (num_tiles,))
        full = random.normal(k, (120,))
        tile_size = 120 // num_tiles
        tiled = torch.cat(
            [random.normal(keys[i], (tile_size,)) for i in range(num_tiles)]
        )
        self.assertEqual(full, tiled)

    def test_1d_determinism(self, device):
        k = random.key(42, device=device)
        keys1 = random.unbind(k, (120,), (10,))
        keys2 = random.unbind(k, (120,), (10,))
        self.assertEqual(keys1, keys2)

    def test_2d_shape(self, device):
        k = random.key(42, device=device)
        splits = (10, 10)
        keys = random.unbind(k, (100, 200), splits)
        # splits=(10,10), tile_shape=(10,20), per-tile rows=10
        # -> shape (*splits, tile_shape[0], 2) = (10, 10, 10, 2)
        self.assertEqual(keys.shape, (10, 10, 10, 2))

    def test_2d_uniform_tile(self, device):
        k = random.key(42, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = random.unbind(k, shape, splits)
        full = random.uniform(k, shape)
        tile = random.uniform(keys[0, 0].unsqueeze(-2), tile_shape)
        self.assertEqual(tile, full[0:10, 0:20])

    def test_2d_uniform_reconstruction(self, device):
        k = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = random.unbind(k, shape, splits)
        full = random.uniform(k, shape)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(random.uniform(keys[r, c].unsqueeze(-2), tile_shape))
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_normal_reconstruction(self, device):
        k = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = random.unbind(k, shape, splits)
        full = random.normal(k, shape)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(random.normal(keys[r, c].unsqueeze(-2), tile_shape))
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_arbitrary_tile(self, device):
        """Verify a non-corner tile matches the full generation."""
        k = random.key(123, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = random.unbind(k, shape, splits)
        full = random.uniform(k, shape)
        for tr, tc in [(3, 7), (9, 9), (0, 5)]:
            tile = random.uniform(keys[tr, tc].unsqueeze(-2), tile_shape)
            expected = full[
                tr * tile_shape[0] : (tr + 1) * tile_shape[0],
                tc * tile_shape[1] : (tc + 1) * tile_shape[1],
            ]
            self.assertEqual(tile, expected)

    def test_3d_shape(self, device):
        k = random.key(42, device=device)
        splits = (3, 4, 4)
        keys = random.unbind(k, (12, 20, 32), splits)
        # tile_shape = (4, 5, 8), per-tile rows = (4, 5)
        # -> shape (*splits, 4, 5, 2) = (3, 4, 4, 4, 5, 2)
        self.assertEqual(keys.shape, (3, 4, 4, 4, 5, 2))

    def test_3d_uniform_reconstruction(self, device):
        k = random.key(77, device=device)
        shape = (12, 20, 32)
        splits = (3, 4, 4)
        tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
        keys = random.unbind(k, shape, splits)
        full = random.uniform(k, shape)
        reconstructed = torch.empty_like(full)
        for t0 in range(splits[0]):
            for t1 in range(splits[1]):
                for t2 in range(splits[2]):
                    tile = random.uniform(keys[t0, t1, t2].unsqueeze(-2), tile_shape)
                    reconstructed[
                        t0 * tile_shape[0] : (t0 + 1) * tile_shape[0],
                        t1 * tile_shape[1] : (t1 + 1) * tile_shape[1],
                        t2 * tile_shape[2] : (t2 + 1) * tile_shape[2],
                    ] = tile
        self.assertEqual(full, reconstructed)

    def test_2d_row_only_split(self, device):
        """Splitting only along rows (single column tile) should work."""
        k = random.key(42, device=device)
        splits = (10, 1)
        keys = random.unbind(k, (100, 200), splits)
        tile_shape = (10, 200)
        # shape: (*splits, tile_shape[0], 2) = (10, 1, 10, 2)
        self.assertEqual(keys.shape, (10, 1, 10, 2))
        full = random.uniform(k, (100, 200))
        tiles = [
            random.uniform(keys[i, 0].unsqueeze(-2), tile_shape) for i in range(10)
        ]
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    @dtypes(torch.float32, torch.float64)
    def test_1d_near_max_offset(self, device, dtype):
        """unbind reconstruction holds when tile offsets wrap past 2^64."""
        seed = 42
        near_max_offset = (1 << 64) - 48
        k = torch.tensor([seed, near_max_offset], dtype=torch.uint64, device=device)
        shape = (120,)
        num_tiles = 10
        tile_size = shape[0] // num_tiles
        keys = random.unbind(k, shape, (num_tiles,), dtype=dtype)
        full_uniform = random.uniform(k, shape, dtype=dtype)
        tiled_uniform = torch.cat(
            [
                random.uniform(keys[i], (tile_size,), dtype=dtype)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full_uniform, tiled_uniform)
        full_normal = random.normal(k, shape, dtype=dtype)
        tiled_normal = torch.cat(
            [
                random.normal(keys[i], (tile_size,), dtype=dtype)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full_normal, tiled_normal)

    def test_error_uneven_split(self, device):
        k = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            random.unbind(k, (100,), (3,))
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            random.unbind(k, (100, 200), (10, 3))

    def test_error_mismatched_lengths(self, device):
        k = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "same length"):
            random.unbind(k, (100, 200), (10,))

    @onlyCUDA
    def test_cross_device_consistency_1d(self, device):
        k_cpu = random.key(42)
        k_cuda = random.key(42, device=device)
        keys_cpu = random.unbind(k_cpu, (120,), (10,))
        keys_cuda = random.unbind(k_cuda, (120,), (10,))
        self.assertEqual(keys_cpu, keys_cuda.cpu())

    @onlyCUDA
    def test_cross_device_consistency_2d(self, device):
        k_cpu = random.key(42)
        k_cuda = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys_cpu = random.unbind(k_cpu, shape, splits)
        keys_cuda = random.unbind(k_cuda, shape, splits)
        tile_cpu = random.uniform(keys_cpu[2, 1].unsqueeze(-2), tile_shape)
        tile_cuda = random.uniform(keys_cuda[2, 1].unsqueeze(-2), tile_shape)
        self.assertEqual(tile_cpu, tile_cuda.cpu())


instantiate_device_type_tests(TestUnbind, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
