# Owner(s): ["module: random"]

import torch
import torch._dynamo.testing
import torch.func._random as random
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import run_tests, TestCase


all_floating_dtypes = floating_types_and(torch.half, torch.bfloat16)


class TestPhiloxKeySplit(TestCase):
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
        keys = random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
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
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    def test_error_invalid_num_splits(self, device):
        key = random.key(42, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 0)
        with self.assertRaises(RuntimeError):
            random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 4)


instantiate_device_type_tests(TestPhiloxKeySplit, globals(), only_for=("cuda"))


class TestPhiloxKeyFoldIn(TestCase):
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
        keys = random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)


instantiate_device_type_tests(TestPhiloxKeyFoldIn, globals(), only_for=("cuda"))


class TestPhiloxNormal(TestCase):
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype):
        key = random.key(42, device=device)
        a = random.normal(key, (1000,), dtype=dtype)
        b = random.normal(key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = random.normal(key1, (1000,), dtype=dtype)
        b = random.normal(key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

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
    def test_batched_keys(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = random.normal(keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = random.normal(keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 6).reshape(2, 3, 1, 2)  # (2, 3, 1, 2)
        result = random.normal(keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = random.normal(keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @dtypes(*all_floating_dtypes)
    def test_key_broadcasting_semantics(self, device, dtype):
        key = random.key(42, device=device)

        # Broadcast key dim: size-1 dims replicate, real dims index keys.
        keys = random.split(key, 3).unsqueeze(0).unsqueeze(-2)  # (1, 3, 1, 2)
        result = random.normal(keys, (4, 3, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])
        for j in range(1, 3):
            self.assertNotEqual(result[0][0], result[0][j])

        # All-broadcast key matches unbatched (all dims are generation).
        batched = random.normal(key.reshape(1, 1, 2), (4, 100), dtype=dtype)
        unbatched = random.normal(key, (400,), dtype=dtype)
        self.assertEqual(batched.flatten(), unbatched)

        # Multiple trailing size-1 dims form the generation axis.
        keys = random.split(key, 4).reshape(4, 1, 1, 2)  # (4, 1, 1, 2)
        result = random.normal(keys, (4, 10, 100), dtype=dtype)
        for i in range(4):
            individual = random.normal(keys[i], (10, 100), dtype=dtype)
            self.assertEqual(result[i], individual)
        keys_flat = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        flat = random.normal(keys_flat, (4, 1000), dtype=dtype)
        self.assertEqual(result.reshape(4, 1000), flat)

        # No generation dims: every element gets its own key.
        keys = random.split(key, 12).reshape(4, 3, 2)  # (4, 3, 2)
        result = random.normal(keys, (4, 3), dtype=dtype)
        for i in range(4):
            for j in range(3):
                individual = random.normal(keys[i][j], (1,), dtype=dtype)
                self.assertEqual(result[i][j], individual.squeeze())

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.normal(key, (100,))

    @dtypes(torch.float32, torch.float64)
    def test_offset_shift_consistency(self, device, dtype):
        """Shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = random.normal(key0, (n,), dtype=dtype)
        for elem_offset in range(1, 4):
            offset = elem_offset * outputs_per_elem
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            result = random.normal(key, (n - elem_offset,), dtype=dtype)
            self.assertEqual(result, ref[elem_offset:])

    def test_error_shape_mismatch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 3).unsqueeze(-2)  # (3, 1, 2)
        with self.assertRaises(RuntimeError):
            random.normal(keys, (2, 100))  # batch dim 2 != 3

    @dtypes(torch.float32, torch.float64)
    def test_offset_overflow(self, device, dtype):
        """After wrapping past 2^64, generation continues from offset 0."""
        seed = 42
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        wrap_at = 5
        near_max = (1 << 64) - wrap_at * outputs_per_elem
        key = torch.tensor([seed, near_max], dtype=torch.uint64, device=device)
        result = random.normal(key, (20,), dtype=dtype)
        # First wrap_at elements come from the stream at near_max.
        self.assertEqual(
            result[:wrap_at],
            random.normal(key, (wrap_at,), dtype=dtype),
        )
        # After the wrap, elements come from the stream at offset 0.
        key_zero = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        self.assertEqual(
            result[wrap_at:],
            random.normal(key_zero, (20 - wrap_at,), dtype=dtype),
        )

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.normal(key, (100,))


instantiate_device_type_tests(TestPhiloxNormal, globals(), only_for=("cuda"))


class TestPhiloxUniform(TestCase):
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype):
        key = random.key(42, device=device)
        a = random.uniform(key, (1000,), dtype=dtype)
        b = random.uniform(key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = random.uniform(key1, (1000,), dtype=dtype)
        b = random.uniform(key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_standard_uniform_statistics(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() > 0.0)
        self.assertTrue(result.max().item() <= 1.0)

    @dtypes(*all_floating_dtypes)
    def test_custom_low_high(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), low=2.0, high=5.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    @dtypes(*all_floating_dtypes)
    def test_batched_keys(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = random.uniform(keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = random.uniform(keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 6).reshape(2, 3, 1, 2)  # (2, 3, 1, 2)
        result = random.uniform(keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = random.uniform(keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @dtypes(*all_floating_dtypes)
    def test_key_broadcasting_semantics(self, device, dtype):
        key = random.key(42, device=device)

        # Broadcast key dim: size-1 dims replicate, real dims index keys.
        keys = random.split(key, 3).unsqueeze(0).unsqueeze(-2)  # (1, 3, 1, 2)
        result = random.uniform(keys, (4, 3, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])
        for j in range(1, 3):
            self.assertNotEqual(result[0][0], result[0][j])

        # All-broadcast key matches unbatched.
        batched = random.uniform(key.reshape(1, 1, 2), (4, 100), dtype=dtype)
        unbatched = random.uniform(key, (400,), dtype=dtype)
        self.assertEqual(batched.flatten(), unbatched)

        # No generation dims: every element gets its own key.
        keys = random.split(key, 12).reshape(4, 3, 2)  # (4, 3, 2)
        result = random.uniform(keys, (4, 3), dtype=dtype)
        for i in range(4):
            for j in range(3):
                individual = random.uniform(keys[i][j], (1,), dtype=dtype)
                self.assertEqual(result[i][j], individual.squeeze())

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.uniform(key, (100,))

    def test_error_shape_mismatch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 3).unsqueeze(-2)  # (3, 1, 2)
        with self.assertRaises(RuntimeError):
            random.uniform(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.uniform(key, (100,))

    @dtypes(torch.float32, torch.float64)
    def test_offset_shift_consistency(self, device, dtype):
        """Shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = random.uniform(key0, (n,), dtype=dtype)
        for elem_offset in range(1, 4):
            offset = elem_offset * outputs_per_elem
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            result = random.uniform(key, (n - elem_offset,), dtype=dtype)
            self.assertEqual(result, ref[elem_offset:])

    @dtypes(torch.float32, torch.float64)
    def test_offset_overflow(self, device, dtype):
        """After wrapping past 2^64, generation continues from offset 0."""
        seed = 42
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        wrap_at = 5
        near_max = (1 << 64) - wrap_at * outputs_per_elem
        key = torch.tensor([seed, near_max], dtype=torch.uint64, device=device)
        result = random.uniform(key, (20,), dtype=dtype)
        # First wrap_at elements come from the stream at near_max.
        self.assertEqual(
            result[:wrap_at],
            random.uniform(key, (wrap_at,), dtype=dtype),
        )
        # After the wrap, elements come from the stream at offset 0.
        key_zero = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        self.assertEqual(
            result[wrap_at:],
            random.uniform(key_zero, (20 - wrap_at,), dtype=dtype),
        )


instantiate_device_type_tests(TestPhiloxUniform, globals(), only_for=("cuda"))


class TestPhiloxCompile(TestCase):
    def test_uniform_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.uniform(key, (100,))

        self.assertEqual(f(key), random.uniform(key, (100,)))

    def test_split_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.split(key, 4)

        self.assertEqual(f(key), random.split(key, 4))

    def test_fold_in_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.fold_in(key, 7)

        self.assertEqual(f(key), random.fold_in(key, 7))

    def test_normal_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.normal(key, (100,))

        self.assertEqual(f(key), random.normal(key, (100,)))

    def test_batched_normal_aot_eager(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            return random.normal(keys, (4, 50))

        self.assertEqual(f(keys), random.normal(keys, (4, 50)))

    def test_split_then_normal_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            keys = random.split(key, 4).unsqueeze(-2)
            return random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), random.normal(random.split(key, 4).unsqueeze(-2), (4, 100))
        )

    def test_fold_in_then_uniform_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            k = random.fold_in(key, 3)
            return random.uniform(k, (100,))

        self.assertEqual(f(key), random.uniform(random.fold_in(key, 3), (100,)))


instantiate_device_type_tests(TestPhiloxCompile, globals(), only_for=("cuda"))


if __name__ == "__main__":
    run_tests()
