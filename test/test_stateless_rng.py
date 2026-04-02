# Owner(s): ["module: random"]

import torch
import torch.func._random as random
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


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

    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaisesRegex(
            NotImplementedError,
            "Could not run .* with arguments from the 'CPU' backend",
        ):
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

    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaisesRegex(
            NotImplementedError,
            "Could not run .* with arguments from the 'CPU' backend",
        ):
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


instantiate_device_type_tests(TestStatelessRNGKey, globals(), only_for=("cuda",))
instantiate_device_type_tests(TestStatelessRNGKeySplit, globals(), only_for=("cuda",))
instantiate_device_type_tests(TestStatelessRNGKeyFoldIn, globals(), only_for=("cuda",))
instantiate_device_type_tests(TestStatelessRNGCompile, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
