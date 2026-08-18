# Owner(s): ["module: random"]

import torch
import torch.func._random as random
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_TRITON


all_floating_dtypes = floating_types_and(torch.half, torch.bfloat16)
all_int_dtypes = [
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


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

    @parametrize("batched", [False, True])
    @onlyAccelerator
    def test_cross_device_consistency(self, device, batched):
        key_cpu = random.key(42)
        key_dev = random.key(42, device=device)
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(key_cpu, 4)  # (4, 2)
            key_dev = random.split(key_dev, 4)
        self.assertEqual(
            random.split(key_cpu, 8),
            random.split(key_dev, 8).cpu(),
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

    def test_data_above_int64_max(self, device):
        # data is interpreted as uint64; values above int64 max must be accepted.
        data = (1 << 64) - 1
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        key_shifted = torch.tensor([42, data], dtype=torch.uint64, device=device)
        self.assertEqual(random.fold_in(key0, data), random.fold_in(key_shifted, 0))

    @parametrize(
        "data",
        [
            subtest(-1, name="neg_one"),
            subtest(-12345, name="mid"),
            subtest(-(1 << 63), name="int64_min"),
        ],
    )
    def test_data_negative_reinterpreted_as_uint64(self, device, data):
        # Negative data is remapped per the docstring: 2**64 + data.
        key = random.key(42, device=device)
        expected = random.fold_in(key, (1 << 64) + data)
        self.assertEqual(random.fold_in(key, data), expected)

    def test_error_invalid_data(self, device):
        key = random.key(42, device=device)
        # int out of the [-2**63, 2**64 - 1] range
        for bad in (1 << 64, -(1 << 63) - 1):
            with self.assertRaisesRegex(ValueError, "data must be in"):
                random.fold_in(key, bad)
        # tensor with the wrong dtype
        with self.assertRaisesRegex(RuntimeError, "data must have dtype uint64"):
            random.fold_in(key, torch.tensor(7, dtype=torch.int64, device=device))
        # tensor with more than one value
        with self.assertRaisesRegex(RuntimeError, "data must be a single value"):
            random.fold_in(key, torch.tensor([1, 2], dtype=torch.uint64, device=device))

    @onlyAccelerator
    def test_error_data_wrong_device(self, device):
        key = random.key(42, device=device)
        # A CPU data tensor with an accelerator key is a device mismatch.
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            random.fold_in(key, torch.tensor(7, dtype=torch.uint64))  # CPU

    @parametrize(
        "data",
        [
            subtest(0, name="zero"),
            subtest(7, name="small"),
            subtest(1 << 63, name="int64_max_plus_one"),
            subtest((1 << 64) - 1, name="uint64_max"),
        ],
    )
    def test_tensor_data_matches_int(self, device, data):
        key = random.split(random.key(42, device=device), 4)  # (4, 2) batched
        expected = random.fold_in(key, data)
        # Both a 0-dim scalar and a (1,) tensor are accepted as a single value.
        scalar = torch.tensor(data, dtype=torch.uint64, device=device)
        one_d = torch.tensor([data], dtype=torch.uint64, device=device)
        self.assertEqual(random.fold_in(key, scalar), expected)
        self.assertEqual(random.fold_in(key, one_d), expected)

    @onlyCUDA
    def test_tensor_data_cuda_graph(self, device):
        # A tensor data is not baked into the graph: mutating it and replaying
        # produces the result for the new value.
        key = random.key(42, device=device)
        data = torch.zeros((), dtype=torch.uint64, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                random.fold_in(key, data)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = random.fold_in(key, data)

        for value in (5, 99):
            data.fill_(value)
            g.replay()
            torch.cuda.synchronize()
            self.assertEqual(out, random.fold_in(key, value))

    @parametrize("batched", [False, True])
    @parametrize("tensor_data", [False, True])
    @onlyAccelerator
    def test_cross_device_consistency(self, device, batched, tensor_data):
        key_cpu = random.key(42)
        key_dev = random.key(42, device=device)
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(key_cpu, 4)  # (4, 2)
            key_dev = random.split(key_dev, 4)
        if tensor_data:
            # Tensor data exercises the .Tensor overload.
            data_cpu = torch.tensor(7, dtype=torch.uint64)
            data_dev = torch.tensor(7, dtype=torch.uint64, device=device)
        else:
            data_cpu = data_dev = 7
        self.assertEqual(
            random.fold_in(key_cpu, data_cpu),
            random.fold_in(key_dev, data_dev).cpu(),
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
    @parametrize("batched", [False, True])
    @onlyAccelerator
    def test_cross_device_uniform_consistency(self, device, dtype, batched):
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(random.key(42), 4).unsqueeze(-2)  # (4, 1, 2)
            key_dev = random.split(random.key(42, device=device), 4).unsqueeze(-2)
            shape = (4, 100)
        else:
            key_cpu = random.fold_in(random.key(42), 7)
            key_dev = random.fold_in(random.key(42, device=device), 7)
            shape = (1000,)
        # Uniform generation uses no transcendentals, so results must be bitwise identical.
        self.assertEqual(
            self._gen("uniform", key_cpu, shape, dtype=dtype),
            self._gen("uniform", key_dev, shape, dtype=dtype).cpu(),
            atol=0,
            rtol=0,
        )

    @dtypes(*all_floating_dtypes)
    @onlyAccelerator
    def test_cross_device_normal_consistency(self, device, dtype):
        key_cpu = random.fold_in(random.key(42), 7)
        key_dev = random.fold_in(random.key(42, device=device), 7)
        # Normal generation uses Box-Muller (log, sin, cos), and CUDA uses fast-math
        # intrinsics (__logf, __sincosf) that differ slightly from CPU std::log / std::sin /
        # std::cos. Results are approximately but not bitwise equal. assertEqual() by default
        # allows for some tolerance in the comparisons.
        self.assertEqual(
            self._gen("normal", key_cpu, (1000,), dtype=dtype),
            self._gen("normal", key_dev, (1000,), dtype=dtype).cpu(),
        )

    # --- bits(): a Python-only spelling of randint() over the full range ---

    @parametrize("dtype", all_int_dtypes)
    def test_bits_matches_randint_full_range(self, device, dtype):
        key = random.key(42, device=device)
        self.assertEqual(
            random.bits(key, (1000,), dtype=dtype),
            random.randint(key, (1000,), low=None, high=None, dtype=dtype),
        )

    def test_bits_default_dtype_is_int32(self, device):
        key = random.key(42, device=device)
        self.assertEqual(random.bits(key, (10,)).dtype, torch.int32)

    @parametrize("dtype", all_int_dtypes)
    def test_bits_inplace(self, device, dtype):
        key = random.key(42, device=device)
        result = torch.empty(1000, dtype=dtype, device=device)
        out = random.bits_(key, result)
        self.assertIs(out, result)
        self.assertEqual(result, random.bits(key, (1000,), dtype=dtype))

    def test_bits_shape_forms(self, device):
        key = random.key(42, device=device)
        expected = random.bits(key, (2, 50))
        self.assertEqual(random.bits(key, 2, 50), expected)
        self.assertEqual(random.bits(key, [2, 50]), expected)


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

    def test_fold_in_tensor_fullgraph(self, device):
        key = random.key(42, device=device)
        # data as a graph input (not a constant) exercises the Tensor overload.
        data = torch.tensor(7, dtype=torch.uint64, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key, data):
            return random.fold_in(key, data)

        self.assertEqual(f(key, data), random.fold_in(key, data))

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

    def test_randint_fullgraph(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.randint(key, (100,), low=-3, high=17)

        self.assertEqual(f(key), random.randint(key, (100,), low=-3, high=17))

    def test_bits_fullgraph(self, device):
        # bits() delegates to randint() in Python, so this also checks the
        # full-range path traces without a graph break.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.bits(key, (100,), dtype=torch.uint64)

        self.assertEqual(f(key), random.bits(key, (100,), dtype=torch.uint64))

    @onlyAccelerator
    @parametrize("op", ["uniform", "normal", "randint"])
    def test_generation_no_extra_clone(self, device, op):
        # Out-of-place generation fully overwrites its output; ensure generation
        # in torch.compile doesn't allocate an extra full-size buffer (i.e.
        # ensure peak ~= output size).
        if torch.device(device).type == "cuda" and not HAS_TRITON:
            self.skipTest("CUDA inductor codegen requires triton")
        gen_fn = getattr(random, op)

        gen_kwargs = {"high": 100} if op == "randint" else {}

        def gen(key, shape):
            return gen_fn(key, shape, **gen_kwargs)

        key = random.key(42, device=device)
        shape = (2048, 2048)  # 16 MiB fp32; an extra clone would ~double peak
        itemsize = (torch.int64 if op == "randint" else torch.float32).itemsize
        out_bytes = shape[0] * shape[1] * itemsize

        @torch.compile(fullgraph=True)
        def f(key):
            return gen(key, shape)

        f(key)  # compile + warm up the allocator
        torch.accelerator.synchronize(device)
        base = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        result = f(key)
        torch.accelerator.synchronize(device)
        extra = torch.accelerator.max_memory_allocated(device) - base

        self.assertEqual(result, gen(key, shape))
        self.assertLess(extra, 1.5 * out_bytes)  # no extra full-size clone

    @onlyAccelerator
    def test_generation_no_corruption_from_buffer_reuse(self, device):
        # Regression test for Inductor buffer reuse corrupting generation.
        if torch.device(device).type == "cuda" and not HAS_TRITON:
            self.skipTest("CUDA inductor codegen requires triton")

        # Keep all generations live until the final sum, exercising whether
        # Inductor reuses an in-place generation's buffer while it is still live.
        def f(key, x):
            rs = [random.uniform(random.fold_in(key, i), x.shape) for i in range(4)]
            for r in rs:
                x = x + r
            return x

        key = random.key(0, device=device)
        x = torch.randn(16, device=device)
        self.assertEqual(torch.compile(f, dynamic=False)(key, x), f(key, x))


class TestStatelessRNGInteger(TestCase):
    @parametrize("dtype", all_int_dtypes)
    def test_basic_shape_and_dtype(self, device, dtype):
        key = random.key(42, device=device)
        result = random.randint(key, (100,), low=0, high=10, dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.device, torch.device(device))

    def test_shape_forms_and_kw_only_bounds(self, device):
        # Shape may be splatted or passed as a sequence; bounds are keyword-only
        # and mean the same thing for randint and randint_.
        key = random.key(42, device=device)
        expected = random.randint(key, (2, 50), low=0, high=10)
        self.assertEqual(random.randint(key, 2, 50, low=0, high=10), expected)
        self.assertEqual(random.randint(key, [2, 50], low=0, high=10), expected)

        result = torch.empty(2, 50, dtype=torch.int32, device=device)
        random.randint_(key, result, low=0, high=10)
        self.assertEqual(result, expected)

        # Bounds are keyword-only in both.
        with self.assertRaises(TypeError):
            random.randint_(key, result, 0, 10)

    @parametrize("dtype", all_int_dtypes)
    def test_default_low_is_zero(self, device, dtype):
        # low defaults to 0 (matching torch.randint), not the dtype's minimum.
        key = random.key(42, device=device)
        result = random.randint(key, (1000,), high=100, dtype=dtype)
        self.assertEqual(
            result, random.randint(key, (1000,), low=0, high=100, dtype=dtype)
        )
        self.assertTrue((result.double() >= 0).all())

    def test_default_dtype_is_int32(self, device):
        key = random.key(42, device=device)
        self.assertEqual(random.randint(key, (10,), high=10).dtype, torch.int32)

    @parametrize("dtype", all_int_dtypes)
    def test_dtype_only_call(self, device, dtype):
        # The most basic call: no bounds at all. low defaults to 0 and high to
        # the dtype's top, which is a power-of-two range and therefore exact,
        # so this must work for every dtype.
        key = random.key(42, device=device)
        result = random.randint(key, (1000,), dtype=dtype)
        self.assertEqual(result.shape, (1000,))
        self.assertEqual(result.dtype, dtype)
        self.assertTrue(all(v >= 0 for v in result.tolist()))

    @parametrize("dtype", [torch.uint32, torch.int32])
    def test_power_of_two_range_allowed(self, device, dtype):
        # Ranges dividing 2**32 evenly are exact, so the bias guard must not
        # reject them however large they are.
        key = random.key(42, device=device)
        for k in (28, 29, 30, 31):
            result = random.randint(key, (100,), low=0, high=2**k, dtype=dtype)
            self.assertTrue(all(0 <= v < 2**k for v in result.to(torch.int64).tolist()))
        # ...but one past a power of two is genuinely biased and still rejected.
        with self.assertRaisesRegex(RuntimeError, "does not divide 2\\^32 evenly"):
            random.randint(key, (100,), low=0, high=2**28 + 1, dtype=dtype)

    @parametrize("dtype", all_int_dtypes)
    def test_determinism(self, device, dtype):
        key = random.key(42, device=device)
        a = random.randint(key, (1000,), high=100, dtype=dtype)
        b = random.randint(key, (1000,), high=100, dtype=dtype)
        self.assertEqual(a, b)

    @parametrize("dtype", all_int_dtypes)
    def test_range_bounds(self, device, dtype):
        key = random.key(42, device=device)
        # Cast to int64 for comparison: uint32/uint64 lack CPU comparison ops.
        result = random.randint(key, (10000,), low=3, high=17, dtype=dtype).to(
            torch.int64
        )
        self.assertTrue((result >= 3).all())
        self.assertTrue((result < 17).all())

    @parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    def test_negative_low(self, device, dtype):
        key = random.key(42, device=device)
        result = random.randint(key, (10000,), low=-8, high=8, dtype=dtype)
        self.assertTrue((result >= -8).all())
        self.assertTrue((result < 8).all())

    def test_covers_full_small_range(self, device):
        # Enough samples over a small range should hit every value.
        key = random.key(42, device=device)
        result = random.randint(key, (10000,), low=0, high=5)
        self.assertEqual(torch.unique(result).tolist(), [0, 1, 2, 3, 4])

    def test_batched_keys(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = random.randint(keys, (4, 100), high=1000)
        self.assertEqual(result.shape, (4, 100))
        for i in range(4):
            self.assertEqual(result[i], random.randint(keys[i], (100,), high=1000))

    def test_error_high_not_greater_than_low(self, device):
        key = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "high must be greater than low"):
            random.randint(key, (10,), low=5, high=5)
        with self.assertRaisesRegex(ValueError, "high must be greater than low"):
            random.randint(key, (10,), low=5, high=3)

    @parametrize("dtype", [torch.uint8, torch.int8, torch.uint16, torch.int16])
    def test_narrow_dtype_matches_32_bit_cast(self, device, dtype):
        # Dtypes narrower than the 32-bit sample draw a full word per element
        # and keep its low bits, so results match generating in 32 bits and
        # casting down (which drops the MSBs).
        key = random.key(42, device=device)
        wide = random.randint(key, (1000,), low=None, dtype=torch.uint32)
        narrow = random.randint(key, (1000,), low=None, dtype=dtype)
        self.assertEqual(narrow, wide.to(dtype))

    @parametrize("dtype", [torch.uint8, torch.int8, torch.uint16, torch.int16])
    def test_narrow_dtype_bounded_matches_32_bit(self, device, dtype):
        # The modulo is applied at the sampled width, not the output width, so a
        # bounded draw also matches the 32-bit result cast down.
        key = random.key(42, device=device)
        wide = random.randint(key, (1000,), low=0, high=100, dtype=torch.uint32)
        narrow = random.randint(key, (1000,), low=0, high=100, dtype=dtype)
        self.assertEqual(narrow, wide.to(dtype))

    @parametrize("dtype", all_int_dtypes)
    def test_no_bounds_is_full_range(self, device, dtype):
        # Omitting both bounds spans the dtype, so samples reach near its limits.
        key = random.key(42, device=device)
        info = torch.iinfo(dtype)
        result = random.randint(key, (1000,), low=None, dtype=dtype).double()
        self.assertGreater(result.max().item(), info.max * 0.9)
        if info.min < 0:
            self.assertLess(result.min().item(), info.min * 0.9)

    @parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
    def test_full_range_covers_negatives(self, device, dtype):
        key = random.key(42, device=device)
        result = random.randint(key, (1000,), low=None, dtype=dtype)
        self.assertTrue((result < 0).any())
        self.assertTrue((result > 0).any())

    @parametrize("dtype", all_int_dtypes)
    def test_explicit_dtype_bounds_match_none(self, device, dtype):
        # Passing a bound equal to the dtype's limit is the same as omitting it.
        key = random.key(42, device=device)
        info = torch.iinfo(dtype)
        expected = random.randint(key, (1000,), low=None, dtype=dtype)
        self.assertEqual(
            random.randint(key, (1000,), low=info.min, dtype=dtype), expected
        )
        self.assertEqual(
            random.randint(key, (1000,), low=None, high=info.max + 1, dtype=dtype),
            expected,
        )
        self.assertEqual(
            random.randint(key, (1000,), low=info.min, high=info.max + 1, dtype=dtype),
            expected,
        )

    @parametrize("dtype", all_int_dtypes)
    def test_only_one_bound_specified(self, device, dtype):
        # Either bound may be given on its own; the other defaults to the dtype's.
        key = random.key(42, device=device)
        info = torch.iinfo(dtype)
        # Compare as Python ints: float64 cannot represent int64 magnitudes.
        low_only = random.randint(key, (1000,), low=info.max - 10, dtype=dtype)
        self.assertTrue(all(v >= info.max - 10 for v in low_only.tolist()))
        high_only = random.randint(
            key, (1000,), low=None, high=info.min + 10, dtype=dtype
        )
        self.assertTrue(all(v < info.min + 10 for v in high_only.tolist()))

    def test_error_bounds_out_of_dtype_range(self, device):
        key = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "out of range for dtype"):
            random.randint(key, (10,), high=2**40, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "out of range for dtype"):
            random.randint(key, (10,), low=-1, dtype=torch.uint32)

    def test_error_wrong_self_dtype(self, device):
        key = random.key(42, device=device)
        result = torch.empty(100, dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have an integer dtype"):
            random.randint_(key, result, high=10)

    @parametrize("dtype", [torch.uint32, torch.int32])
    def test_error_range_too_large_for_32_bit(self, device, dtype):
        key = random.key(42, device=device)
        # A range that does not divide 2^32 is rejected at or above the limit.
        with self.assertRaisesRegex(RuntimeError, "does not divide 2\\^32 evenly"):
            random.randint(key, (10,), low=0, high=2**28 + 1, dtype=dtype)
        # Just under the limit is accepted, even though it also does not divide.
        random.randint(key, (10,), low=0, high=2**28 - 1, dtype=dtype)

    @parametrize("dtype", [torch.uint64, torch.int64])
    def test_large_range_allowed_for_64_bit(self, device, dtype):
        # No bias guard at 64 bits: a range far past 2^28 that does not divide
        # 2^64 evenly is still accepted.
        key = random.key(42, device=device)
        result = random.randint(key, (1000,), low=0, high=10**12, dtype=dtype)
        self.assertTrue((result.to(torch.int64) < 10**12).all())

    def test_error_op_empty_range(self, device):
        # low == high is empty, not the full range, even though it would give a
        # zero-width range in the kernel's unsigned arithmetic.
        key = random.key(42, device=device)
        result = torch.empty(100, dtype=torch.int64, device=device)
        with self.assertRaisesRegex(RuntimeError, r"\[low, high\) must be non-empty"):
            torch.ops.aten._philox_randint_(result, key, 5, 5)

    def test_op_resolves_partial_bounds(self, device):
        # The op resolves an absent bound to the dtype's limit, matching what
        # randint() does in Python.
        key = random.key(42, device=device)
        result = torch.empty(1000, dtype=torch.int64, device=device)
        torch.ops.aten._philox_randint_(result, key, 0, None)
        self.assertEqual(result, random.randint(key, (1000,), low=0, dtype=torch.int64))
        torch.ops.aten._philox_randint_(result, key, None, 0)
        self.assertEqual(
            result,
            random.randint(key, (1000,), low=None, high=0, dtype=torch.int64),
        )

    @parametrize("dtype", all_int_dtypes)
    def test_inplace(self, device, dtype):
        key = random.key(42, device=device)
        result = torch.empty(1000, dtype=dtype, device=device)
        out = random.randint_(key, result, high=100)
        self.assertIs(out, result)
        self.assertEqual(result, random.randint(key, (1000,), high=100, dtype=dtype))

    @parametrize("dtype", all_int_dtypes)
    def test_empty_output(self, device, dtype):
        key = random.key(42, device=device)
        for shape in [(0,), (3, 0)]:
            result = random.randint(key, shape, high=10, dtype=dtype)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, dtype)

    @parametrize("dtype", all_int_dtypes)
    @parametrize("layout", ["contiguous", "noncontiguous", "unaligned"])
    def test_inplace_layouts(self, device, dtype, layout):
        # Exercises the contiguous()/clone() copy-back path per dtype; the
        # vectorized store width is epc * itemsize, so it differs by dtype.
        key = random.key(42, device=device)
        if layout == "contiguous":
            result = torch.empty(1000, dtype=dtype, device=device)
        elif layout == "noncontiguous":
            result = torch.empty(2000, dtype=dtype, device=device)[::2]
        else:
            result = torch.empty(1001, dtype=dtype, device=device)[1:]
        out = random.randint_(key, result, high=100)
        self.assertIs(out, result)
        self.assertEqual(result, random.randint(key, (1000,), high=100, dtype=dtype))

    @parametrize("dtype", all_int_dtypes)
    def test_batched_keys_partial_chunk(self, device, dtype):
        # elems_per_key that is not a multiple of the per-call element count
        # exercises the multi-key kernel's scalar (non-vectorized) store path.
        key = random.key(42, device=device)
        keys = random.split(key, 3).unsqueeze(-2)  # (3, 1, 2)
        result = random.randint(keys, (3, 7), high=100, dtype=dtype)
        self.assertEqual(result.shape, (3, 7))
        for i in range(3):
            self.assertEqual(
                result[i], random.randint(keys[i], (7,), high=100, dtype=dtype)
            )

    @parametrize("dtype", all_int_dtypes)
    def test_different_keys_produce_different_values(self, device, dtype):
        a = random.randint(random.key(1, device=device), (1000,), low=None, dtype=dtype)
        b = random.randint(random.key(2, device=device), (1000,), low=None, dtype=dtype)
        self.assertNotEqual(a, b)

    @parametrize(
        "signed_dtype,unsigned_dtype",
        [
            (torch.int8, torch.uint8),
            (torch.int16, torch.uint16),
            (torch.int32, torch.uint32),
            (torch.int64, torch.uint64),
        ],
    )
    def test_full_range_signed_matches_unsigned(
        self, device, signed_dtype, unsigned_dtype
    ):
        # Signed dtypes reinterpret the same raw bits as the unsigned dtype.
        key = random.key(42, device=device)
        signed = random.randint(key, (1000,), low=None, dtype=signed_dtype)
        unsigned = random.randint(key, (1000,), low=None, dtype=unsigned_dtype)
        self.assertEqual(signed.view(unsigned_dtype), unsigned)

    def test_full_range_uint64_packs_two_uint32(self, device):
        # Each uint64 packs a consecutive pair of uint32 outputs:
        # uint64[i] == (uint32[2i] << 32) | uint32[2i + 1]. In little-endian
        # memory that reads back as the uint32 pairs swapped.
        key = random.key(42, device=device)
        n = 128
        b32 = random.randint(key, (2 * n,), dtype=torch.uint32)
        b64 = random.randint(key, (n,), dtype=torch.uint64)
        self.assertEqual(
            b64.view(torch.uint32), b32.reshape(-1, 2).flip(-1).reshape(-1)
        )

    @parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32, torch.uint64])
    def test_full_range_statistically_uniform(self, device, dtype):
        key = random.key(42, device=device)
        result = random.randint(key, (100000,), dtype=dtype)
        width = torch.iinfo(dtype).bits
        normalized = result.double() / float(2**width)
        self.assertTrue(abs(normalized.mean().item() - 0.5) < 0.01)

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        result = torch.empty(100, dtype=torch.uint32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            random.randint_(key, result, high=10)

    @onlyAccelerator
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_dev = random.key(42, device=device)
        # Integer reduction is deterministic, so results are bitwise identical.
        self.assertEqual(
            random.randint(key_cpu, (1000,), low=-5, high=100),
            random.randint(key_dev, (1000,), low=-5, high=100).cpu(),
        )


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
instantiate_device_type_tests(
    TestStatelessRNGInteger, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
