# Owner(s): ["module: cuda"]

import collections
import contextlib
import ctypes
import gc
import io
import queue
import sys
import tempfile
import threading
import unittest
from itertools import chain, repeat
from typing import NamedTuple, Union

import torch
import torch.cuda.comm as comm
from torch.nn.parallel import scatter_gather
from torch.testing._internal.common_cuda import (
    _create_scaling_case,
    _create_scaling_models_optimizers,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    IS_JETSON,
    IS_REMOTE_GPU,
    IS_SANDCASTLE,
    NoTest,
    run_tests,
    serialTest,
    skipCUDANonDefaultStreamIf,
    TEST_CUDA,
    TestCase,
)


TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


class TestCudaMultiGPU(TestCase):
    FIFTY_MIL_CYCLES = 50000000

    def _check_memory_stat_consistency(self):
        snapshot = torch.cuda.memory_snapshot()

        expected_each_device = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        for segment in snapshot:
            expandable = segment["is_expandable"]
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            if not expandable:
                expected["segment.all.current"] += 1
                expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment[
                "allocated_size"
            ]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            expected["requested_bytes.all.current"] += segment["requested_size"]
            expected["requested_bytes." + pool_str + ".current"] += segment[
                "requested_size"
            ]

            sum_requested = 0
            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    sum_requested += block["requested_size"]
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split and not expandable:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block[
                        "size"
                    ]

            self.assertEqual(sum_requested, segment["requested_size"])

        for device, expected in expected_each_device.items():
            stats = torch.cuda.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

    def test_cuda_synchronize(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize("cuda")
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(torch.device("cuda:0"))

        if TEST_MULTIGPU:
            torch.cuda.synchronize("cuda:1")
            torch.cuda.synchronize(1)
            torch.cuda.synchronize(torch.device("cuda:1"))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize("cpu")

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch.cuda.current_device()

        m0 = torch.cuda.memory_allocated(device)
        last_m_arr = [torch.cuda.memory_allocated(device)]
        max_m_arr = [torch.cuda.max_memory_allocated(device)]
        last_r_arr = [torch.cuda.memory_reserved(device)]
        max_r_arr = [torch.cuda.max_memory_reserved(device)]

        def alloc(*size):
            with torch.cuda.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch.cuda.FloatTensor(*size)

        def assert_change(comp=1, empty_cache=False, reset_peak=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch.cuda.memory_allocated(device)
            new_max_m = torch.cuda.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_r = torch.cuda.memory_reserved(device)
            new_max_r = torch.cuda.max_memory_reserved(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_r, new_max_r)
            self.assertGreaterEqual(new_max_r, max_r_arr[0])
            last_r_arr[0] = new_r
            max_r_arr[0] = new_max_r

            stat_key_n_sync = "num_sync_all_streams"
            stat_key_n_alloc = "num_device_alloc"
            stat_key_n_free = "num_device_free"
            if empty_cache:
                num_sync_1 = torch.cuda.memory_stats(device).get(stat_key_n_sync, -1)
                self.assertGreaterEqual(num_sync_1, 0)
                num_alloc_1 = torch.cuda.memory_stats(device).get(stat_key_n_alloc, -1)
                # if current memory usage is greater than zero we must have
                # allocated something
                self.assertGreaterEqual(num_alloc_1, 0 if new_m == 0 else 1)
                num_free_1 = torch.cuda.memory_stats(device).get(stat_key_n_free, -1)
                self.assertGreaterEqual(num_free_1, 0)
                # empty_cache will enforce the call of release_cached_blocks
                torch.cuda.empty_cache()
                num_sync_2 = torch.cuda.memory_stats(device).get(stat_key_n_sync, -1)
                self.assertEqual(num_sync_1 + 1, num_sync_2)
                num_alloc_2 = torch.cuda.memory_stats(device).get(stat_key_n_alloc, -1)
                self.assertGreaterEqual(num_alloc_2, num_alloc_1)
                num_free_2 = torch.cuda.memory_stats(device).get(stat_key_n_free, -1)
                self.assertGreaterEqual(num_free_2, num_free_1)

                new_r = torch.cuda.memory_reserved(device)
                new_max_r = torch.cuda.max_memory_reserved(device)
                self.assertLessEqual(new_r, last_r_arr[0])
                self.assertLessEqual(new_r, new_max_r)
                self.assertEqual(new_max_r, max_r_arr[0])
                last_r_arr[0] = new_r

            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
                self.assertEqual(torch.cuda.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.cuda.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch.cuda.memory_reserved(device), last_r_arr[0])
                self.assertEqual(torch.cuda.max_memory_reserved(device), last_r_arr[0])
                max_r_arr[0] = last_r_arr[0]

        assert_change(0)
        assert_change(0, reset_peak=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch.cuda.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_peak=(i % 2 == 0))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        assert_change(0, reset_peak=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_peak=True)
        assert_change(0)
        self.assertEqual(torch.cuda.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_peak=True)
        self.assertEqual(torch.cuda.memory_allocated(device), m0)

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @serialTest()
    def test_memory_stats(self):
        gc.collect()
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_memory_stats_multigpu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device="cuda:0", N=35)
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("cuda:1"), N=35
        )
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("cuda:1"), N=35
        )
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = torch.inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_autogpu(self):
        x = torch.randn(5, 5).cuda()
        y = torch.randn(5, 5).cuda()
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(x.get_device(), 0)
        with torch.cuda.device(1):
            z = torch.randn(5, 5).cuda()
            self.assertEqual(z.get_device(), 1)
            q = x.add(y)
            self.assertEqual(q.get_device(), 0)
            w = torch.randn(5, 5).cuda()
            self.assertEqual(w.get_device(), 1)
            self.assertEqual(y.cuda().get_device(), 1)
        z = z.cuda()
        self.assertEqual(z.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_new(self):
        x = torch.randn(3, 3).cuda()
        self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
        self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

        with torch.cuda.device(1):
            self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
            self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_device(self):
        x = torch.randn(5, 5).cuda()
        with torch.cuda.device(1):
            y = x.cuda()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.cuda(), y)
            z = y.cuda(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.cuda(0), z)

        x = torch.randn(5, 5)
        with torch.cuda.device(1):
            y = x.cuda()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.cuda(), y)
            z = y.cuda(0)

            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.cuda(0), z)

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch.cuda.Stream(device=x.device)
        s1 = torch.cuda.Stream(device=y.device)
        s2 = torch.cuda.Stream(device=x.device)
        s3 = torch.cuda.Stream(device=y.device)

        # same dst stream different src streams
        with torch.cuda.stream(s0):
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s1):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s2), torch.cuda.stream(s1):
            y.copy_(x)

        s1.synchronize()
        # The copy() is synchronized on the current streams of both src and dst.
        # In the above test, the _sleep() op on s0 will not block the copy() on
        # s2, but both copies are synchronized on s1 in the dst device. Hence,
        # x is copied to y after x_plus_one is copied to y. If x and y are on
        # the same device, both copy() ops are synchronized on s1.
        self.assertEqual(y, x)

        # same src stream different dst streams
        with torch.cuda.stream(s1):
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s0):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s3), torch.cuda.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_streams(self):
        d0 = torch.device("cuda:0")
        x0 = torch.zeros(5, 5, device=d0)

        d1 = torch.device("cuda:1")
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)

        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_cat_autogpu(self):
        x = torch.randn(4, 4).cuda(1)
        y = torch.randn(4, 4).cuda(1)
        z = torch.cat([x, y], 0)
        self.assertEqual(z.get_device(), x.get_device())

    @unittest.skipIf(torch.cuda.device_count() >= 10, "Loading a cuda:9 tensor")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:9' restore location
        tensor = torch.randn(2, device="cuda")
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # NB: this might not work in the future if serialization changes
        buf = io.BytesIO(buf.getvalue().replace(b"cuda:0", b"cuda:9"))

        msg = r"Attempting to deserialize object on CUDA device 9"
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization_remap(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]

        def gpu_remap(storage, location):
            if location == "cuda:1":
                return storage.cuda(0)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location=gpu_remap)

        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization_remap_dict(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location={"cuda:1": "cuda:0"})
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_storage_clone(self):
        x = torch.randn(4, 4, device="cuda:1").storage()
        y = x.clone()
        self.assertEqual(x.get_device(), y.get_device())
        for t in ["byte", "char", "short", "int", "long", "half", "double"]:
            self.assertEqual(getattr(x, t)().get_device(), x.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_cuda_set_device(self):
        x = torch.randn(5, 5)
        with torch.cuda.device(1):
            self.assertEqual(x.cuda().get_device(), 1)
            torch.cuda.set_device(0)
            self.assertEqual(x.cuda().get_device(), 0)
            with torch.cuda.device(1):
                self.assertEqual(x.cuda().get_device(), 1)
            self.assertEqual(x.cuda().get_device(), 0)
            torch.cuda.set_device(1)
        self.assertEqual(x.cuda().get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_current_stream(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.current_stream(device=1)
        s2 = torch.cuda.current_stream(device=0)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s2)

        with torch.cuda.device(d1):
            s0 = torch.cuda.current_stream()
            s1 = torch.cuda.current_stream(1)
            s2 = torch.cuda.current_stream(d0)

        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s1)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but got: cpu"):
            torch.cuda.current_stream(torch.device("cpu"))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        with torch.cuda.device(d0):
            s0 = torch.cuda.default_stream()

        with torch.cuda.device(d1):
            s1 = torch.cuda.default_stream()

        s2 = torch.cuda.default_stream(device=0)
        s3 = torch.cuda.default_stream(d1)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch.cuda.device(d0):
            self.assertEqual(torch.cuda.current_stream(), s0)

        with torch.cuda.device(d1):
            self.assertEqual(torch.cuda.current_stream(), s1)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but got: cpu"):
            torch.cuda.default_stream(torch.device("cpu"))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_device(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        e0 = torch.cuda.Event()

        self.assertEqual(None, e0.device)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device("cuda:0"))
        self.assertEqual(e0.device, torch.device("cuda:0"))
        self.assertEqual(s1.device, torch.device("cuda:1"))
        self.assertEqual(e1.device, torch.device("cuda:1"))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_context(self):
        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.Stream(device=1)
        s2 = torch.cuda.Stream(device=0)

        with torch.cuda.device(s1.device):
            prev_stream_on_cuda1 = torch.cuda.current_stream()

        self.assertEqual(torch.cuda.current_stream(), s0)
        self.assertEqual(0, torch.cuda.current_device())
        with torch.cuda.stream(s1):
            self.assertEqual(torch.cuda.current_stream(), s1)
            self.assertEqual(1, torch.cuda.current_device())
            with torch.cuda.stream(s2):
                self.assertEqual(torch.cuda.current_stream(), s2)
                self.assertEqual(0, torch.cuda.current_device())
                with torch.cuda.stream(s0):
                    self.assertEqual(torch.cuda.current_stream(), s0)
                    self.assertEqual(0, torch.cuda.current_device())
                self.assertEqual(torch.cuda.current_stream(), s2)
                self.assertEqual(0, torch.cuda.current_device())
            self.assertEqual(torch.cuda.current_stream(), s1)
            self.assertEqual(1, torch.cuda.current_device())

        with torch.cuda.device(s1.device):
            self.assertEqual(prev_stream_on_cuda1, torch.cuda.current_stream())

        self.assertEqual(torch.cuda.current_stream(), s0)
        self.assertEqual(0, torch.cuda.current_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu(self):
        default_stream = torch.cuda.current_stream()
        self.assertEqual(default_stream.device, torch.device("cuda:0"))
        stream = torch.cuda.Stream(device=1)
        self.assertEqual(stream.device, torch.device("cuda:1"))
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.current_stream().device, torch.device("cuda:1"))
            self.assertNotEqual(torch.cuda.current_stream(), default_stream)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu_query(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)

        self.assertTrue(s0.query())
        self.assertFalse(s1.query())

        with torch.cuda.device(d0):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        with torch.cuda.device(d1):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        # deliberately using a different device
        with torch.cuda.device(d0):
            s1.synchronize()

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

        with torch.cuda.device(d0):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

        with torch.cuda.device(d1):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu_eq(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            s1 = torch.cuda.current_stream()

        with torch.cuda.device(d1):
            s2 = torch.cuda.current_stream()
            s3 = torch.cuda.current_stream()

        self.assertTrue(s0 == s0)
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)

        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.cuda_stream, s1.cuda_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.cuda_stream, s3.cuda_stream)
        self.assertNotEqual(s0.device, s3.device)

        self.assertEqual(hash(s0), hash(s1))
        self.assertEqual(hash(s2), hash(s3))
        self.assertNotEqual(hash(s0), hash(s3))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_streams_priority(self):
        low, high = torch.cuda.Stream.priority_range()
        s0 = torch.cuda.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device("cuda:0"), s0.device)

        s1 = torch.cuda.Stream(device=1, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device("cuda:1"), s1.device)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_tensor_device(self):
        self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 0)
        self.assertEqual(torch.cuda.FloatTensor(1, device=1).get_device(), 1)
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 1)
            self.assertEqual(torch.cuda.FloatTensor(1, device=0).get_device(), 0)
            self.assertEqual(torch.cuda.FloatTensor(1, device=None).get_device(), 1)

    @staticmethod
    def _stream_synchronize(self, spin_time_cycles):
        s = torch.cuda.current_stream()
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        e_tik.record(s)
        torch.cuda._sleep(spin_time_cycles)
        e_tok.record(s)
        s.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_synchronize(self, spin_time_cycles):
        s = torch.cuda.current_stream()
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        e_tik.record(s)
        torch.cuda._sleep(spin_time_cycles)
        s.record_event(e_tok)
        e_tok.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_wait(self, spin_time_cycles):
        s0 = torch.cuda.current_stream()
        s1 = torch.cuda.Stream()
        e_tik = torch.cuda.Event(blocking=True, enable_timing=True)
        e_tok = torch.cuda.Event(blocking=True, enable_timing=True)

        e_tik.record(s0)
        torch.cuda._sleep(spin_time_cycles - 10)
        e_sync = torch.cuda.Event(blocking=True)
        e_sync.record()
        e_sync.wait(s1)
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10)
        s1.synchronize()
        e_tok.record()
        e_tok.synchronize()

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())
        self.assertTrue(e_sync.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _test_stream_event_nogil(self, sync_func, p2c, c2p):
        with torch.cuda.device("cuda:1"):
            c2p.put(0)
            p2c.get()
            c2p.put(sync_func(self, TestCudaMultiGPU.FIFTY_MIL_CYCLES))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_nogil(self):
        for sync_func in [
            TestCudaMultiGPU._stream_synchronize,
            TestCudaMultiGPU._event_synchronize,
            TestCudaMultiGPU._event_wait,
        ]:
            p2c = queue.Queue()
            c2p = queue.Queue()
            e_tik = torch.cuda.Event(enable_timing=True)
            e_tok = torch.cuda.Event(enable_timing=True)

            t = threading.Thread(
                target=TestCudaMultiGPU._test_stream_event_nogil,
                args=(self, sync_func, p2c, c2p),
            )
            t.daemon = True
            t.start()

            c2p.get()
            with torch.cuda.device("cuda:0"):
                e_tik.record()
                p2c.put(0)
                parent_time = sync_func(self, TestCudaMultiGPU.FIFTY_MIL_CYCLES)
                child_time = c2p.get()
                e_tok.record()
                e_tok.synchronize()
                total_time = e_tik.elapsed_time(e_tok)

            # Without GIL, synchronizations in parent and child threads can
            # overlap. The total execution time should be a little bit longer
            # than spinning fifty million cycles and much shorter than twice of
            # that. However, testing absolute execution time is not reliable as
            # it may vary on different hardware in different environments.
            # Therefore, this test uses relative comparisons, checking if the
            # sum of parent and child threads execution time is greater than the
            # real execution time by least 30%.
            self.assertGreater(parent_time + child_time, total_time * 1.3)

    # This test is flaky for ROCm, see issue #62602
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_events_wait(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            e0 = torch.cuda.Event()
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()

        self.assertFalse(s0.query())
        self.assertTrue(s1.query())

        s1.wait_event(e0)
        s1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_events_multi_gpu_query(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = s0.record_event()
            s0.synchronize()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            e1 = s1.record_event()

        self.assertTrue(e0.query())
        self.assertFalse(e1.query())

        with torch.cuda.device(d0):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        with torch.cuda.device(d1):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        # deliberately using a different device
        with torch.cuda.device(d0):
            e1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(e1.query())

        with torch.cuda.device(d0):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

        with torch.cuda.device(d1):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_events_multi_gpu_elapsed_time(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(10)
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            s1.record_event(e1)

        e0.synchronize()
        e1.synchronize()
        with torch.cuda.device(d0):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch.cuda.device(d1):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e2 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            s0.record_event(e2)
            s0.synchronize()

        self.assertGreater(e0.elapsed_time(e2), 0)

        # deliberately calling from a different device
        with torch.cuda.device(d1):
            self.assertGreater(e0.elapsed_time(e2), 0)

    @contextlib.contextmanager
    def _get_external_stream(self, device):
        cudart = torch.cuda.cudart()
        stream = ctypes.c_ulonglong(0)
        stream_p = ctypes.POINTER(ctypes.c_void_p)(stream)
        stream_p_int = ctypes.cast(stream_p, ctypes.c_void_p).value
        with device:
            try:
                out = cudart.cudaStreamCreate(stream_p_int)
                self.assertEqual(out, 0)
                self.assertNotEqual(stream.value, 0)
                yield stream.value
            finally:
                out = cudart.cudaStreamDestroy(stream.value)
                self.assertEqual(out, 0)

    def test_external_streams(self):
        device = torch.cuda.device(0)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch.cuda.ExternalStream(stream_v)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)
            ext_stream = torch.cuda.get_stream_from_external(stream_v, device)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_external_streams_multi_device(self):
        device = torch.cuda.device(1)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch.cuda.ExternalStream(stream_v, device=device)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)
            ext_stream = torch.cuda.get_stream_from_external(stream_v, device)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_caching_pinned_memory_multi_gpu(self):
        # checks that the events preventing pinned memory from being reused
        # too early are recorded on the correct GPU
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        gpu_tensor0 = torch.cuda.FloatTensor([0], device=0)
        gpu_tensor1 = torch.cuda.FloatTensor([0], device=1)

        with torch.cuda.device(1):
            torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
            gpu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg="allocation reused too soon")

        with torch.cuda.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(gpu_tensor1[0], 1)
        self.assertEqual(gpu_tensor0[0], 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_get_set_rng_state_all(self):
        states = torch.cuda.get_rng_state_all()
        before0 = torch.cuda.FloatTensor(100, device=0).normal_()
        before1 = torch.cuda.FloatTensor(100, device=1).normal_()
        torch.cuda.set_rng_state_all(states)
        after0 = torch.cuda.FloatTensor(100, device=0).normal_()
        after1 = torch.cuda.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, atol=0, rtol=0)
        self.assertEqual(before1, after1, atol=0, rtol=0)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_rng_state_offset(self):
        before = torch.cuda.get_rng_state()
        torch.cuda._set_rng_state_offset(100)
        offset = torch.cuda._get_rng_state_offset()
        torch.cuda.set_rng_state(before)
        self.assertEqual(offset, 100)

    # Verifies that mem_get_info works, including when called for a different device
    def test_mem_get_info(self):
        def _test(device: Union[str, int, torch.device]):
            # Prevent PyTorch from reusing the allocated memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            before_free_bytes, before_available_bytes = torch.cuda.mem_get_info(device)
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            t = torch.randn(1024 * 1024 * 8, device=device)  # noqa: F841

            if IS_JETSON:
                # w/o syncing, mem_get_info will run before memory allocated has actually increased.
                # This race condition causes consistent failure
                torch.cuda.synchronize()
            after_free_bytes, after_available_bytes = torch.cuda.mem_get_info(device)

            self.assertLess(after_free_bytes, before_free_bytes)
            self.assertEqual(before_available_bytes, after_available_bytes)

        # Test calls with different device representations
        _test(0)
        _test(torch.device("cuda"))
        _test(torch.device("cuda:0"))
        _test("cuda")
        _test("cuda:0")
        if TEST_MULTIGPU:
            _test(1)
            _test(torch.device("cuda:1"))
            _test("cuda:1")

    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_cuda_memory_leak_detection(self):
        l = []

        @self.wrap_with_cuda_memory_check
        def no_leak():
            pass

        @self.wrap_with_cuda_memory_check
        def leak_gpu0():
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("cuda:0")))

        no_leak()
        regex = r"CUDA driver API confirmed .+ on device 0.+"
        if IS_JETSON:
            try:
                leak_gpu0()
            except RuntimeError as e:
                import re

                assert re.match(regex, str(e)), str(e) + "\n does not match: \n" + regex
        else:
            # assertRaisesRegex does not pass with Python for Jetson,
            # even though the RuntimeError matches regex using re.match
            with self.assertRaisesRegex(RuntimeError, regex):
                leak_gpu0()

        if TEST_MULTIGPU:

            @self.wrap_with_cuda_memory_check
            def leak_gpu1():
                # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
                l.append(torch.randn(1024 * 1024 * 8, device=torch.device("cuda:1")))

            with self.assertRaisesRegex(
                RuntimeError, r"CUDA driver API confirmed .+ on device 1.+"
            ):
                leak_gpu1()

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_streaming_backwards_device_transfer(self):
        # This function must run with non-default current streams on all devices, otherwise it's meaningless.
        # The intention is to test that to()'s backward (CopyBackward) interacts properly with the
        # synchronization logic in torch/csrc/autograd/input_buffer.cpp.
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")

        # Unfortunately I need to make the tensors largeish.
        # Bigger tensors = longer D2D transfers = more likely to expose races.
        size = 2**26

        a = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)
        b = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)

        # Here to_backward_recipient = a*b is used only once, so MulBackward's InputBuffer slot only expects 1 input.
        # This tests the situation where we don't call InputBuffer::accumulate for MulBackward's InputBuffer.
        to_backward_recipient = a * b
        s = to_backward_recipient.to(device="cuda:0").sum()
        torch.cuda.synchronize(device=dev0)
        torch.cuda.synchronize(device=dev1)
        s.backward()
        self.assertTrue(a.grad.sum().item() == size)
        self.assertTrue(b.grad.sum().item() == size)

        # Here to_backward_recipient = a*b is used twice, so MulBackward's InputBuffer slot expects 2 inputs.
        # This tests the situation where we do call InputBuffer::accumulate for MulBackward's InputBuffer.
        a.grad = None
        b.grad = None
        to_backward_recipient = a * b
        # Multiply by 2 here so to's backward creates gradient values that are different from the case above,
        # to mitigate weirdness if the caching allocator happens to reuse memory regions that were populated
        # with 1s by the case above
        s0 = to_backward_recipient.to(device="cuda:0").sum() * 2.0
        s1 = to_backward_recipient.to(device="cuda:0").sum() * 2.0
        torch.cuda.synchronize(device=dev0)
        torch.cuda.synchronize(device=dev1)
        s0.backward(retain_graph=True)
        s1.backward()
        self.assertTrue(a.grad.sum().item() == 4 * size)
        self.assertTrue(b.grad.sum().item() == 4 * size)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @unittest.skipIf(IS_SANDCASTLE or IS_REMOTE_GPU, "Does not work on Sandcastle")
    def test_cuda_init_race(self):
        # See https://github.com/pytorch/pytorch/issues/16559
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-c",
                """\
import torch
import threading

def worker(rank):
    torch.tensor([1.]).cuda(rank)

t1 = threading.Thread(target=worker, args=(0,))
t2 = threading.Thread(target=worker, args=(1,))
t1.start()
t2.start()
""",
            ]
        )

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_grad_scaling_device_as_key(self):
        # Ensure that different instances of "device" objects that point to the same device
        # are treated as identical keys by dicts.  GradScaler relies on this behavior, and may
        # error otherwise in a way that's difficult to detect (a silent performance hit).
        d = {}
        t = torch.empty((1,), device="cuda:0")
        dev0a = torch.device("cuda:0")
        dev0b = torch.device("cuda:0")
        dev1a = torch.device("cuda:1")
        dev1b = torch.device("cuda:1")

        self.assertTrue(hash(dev0a) == hash(dev0b))
        self.assertTrue(hash(dev1a) == hash(dev1b))

        d[dev0a] = "0a"
        d[dev0b] = "0b"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "0b")
        d[t.device] = "t"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "t")

        d[dev1a] = "1a"
        d[dev1b] = "1b"
        self.assertTrue(len(d) == 2)
        self.assertTrue(d[dev1a] == "1b")

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_grad_scaling_scale(self):
        scaler = torch.amp.GradScaler(device="cuda", init_scale=2.0)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:0")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (
            t1.clone(),
            (t0.clone(), t1.clone()),
            [t0.clone(), (t1.clone(), t0.clone())],
        )
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 8.0
            and outputs[1][0] == 8.0
            and outputs[1][1] == 8.0
            and outputs[2][0] == 8.0
            and outputs[2][1][0] == 8.0
            and outputs[2][1][1] == 8.0
        )
        self.assertTrue(scaler._scale.device == t1.device)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_grad_scaling_multigpu(self):
        # Same as above, but runs some of the models on device 1.
        # GradScaler should transparently handle losses and gradients on multiple devices.
        # This test could be combined with the test above, but I think it makes sense to treat
        # multi-GPU operations separately.
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")

        for enabled in True, False:
            (
                mod_control0,
                mod_scaling0,
                opt_control0,
                opt_scaling0,
                data,
                loss_fn,
                skip_iter,
            ) = _create_scaling_case()
            (
                mod_control1,
                mod_scaling1,
                opt_control1,
                opt_scaling1,
            ) = _create_scaling_models_optimizers(device=dev1)

            scaler = torch.amp.GradScaler(
                device="cuda",
                init_scale=128.0,
                growth_factor=2.0,
                enabled=enabled,
                growth_interval=1,
            )

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input.to(dev1))
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1.to(dev0), target)
                    loss1 = loss_fn(
                        0.6 * output0.to(dev1) - 0.4 * output1, target.to(dev1)
                    )

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        # Make sure the found_infs were collected properly across optimizers and devices.
                        if scaler.is_enabled():
                            self.assertTrue(
                                len(scaler._found_inf_per_device(optimizer0)) == 1
                            )
                            self.assertTrue(
                                len(scaler._found_inf_per_device(optimizer1)) == 1
                            )
                            self.assertTrue(
                                scaler._found_inf_per_device(optimizer0)[dev0].item()
                                == 0.0
                            )
                            self.assertTrue(
                                scaler._found_inf_per_device(optimizer1)[dev1].item()
                                == float(i == skip_iter)
                            )

                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(
                scaler.get_scale()
                == (
                    128.0
                    * scaler.get_growth_factor() ** 3
                    * scaler.get_backoff_factor() ** 1
                )
                if enabled
                else 1.0
            )

            # Copy mod_control1 and mod_scaling1 back the device 0 for comparison
            mod_control1.to(dev0)
            mod_scaling1.to(dev0)

            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                self.assertEqual(c, s, rtol=1e-5, atol=1e-7)

    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    def test_cuda_device_memory_allocated(self):
        from torch.cuda import memory_allocated

        device_count = torch.cuda.device_count()
        current_alloc = [memory_allocated(idx) for idx in range(device_count)]
        _x = torch.ones(10, device="cuda:0")
        self.assertGreater(memory_allocated(0), current_alloc[0])
        self.assertTrue(
            all(
                memory_allocated(torch.cuda.device(idx)) == current_alloc[idx]
                for idx in range(1, device_count)
            )
        )


class TestCudaComm(TestCase):
    def _test_broadcast(self, input):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        # test regular
        results = comm.broadcast(input, (0, 1))
        for i, t in enumerate(results):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)
            if (
                input.is_cuda and input.get_device() == i
            ):  # test not copying on same device
                self.assertEqual(t.data_ptr(), input.data_ptr())
        # test out=
        for inplace in [True, False]:
            if inplace:
                outputs = [
                    torch.empty_like(input, device=0),
                    torch.empty_like(input, device=1),
                ]
            else:
                outputs = [input.cuda(0), torch.empty_like(input, device=1)]
            results = comm.broadcast(input, out=outputs)
            for r, o in zip(results, outputs):
                self.assertIs(r, o)
            for i, t in enumerate(results):
                self.assertEqual(t.get_device(), i)
                self.assertEqual(t, input)
        # test error msg
        with self.assertRaisesRegex(
            RuntimeError, r"Exactly one of 'devices' and 'out'"
        ):
            comm.broadcast(input, (0, 1), out=outputs)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to be CUDA tensors, but output tensor at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cpu()])
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to have same shape as the source .+ at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cuda(1).unsqueeze(0)])

    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).cuda())

    def _test_broadcast_coalesced(self, tensors, buffer_size):
        b_tensors = [comm.broadcast(t, (0, 1)) for t in tensors]
        for (_, bt), t in zip(b_tensors, tensors):
            self.assertEqual(bt.get_device(), 1)
            self.assertEqual(bt, t)
            self.assertIsInstance(bt, type(t))

        bc_tensors = comm.broadcast_coalesced(tensors, (0, 1), buffer_size=buffer_size)
        bc_tensors_t = list(zip(*bc_tensors))
        self.assertEqual(b_tensors, bc_tensors_t)
        for (_, bt), (_, bct) in zip(b_tensors, bc_tensors_t):
            self.assertEqual(bt.get_device(), bct.get_device())
            self.assertIsInstance(bct, type(bt))

        # check that tensors on device[0] are returned as-is
        for out_tensors in (b_tensors, bc_tensors_t):
            for inp_t, (out_t, _) in zip(tensors, out_tensors):
                self.assertIs(inp_t, out_t)

        # check that the tensors not on device[0] have different version counters
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for _, t in bc_tensors_t]
        for old_version, (_, t) in zip(versions, bc_tensors_t):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # Note: fails sometimes on the CI, passes on dual gfx906
    def test_broadcast_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            self.genSparseTensor((2, 3), 2, 1, False, "cuda", torch.float64)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            self.genSparseTensor((2, 3), 2, 10, False, "cuda", torch.float64)[0],
            self.genSparseTensor((2, 3), 2, 5, False, "cuda", torch.float64)[0],
            self.genSparseTensor((3, 3), 2, 7, False, "cuda", torch.int64)[0],
            self.genSparseTensor((2, 3), 2, 2, False, "cuda", torch.float32)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            self.genSparseTensor((2, 7), 2, 3, False, "cuda", torch.int64)[0],
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast_coalesced_empty_tensors(self):
        tensors = [
            torch.tensor([]).byte().cuda(),
            torch.randn(5).cuda(),
            torch.randn(5).double().cuda(),
        ]
        self._test_broadcast_coalesced(tensors, 256)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        x_cuda = x.cuda(0)
        y_cuda = y.cuda(1)
        result = comm.reduce_add((x_cuda, y_cuda))
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.cpu(), x + y)

    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        dup_tensors = [tensors, [t.cuda(1) for t in tensors]]

        r_tensors = [comm.reduce_add(t) for t in zip(*dup_tensors)]
        for r, t in zip(r_tensors, tensors):
            self.assertEqualTypeString(r, t)
            self.assertEqual(r.coalesce() if r.is_sparse else r, t * 2)

        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)
        self.assertEqual(r_tensors, rc_tensors)
        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqualTypeString(rc, r)

        # Since we have both cuda:0 and cuda:1 inputs, the outputs must be new.
        # We can check that they have different version counters.
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for t in rc_tensors]
        for old_version, t in zip(versions, rc_tensors):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            self.genSparseTensor((2, 3), 2, 1, False, "cuda", torch.float64)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            self.genSparseTensor((2, 3), 2, 10, False, "cuda", torch.float64)[0],
            self.genSparseTensor((2, 3), 2, 5, False, "cuda", torch.float64)[0],
            self.genSparseTensor((3, 3), 2, 7, False, "cuda", torch.int64)[0],
            self.genSparseTensor((2, 3), 2, 2, False, "cuda", torch.float32)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            self.genSparseTensor((2, 7), 2, 3, False, "cuda", torch.int64)[0],
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            torch.randn(numel).cuda(),
        ]
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)

    def _test_scatter(self, input, chunk_sizes=None, dim=0):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        if chunk_sizes is None:
            ref_chunk_sizes = tuple(repeat(input.size(dim) // 2, 2))
        else:
            ref_chunk_sizes = chunk_sizes

        # test regular
        result = comm.scatter(input, (0, 1), chunk_sizes, dim)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end
            if r.device == input.device:
                self.assertEqual(
                    r.data_ptr(), input.data_ptr()
                )  # for target @ same device, a view should be returned

        # test out
        out = [torch.empty_like(t) for t in result]
        result = comm.scatter(input, dim=dim, out=out)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            self.assertIs(r, out[i])
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end

        # test error msg
        if chunk_sizes is not None:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected devices and chunk_sizes to be of same length"
            ):
                comm.scatter(
                    input,
                    [0 for _ in range(len(chunk_sizes) + 1)],
                    dim=dim,
                    chunk_sizes=chunk_sizes,
                )
        with self.assertRaisesRegex(RuntimeError, r"'devices' must not be specified"):
            comm.scatter(input, (0, 1), dim=dim, out=out)
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one device to scatter to"
        ):
            comm.scatter(input, (), dim=dim)
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one output tensor to scatter to"
        ):
            comm.scatter(input, dim=dim, out=[])
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to be CUDA tensors, but output tensor at index 0",
        ):
            comm.scatter(input, dim=dim, out=([out[0].cpu()] + out[1:]))
        with self.assertRaisesRegex(
            RuntimeError, r"Output tensor at index 0 has incorrect shape"
        ):
            comm.scatter(input, dim=dim, out=([out[0].unsqueeze(0)] + out[1:]))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Total size for output tensors along scatter dim \d+ does not match",
        ):
            index = [slice(None, None) for _ in range(input.dim())]
            index[dim] = slice(1, None)
            comm.scatter(input, dim=dim, out=([out[0][tuple(index)]] + out[1:]))

    def test_scatter_cpu(self):
        self._test_scatter(torch.randn(4, 4), dim=0)

    def test_scatter_cpu_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=1)

    def test_scatter_cpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=-2)

    def test_scatter_cpu_sizes(self):
        self._test_scatter(torch.randn(6, 4), chunk_sizes=(2, 4))

    def test_scatter_gpu(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=0)

    def test_scatter_gpu_dim(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=1)

    def test_scatter_gpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4).cuda(), dim=-2)

    def test_scatter_gpu_sizes(self):
        self._test_scatter(torch.randn(6, 4).cuda(), chunk_sizes=(2, 4))

    def _test_gather(self, dim):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        x = torch.randn(2, 5, device=0)
        y = torch.randn(2, 5, device=1)
        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = torch.Size(expected_size)

        destinations = [None, torch.device("cuda:0"), torch.device("cpu")]
        if torch.cuda.device_count() > 2:
            destinations.append(torch.device("cuda:2"))
        with torch.cuda.device(1):
            for destination in destinations:
                if destination is None:
                    expected_device = torch.device("cuda", torch.cuda.current_device())
                else:
                    expected_device = destination
                for use_out in [True, False]:
                    if use_out:
                        out = torch.empty(expected_size, device=expected_device)
                        result = comm.gather((x, y), dim, out=out)
                        self.assertIs(out, result)
                    else:
                        result = comm.gather((x, y), dim, destination=destination)

                    self.assertEqual(result.device, expected_device)
                    self.assertEqual(result.size(), expected_size)

                    index = [slice(None, None), slice(None, None)]
                    index[dim] = slice(0, x.size(dim))
                    self.assertEqual(result[tuple(index)], x)
                    index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
                    self.assertEqual(result[tuple(index)], y)

        # test error msg
        with self.assertRaisesRegex(
            RuntimeError, r"'destination' must not be specified"
        ):
            comm.gather(
                (x, y),
                dim,
                destination="cpu",
                out=torch.empty(expected_size, device="cpu"),
            )
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one tensor to gather from"
        ):
            comm.gather(())
        with self.assertRaisesRegex(
            RuntimeError, r"Expected all input tensors to be CUDA tensors, "
        ):
            comm.gather((x.cpu(), y))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all input tensors to have the same number of dimensions",
        ):
            comm.gather((x, y.unsqueeze(0)))
        with self.assertRaisesRegex(
            RuntimeError, r"Input tensor at index 1 has invalid shape"
        ):
            if dim in [0, -2]:
                comm.gather((x, y[:, 1:]), dim=dim)
            elif dim in [1, -1]:
                comm.gather((x, y[1:, :]), dim=dim)

    def test_gather(self):
        self._test_gather(0)

    def test_gather_dim(self):
        self._test_gather(1)

    def test_gather_neg_dim(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_memory_format_scatter_gather(self):
        nhwc = torch.randn((10, 3, 32, 32), device="cpu").contiguous(
            memory_format=torch.channels_last
        )
        results = torch.cuda.comm.scatter(nhwc, (0, 1), None, 0)
        for result in results:
            self.assertFalse(result.is_contiguous())
            self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

        gathered = torch.cuda.comm.gather(results)
        self.assertTrue(gathered.is_contiguous(memory_format=torch.channels_last))

    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    def test_scatter_namedtuple(self):
        # tests ability to scatter namedtuples and retrieve a list where each
        # element is of the expected namedtuple type.
        fields = ("a", "b")
        TestNamedTupleInput_0 = collections.namedtuple("NamedTuple", fields)
        num_gpus = torch.cuda.device_count()
        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=0)
        a_tensors_for_gpu = [a[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
        b_tensors_for_gpu = [b[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]

        inp = TestNamedTupleInput_0(a, b)
        target_gpus = [torch.device(i) for i in range(num_gpus)]
        scatter_out = scatter_gather.scatter(inp, target_gpus)

        for i, x in enumerate(scatter_out):
            self.assertTrue(isinstance(x, type(inp)))
            self.assertEqual(x._fields, fields)
            expected_a = a_tensors_for_gpu[i]
            expected_b = b_tensors_for_gpu[i]
            self.assertEqual(expected_a, x.a)
            self.assertEqual(expected_b, x.b)

        class TestNamedTupleInput_1(NamedTuple):
            a: torch.tensor
            b: torch.tensor

        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=0)
        a_tensors_for_gpu = [a[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
        b_tensors_for_gpu = [b[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
        inp = TestNamedTupleInput_1(a, b)

        scatter_out = scatter_gather.scatter(inp, target_gpus)
        for i, x in enumerate(scatter_out):
            self.assertTrue(isinstance(x, type(inp)))
            self.assertEqual(x._fields, fields)
            expected_a = a_tensors_for_gpu[i]
            expected_b = b_tensors_for_gpu[i]
            self.assertEqual(expected_a, x.a)
            self.assertEqual(expected_b, x.b)

    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    def test_gather_namedtuple(self):
        # tests ability to gather a list of namedtuples and return a namedtuple where each
        # element is of the expected tensor type.
        fields = ["a", "b"]
        TestNamedTupleInput_0 = collections.namedtuple("NamedTuple", fields)

        num_gpus = torch.cuda.device_count()
        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=1)
        out1 = TestNamedTupleInput_0(a, b)

        a = torch.rand(num_gpus * 2, device=1)
        b = torch.rand(num_gpus * 2, device=0)
        out2 = TestNamedTupleInput_0(a, b)

        outputs = [out1, out2]

        out = scatter_gather.gather(outputs, "cpu")  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))  # x must be a tensor
            cat = torch.cat((outputs[0][i].to("cpu"), outputs[1][i].to("cpu")))
            self.assertTrue(torch.equal(x, cat))

        out = scatter_gather.gather(outputs, 0)  # test on GPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            self.assertTrue(torch.equal(x, cat))

        class TestNamedTupleInput_1(NamedTuple):
            a: torch.tensor
            b: torch.tensor

        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=1)
        out1 = TestNamedTupleInput_1(a, b)

        a = torch.rand(num_gpus * 2, device=1)
        b = torch.rand(num_gpus * 2, device=0)
        out2 = TestNamedTupleInput_1(a, b)

        outputs = [out1, out2]

        out = scatter_gather.gather(outputs, 0)  # test on GPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            self.assertTrue(torch.equal(x, cat))

        out = scatter_gather.gather(outputs, "cpu")  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to("cpu"), outputs[1][i].to("cpu")))
            self.assertTrue(torch.equal(x, cat))


instantiate_parametrized_tests(TestCudaMultiGPU)


if __name__ == "__main__":
    run_tests()
