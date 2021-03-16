from itertools import repeat, chain, product
from typing import NamedTuple
import collections
import gc
import io
import os
import pickle
import queue
import sys
import tempfile
import threading
import unittest

import torch
import torch.cuda
import torch.cuda.comm as comm
from torch import multiprocessing as mp
from torch.nn.parallel import scatter_gather
from torch.utils.checkpoint import checkpoint_sequential
from torch._six import inf, nan

from test_torch import AbstractTestCases

from torch.testing._internal.common_methods_invocations import tri_tests_args, tri_large_tests_args, \
    _compare_trilu_indices, _compare_large_trilu_indices
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, run_tests, \
    NO_MULTIPROCESSING_SPAWN, skipIfRocm, load_tests, IS_REMOTE_GPU, IS_SANDCASTLE, IS_WINDOWS, \
    slowTest, skipCUDANonDefaultStreamIf, TEST_WITH_ROCM, TEST_NUMPY
from torch.testing._internal.autocast_test_lists import AutocastTestLists

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA and TEST_MULTIGPU from torch.testing._internal.common_cuda here,
# because if we do that, the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811

TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_CUDNN = TEST_CUDA
if TEST_CUDA:
    torch.ones(1).cuda()  # initialize cuda context
    TEST_CUDNN = TEST_CUDA and (TEST_WITH_ROCM or
                                torch.backends.cudnn.is_acceptable(torch.tensor(1., device=torch.device('cuda:0'))))
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9

types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.CharTensor,
    torch.ByteTensor,
    torch.HalfTensor,
]


def make_sparse_tensor(t, n, *sizes):
    assert t.is_sparse
    tensor = t()
    i = tensor._indices()
    i = i.new(len(sizes), n).copy_(
        torch.cat([torch.LongTensor(1, n).random_(s) for s in sizes], 0))
    v = tensor._values()
    v = v.new(n).copy_(torch.randn(n))
    return t(i, v, torch.Size(sizes))

_cycles_per_ms = None


def get_cycles_per_ms():
    """Approximate number of cycles per millisecond for torch.cuda._sleep"""
    global _cycles_per_ms
    if _cycles_per_ms is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        _cycles_per_ms = 1000000 / start.elapsed_time(end)
    return _cycles_per_ms


class TestCuda(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super(TestCuda, self).setUp()
        self.autocast_lists = AutocastTestLists(torch.device('cuda:0'))

    def tearDown(self):
        del self.autocast_lists
        super(TestCuda, self).tearDown()

    def _check_memory_stat_consistency(self):
        snapshot = torch.cuda.memory_snapshot()

        expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

        for segment in snapshot:
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            expected["segment.all.current"] += 1
            expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]

        for device, expected in expected_each_device.items():
            stats = torch.cuda.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

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

            if empty_cache:
                torch.cuda.empty_cache()
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

    def test_cudart_register(self):
        t = torch.ones(20)
        self.assertFalse(t.is_pinned())
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        self.assertEqual(r, 0)
        self.assertTrue(t.is_pinned())
        r = cudart.cudaHostUnregister(t.data_ptr())
        self.assertEqual(r, 0)
        self.assertFalse(t.is_pinned())

    def test_memory_stats(self):
        gc.collect()
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    def test_memory_allocation(self):
        gc.collect()
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.cuda.memory_allocated()
            mem = torch.cuda.caching_allocator_alloc(size)
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                self.assertEqual(torch.cuda.memory_allocated(), prev)

    def test_check_error(self):
        # Assert this call doesn't raise.
        torch.cuda.check_error(0)

        with self.assertRaisesRegex(torch.cuda.CudaError,
                                    "out of memory|hipErrorOutOfMemory"):
            torch.cuda.check_error(2)

    def test_cuda_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    def test_cuda_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

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
        gen0 = self._test_memory_stats_generator(self, device='cuda:0', N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device='cuda')

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 8000000000.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device='cuda')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    def test_set_per_process_memory_fraction(self):
        # test invalid fraction value.
        with self.assertRaisesRegex(TypeError, "Invalid type"):
            torch.cuda.set_per_process_memory_fraction(int(1))
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.cuda.set_per_process_memory_fraction(-0.1)
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.cuda.set_per_process_memory_fraction(2.0)

        tensor = torch.zeros(1024, device='cuda')
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(0.5, 0)

        # test 0.499 allocation is ok.
        application = int(total_memory * 0.499) - torch.cuda.max_memory_reserved()
        tmp_tensor = torch.empty(application, dtype=torch.int8, device='cuda')
        del tmp_tensor
        torch.cuda.empty_cache()

        application = int(total_memory * 0.5)
        # it will get OOM when try to allocate more than half memory.
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            torch.empty(application, dtype=torch.int8, device='cuda')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

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
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
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
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s0):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s3), torch.cuda.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_streams(self):
        d0 = torch.device('cuda:0')
        x0 = torch.zeros(5, 5, device=d0)

        d1 = torch.device('cuda:1')
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)

        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_to_non_blocking(self):
        stream = torch.cuda.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            torch.cuda.synchronize()
            # Pushes an 0.1 second spin to stream so if the copy is non blocking,
            # stream will almost surely be active when we query().
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            b = a.to(device=dst, non_blocking=non_blocking)
            self.assertEqual(stream.query(), not non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in product(("cuda", "cpu"), (True, False)):
            # Creates source on the opposite device from destination.
            src = torch.randn(1000000,
                              device="cuda" if dst == "cpu" else "cpu",
                              pin_memory=True if dst == "cuda" else False)
            _test_to_non_blocking(src, try_non_blocking, dst)

    def test_to_cpu_blocking_by_default(self):
        src = torch.randn(1000000, device="cuda")
        torch.cuda.synchronize()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        dst = src.to(device="cpu")
        self.assertEqual(torch.cuda.current_stream().query(), True)
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.cuda.IntStorage))
        q_copy[1].fill_(10)
        self.assertTrue(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    def test_cublas_allow_tf32_get_set(self):
        orig = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        torch.backends.cuda.matmul.allow_tf32 = orig

    def test_cudnn_allow_tf32_get_set(self):
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().double(), torch.cuda.DoubleTensor)
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().double(), torch.cuda.DoubleStorage)
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    @unittest.skip("was disabled due to not enough memory, but actually it always fail")
    def test_arithmetic_large_tensor(self):
        x = torch.empty(2**30, device='cuda')

        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        x += 1
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    def test_gather_bool(self):
        t = torch.tensor([[False, True], [True, True]], device='cuda')
        self.assertEqual(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device='cuda')),
                         torch.tensor([[False, False], [True, True]], device='cuda'))

    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_cat_autogpu(self):
        x = torch.randn(4, 4).cuda(1)
        y = torch.randn(4, 4).cuda(1)
        z = torch.cat([x, y], 0)
        self.assertEqual(z.get_device(), x.get_device())

    @unittest.skipIf(torch.cuda.device_count() >= 10, "Loading a cuda:9 tensor")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:9' restore location
        tensor = torch.randn(2, device='cuda')
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # NB: this might not work in the future if serialization changes
        buf = io.BytesIO(buf.getvalue().replace(b'cuda:0', b'cuda:9'))

        msg = r'Attempting to deserialize object on CUDA device 9'
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    def test_specify_improper_device_name(self):
        import os
        fname = "tempfile.pt"
        try:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                torch.save([torch.nn.Parameter(torch.randn(10, 10))], fname,
                           _use_new_zipfile_serialization=True)
                torch.load(fname, 'cuda0')
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_get_device_index(self):
        from torch.cuda._utils import _get_device_index
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index('cuda0', optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            cpu_device = torch.device('cpu')
            _get_device_index(cpu_device, optional=True)

    def test_serialization_array_with_empty(self):
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_serialization_remap(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]

        def gpu_remap(storage, location):
            if location == 'cuda:1':
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
            x_copy = torch.load(f, map_location={'cuda:1': 'cuda:0'})
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_multigpu_storage_clone(self):
        x = torch.randn(4, 4, device='cuda:1').storage()
        y = x.clone()
        self.assertEqual(x.get_device(), y.get_device())
        for t in ['byte', 'char', 'short', 'int', 'long', 'half', 'double']:
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

    def test_cuda_synchronize(self):
        torch.cuda.synchronize()
        torch.cuda.synchronize('cuda')
        torch.cuda.synchronize('cuda:0')
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(torch.device('cuda:0'))

        if TEST_MULTIGPU:
            torch.cuda.synchronize('cuda:1')
            torch.cuda.synchronize(1)
            torch.cuda.synchronize(torch.device('cuda:1'))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize("cpu")

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_current_stream(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

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

        with self.assertRaisesRegex(ValueError,
                                    "Expected a cuda device, but got: cpu"):
            torch.cuda.current_stream(torch.device('cpu'))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

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

        with self.assertRaisesRegex(ValueError,
                                    "Expected a cuda device, but got: cpu"):
            torch.cuda.default_stream(torch.device('cpu'))

    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        tensor2 = tensor1.cuda(non_blocking=True) + 1
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_device(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')
        e0 = torch.cuda.Event()

        self.assertEqual(None, e0.device)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device('cuda:0'))
        self.assertEqual(e0.device, torch.device('cuda:0'))
        self.assertEqual(s1.device, torch.device('cuda:1'))
        self.assertEqual(e1.device, torch.device('cuda:1'))

    def test_stream_event_repr(self):
        s = torch.cuda.current_stream()
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.cuda.Event" in e.__repr__())

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
        self.assertEqual(default_stream.device, torch.device('cuda:0'))
        stream = torch.cuda.Stream(device=1)
        self.assertEqual(stream.device, torch.device('cuda:1'))
        with torch.cuda.device(1):
            self.assertEqual(
                torch.cuda.current_stream().device, torch.device('cuda:1'))
            self.assertNotEqual(torch.cuda.current_stream(), default_stream)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu_query(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)

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
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

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
        self.assertEqual(torch.device('cuda:0'), s0.device)

        s1 = torch.cuda.Stream(device=1, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device('cuda:1'), s1.device)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_tensor_device(self):
        self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 0)
        self.assertEqual(torch.cuda.FloatTensor(1, device=1).get_device(), 1)
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 1)
            self.assertEqual(torch.cuda.FloatTensor(1, device=0).get_device(), 0)
            self.assertEqual(torch.cuda.FloatTensor(1, device=None).get_device(), 1)

    def test_events(self):
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

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
        with torch.cuda.device('cuda:1'):
            c2p.put(0)
            p2c.get()
            c2p.put(sync_func(self, TestCuda.FIFTY_MIL_CYCLES))

    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_nogil(self):
        for sync_func in [TestCuda._stream_synchronize,
                          TestCuda._event_synchronize,
                          TestCuda._event_wait]:
            p2c = queue.Queue()
            c2p = queue.Queue()
            e_tik = torch.cuda.Event(enable_timing=True)
            e_tok = torch.cuda.Event(enable_timing=True)

            t = threading.Thread(
                target=TestCuda._test_stream_event_nogil,
                args=(self, sync_func, p2c, c2p))
            t.daemon = True
            t.start()

            c2p.get()
            with torch.cuda.device('cuda:0'):
                e_tik.record()
                p2c.put(0)
                parent_time = sync_func(self, TestCuda.FIFTY_MIL_CYCLES)
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
            # real execution time by least 40%.
            self.assertGreater(parent_time + child_time, total_time * 1.4)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_events_wait(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
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
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = s0.record_event()
            s0.synchronize()

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
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
    @skipIfRocm
    def test_events_multi_gpu_elapsed_time(self):
        d0 = torch.device('cuda:0')
        d1 = torch.device('cuda:1')

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()
            e0 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(10)
            s0.record_event(e0)

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
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
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            s0.record_event(e2)
            s0.synchronize()

        self.assertGreater(e0.elapsed_time(e2), 0)

        # deliberately calling from a different device
        with torch.cuda.device(d1):
            self.assertGreater(e0.elapsed_time(e2), 0)

    def test_record_stream(self):
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.cuda.FloatTensor(t.size())
        stream = torch.cuda.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            with torch.cuda.stream(stream):
                tmp = t.cuda(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.cuda.current_stream().wait_stream(stream)
            tmp.record_stream(torch.cuda.current_stream())
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with torch.cuda.stream(stream):
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(tmp2.data_ptr(), ptr[0], msg='allocation re-used to soon')

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        # Check that the block will be re-used after the main stream finishes
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(stream):
            tmp3 = torch.cuda.FloatTensor(t.size())
            self.assertEqual(tmp3.data_ptr(), ptr[0], msg='allocation not re-used')

    def test_record_stream_on_shifted_view(self):
        # See issue #27366

        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        stream_alloc = torch.cuda.Stream()
        with torch.cuda.stream(stream_alloc):
            base = torch.cuda.FloatTensor([10, 10])

        # Record another stream on a shifted view tensor.
        view = base[5:]
        assert view.storage_offset() > 0

        stream_record = torch.cuda.Stream()
        with torch.cuda.stream(stream_record):
            torch.cuda._sleep(int(50 * get_cycles_per_ms()))

        view.record_stream(stream_record)

        # Delete those tensors to make the block free soon.
        data_ptr = base.data_ptr()
        del base, view

        # A new tensor should not be allocated to the block above.
        stream_alloc.synchronize()

        with torch.cuda.stream(stream_alloc):
            try_realloc = torch.cuda.FloatTensor([10, 10])

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)

    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        cycles_per_ms = get_cycles_per_ms()

        # check that allocations are re-used after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, msg='allocation not reused')

        # check that the allocation is not re-used if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg='allocation re-used too soon')
        self.assertEqual(list(gpu_tensor), [1])

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_caching_pinned_memory_multi_gpu(self):
        # checks that the events preventing pinned memory from being re-used
        # too early are recorded on the correct GPU
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        gpu_tensor0 = torch.cuda.FloatTensor([0], device=0)
        gpu_tensor1 = torch.cuda.FloatTensor([0], device=1)

        with torch.cuda.device(1):
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            gpu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg='allocation re-used too soon')

        with torch.cuda.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(gpu_tensor1[0], 1)
        self.assertEqual(gpu_tensor0[0], 2)

    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cuda_malloc_retry. see issue #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device='cuda')

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device='cuda')
            with torch.cuda.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.cuda.empty_cache()

    # Tests for historic illegal memory access, see #17040.
    def test_reduction_gpu_memory_accessing(self):
        x = torch.ones(512, 8, dtype=torch.float32, device='cuda')
        torch.sum(x, 0)

    def test_sum_fp16(self):
        x = torch.zeros(10, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        x = torch.ones(65504, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(), 65504)
        self.assertEqual(x.sum(dtype=torch.float32), 65504)

        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device='cuda', dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    def test_mean_fp16(self):
        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        x = torch.ones(65536, device='cuda', dtype=torch.float16)
        self.assertEqual(x.mean(dtype=torch.float32), 1)

    def test_prod_large(self):
        # tests global reduction (should_global_reduce = true) in case of non-zero identity element
        x = torch.ones(240000, device='cuda', dtype=torch.float32)
        self.assertEqual(x.prod(), 1)

    def test_multinomial_ext(self):
        # Test two corner cases from older PyTorch (Issue #4858)
        freqs = torch.cuda.FloatTensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.03178183361887932, 0.027680952101945877, 0.033176131546497345,
            0.046052902936935425, 0.07742464542388916, 0.11543981730937958,
            0.14148041605949402, 0.15784293413162231, 0.13180233538150787,
            0.08271478116512299, 0.049702685326337814, 0.027557924389839172,
            0.018125897273421288, 0.011851548217236996, 0.010252203792333603,
            0.007422595750540495, 0.005372154992073774, 0.0045109698548913,
            0.0036087757907807827, 0.0035267581697553396, 0.0018864056328311563,
            0.0024605290964245796, 0.0022964938543736935, 0.0018453967059031129,
            0.0010662291897460818, 0.0009842115687206388, 0.00045109697384759784,
            0.0007791675161570311, 0.00020504408166743815, 0.00020504408166743815,
            0.00020504408166743815, 0.00012302644609007984, 0.0,
            0.00012302644609007984, 4.100881778867915e-05, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0])

        torch.cuda.manual_seed(11042)
        sample = torch.multinomial(freqs, 1000, True)
        self.assertNotEqual(freqs[sample].min(), 0)

        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        p[:, 1] = 1
        torch.cuda.manual_seed(5214)
        r = torch.multinomial(p, 1)
        self.assertNotEqual(r.min().item(), 0)

        # test corner case from Issue #13867
        torch.cuda.manual_seed(33)
        probs = torch.randn(1000000, device='cuda').clamp(min=0) * 3e-5
        samples = probs.multinomial(1000000, replacement=True)
        self.assertGreater(probs[samples].min().item(), 0)

    @staticmethod
    def mute():
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())

    def _spawn_method(self, method, arg):
        ctx = mp.get_context("spawn")
        with ctx.Pool(1, initializer=self.mute) as pool:
            errors = pool.map(method, [arg])
            for e in errors:
                if 'device-side assert triggered' not in str(e):
                    self.fail(e)

    @staticmethod
    def _test_multinomial_invalid_probs_cuda(probs):
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.multinomial(probs.to('cuda'), 2, replacement=True)
                torch.cuda.synchronize()
            return False  # Should not be reached
        except RuntimeError as e:
            return e

    @slowTest
    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @skipIfRocm
    def test_multinomial_invalid_probs_cuda(self):
        test_method = TestCuda._test_multinomial_invalid_probs_cuda
        self._spawn_method(test_method, torch.Tensor([1, -1, 1]))
        self._spawn_method(test_method, torch.Tensor([1, inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, -inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, 1, nan]))

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_huge_index(self):
        src = torch.empty(15000000, 45, device='cuda', dtype=torch.long).random_(0, 2**22)
        idx = torch.randperm(src.shape[0], device='cuda')
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_tensor_gather(self):
        AbstractTestCases._TestTorchMixin._test_gather(self, lambda t: t.cuda(), False)

    def test_tensor_scatter(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', test_bounds=False)

    def test_tensor_scatterAdd(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_add_', test_bounds=False)

    def test_scatter_add_mult_index_base(self):
        AbstractTestCases._TestTorchMixin._test_scatter_add_mult_index_base(self, lambda t: t.cuda())

    def test_tensor_scatterFill(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(),
                                                             'scatter_', True, test_bounds=False)

    def test_tensor_scatter_complex(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(),
                                                             'scatter_', test_bounds=False, test_complex=True)

    def test_tensor_scatterAdd_complex(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(),
                                                             'scatter_add_', test_bounds=False, test_complex=True)

    def test_tensor_scatterFill_complex(self):
        AbstractTestCases._TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(),
                                                             'scatter_', True, test_bounds=False, test_complex=True)

    def test_min_max_inits(self):
        # Testing if THC_reduceAll received the correct index initialization.
        # This affects the result of THC_reduceAll operations at extreme values
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

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

    def test_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()

    def test_bincount_ext(self):
        # ensure CUDA code coverage
        input_size = (5000,)
        w = torch.randn(input_size, dtype=torch.double, device='cuda')
        w_cpu = w.cpu()
        # test shared memory impl
        t = torch.randint(50, input_size, dtype=torch.int8, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test multi block memory impl
        # see `THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM` in SummaryOps.cu
        t = torch.randint(500, input_size, dtype=torch.int64, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test global memory impl
        # see `THRESH_NUMBER_BINS_FOR_GLOBAL_MEM` in SummaryOps.cu
        t = torch.randint(2000, input_size, dtype=torch.int64, device='cuda')
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device='cuda')
        # 35488 * 65536 as int32 would cause overflow to negative value
        # giving negative bin offset
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_tiny_half_norm_(self):
        a = torch.arange(25).cuda().float()
        a /= 100000000
        b = a.half()
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        a = torch.ones(65536).cuda().half()
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_cuda_memory_leak_detection(self):
        l = []

        @self.wrap_with_cuda_memory_check
        def no_leak():
            pass

        @self.wrap_with_cuda_memory_check
        def leak_gpu0():
            l.append(torch.tensor(10, device=torch.device("cuda:0")))

        no_leak()

        with self.assertRaisesRegex(AssertionError, r"leaked \d+ bytes CUDA memory on device 0"):
            leak_gpu0()

        if TEST_MULTIGPU:
            @self.wrap_with_cuda_memory_check
            def leak_gpu1():
                l.append(torch.tensor(10, device=torch.device("cuda:1")))

            with self.assertRaisesRegex(AssertionError, r"leaked \d+ bytes CUDA memory on device 1"):
                leak_gpu1()

    def test_cuda_memory_leak_detection_propagates_errors(self):
        with self.assertRaisesRegex(RuntimeError, r"The size of tensor a \(3\) must match"):
            with self.assertLeaksNoCudaTensors():
                x = torch.randn(3, 1, device='cuda')
                y = torch.randn(2, 1, device='cuda')
                z = x + y

    def test_trilu_indices(self):
        for test_args in tri_tests_args:
            _compare_trilu_indices(self, *test_args, device='cuda')

        # test default options
        x = torch.ones(
            3, 3, dtype=torch.long, device='cuda', layout=torch.strided)
        self.assertEqual(
            x.tril(0).nonzero().transpose(0, 1),
            torch.tril_indices(3, 3, device='cuda'))
        self.assertEqual(
            x.triu(0).nonzero().transpose(0, 1),
            torch.triu_indices(3, 3, device='cuda'))

    def test_large_trilu_indices(self):
        for test_args in tri_large_tests_args:
            _compare_large_trilu_indices(self, *test_args, device='cuda')

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    def test_cuda_kernel_loop_overflow(self):
        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**30 + 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**30]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**30], expected)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_cuda_kernel_loop_overflow_large(self):
        # Make sure input.numel() > INT_MAX is handled:
        x = torch.randn(1, 1, 1, 2**31, dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            y = torch.nn.functional.avg_pool2d(x, kernel_size=1)

        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**31 - 2]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**31 - 2], expected)

    @skipCUDANonDefaultStreamIf(True)
    def test_streaming_backwards_sync(self):
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()

        class MultiplyInStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(torch.cuda.current_stream(), stream)
                # delays the operation in the the background stream
                torch.cuda._sleep(1000 * 1000)
                return grad * 2

        x = torch.randn(5, 5, device='cuda', requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x)
            output.sum().backward()

        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        self.assertEqual(torch.cuda.current_stream(), default_stream)

    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    def test_streaming_backwards_multiple_streams(self):

        class StreamModel(torch.nn.Module):
            def __init__(self):
                super(StreamModel, self).__init__()
                self.event = torch.cuda.Event()
                self.stream0 = torch.cuda.Stream()
                self.stream1 = torch.cuda.Stream()

            def forward(self, x):
                x0 = x.clone()
                torch._C._cuda_setStream(self.stream0._cdata)
                y0 = x0 * 2
                self.event.record(stream=torch.cuda.current_stream())

                torch._C._cuda_setStream(self.stream1._cdata)
                y1 = x * 3
                self.stream1.wait_event(self.event)
                return y0 + y1

        stream = torch.cuda.Stream()

        def accum_hook(grad):
            self.assertEqual(torch.cuda.current_stream(), stream)

        with torch.cuda.stream(stream):
            x = torch.randn(5, 5, device='cuda', requires_grad=True)
            x.register_hook(accum_hook)
            torch.cuda.current_stream().wait_stream(stream)
            model = StreamModel().cuda()
            model(x).sum().backward()

        self.assertEqual(x.grad, torch.ones_like(x) * 5)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
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
        s0 = to_backward_recipient.to(device="cuda:0").sum() * 2.
        s1 = to_backward_recipient.to(device="cuda:0").sum() * 2.
        torch.cuda.synchronize(device=dev0)
        torch.cuda.synchronize(device=dev1)
        s0.backward(retain_graph=True)
        s1.backward()
        self.assertTrue(a.grad.sum().item() == 4 * size)
        self.assertTrue(b.grad.sum().item() == 4 * size)

    def test_streaming_backward_sync_graph_root(self):
        # This function tests if bwd ops running on a side stream properly sync with the GraphRoot.
        # The potential bug it targets is a race condition. The test uses multiple trials and
        # torch.cuda._sleep such that if the race condition exists, the test will almost certainly fail,
        # but there's a chance it may spuriously pass. Passing does not guarantee the backend is bug-free,
        # but failure does guarantee there is a bug.
        fwd_bwd_op_stream = torch.cuda.Stream()
        bwd_ambient_stream = torch.cuda.Stream()
        # We need these streams to be different otherwise the test is meaningless.
        self.assertTrue(fwd_bwd_op_stream != bwd_ambient_stream)

        size = int(1e3)

        a = torch.full((size,), 2.0, device="cuda", requires_grad=True)
        b = torch.full((size,), 3.0, device="cuda", requires_grad=True)

        # I don't think we need any manual record_streams below.
        # a and b remain in scope for the entire test.
        # c and grad remain in scope for each iteration, and there's a full sync between iterations.
        for trial in range(5):
            torch.cuda.synchronize()
            a.grad = b.grad = None
            with torch.cuda.stream(fwd_bwd_op_stream):
                c = a * b

            with torch.cuda.stream(bwd_ambient_stream):
                torch.cuda.synchronize()
                # Long-running dummy kernel on bwd_ambient_stream delays filling of grad
                torch.cuda._sleep(int(50 * get_cycles_per_ms()))
                # Fills grad on bwd_ambient_stream
                grad = torch.full((size,), float(trial + 1), device="cuda")

                # Bwd ops still run on fwd_bwd_ops_stream, so the following will likely fail if
                # bwd ops don't sync with bwd_ambient_stream before consuming grad.
                torch.autograd.backward(tensors=c, grad_tensors=grad)

                # See https://github.com/pytorch/pytorch/issues/47028
                # assertEquals below run on bwd_ambient_stream, so this test may also fail
                # if backward() fails to sync with bwd_ambient_stream at the end.
                # Synchronizing here works around the issue until a proper fix can be made.
                torch.cuda.synchronize()
                with torch.no_grad():
                    self.assertEqual(a.grad, grad * b)
                    self.assertEqual(b.grad, grad * a)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @unittest.skipIf(IS_SANDCASTLE or IS_REMOTE_GPU, "Does not work on Sandcastle")
    def test_cuda_init_race(self):
        # See https://github.com/pytorch/pytorch/issues/16559
        import subprocess
        subprocess.check_call([sys.executable, '-c', """\
import torch
import threading

def worker(rank):
    torch.tensor([1.]).cuda(rank)

t1 = threading.Thread(target=worker, args=(0,))
t2 = threading.Thread(target=worker, args=(1,))
t1.start()
t2.start()
"""])

    # ROCm doesn't support device side asserts
    @skipIfRocm
    def test_fixed_cuda_assert_async(self):
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([], device="cuda"))
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0], device="cuda"))

        torch._assert_async(torch.tensor(1, device="cuda"))
        torch._assert_async(torch.tensor(0.1, device="cuda"))
        torch._assert_async(torch.tensor(-0.1, device="cuda"))
        torch._assert_async(torch.tensor(True, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0.1j, device="cuda"))

        fail_stmts = [
            "torch._assert_async(torch.tensor(0, device='cuda'))",
            "torch._assert_async(torch.tensor(0.0, device='cuda'))",
            "torch._assert_async(torch.tensor(False, device='cuda'))",
            "torch._assert_async(torch.tensor(0 + 0j, device='cuda'))",
        ]

        import subprocess
        for stmt in fail_stmts:
            with self.subTest(stmt=stmt):
                r = subprocess.call([sys.executable, '-c', f"""\
import torch

{stmt}
torch.cuda.synchronize()
"""])
                self.assertTrue(r != 0)


    def test_grad_scaling_unscale(self, dtype=torch.float):
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device="cuda:0")
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="cuda:0")

        size = 10
        g = torch.full((size, size), 4.0, dtype=dtype, device="cuda:0")
        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertTrue(torch.allclose(grad, torch.ones_like(grad), atol=1e-7))

        # Passing lists with mismatched devices or dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call should raise errors.
        with self.assertRaisesRegex(RuntimeError, r"must have the same dtype"):
            torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(dtype=torch.float16)],
                                                             found_inf,
                                                             inv_scale)

        if TEST_MULTIGPU:
            with self.assertRaisesRegex(RuntimeError, r"scaled_grads must be on the same device."):
                torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(device="cuda:1")],
                                                                 found_inf,
                                                                 inv_scale)

        # Creates a list of grads with mismatched dtypes and devices, to ensure
        # scaler._unscale_grads_ organizes grads by dtype and device before calling
        # _amp_foreach_non_finite_check_and_unscale_ on each set.
        # If inject_inf >= 0, writes an inf into one grad for _unscale_grads_ to find.
        def perfect_storm_grads(inject_inf):
            grads = [g.clone(), g.clone()[:, :5], g.to(dtype=torch.float16), g.to(dtype=torch.float16)]
            if TEST_MULTIGPU:
                grads += [g.to(device="cuda:1"),
                          g.to(device="cuda:1")[:, :5],
                          g.to(device="cuda:1", dtype=torch.float16),
                          g.to(device="cuda:1", dtype=torch.float16)]
            if inject_inf >= 0:
                grads[inject_inf][2, 2] = float('inf')
            return grads

        scaler = torch.cuda.amp.GradScaler()
        dummy_params = [torch.empty_like(g) for g in perfect_storm_grads(-1)]
        dummy_opt = torch.optim.SGD(dummy_params, lr=1.)

        # Ensures the inf/nan checking can find an inf injected onto any grad in the perfect storm.
        for inject_inf in range(-1, len(dummy_params)):
            found_inf = torch.full((1,), 0.0, dtype=torch.float, device="cuda:0")
            grads = perfect_storm_grads(inject_inf)
            for i, p in enumerate(dummy_params):
                p.grad = grads[i]
            found_inf_per_device = scaler._unscale_grads_(dummy_opt, inv_scale, found_inf, True)
            if inject_inf < 0:
                # No inf was injected, ensures unscaling worked normally.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 0)
                for grad in grads:
                    self.assertTrue(torch.allclose(grad, torch.ones_like(grad), atol=1e-7))
            else:
                # inf was injected, ensures inf was found.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 1)

    def test_grad_scaling_update_scale(self, device="cuda", dtype=torch.float):
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = torch.full((1,), 4.0, dtype=dtype, device=device)
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="cuda:0")

        # Simulates 2 consecutive unskipped iterations
        scale = torch._amp_update_scale(growth_tracker, scale, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 1)
        self.assertEqual(scale, 4.0)
        scale = torch._amp_update_scale(growth_tracker, scale, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 8.0)

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        scale = torch._amp_update_scale(growth_tracker, scale, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 2.0)

    def test_grad_scaling_unscale_sparse(self, device="cuda", dtype=torch.float):
        scaler = torch.cuda.amp.GradScaler()

        inv_scale = torch.full((1,), 0.25, dtype=dtype, device=device)
        found_inf = torch.empty((1,), dtype=dtype, device=device)
        cur = found_inf.device

        # As of d0c925f (4/16/20), docs are unclear about best API for sparse cuda tensor construction.
        # https://pytorch.org/docs/master/tensors.html shows torch.sparse_coo_tensor(...), but it has no docstring.
        # The same page shows several tensors with layout=torch.sparse_coo, but no constructors using that layout.
        # Meanwhile, https://pytorch.org/docs/master/sparse.html shows torch.sparse.FloatTensor(...), which looks
        # legacy and does not accept a device="cuda" kwarg.  Going with torch.sparse_coo_tensor.
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]], device="cuda", dtype=torch.int64)
        v = torch.tensor([16., 32., 64.], device="cuda", dtype=torch.float)
        s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device="cuda", dtype=dtype)

        p = s.clone()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.)

        p.grad = s.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertTrue(torch.allclose(p.grad.to_dense(), (s / 4).to_dense()))

        v = torch.FloatTensor([16., 32., float('inf')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device="cuda", dtype=dtype)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        v = torch.FloatTensor([16., 32., float('nan')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device="cuda", dtype=dtype)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        p = s.clone().half()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.)

        p.grad = s.clone().half()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertTrue(torch.allclose(p.grad.to_dense(), (s.half() / 4).to_dense()))

        # Creates fp16 sparse tensor with duplicated indices (uncoalesced).  The uncoalesced representation
        # does not overflow in fp16, but the coalesced representation would, because 64000 + 64000 > fp16 max.
        # _amp_non_finite_check_and_unscale_ should report an overflow here.
        i = torch.LongTensor([[0, 1, 0],
                              [2, 0, 2]])
        v = torch.FloatTensor([64000., 32., 64000.])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device="cuda", dtype=torch.float16)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 1.0)

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
        scaler = torch.cuda.amp.GradScaler(init_scale=2.)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:0")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), (t1.clone(), t0.clone())])
        outputs = scaler.scale(outputs)
        self.assertTrue(outputs[0] == 8.0 and outputs[1][0] == 8.0 and outputs[1][1] == 8.0 and
                        outputs[2][0] == 8.0 and outputs[2][1][0] == 8.0 and outputs[2][1][1] == 8.0)
        self.assertTrue(scaler._scale.device == t1.device)

    def test_grad_scaling_state_dict(self):
        for lazy_init_scale in True, False:
            s0 = torch.cuda.amp.GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            s1 = torch.cuda.amp.GradScaler(init_scale=6., growth_factor=7., backoff_factor=.8, growth_interval=1)

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device="cuda:0"))
                self.assertTrue(isinstance(s1._scale, torch.cuda.FloatTensor))

            s1.load_state_dict(s0.state_dict())

            self.assertEqual(s1.get_scale(), 3.)
            self.assertEqual(s1.get_growth_factor(), 4.)
            self.assertEqual(s1.get_backoff_factor(), .5)
            self.assertEqual(s1.get_growth_interval(), 2)
            self.assertEqual(s1._init_growth_tracker, 0)

    def _create_scaling_models_optimizers(self, device="cuda"):
        # Create a module+optimizer that will use scaling, and a control module+optimizer
        # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
        mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.data.copy_(c.data)

        opt_control = torch.optim.SGD(mod_control.parameters(), lr=1.0)
        opt_scaling = torch.optim.SGD(mod_scaling.parameters(), lr=1.0)

        return mod_control, mod_scaling, opt_control, opt_scaling

    def _create_scaling_case(self, device="cuda", dtype=torch.float):
        data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

        loss_fn = torch.nn.MSELoss().cuda()

        skip_iter = 2

        return self._create_scaling_models_optimizers(device=device) + (data, loss_fn, skip_iter)

    # _run_scaling_case generalizes some single-optimizer test logic to avoid too much copy-pasting below.
    def _run_scaling_case(self, run, unskipped, skipped, atol=1e-7):
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter = self._create_scaling_case()

            # For functionality, test with a modest initial scale, and an unrealistically-large growth factor
            # so any potential errors with the growth factor handling will be magnified.
            scaler = torch.cuda.amp.GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            _ = run(data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            # If scaling was enabled, the scale factor should have been multiplied by the growth factor
            # len(data) - skipped times and the backoff factor "skipped" times.
            if enabled:
                net_growth = scaler.get_growth_factor()**unskipped if unskipped > 0 else 1.0
                net_backoff = scaler.get_backoff_factor()**skipped if skipped > 0 else 1.0
                self.assertTrue(scaler.get_scale() == (128. * net_growth * net_backoff))
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                self.assertTrue(torch.allclose(c, s, atol=atol))

    # Compares no scaling + no autocasting against scaling + autocasting.
    def test_grad_scaling_autocast(self):
        try_pickle = False

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=try_scaling_api):
                    output = model(input)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)
        # this will be picked up by try_pickle within run():
        try_pickle = True
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    def test_grad_scaling_clipping(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_clipping_separate_unscale(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    @unittest.skipIf(IS_WINDOWS, 'FIXME: fix this test for Windows')
    def test_grad_scaling_penalty(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(scaler.scale(loss),
                                                      model.parameters(), create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float('inf'))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_accumulation(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            iters_to_accumulate = 2
            for i, (input, target) in enumerate(data):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(run, unskipped=2, skipped=0)

    def test_grad_scaling_multiple(self):
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                self._create_scaling_case()
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                self._create_scaling_models_optimizers()

            scaler = torch.cuda.amp.GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float('inf'))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
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
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                self.assertTrue(torch.allclose(c, s, atol=1e-7))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    def test_grad_scaling_multigpu(self):
        # Same as above, but runs some of the models on device 1.
        # GradScaler should transparently handle losses and gradients on multiple devices.
        # This test could be combined with the test above, but I think it makes sense to treat
        # multi-GPU operations separately.
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")

        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                self._create_scaling_case()
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                self._create_scaling_models_optimizers(device=dev1)

            scaler = torch.cuda.amp.GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input.to(dev1))
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1.to(dev0), target)
                    loss1 = loss_fn(0.6 * output0.to(dev1) - 0.4 * output1, target.to(dev1))

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float('inf'))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        # Make sure the found_infs were collected properly across optimizers and devices.
                        if scaler.is_enabled():
                            self.assertTrue(len(scaler._found_inf_per_device(optimizer0)) == 1)
                            self.assertTrue(len(scaler._found_inf_per_device(optimizer1)) == 1)
                            self.assertTrue(scaler._found_inf_per_device(optimizer0)[dev0].item() == 0.)
                            self.assertTrue(scaler._found_inf_per_device(optimizer1)[dev1].item() ==
                                            float(i == skip_iter))

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
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            # Copy mod_control1 and mod_scaling1 back the device 0 for comparison
            mod_control1.to(dev0)
            mod_scaling1.to(dev0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                self.assertTrue(torch.allclose(c, s, atol=1e-7))

    def test_cublas_multiple_threads_same_device(self):
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device='cuda')
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device='cuda')

            threads = [threading.Thread(target=_worker,
                                        args=(t,)) for t in range(num_threads)]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @skipIfRocm
    def test_cudnn_multiple_threads_same_device(self):
        # This function is intended to test the lazy creation and reuse of per-thread
        # cudnn handles on each device in aten/src/ATen/cudnn/Handles.cpp.
        # Failure here likely indicates something wrong with that logic.
        weight = torch.ones((1, 1, 2, 2), device='cuda')

        results = {}

        num_threads = 2
        trials = 3
        test_iters = 1000
        barrier = threading.Barrier(num_threads)

        with torch.backends.cudnn.flags(enabled=True):
            def _worker(t):
                my_stream = torch.cuda.Stream()
                # Hard sync so we don't need to worry about creating and using tensors
                # across streams or the fact that default streams are thread-local.
                # Those issues are not the target of this test.
                torch.cuda.synchronize()
                # Line up threads to increase likelihood of race conditions.
                barrier.wait()
                with torch.cuda.stream(my_stream):
                    for _ in range(test_iters):
                        # If all threads are sharing the same cudnn handle,
                        # the following sequence may occur:
                        # thread 0 calls setCuDNNStreamToCurrent()
                        # thread 1 calls setCuDNNStreamToCurrent()
                        # thread 0 launches its raw convolution, which it thinks is in
                        #          its own stream, but is actually in thread 1's stream.
                        # thread 0 enqueues its div_, which IS is its own stream,
                        #          but now races with its convolution.
                        results[t] = torch.nn.functional.conv2d(results[t], weight, padding=0)
                        results[t].div_(4.0)
                torch.cuda.synchronize()

            for _ in range(trials):
                for t in range(num_threads):
                    results[t] = torch.ones((1, 1, 2048, 2048), device='cuda')

                threads = [threading.Thread(target=_worker,
                                            args=(t,)) for t in range(num_threads)]

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for t in range(num_threads):
                    self.assertEqual(results[t].sum().item(),
                                     (2048 - test_iters) * (2048 - test_iters))

    def test_cusparse_multiple_threads_same_device(self):
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 500

        def ones_sparse(size):
            a = torch.arange(size, device='cuda')
            indices = torch.cartesian_prod(a, a).t()
            values = torch.ones(size * size, device='cuda')
            return torch.sparse_coo_tensor(indices, values)

        weight = ones_sparse(size)
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = weight.mm(results[t])
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device='cuda')

            threads = [threading.Thread(target=_worker,
                                        args=(t,)) for t in range(num_threads)]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    def _run_autocast_outofplace(self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.is_autocast_enabled())
        with torch.cuda.amp.autocast():
            self.assertTrue(torch.is_autocast_enabled())

            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(out_type == output.dtype,
                                    "autocast for torch.{} produced {}, should produce {}"
                                    .format(op, output.dtype, out_type))

            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(out_type == output_method.dtype,
                                    "autocast for torch.{} produced {}, should produce torch.{}"
                                    .format(op, output_method.dtype, out_type))

            self.assertTrue((output is not None) or (output_method is not None),
                            "{} not found as an attribute on either Tensor or the requested module {}".format(
                            op, module))

            # Accounts for ops that return Tensors, iterables, and other non-Tensors.
            # For example, lstm_cell returns a tuple and equal returns bool.
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all(compare(f, s) for f, s in zip(first, second))
                else:
                    return first == second

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (output_method is not None):
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(comparison, "torch.{0} result did not match Tensor.{0} result".format(op))

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            output_to_compare = output if output is not None else output_method
            with torch.cuda.amp.autocast(enabled=False):
                self.assertFalse(torch.is_autocast_enabled())

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
                else:
                    control = getattr(args[0].to(run_as_type), op)(*cast(args[1:], run_as_type), **add_kwargs)
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, "torch.{} result did not match control".format(op))
            self.assertTrue(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled())

    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_torch_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                if not skip_test:
                    self._run_autocast_outofplace(op, args, torch.float16)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, add_kwargs=maybe_kwargs)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_nn_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.nn_fp16:
                self._run_autocast_outofplace(op, args, torch.float16, module=torch._C._nn)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_nn_fp32(self):
        for op, args in self.autocast_lists.nn_fp32:
            self._run_autocast_outofplace(op, args, torch.float32, module=torch._C._nn)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_methods_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.methods_fp16:
                self._run_autocast_outofplace(op, args, torch.float16, module=None)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_methods_fp32(self):
        for op, args in self.autocast_lists.methods_fp32:
            self._run_autocast_outofplace(op, args, torch.float32, module=None)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, module=None, out_type=out_type)

    def test_autocast_banned(self):
        with torch.cuda.amp.autocast():
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    def test_autocast_ignored_types(self):
        with torch.cuda.amp.autocast():
            for ignore_type in (torch.double, torch.int32):
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                c_16 = torch.ones((8, 8), dtype=torch.float16, device="cuda:0")

                # Tests if CastPolicy::fp16 ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with self.assertRaises(RuntimeError):
                        torch.mm(a_ignore, c_16)
                    with torch.cuda.amp.autocast(enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    self.assertTrue(torch.mm(a_ignore, b_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32 ops ignore double and int
                with torch.cuda.amp.autocast(enabled=False):
                    type_no_autocast = torch.pow(a_ignore, 2.0).dtype
                self.assertTrue(torch.pow(a_ignore, 2.0).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_set_opt_dtype ops ignore double and int
                with torch.cuda.amp.autocast(enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_append_dtype ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.cuda.amp.autocast(enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)

    def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd
            def forward(ctx, a, b):
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad):
                self.assertTrue(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)

        with torch.cuda.amp.autocast():
            output = mymm(x, y)
            self.assertTrue(output.dtype is torch.float16)
            loss = output.sum()
        loss.backward()

    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.cuda.amp.custom_bwd
            def backward(ctx, grad):
                self.assertFalse(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float16, requires_grad=True)
        # Puts one input tensor in a nested container.  y's contained Tensor won't receive a gradient,
        # because torch.autograd.Function can't hand gradients back to non-Tensor forward arguments.
        # Sets requires_grad=False explicitly so we don't lie about expecting a gradient.
        y = (0, {0: torch.randn((8, 8), device="cuda", dtype=torch.float16, requires_grad=False)})

        with torch.cuda.amp.autocast():
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()

    def test_autocast_cat_jit(self):
        # Reported at https://github.com/pytorch/pytorch/issues/38958

        class Model(torch.nn.Module):
            def forward(self):
                a = torch.randn(1)
                b = torch.randn(1)
                c = torch.cat((a, b), 0)
                d = torch.stack([c, c], 0)
                return d

        # The JIT here doesn't really matter, we just need to call
        # cat via the boxed API
        model = Model()
        model_jit_script = torch.jit.script(model)

        with torch.cuda.amp.autocast(True):
            model()
            model_jit_script()

    # cudnn RNNs require special backend handling (weights are cast to FP16 and reflattened)
    # so they get a dedicated test.
    # Despite the large number of RNN cases it tries, the test takes < 15 seconds on a Titan V (similar to V100).
    @skipIfRocm
    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_autocast_rnn(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # seq, batch, features, hidden size
            clses = ("RNN", "GRU", "LSTM")
            T, B, F, H = 3, 4, 5, 6
            dtypes = (torch.float16, torch.float32)
            input_layouts = ("seq_first", "batch_first", "packed")

            for (cls, num_layers, bias, input_layout, bidirectional, try_nonpreflattened_weights,
                 input_dtype, hidden_dtype, weight_dtype) in \
                    product(clses, (1, 2), (True, False), input_layouts, (True, False), (True, False),
                            dtypes, dtypes, dtypes):
                if input_layout == "seq_first":
                    batch_first = False
                    x = torch.randn((T, B, F), device="cuda", dtype=input_dtype)
                elif input_layout == "batch_first":
                    batch_first = True
                    x = torch.randn((B, T, F), device="cuda", dtype=input_dtype)
                elif input_layout == "packed":
                    batch_first = False
                    x = torch.randn((T, B, F), device="cuda", dtype=input_dtype)
                    x = torch.nn.utils.rnn.pack_padded_sequence(torch.randn((T, B, F),
                                                                            device="cuda", dtype=input_dtype),
                                                                lengths=(3, 2, 1, 3),
                                                                enforce_sorted=False)

                rnn = getattr(torch.nn, cls)(F, H, num_layers=num_layers, bidirectional=bidirectional,
                                             bias=bias, batch_first=batch_first).cuda().to(dtype=weight_dtype)

                if try_nonpreflattened_weights:
                    for p in rnn.parameters():
                        with torch.no_grad():
                            p.set_(p.clone())

                h = torch.randn((num_layers * (2 if bidirectional else 1), B, H),
                                device="cuda", dtype=hidden_dtype)
                if cls == "LSTM":
                    c = torch.randn((num_layers * (2 if bidirectional else 1), B, H),
                                    device="cuda", dtype=hidden_dtype)
                    h = (h, c)

                with torch.cuda.amp.autocast():
                    out, h_out = rnn(x, h)
                out = out.data if input_layout == "packed" else out
                self.assertEqual(out.dtype, torch.float16)
                # Autocast wrapper requires at::_cudnn_rnn is autograd-exposed.  This check can't guarantee
                # at::_cudnn_rnn is autograd-exposed, but if it fires, it indicates some funny business has
                # occurred and we should double check that at::_cudnn_rnn remains autograd-exposed.
                self.assertEqual(out.grad_fn.name(), "CudnnRnnBackward")
                out.sum().backward()
                grads = [p.grad.clone() for p in rnn.parameters()]

                rnn.zero_grad()

                if cls == "LSTM":
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(x.half(), (h[0].half(), h[1].half()))
                else:
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(x.half(), h.half())
                out_control = out_control.data if input_layout == "packed" else out_control
                out_control.sum().backward()
                grads_control = [p.grad.clone() for p in rnn.parameters()]

                # Compares with default tolerances, even for FP16 execution.  Barring nondeterminism,
                # autocast and control results should be bitwise identical.
                self.assertEqual(out, out_control)

                if cls == "LSTM":
                    self.assertTrue(h_out[0].dtype is torch.float16 and h_out[1].dtype is torch.float16)
                    self.assertEqual(h_out[0], h_out_control[0])
                    self.assertEqual(h_out[1], h_out_control[1])
                else:
                    self.assertEqual(h_out.dtype, torch.float16)
                    self.assertEqual(h_out, h_out_control)
                for grad, grad_control in zip(grads, grads_control):
                    self.assertEqual(grad.half(), grad_control)

    def test_autocast_cache_leak(self):
        # Reported at https://github.com/pytorch/pytorch/issues/48049
        # Test is used to check, if autocast recaches the same parameters
        # when executed in a `torch.no_grad()` block.

        linear = torch.nn.Linear(10, 10).to('cuda')
        data = torch.randn(1, 10, device='cuda')

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out = linear(data)
                first_iter_mem = torch.cuda.memory_allocated()
                for _ in range(3):
                    out = linear(data)
                self.assertTrue(first_iter_mem == torch.cuda.memory_allocated())

    def test_autocast_checkpointing(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8),
                                    torch.nn.Linear(8, 8),
                                    torch.nn.Linear(8, 8)).cuda()
        input = torch.rand((8, 8), device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.cuda.amp.autocast():
            output = checkpoint_sequential(model, 2, input)
        self.assertTrue(output.requires_grad)
        self.assertTrue(output.dtype is torch.float16)
        output.sum().backward()

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_max_large_axis(self):
        x = torch.zeros(2**32, device='cuda', dtype=torch.int8)
        x[-1] = 1
        val, idx = x.max(0)
        self.assertEqual(val, 1)
        self.assertEqual(idx, x.shape[0] - 1)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_to_numpy(self):
        self.assertRaises(TypeError, lambda: torch.empty(1, device="cuda").numpy())

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_capture_simple(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda._Graph()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertTrue(b.sum().item() == 11000.)

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_rng_functional(self):
        ops_with_kwargs = ((torch.nn.functional.dropout, {"p": 0.1}),
                           (torch.nn.functional.rrelu, {"training": True}),)
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="cuda", dtype=torch.float)

            # Control
            torch.cuda.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda._Graph()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_rng_distributions(self):
        size = 10000
        input = torch.rand((size,), device="cuda", dtype=torch.float)
        alloc = torch.empty((size,), device="cuda", dtype=torch.float)

        # Torch ops to test with sample args (tuple) and kwargs (dict)
        torch_with_args = (("bernoulli", (input.clone(),), {}),
                           # multinomial uses some uncapturable CUDA calls.
                           # TODO: reenable multinomial tests if/when the implementation is capturable.
                           # ("multinomial", (input.clone(), size, True), {}),
                           # ("multinomial", (input.clone(), size // 2, False), {}),
                           ("normal", (input.clone() + 1, input.clone()), {}),
                           ("poisson", (input.clone(),), {}),
                           ("rand", (size,), {"device": "cuda", "dtype": torch.float}),
                           ("randint", (0, 3, (size,)), {"device": "cuda", "dtype": torch.float}),
                           ("randn", (size,), {"device": "cuda", "dtype": torch.float}),)

        # Tensor methods to test with sample args (tuple)
        tensor_with_args = (("bernoulli_", (input.clone(),)),
                            ("cauchy_", ()),
                            ("exponential_", ()),
                            ("geometric_", (0.3,)),
                            ("log_normal_", ()),
                            ("normal_", ()),
                            ("random_", ()),
                            ("uniform_", ()),)

        def run(module, op, args, kwargs):
            torch.cuda.manual_seed(5)

            # Each path runs a dummy op to increment the state a bit before creating controls.
            if (module == "torch"):
                dummy = getattr(torch, op)(*args, **kwargs)
                control1 = getattr(torch, op)(*args, **kwargs)
                control2 = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                control1 = alloc.clone()
                control2 = alloc.clone()
                getattr(dummy, op)(*args)
                getattr(control1, op)(*args)
                getattr(control2, op)(*args)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda._Graph()
                if (module == "torch"):
                    g.capture_begin()
                    t1 = getattr(torch, op)(*args, **kwargs)
                    t2 = getattr(torch, op)(*args, **kwargs)
                    g.capture_end()
                else:
                    t1 = alloc.clone()
                    t2 = alloc.clone()
                    g.capture_begin()
                    getattr(t1, op)(*args)
                    getattr(t2, op)(*args)
                    g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            try:
                self.assertNotEqual(control1, t1)
                self.assertNotEqual(control2, t2)
            except Exception as e:
                raise RuntimeError("Failed on " + module + "." + op) from e

            # Runs a dummy op prelude, as for controls, to make sure replay()
            # picks up the dummy op's state increment.
            if module == "torch":
                dummy = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                getattr(dummy, op)(*args)

            # Runs RNG ops that fill t1 and t2.
            g.replay()

            try:
                self.assertEqual(control1, t1)
                self.assertEqual(control2, t2)
            except Exception as e:
                raise RuntimeError("Failed on " + module + "." + op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op_with_args in torch_with_args:
            run("torch", *op_with_args)

        for meth_with_args in tensor_with_args:
            # Adds an empty dict for kwargs, which none of the Tensor methods use
            run("Tensor", *(meth_with_args + ({},)))

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_two_successive(self):
        torch.cuda.empty_cache()

        size = 1000
        kSmallBuffer = 2097152

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda._Graph()
            g1 = torch.cuda._Graph()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda._graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g1.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # mixes unrelated eager ops with replays
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            if share_mem != "Don't share":
                self.assertEqual(reserved_no_sharing - torch.cuda.memory_stats()["reserved_bytes.all.current"],
                                 kSmallBuffer)
            else:
                reserved_no_sharing = torch.cuda.memory_stats()["reserved_bytes.all.current"]

            del a, b, c, g0, g1
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_concurrent_replay(self):
        torch.cuda.empty_cache()

        size = 1000000  # largeish to help expose race conditions

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda._Graph()
            g1 = torch.cuda._Graph()

            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda._graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                c = a.clone()
                for _ in range(5):
                    c = func_with_temps(c, 2)
                g1.capture_end()

            # To reproduce data corruption, I need g0 and g1's kernels to run concurrently.
            # But replay() (especially cudaGraphLaunch) can incur significant CPU overhead.
            # The following pattern helps align device-side execution of g0 and g1's kernels.
            torch.cuda.synchronize()
            with torch.cuda.stream(s0):
                torch.cuda._sleep(1000000)
                s1.wait_stream(s0)
                g0.replay()
            with torch.cuda.stream(s1):
                g1.replay()
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.current_stream().wait_stream(s1)

            if share_mem != "Don't share":
                # Confirms concurrent replays using the same mempool corrupted each other.
                self.assertNotEqual(b.sum().item(), size * 94)
                self.assertNotEqual(c.sum().item(), size * 156)
            else:
                # Confirms concurrent replays using different mempools did not corrupt each other.
                self.assertEqual(b.sum().item(), size * 94)
                self.assertEqual(c.sum().item(), size * 156)

            del a, b, c, g0, g1
            # Tensors used across streams (a, b, c) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_three_successive(self):
        torch.cuda.empty_cache()

        size = 1000

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            a = torch.ones((size,), device="cuda")

            g0 = torch.cuda._Graph()
            g1 = torch.cuda._Graph()
            g2 = torch.cuda._Graph()

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (torch.cuda._graph_pool_handle(),) if share_mem == "via graph_pool_handle()" else ()
                g0.capture_begin(*g0_args)
                b = a.clone()
                c = b + 1
                d = b + 2
                g0.capture_end()

                args = (g0.pool(),) if share_mem == "via pool()" else g0_args

                g1.capture_begin(*args)
                e = c + 3
                del c
                g1.capture_end()

                g2.capture_begin(*args)
                f = d + 4
                g2.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # Tests that replaying in capture order is valid
            g0.replay()
            g1.replay()
            g2.replay()

            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            # Tests that replaying as g0, g2, g1 is only valid if they don't share a pool
            g0.replay()
            g2.replay()
            g1.replay()

            # If share_mem is True, g2's capture should have reused c's memory for f. We replayed g2 then g1,
            # so we expect g1's captured "e = c + 3" mistakenly filled e with "f's vals + 3".
            self.assertEqual(e.sum().item(), size * (7 + 3) if share_mem != "Don't share" else size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            del a, b, d, e, f, g0, g1, g2
            # Tensors used across streams (a, e, f) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf((not TEST_CUDA) or
                     TEST_WITH_ROCM or
                     int(torch.version.cuda.split(".")[0]) < 11, "CUDA >= 11.0 required for graphs")
    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        kSmallSize = 1048576
        kSmallBuffer = 2097152
        kLargeBuffer = 20971520
        kMinLargeAlloc = 10485760
        kRoundLarge = 2097152

        elem = 4

        # this was annoying to write but stresses the expectations pretty rigorously
        cases = ((512 // elem, 1, kSmallBuffer, kSmallBuffer, "small_pool"),
                 (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "small_pool"),
                 ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
                 ((kMinLargeAlloc - 512) // elem, 2, 2 * kLargeBuffer, kLargeBuffer, "large_pool"),
                 ((kMinLargeAlloc + 512) // elem, 3,
                  3 * (kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)),
                  kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                  "large_pool"),)

        stats_to_check = ("segment.",
                          "reserved_bytes.",
                          "active.",
                          "active_bytes.")

        gc.collect()
        torch.cuda.empty_cache()

        s = torch.cuda.Stream()

        for (numel,
             delta_cudaMallocs,
             delta_cudaMalloc_bytes,
             delta_cudaMalloc_bytes_post_del_g,
             pool_string) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 2  # one from "b" plus a sneaky one from CUDAGraph's one-element rng offset holder
                delta_active_bytes = numel * elem + 512  # + 512 for CUDAGraph's rng offset holder
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.cuda._Graph()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # Allocation stat estimates assume input is created on the same stream as capture_begin()
                # (in other words, the same stream silo as the rng offset holder, which is not allocated from the
                # capture's private pool).
                a = torch.ones((numel,), device="cuda")

                precapture_stats = torch.cuda.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.cuda.memory_stats()

            expecteds = (delta_cudaMallocs,
                         delta_cudaMalloc_bytes,
                         delta_active_blocks,
                         delta_active_bytes)
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]
                    self.assertEqual(current, expected, "Pre to post capture delta of " +
                                     stat + " = {}, expected = {}, numel = {}".format(current, expected, numel))

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.cuda.empty_cache()

            del g
            gc.collect()
            torch.cuda.empty_cache()
            postdel_stats = torch.cuda.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_cudaMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]
                self.assertEqual(current, expected, "Pre capture to post graph delete delta of " +
                                 stat + " = {}, expected = {}, numel = {}".format(current, expected, numel))

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def test_batch_norm_gather_stats(self):
        input = torch.randn(1, 3, 3, 3, device='cuda')
        mean, invstd = torch.batch_norm_gather_stats(
            input, mean=torch.ones(2, 3, device='cuda'), invstd=torch.ones(2, 3, device='cuda'),
            running_mean=None, running_var=None  , momentum=.1, eps=1e-5, count=2
        )
        self.assertEqual(mean, torch.ones(3, device='cuda'))
        self.assertEqual(invstd, torch.ones(3, device='cuda'))

    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    def test_cuda_device_memory_allocated(self):
        from torch.cuda import memory_allocated
        device_count = torch.cuda.device_count()
        current_alloc = [memory_allocated(idx) for idx in range(device_count)]
        x = torch.ones(10, device="cuda:0")
        self.assertTrue(memory_allocated(0) > current_alloc[0])
        self.assertTrue(all(memory_allocated(torch.cuda.device(idx)) == current_alloc[idx] for idx in range(1, device_count)))


class TestCudaComm(TestCase):
    def _test_broadcast(self, input):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        # test regular
        results = comm.broadcast(input, (0, 1))
        for i, t in enumerate(results):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)
            if input.is_cuda and input.get_device() == i:  # test not copying on same device
                self.assertEqual(t.data_ptr(), input.data_ptr())
        # test out=
        for inplace in [True, False]:
            if inplace:
                outputs = [torch.empty_like(input, device=0), torch.empty_like(input, device=1)]
            else:
                outputs = [input.cuda(0), torch.empty_like(input, device=1)]
            results = comm.broadcast(input, out=outputs)
            for r, o in zip(results, outputs):
                self.assertIs(r, o)
            for i, t in enumerate(results):
                self.assertEqual(t.get_device(), i)
                self.assertEqual(t, input)
        # test error msg
        with self.assertRaisesRegex(RuntimeError, r"Exactly one of 'devices' and 'out'"):
            comm.broadcast(input, (0, 1), out=outputs)
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to be CUDA tensors, but output tensor at index 1"):
            comm.broadcast(input, out=[input.cuda(0), input.cpu()])
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to have same shape as the source .+ at index 1"):
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
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 1, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 10, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 5, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 7, 3, 3),
            make_sparse_tensor(torch.cuda.sparse.FloatTensor, 2, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 3, 2, 7),
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
            torch.randn(5).double().cuda()
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
            self.assertEqual(r, t * 2)

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
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 1, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 10, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.DoubleTensor, 5, 2, 3),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 7, 3, 3),
            make_sparse_tensor(torch.cuda.sparse.FloatTensor, 2, 2, 3),
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            make_sparse_tensor(torch.cuda.sparse.LongTensor, 3, 2, 7),
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
                self.assertEqual(r.data_ptr(), input.data_ptr())  # for target @ same device, a view should be returned

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
            with self.assertRaisesRegex(RuntimeError, r"Expected devices and chunk_sizes to be of same length"):
                comm.scatter(input, [0 for _ in range(len(chunk_sizes) + 1)], dim=dim, chunk_sizes=chunk_sizes)
        with self.assertRaisesRegex(RuntimeError, r"'devices' must not be specified"):
            comm.scatter(input, (0, 1), dim=dim, out=out)
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one device to scatter to"):
            comm.scatter(input, (), dim=dim)
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one output tensor to scatter to"):
            comm.scatter(input, dim=dim, out=[])
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to be CUDA tensors, but output tensor at index 0"):
            comm.scatter(input, dim=dim, out=([out[0].cpu()] + out[1:]))
        with self.assertRaisesRegex(RuntimeError, r"Output tensor at index 0 has incorrect shape"):
            comm.scatter(input, dim=dim, out=([out[0].unsqueeze(0)] + out[1:]))
        with self.assertRaisesRegex(RuntimeError, r"Total size for output tensors along scatter dim \d+ does not match"):
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

        destinations = [None, torch.device('cuda:0'), torch.device('cpu')]
        if torch.cuda.device_count() > 2:
            destinations.append(torch.device('cuda:2'))
        with torch.cuda.device(1):
            for destination in destinations:
                if destination is None:
                    expected_device = torch.device('cuda', torch.cuda.current_device())
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
        with self.assertRaisesRegex(RuntimeError, r"'destination' must not be specified"):
            comm.gather((x, y), dim, destination='cpu', out=torch.empty(expected_size, device='cpu'))
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one tensor to gather from"):
            comm.gather(())
        with self.assertRaisesRegex(RuntimeError, r"Expected all input tensors to be CUDA tensors, "):
            comm.gather((x.cpu(), y))
        with self.assertRaisesRegex(RuntimeError, r"Expected all input tensors to have the same number of dimensions"):
            comm.gather((x, y.unsqueeze(0)))
        with self.assertRaisesRegex(RuntimeError, r"Input tensor at index 1 has invalid shape"):
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
        nhwc = torch.randn((10, 3, 32, 32), device='cpu').contiguous(memory_format=torch.channels_last)
        results = torch.cuda.comm.scatter(nhwc, (0, 1), None, 0)
        for result in results:
            self.assertFalse(result.is_contiguous())
            self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

        gathered = torch.cuda.comm.gather(results)
        self.assertTrue(gathered.is_contiguous(memory_format=torch.channels_last))


    def test_matmul_device_mismatch(self):
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        with self.assertRaisesRegex(RuntimeError, "expected (it|them) to be on GPU"):
            cpu @ cuda
        with self.assertRaisesRegex(RuntimeError, "expected (it|them) to be on GPU"):
            cuda @ cpu

        for s, m1, m2 in product((cpu, cuda), repeat=3):
            if s.device == m1.device == m2.device:
                torch.addmm(s, m1, m2)
            else:
                with self.assertRaisesRegex(RuntimeError, "expected (it|them) to be on GPU"):
                    torch.addmm(s, m1, m2)

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
        fields = ['a', 'b']
        TestNamedTupleInput_0 = collections.namedtuple('NamedTuple', fields)

        num_gpus = torch.cuda.device_count()
        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=1)
        out1 = TestNamedTupleInput_0(a, b)

        a = torch.rand(num_gpus * 2, device=1)
        b = torch.rand(num_gpus * 2, device=0)
        out2 = TestNamedTupleInput_0(a, b)

        outputs = [out1, out2]

        out = scatter_gather.gather(outputs, 'cpu')  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))  # x must be a tensor
            cat = torch.cat((outputs[0][i].to('cpu'), outputs[1][i].to('cpu')))
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

        out = scatter_gather.gather(outputs, 'cpu')  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to('cpu'), outputs[1][i].to('cpu')))
            self.assertTrue(torch.equal(x, cat))

if __name__ == '__main__':
    run_tests()
