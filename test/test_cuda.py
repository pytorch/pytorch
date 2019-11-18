import collections
import io
import tempfile
import unittest
import sys
from itertools import repeat
import os
from contextlib import contextmanager
import threading
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue

import torch
import torch.cuda
import torch.cuda.comm as comm
from torch import multiprocessing as mp
from torch._six import inf, nan

from test_torch import _TestTorchMixin

from common_methods_invocations import tri_tests_args, tri_large_tests_args, \
    _compare_trilu_indices, _compare_large_trilu_indices
from common_utils import TestCase, get_gpu_type, freeze_rng_state, run_tests, \
    PY3, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN, skipIfRocm, \
    load_tests, slowTest, skipCUDANonDefaultStreamIf, TEST_WITH_ROCM

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA and TEST_MULTIGPU from common_cuda here,
# because if we do that, the TEST_CUDNN line from common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if not TEST_CUDA:
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811

TEST_MAGMA = TEST_CUDA
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_CUDNN = TEST_CUDA
if TEST_CUDA:
    torch.ones(1).cuda()  # has_magma shows up after cuda is initialized
    TEST_CUDNN = TEST_CUDA and (TEST_WITH_ROCM or torch.backends.cudnn.is_acceptable(torch.tensor(1., device=torch.device('cuda:0'))))
    TEST_MAGMA = torch.cuda.has_magma
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

    def test_memory_stats(self):
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

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

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 80.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 80, dtype=torch.int8, device='cuda')

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
            self.assertFalse(event.query())
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, 0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], 0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.cuda.IntStorage))
        q_copy[1].fill_(10)
        self.assertTrue(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

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

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
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

    def _test_broadcast(self, input):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        result = comm.broadcast(input, (0, 1))
        for i, t in enumerate(result):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)
            if input.is_cuda and input.get_device() == i:
                self.assertEqual(t.data_ptr(), input.data_ptr())

    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).cuda())

    @staticmethod
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
        self._test_broadcast_coalesced(self, tensors, num_bytes * 5 // 2)

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
        self._test_broadcast_coalesced(self, tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        x_cuda = x.cuda(0)
        y_cuda = y.cuda(1)
        result = comm.reduce_add((x_cuda, y_cuda))
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.cpu(), x + y)

    @staticmethod
    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        dup_tensors = [tensors, list(map(lambda t: t.cuda(1), tensors))]

        r_tensors = list(map(comm.reduce_add, zip(*dup_tensors)))
        for r, t in zip(r_tensors, tensors):
            self.assertEqual(r.get_device(), t.get_device())
            self.assertEqual(r, t * 2)
            self.assertEqual(r.type(), t.type())

        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)
        self.assertEqual(r_tensors, rc_tensors)
        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqual(rc.get_device(), r.get_device())
            self.assertEqual(rc.type(), r.type())

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
        self._test_reduce_add_coalesced(self, tensors, num_bytes * 5 // 2)

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
        self._test_reduce_add_coalesced(self, tensors, num_bytes * 5 // 2)

    def _test_scatter(self, input, chunk_sizes=None, dim=0):
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        result = comm.scatter(input, (0, 1), chunk_sizes, dim)
        self.assertEqual(len(result), 2)
        if chunk_sizes is None:
            chunk_sizes = tuple(repeat(input.size(dim) // 2, 2))
        chunk_start = 0
        for i, r in enumerate(result):
            chunk_end = chunk_start + chunk_sizes[i]
            index = [slice(None, None), slice(None, None)]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input[tuple(index)], 0)
            chunk_start = chunk_end

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
        x = torch.randn(2, 5).cuda(0)
        y = torch.randn(2, 5).cuda(1)
        result = comm.gather((x, y), dim)

        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = torch.Size(expected_size)
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.size(), expected_size)

        index = [slice(None, None), slice(None, None)]
        index[dim] = slice(0, x.size(dim))
        self.assertEqual(result[tuple(index)], x)
        index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
        self.assertEqual(result[tuple(index)], y)

        # Bool test case
        t = torch.tensor([[False, True], [True, True]], device='cuda')
        self.assertEqual(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device='cuda')),
                         torch.tensor([[False, False], [True, True]], device='cuda'))

    def test_gather(self):
        self._test_gather(0)

    def test_gather_dim(self):
        self._test_gather(1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_memory_format_scatter_gather(self):
        nhwc = torch.randn((10, 3, 32, 32), device='cpu').contiguous(memory_format=torch.channels_last)
        results = torch.cuda.comm.scatter(nhwc, (0, 1), None, 0)
        for result in results:
            self.assertFalse(result.is_contiguous())
            self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

        gathered = torch.cuda.comm.gather(results)
        self.assertTrue(gathered.is_contiguous(memory_format=torch.channels_last))

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

    def test_bernoulli(self):
        _TestTorchMixin._test_bernoulli(self, torch.float32, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float32, torch.float16, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float16, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.float16, torch.float16, 'cuda')
        # test that it works with integral tensors
        _TestTorchMixin._test_bernoulli(self, torch.uint8, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.uint8, torch.float16, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.int64, torch.float64, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.int64, torch.float16, 'cuda')
        # test that it works with bool tensors
        _TestTorchMixin._test_bernoulli(self, torch.bool, torch.float16, 'cuda')
        _TestTorchMixin._test_bernoulli(self, torch.int64, torch.float16, 'cuda')

    @unittest.skipIf(torch.cuda.device_count() >= 10, "Loading a cuda:9 tensor")
    @unittest.skipIf(not PY3, "Tensor was serialized with Python 3")
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

    def test_is_tensor(self):
        for t in types:
            tensor = get_gpu_type(t)()
            self.assertTrue(torch.is_tensor(tensor))
        self.assertTrue(torch.is_tensor(torch.cuda.HalfTensor()))

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
        # copy 10 MB tensor from CPU-GPU which should take some time
        tensor1 = torch.ByteTensor(10000000).pin_memory()
        tensor2 = tensor1.cuda(non_blocking=True)
        self.assertFalse(default_stream.query())
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
    # Note: fails sometimes on the CI, passes on dual gfx906
    @skipIfRocm
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
    @skipIfRocm
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
        s1.record_event(e_tok)

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

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # Flaky on the ROCm CI
    @skipIfRocm
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
            self.assertNotEqual(tmp2.data_ptr(), ptr[0], 'allocation re-used to soon')

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        # Check that the block will be re-used after the main stream finishes
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(stream):
            tmp3 = torch.cuda.FloatTensor(t.size())
            self.assertEqual(tmp3.data_ptr(), ptr[0], 'allocation not re-used')

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
        self.assertEqual(t.data_ptr(), ptr, 'allocation not reused')

        # check that the allocation is not re-used if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')
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
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')

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

    @skipIfRocm
    def test_fft_ifft_rfft_irfft(self):
        _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        @contextmanager
        def plan_cache_max_size(n, device=None):
            if device is None:
                plan_cache = torch.backends.cuda.cufft_plan_cache
            else:
                plan_cache = torch.backends.cuda.cufft_plan_cache[device]
            original = plan_cache.max_size
            plan_cache.max_size = n
            yield
            plan_cache.max_size = original

        with plan_cache_max_size(max(1, torch.backends.cuda.cufft_plan_cache.size - 10)):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        with plan_cache_max_size(0):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        torch.backends.cuda.cufft_plan_cache.clear()

        # check that stll works after clearing cache
        with plan_cache_max_size(10):
            _TestTorchMixin._test_fft_ifft_rfft_irfft(self, device=torch.device('cuda'))

        with self.assertRaisesRegex(RuntimeError, r"must be non-negative"):
            torch.backends.cuda.cufft_plan_cache.max_size = -1

        with self.assertRaisesRegex(RuntimeError, r"read-only property"):
            torch.backends.cuda.cufft_plan_cache.size = -1

        with self.assertRaisesRegex(RuntimeError, r"but got device with index"):
            torch.backends.cuda.cufft_plan_cache[torch.cuda.device_count() + 10]

        if TEST_MULTIGPU:
            # Test that different GPU has different cache
            x0 = torch.randn(2, 3, 3, device='cuda:0')
            x1 = x0.cuda(1)
            self.assertEqual(x0.rfft(2), x1.rfft(2))
            # If a plan is used across different devices, the following line (or
            # the assert above) would trigger illegal memory access. Other ways
            # to trigger the error include
            #   (1) setting CUDA_LAUNCH_BLOCKING=1 (pytorch/pytorch#19224) and
            #   (2) printing a device 1 tensor.
            x0.copy_(x1)

            # Test that un-indexed `torch.backends.cuda.cufft_plan_cache` uses current device
            with plan_cache_max_size(10, device='cuda:0'):
                with plan_cache_max_size(11, device='cuda:1'):
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                    self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                    with torch.cuda.device(1):
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(0):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0

                self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                with torch.cuda.device(1):
                    with plan_cache_max_size(11):  # default is cuda:1
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(0):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1

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
                torch.multinomial(probs.to('cuda'), 2)
                torch.cuda.synchronize()
            return False  # Should not be reached
        except RuntimeError as e:
            return e

    @slowTest
    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(IS_WINDOWS, 'FIXME: CUDA OOM error on Windows')
    @unittest.skipIf(not PY3,
                     "spawn start method is not supported in Python 2, \
                     but we need it for creating another process with CUDA")
    @skipIfRocm
    def test_multinomial_invalid_probs_cuda(self):
        test_method = TestCuda._test_multinomial_invalid_probs_cuda
        self._spawn_method(test_method, torch.Tensor([1, -1, 1]))
        self._spawn_method(test_method, torch.Tensor([1, inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, -inf, 1]))
        self._spawn_method(test_method, torch.Tensor([1, 1, nan]))
        self._spawn_method(test_method, torch.Tensor([0, 1, 0]))

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    def test_huge_index(self):
        src = torch.empty(15000000, 45, device='cuda', dtype=torch.long).random_(0, 2**22)
        idx = torch.randperm(src.shape[0], device='cuda')
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_tensor_gather(self):
        _TestTorchMixin._test_gather(self, lambda t: t.cuda(), False)

    def test_tensor_scatter(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', test_bounds=False)

    def test_tensor_scatterAdd(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_add_', test_bounds=False)

    def test_tensor_scatterFill(self):
        _TestTorchMixin._test_scatter_base(self, lambda t: t.cuda(), 'scatter_', True, test_bounds=False)

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
        self.assertEqual(before0, after0, 0)
        self.assertEqual(before1, after1, 0)

    @skipIfRocm
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

    # Note: This test fails on ROCm CI gfx900 but passes on gfx906
    @skipIfRocm
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

    @skipIfRocm
    @unittest.skipIf(not PY3, "Barrier is unavailable before Python3")
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
    @unittest.skipIf(not PY3, "Barrier is unavailable before Python3")
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

    @skipIfRocm
    @unittest.skipIf(not PY3, "Barrier is unavailable before Python3")
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


if __name__ == '__main__':
    run_tests()
