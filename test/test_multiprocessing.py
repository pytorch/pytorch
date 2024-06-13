# Owner(s): ["module: multiprocessing"]

import contextlib
import copy
import gc
import os
import sys
import time
import unittest
from sys import platform

import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.utils.hooks
from torch.nn import Parameter
from torch.testing._internal.common_cuda import IS_JETSON
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TSAN,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_REPEATS = 30
HAS_SHM_FILES = os.path.isdir("/dev/shm")
MAX_WAITING_TIME_IN_SECONDS = 30

TEST_CUDA_IPC = (
    torch.cuda.is_available()
    and sys.platform != "darwin"
    and sys.platform != "win32"
    and not IS_JETSON
    and not TEST_WITH_ROCM
)  # https://github.com/pytorch/pytorch/issues/90940

TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1

if TEST_CUDA_IPC:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


class SubProcess(mp.Process):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.daemon = True

    def run(self):
        self.tensor.add_(3)


def _test_cuda_ipc_deadlock_actor(queue, iterations):
    for i in range(iterations):
        if not queue.empty():
            queue.get()
        time.sleep(0.01)


def _test_cuda_ipc_deadlock_learner(queue, iterations):
    net = torch.nn.LSTM(1, 1).cuda()
    for i in range(iterations):
        if not queue.full():
            queue.put(copy.deepcopy(net.state_dict()))
        time.sleep(0.01)


def simple_fill(queue, event):
    data = queue.get()
    data[0][:] = 4
    event.set()


def simple_pool_fill(tensor):
    tensor.fill_(4)
    return tensor.add(1)


def send_tensor(queue, event, device, dtype):
    t = torch.ones(5, 5, device=device, dtype=dtype)
    queue.put(t)
    queue.put(t)
    event.wait()


def send_and_delete_tensors(queue, event, device, dtype, count, size=5):
    for i in range(count):
        t = torch.full([size], i, device=device, dtype=dtype)
        queue.put(t)
        del t
    event.wait()


def receive_and_send_sum(queue, out_queue, event, device, dtype, count, size=5):
    s = torch.full([size], 0, device=device, dtype=dtype)
    for i in range(count):
        t = queue.get()
        s += t
    out_queue.put(s)
    event.wait()


def receive_and_send(queue, out_queue, event, count):
    for i in range(count):
        t = queue.get()
        out_queue.put(t.clone())
    event.wait()


def sum_tensors(inq, outq):
    with torch.cuda.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put(
                (
                    tensor.sum().item(),
                    tensor.get_device(),
                    tensor.numel(),
                    tensor.storage().size(),
                )
            )


def queue_get_exception(inqueue, outqueue):
    os.close(2)  # hide expected error message
    try:
        torch.zeros(5, 5).cuda()
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put("no exception")


# Multiply by two in a separate stream
def cuda_multiply_two(queue, ready, done):
    ready.set()
    with torch.cuda.stream(torch.cuda.Stream()):
        cuda_event, tensor = queue.get()
        cuda_event.wait()
        tensor.mul_(2)
        cuda_event.record()
        done.set()
        del cuda_event


def requires_grad_variable_sharing(queue, ready):
    var = queue.get()
    ready.set()
    queue.put(var.requires_grad)


def integer_parameter_serialization(iparam):
    iparam + 1


def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.arange(1.0, 26, device=device).view(5, 5)
    expected_var[0, 0] = 1000
    is_ok = var.data.equal(expected_var)
    var.data[:] = torch.ones(5, 5, device=device)

    is_ok &= var.grad is None
    is_ok &= not var._backward_hooks
    if is_parameter:
        is_ok &= type(var) == Parameter
    else:
        is_ok &= type(var) == torch.Tensor
    var._grad = torch.ones(5, 5, device=device)

    queue.put(is_ok)


def mixed_type_producer(queue, event):
    for _ in range(10):
        float_tensor = torch.ones(2, 2).float().cuda()
        byte_tensor = torch.zeros(2, 2).byte().cuda()

        queue.put(float_tensor)
        queue.put(byte_tensor)
        event.wait()
        event.clear()


def simple_autograd_function(a=1):
    torch.rand(3).requires_grad_(True).mean().backward()
    return a**2


@contextlib.contextmanager
def fs_sharing():
    prev_strategy = mp.get_sharing_strategy()
    mp.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        mp.set_sharing_strategy(prev_strategy)


class leak_checker:
    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        if args[0] is None:
            # Check that the 10th available file-descriptor at the end of the
            # test is no more than 4 higher than the 10th available at the
            # start. This attempts to catch file descriptor leaks, but allows
            # one-off initialization that may use up a file descriptor
            # TODO: Disabled because this check is too flaky
            # available_fds = self._get_next_fds(10)
            # self.test_case.assertLessEqual(
            #     available_fds[-1] - self.next_fds[-1], 5)
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)

    def _get_next_fds(self, n=1):
        # dup uses the lowest-numbered unused descriptor for the new descriptor
        fds = [os.dup(0) for i in range(n)]
        for fd in fds:
            os.close(fd)
        return fds

    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:
            return False

        result = self._has_shm_files()
        if not result or mp.get_sharing_strategy() != "file_system" or not wait:
            return result

        total_waiting_time = 0
        waiting_time = 0.5

        while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and result:
            time.sleep(waiting_time)
            total_waiting_time += waiting_time
            result = self._has_shm_files()

        return result

    def _has_shm_files(self):
        gc.collect()
        names = ["torch_" + str(pid) for pid in self.checked_pids]
        for filename in os.listdir("/dev/shm"):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestMultiprocessing(TestCase):
    def tearDown(self):
        # This will keep tests isolated from each-other
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    def _test_sharing(self, ctx=mp, device="cpu", dtype=torch.float, repeat=1):
        def test_fill():
            x = torch.zeros(5, 5).to(device, dtype)
            q = ctx.Queue()
            e = ctx.Event()

            data = [x, x[:, 1]]
            q.put(data)

            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            total_waiting_time = 0
            waiting_time = 0.5
            is_set = False
            # Once the child process is done, it will set the event to notify the
            # parent accordingly
            while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and not is_set:
                time.sleep(waiting_time)
                total_waiting_time += waiting_time
                is_set = e.is_set()

            self.assertTrue(is_set)
            self.assertTrue(data[0].eq(4).all())
            self.assertTrue(data[1].eq(4).all())

            p.join(100)
            self.assertFalse(p.is_alive())

        def test_receive():
            q = ctx.Queue()
            e = ctx.Event()

            p = ctx.Process(target=send_tensor, args=(q, e, device, dtype))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            t1 = q.get()
            t2 = q.get()
            self.assertTrue(t1.eq(1).all())
            s1 = t1.storage()
            s2 = t2.storage()
            self.assertEqual(type(s1), type(s2))
            self.assertEqual(s1.data_ptr(), s1.data_ptr())
            self.assertEqual(s1, s2)

            # We need to delete this tensors to allow producer (child process)
            # collect them properly
            del t1, t2

            # Mark the event as done and join the process
            e.set()
            p.join(100)
            self.assertFalse(p.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()

    def _test_preserve_sharing(self, ctx=mp, repeat=1):
        def do_test():
            x = torch.randn(5, 5)
            data = [x.storage(), x, x[2], x[:, 1]]
            q = ctx.Queue()
            q.put(data)
            new_data = q.get(timeout=1)
            self.assertEqual(new_data, data, atol=0, rtol=0)
            storage_cdata = data[0]._cdata
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            for t in new_data[1:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)

        with leak_checker(self):
            for _ in range(repeat):
                do_test()

    def _test_pool(self, ctx=mp, repeat=1):
        def do_test():
            p = ctx.Pool(2)
            for proc in p._pool:
                lc.check_pid(proc.pid)

            buffers = [torch.zeros(2, 2) for i in range(4)]
            results = p.map(simple_pool_fill, buffers, 1)
            self.assertEqual(len(results), len(buffers))
            for r in results:
                self.assertEqual(r, torch.ones(2, 2) * 5, atol=0, rtol=0)
            for b in buffers:
                self.assertEqual(b, torch.ones(2, 2) * 4, atol=0, rtol=0)

            p.close()
            p.join()

        with leak_checker(self) as lc:
            for _ in range(repeat):
                do_test()

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326",
    )
    def test_fd_sharing(self):
        self._test_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_fd_preserve_sharing(self):
        self._test_preserve_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_fd_pool(self):
        self._test_pool(repeat=TEST_REPEATS)

    @unittest.skipIf(
        TEST_WITH_ASAN,
        "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326",
    )
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    def test_fs_sharing(self):
        with fs_sharing():
            # The test works but is very slow on MacOS, see https://github.com/pytorch/pytorch/pull/93183,
            # so run it only once there. The delay is in waiting for the child process to terminate (join)
            repeat = 1 if IS_MACOS else TEST_REPEATS
            self._test_sharing(repeat=repeat)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    def test_fs_preserve_sharing(self):
        with fs_sharing():
            self._test_preserve_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    def test_fs_pool(self):
        with fs_sharing():
            self._test_pool(repeat=TEST_REPEATS)

    @unittest.skipIf(not HAS_SHM_FILES, "don't not how to check if shm files exist")
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "Fail to clean up temporary /dev/shm/torch_* file, see https://github.com/pytorch/pytorch/issues/91467",
    )
    def test_fs(self):
        def queue_put():
            x = torch.DoubleStorage(4)
            q = mp.Queue()
            self.assertFalse(lc.has_shm_files())
            q.put(x)
            time.sleep(0.05)  # queue serializes asynchronously
            self.assertTrue(lc.has_shm_files(wait=False))
            q.get()

        with fs_sharing(), leak_checker(self) as lc:
            for _ in range(TEST_REPEATS):
                queue_put()

    def test_inherit_tensor(self):
        t = torch.zeros(5, 5)
        p = SubProcess(t.share_memory_())
        p.start()
        p.join(2)
        if p.exitcode is None:
            print("test_inherit_tensor: SubProcess too slow")
        else:
            self.assertEqual(t, torch.ones(5, 5) * 3, atol=0, rtol=0)

    @unittest.skipIf(IS_WINDOWS, "Test needs to use fork multiprocessing")
    def test_autograd_errors(self):
        ctx = mp.get_context("fork")
        simple_autograd_function()
        # Autograd only uses thread when GPUs are involved
        if (
            torch.cuda.is_available()
            or torch.backends.mps.is_available()
            or torch.xpu.is_available()
        ):
            with self.assertRaisesRegex(RuntimeError, r"Unable to handle autograd"):
                with ctx.Pool(3) as pool:
                    pool.map(simple_autograd_function, [1, 2, 3])
        else:
            with ctx.Pool(3) as pool:
                pool.map(simple_autograd_function, [1, 2, 3])

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN, "Test needs to use spawn multiprocessing"
    )
    def test_autograd_fine_with_spawn(self):
        ctx = mp.get_context("spawn")
        simple_autograd_function()
        with ctx.Pool(3) as pool:
            pool.map(simple_autograd_function, [1, 2, 3])

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_simple(self):
        torch.cuda.FloatTensor([1])  # initialize CUDA outside of leak checker
        self._test_sharing(mp.get_context("spawn"), "cuda", torch.float)

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_memory_allocation(self):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        e = ctx.Event()
        p = ctx.Process(
            target=send_and_delete_tensors, args=(q, e, "cuda", torch.int, 5)
        )
        p.start()
        t = []
        for _ in range(5):
            t.append(q.get())
        self.assertEqual(t[0], torch.full([5], 0, dtype=torch.int32))
        del t
        e.set()
        p.join(1)

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_ipc_deadlock(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(1)
        processes = dict(
            a=ctx.Process(target=_test_cuda_ipc_deadlock_actor, args=(queue, 100)),
            l=ctx.Process(target=_test_cuda_ipc_deadlock_learner, args=(queue, 100)),
        )

        for p in processes.values():
            p.start()

        for p in processes.values():
            p.join(10)

        for p in processes.values():
            self.assertFalse(p.is_alive())

    @slowTest
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_send_many(self, name=None, size=5, count=100000):
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        q3 = ctx.Queue()
        e1 = ctx.Event()
        e2 = ctx.Event()
        e3 = ctx.Event()
        p1 = ctx.Process(
            target=send_and_delete_tensors,
            args=(q1, e1, "cuda", torch.long, count, size),
        )
        p2 = ctx.Process(target=receive_and_send, args=(q1, q2, e2, count))
        p3 = ctx.Process(
            target=receive_and_send_sum,
            args=(q2, q3, e3, "cuda", torch.long, count, size),
        )
        p1.start()
        p2.start()
        p3.start()
        result = q3.get()
        self.assertEqual(result[0], int(count * (count - 1) / 2))
        del result
        e1.set()
        e2.set()
        e3.set()
        p1.join(1)
        p2.join(1)
        p3.join(1)

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    def test_cuda_small_tensors(self):
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context("spawn")
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).cuda(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for _ in range(5):
            results.append(outq.get())
        p.join()

        for i, _tensor in enumerate(tensors):
            v, device, tensor_size, storage_size = results[i]
            self.assertEqual(v, torch.arange(i * 5.0, (i + 1) * 5).sum())
            self.assertEqual(device, i % 2)
            self.assertEqual(tensor_size, 5)

            # You might think this should be the case, but it's not!  After
            # data from the CUDA caching allocator goes through IPC, the
            # size of the storage is the size of the *cached cudaMalloc for
            # the entire memory block* of the storage, not just the storage.
            # See Note [CUDA IPC and the caching allocator] for more info
            #
            # self.assertEqual(storage_size, 5)

        # Collect current process (producer) files, make sure nothing holds
        # ref to the sent tensors
        del _tensor
        del tensors

        # We need to collect, as CUDA MP implementation holds one shared
        # memory 'file' for performance reason
        torch.cuda.ipc_collect()

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_bad_call(self):
        # Initialize CUDA
        t = torch.zeros(5, 5).cuda().cpu()
        inq = mp.Queue()
        outq = mp.Queue()
        p = mp.Process(target=queue_get_exception, args=(inq, outq))
        p.start()
        inq.put(t)
        p.join()
        self.assertIsInstance(outq.get(), RuntimeError)

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrong_cuda_fork(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.cuda.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        x = torch.rand(20, 2).cuda()
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize CUDA in forked subprocess.")

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        ready = ctx.Event()
        done = ctx.Event()
        p = ctx.Process(target=cuda_multiply_two, args=(queue, ready, done))
        p.start()

        ready.wait()
        with torch.cuda.stream(torch.cuda.Stream()):
            tensor = torch.cuda.FloatTensor([1, 1, 1, 1])
            # Use a sleep kernel to test events. Without the event, the
            # multiply happens before the add.
            event = torch.cuda.Event(interprocess=True)
            torch.cuda._sleep(20000000)  # about 30 ms
            tensor.add_(1)
            event.record()
            queue.put((event, tensor))
            done.wait()  # must wait until subprocess records event
            event.synchronize()
            self.assertEqual(list(tensor), [4, 4, 4, 4])
        p.join()

    @staticmethod
    def _test_event_multiprocess_child(event, p2c, c2p):
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        event.synchronize()
        c2p.put(1)  # notify parent synchronization is done

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_multiprocess(self):
        event = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.assertTrue(event.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_multiprocess_child,
            args=(event, p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for until child process is ready
        torch.cuda._sleep(50000000)  # spin for about 50 ms
        event.record()
        p2c.put(0)  # notify child event is recorded

        self.assertFalse(event.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(event.query())
        p.join()

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    def test_event_handle_multi_gpu(self):
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        with torch.cuda.device(d0):
            e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        with torch.cuda.device(d1):
            # create handle on different device from un-recorded event
            e0.ipc_handle()

        with torch.cuda.device(d0):
            e1 = torch.cuda.Event(enable_timing=False, interprocess=True)
            stream = torch.cuda.Stream()
            torch.cuda._sleep(50000000)  # spin for about 50 ms
            e1.record(stream)

        with torch.cuda.device(d1):
            # create handle on different device from recorded event
            e1.ipc_handle()

    @staticmethod
    def _test_event_handle_importer_consumer(handle, p2c, c2p):
        e1 = torch.cuda.Event.from_ipc_handle(0, handle)
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        e1.synchronize()
        c2p.put(1)  # notify synchronization is done in child
        p2c.get()  # wait for parent to finish before destructing child event

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_importer(self):
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.assertTrue(e0.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        torch.cuda._sleep(50000000)  # spin for about 50 ms
        e0.record()
        p2c.put(0)  # notify child event is recorded

        self.assertFalse(e0.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(e0.query())
        p2c.put(1)  # notify child that parent is done
        p.join()

    @staticmethod
    def _test_event_handle_exporter_consumer(handle, p2c, c2p):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            e1 = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), handle)
            torch.cuda._sleep(50000000)  # spin for about 50 ms
            e1.record()
            c2p.put(0)
            # wait for parent process finished synchronization before
            # destructing e1
            p2c.get()

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_exporter(self):
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()
        # wait for event in child process is recorded
        c2p.get()

        self.assertFalse(e0.query())
        e0.synchronize()
        self.assertTrue(e0.query())
        p2c.put(0)
        p.join()

    def _test_empty_tensor_sharing(self, dtype, device):
        q = mp.Queue()
        empty = torch.tensor([], dtype=dtype, device=device)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(out, empty)

    def test_empty_tensor_sharing(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device("cpu"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_empty_tensor_sharing_cuda(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device("cuda"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("cuda"))

    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        device = "cuda" if var.is_cuda else "cpu"

        ready = ctx.Event()
        master_modified = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(
            target=autograd_sharing,
            args=(queue, ready, master_modified, device, is_parameter),
        )
        p.daemon = True
        p.start()

        # This would cause an error if we tried to serialize the hooks,
        # because it's a closure and pickle doesn't support closures.
        @torch.utils.hooks.unserializable_hook
        def hook(*unused):
            pass

        if var.requires_grad:
            var.register_hook(hook)
        var._grad = torch.zeros(5, 5, device=device)
        queue.put(var)

        ready.wait()
        var.data[0, 0] = 1000
        var.grad.data[:] = torch.ones(5, 5, device=device) * 4
        master_modified.set()

        worker_ok = queue.get()
        self.assertTrue(worker_ok)

        self.assertEqual(var.data, torch.ones(5, 5, device=device))
        self.assertEqual(var.grad.data, torch.ones(5, 5, device=device) * 4)
        p.join(100)
        self.assertFalse(p.is_alive())

    # Check sharing a cudaMalloc allocation with different types of storage.
    # (Issue #11422)
    def _test_mixed_types_cuda_sharing(self, ctx=mp):
        all_ones = torch.ones(2, 2).float()
        all_zeros = torch.zeros(2, 2).byte()
        queue = ctx.Queue()
        event = ctx.Event()

        p = ctx.Process(target=mixed_type_producer, args=(queue, event))

        p.start()

        for _ in range(10):
            float_tensor = queue.get()
            byte_tensor = queue.get()
            self.assertEqual(float_tensor, all_ones)
            self.assertEqual(byte_tensor, all_zeros)
            del float_tensor, byte_tensor
            event.set()

        time.sleep(5)
        p.join()

    @unittest.skipIf(
        TEST_WITH_ASAN,
        "non-deterministically hangs with ASAN https://github.com/pytorch/pytorch/issues/94024",
    )
    def test_variable_sharing(self):
        for requires_grad in [True, False]:
            var = torch.arange(1.0, 26).view(5, 5).requires_grad_(requires_grad)
            self._test_autograd_sharing(var)

    # See https://github.com/pytorch/pytorch/issues/14997
    @unittest.skipIf(TEST_WITH_ASAN, "non-deterministically hangs with ASAN")
    def test_leaf_variable_sharing(self):
        devices = ["cpu"]
        if torch.cuda.is_available() and not NO_MULTIPROCESSING_SPAWN and TEST_CUDA_IPC:
            devices.append("cuda")
        for device in devices:
            for requires_grad in [True, False]:
                var = (
                    torch.arange(1.0, 26, device=device)
                    .view(5, 5)
                    .requires_grad_(requires_grad)
                )
                self.assertTrue(var.is_leaf)
                ctx = mp.get_context("spawn") if device == "cuda" else mp
                ready = ctx.Event()
                queue = ctx.Queue()
                p = ctx.Process(
                    target=requires_grad_variable_sharing, args=(queue, ready)
                )
                p.daemon = True
                p.start()
                queue.put(var)
                ready.wait()
                worker_requires_grad = queue.get()
                self.assertTrue(worker_requires_grad == requires_grad)

    def test_non_leaf_variable_sharing(self):
        devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
        for device in devices:
            var0 = torch.arange(1.0, 26, device=device).view(5, 5).requires_grad_(True)
            var = var0 * 2
            # Don't use a regular Queue; it uses a background thread (which
            # means we can't catch the exceptions)
            queue = mp.SimpleQueue()
            self.assertRaisesRegex(
                RuntimeError, r"requires_grad", lambda: queue.put(var)
            )

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_variable_sharing(self):
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device="cuda")
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self._test_autograd_sharing(var, mp.get_context("spawn"))

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_mixed_types_cuda_sharing(self):
        self._test_mixed_types_cuda_sharing(mp.get_context("spawn"))

    def test_parameter_sharing(self):
        param = Parameter(torch.arange(1.0, 26).view(5, 5))
        self._test_autograd_sharing(param, is_parameter=True)

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_parameter_sharing(self):
        param = Parameter(torch.arange(1.0, 26, device="cuda").view(5, 5))
        self._test_autograd_sharing(param, mp.get_context("spawn"), is_parameter=True)

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    def test_integer_parameter_serialization_cpu(self):
        self._test_integer_parameter_serialization(device="cpu")

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_integer_parameter_serialization_cuda(self):
        self._test_integer_parameter_serialization(device="cuda")

    def _test_integer_parameter_serialization(self, device):
        param = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int64, device=device), requires_grad=False
        )

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=integer_parameter_serialization, args=(param,))
        p.start()
        p.join()

        self.assertEqual(
            0,
            p.exitcode,
            msg=f'Failed to serialize successfully for "{device}" device!',
        )

    def test_empty_shared(self):
        t = torch.tensor([])
        t.share_memory_()

    def _test_is_shared(self):
        t = torch.randn(5, 5)
        self.assertFalse(t.is_shared())
        t.share_memory_()
        self.assertTrue(t.is_shared())

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_is_shared(self):
        self._test_is_shared()

    def test_fs_is_shared(self):
        with fs_sharing():
            self._test_is_shared()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_is_shared_cuda(self):
        t = torch.randn(5, 5).cuda()
        self.assertTrue(t.is_shared())

    @unittest.skipIf(
        sys.platform != "linux",
        "Only runs on Linux; requires prctl(2)",
    )
    def test_set_thread_name(self):
        name = "test name"
        mp._set_thread_name(name)
        self.assertEqual(mp._get_thread_name(), name)


if __name__ == "__main__":
    run_tests()
