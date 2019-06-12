import contextlib
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
from common_utils import (TestCase, run_tests, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN, TEST_WITH_ASAN,
                          load_tests)
from multiprocessing.reduction import ForkingPickler

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

TEST_REPEATS = 30
HAS_SHM_FILES = os.path.isdir('/dev/shm')
TEST_CUDA_IPC = torch.cuda.is_available() and \
    sys.version_info[0] == 3 and \
    sys.platform != 'darwin' and \
    sys.platform != 'win32'
TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1


class SubProcess(mp.Process):
    def __init__(self, tensor):
        super(SubProcess, self).__init__()
        self.tensor = tensor
        self.daemon = True

    def run(self):
        self.tensor.add_(3)


def simple_fill(queue, event):
    data = queue.get()
    data[0][:] = 4
    event.set()


def simple_pool_fill(tensor):
    tensor.fill_(4)
    return tensor.add(1)


def send_tensor(queue, event, tp):
    t = torch.ones(5, 5).type(tp)
    queue.put(t)
    queue.put(t)
    event.wait()


def call_backward():
    x = torch.randn(3, 3, requires_grad=True)
    x.sum().backward()


def sum_tensors(inq, outq):
    with torch.cuda.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put((tensor.sum().item(), tensor.get_device(),
                      tensor.numel(), tensor.storage().size()))


def queue_get_exception(inqueue, outqueue):
    os.close(2)  # hide expected error message
    try:
        torch.zeros(5, 5).cuda()
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put('no exception')


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


def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.arange(1., 26, device=device).view(5, 5)
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


@contextlib.contextmanager
def fs_sharing():
    prev_strategy = mp.get_sharing_strategy()
    mp.set_sharing_strategy('file_system')
    try:
        yield
    finally:
        mp.set_sharing_strategy(prev_strategy)


class leak_checker(object):

    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
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
        if result and mp.get_sharing_strategy() == 'file_system' and wait:
            time.sleep(0.5)
            return self._has_shm_files()
        return result

    def _has_shm_files(self):
        gc.collect()
        names = list('torch_' + str(pid) for pid in self.checked_pids)
        for filename in os.listdir('/dev/shm'):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


class TestMultiprocessing(TestCase):

    def _test_sharing(self, ctx=mp, type=torch.FloatTensor, repeat=1):
        def test_fill():
            x = torch.zeros(5, 5).type(type)
            q = ctx.Queue()
            e = ctx.Event()
            data = [x, x[:, 1]]
            q.put(data)
            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()
            e.wait(10)
            self.assertTrue(e.is_set())
            self.assertTrue(data[0].eq(4).all())
            self.assertTrue(data[1].eq(4).all())
            p.join(1)
            self.assertFalse(p.is_alive())

        def test_receive():
            q = ctx.Queue()
            e = ctx.Event()
            p = ctx.Process(target=send_tensor, args=(q, e, type))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()
            t1 = q.get()
            t2 = q.get()
            self.assertTrue(t1.eq(1).all())
            self.assertTrue(id(t1.storage()) == id(t2.storage()))
            e.set()
            p.join(1)
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
            self.assertEqual(new_data, data, 0)
            storage_cdata = data[0]._cdata
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            for t in new_data[1:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)

        with leak_checker(self):
            for i in range(repeat):
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
                self.assertEqual(r, torch.ones(2, 2) * 5, 0)
            for b in buffers:
                self.assertEqual(b, torch.ones(2, 2) * 4, 0)

            p.close()
            p.join()

        with leak_checker(self) as lc:
            for i in range(repeat):
                do_test()

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on macOS")
    @unittest.skipIf(TEST_WITH_ASAN,
                     "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326")
    def test_fd_sharing(self):
        self._test_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on macOS")
    def test_fd_preserve_sharing(self):
        self._test_preserve_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on macOS")
    def test_fd_pool(self):
        self._test_pool(repeat=TEST_REPEATS)

    @unittest.skipIf(TEST_WITH_ASAN,
                     "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326")
    def test_fs_sharing(self):
        with fs_sharing():
            self._test_sharing(repeat=TEST_REPEATS)

    def test_fs_preserve_sharing(self):
        with fs_sharing():
            self._test_preserve_sharing(repeat=TEST_REPEATS)

    def test_fs_pool(self):
        with fs_sharing():
            self._test_pool(repeat=TEST_REPEATS)

    @unittest.skipIf(not HAS_SHM_FILES, "don't not how to check if shm files exist")
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
        p.join(1)
        self.assertEqual(t, torch.ones(5, 5) * 3, 0)

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_cuda(self):
        torch.cuda.FloatTensor([1])  # initialize CUDA outside of leak checker
        self._test_sharing(mp.get_context('spawn'), torch.cuda.FloatTensor)

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    @unittest.skipIf(not TEST_MULTIGPU, 'found only 1 GPU')
    def test_cuda_small_tensors(self):
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context('spawn')
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5., (i + 1) * 5).cuda(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for i in range(5):
            results.append(outq.get())
        p.join()

        for i, tensor in enumerate(tensors):
            v, device, tensor_size, storage_size = results[i]
            self.assertEqual(v, torch.arange(i * 5., (i + 1) * 5).sum())
            self.assertEqual(device, i % 2)
            self.assertEqual(tensor_size, 5)
            # You might think this should be the case, but it's not!  After
            # data from the CUDA caching allocator goes through IPC, the
            # size of the storage is the size of the *cached cudaMalloc for
            # the entire memory block* of the storage, not just the storage.
            # See Note [CUDA IPC and the caching allocator] for more info
            #
            # self.assertEqual(storage_size, 5)

    @unittest.skipIf(IS_WINDOWS, 'not applicable to Windows (only fails with fork)')
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
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

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_event(self):
        ctx = mp.get_context('spawn')
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

    def _test_empty_tensor_sharing(self, dtype, device):
        q = mp.Queue()
        empty = torch.tensor([], dtype=dtype, device=device)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(out, empty)

    def test_empty_tensor_sharing(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device('cpu'))
        self._test_empty_tensor_sharing(torch.int64, torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_empty_tensor_sharing_cuda(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device('cuda'))
        self._test_empty_tensor_sharing(torch.int64, torch.device('cuda'))

    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        device = 'cuda' if var.is_cuda else 'cpu'

        ready = ctx.Event()
        master_modified = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(target=autograd_sharing, args=(queue, ready, master_modified, device, is_parameter))
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
        p.join(1)
        self.assertFalse(p.is_alive())

    def test_variable_sharing(self):
        for requires_grad in [True, False]:
            var = torch.arange(1., 26).view(5, 5).requires_grad_(requires_grad)
            self._test_autograd_sharing(var)

    def test_leaf_variable_sharing(self):
        devices = ['cpu']
        if torch.cuda.is_available() and not NO_MULTIPROCESSING_SPAWN and TEST_CUDA_IPC:
            devices.append('cuda')
        for device in devices:
            for requires_grad in [True, False]:
                var = torch.arange(1., 26, device=device).view(5, 5).requires_grad_(requires_grad)
                self.assertTrue(var.is_leaf)
                ctx = mp.get_context('spawn') if device == 'cuda' else mp
                ready = ctx.Event()
                queue = ctx.Queue()
                p = ctx.Process(target=requires_grad_variable_sharing, args=(queue, ready))
                p.daemon = True
                p.start()
                queue.put(var)
                ready.wait()
                worker_requires_grad = queue.get()
                self.assertTrue(worker_requires_grad == requires_grad)

    def test_non_leaf_variable_sharing(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            var0 = torch.arange(1., 26, device=device).view(5, 5).requires_grad_(True)
            var = var0 * 2
            # Don't use a regular Queue; it uses a background thread (which
            # means we can't catch the exceptions)
            queue = mp.SimpleQueue()
            self.assertRaisesRegex(RuntimeError, r'requires_grad', lambda: queue.put(var))

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_cuda_variable_sharing(self):
        for requires_grad in [True, False]:
            var = torch.arange(1., 26, device='cuda').view(5, 5).requires_grad_(requires_grad)
            self._test_autograd_sharing(var, mp.get_context('spawn'))

    def test_parameter_sharing(self):
        param = Parameter(torch.arange(1., 26).view(5, 5))
        self._test_autograd_sharing(param, is_parameter=True)

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_cuda_parameter_sharing(self):
        param = Parameter(torch.arange(1., 26, device='cuda').view(5, 5))
        self._test_autograd_sharing(param, mp.get_context('spawn'), is_parameter=True)

    def test_empty_shared(self):
        t = torch.Tensor()
        t.share_memory_()

    def _test_is_shared(self):
        t = torch.randn(5, 5)
        self.assertFalse(t.is_shared())
        t.share_memory_()
        self.assertTrue(t.is_shared())

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on macOS")
    def test_is_shared(self):
        self._test_is_shared()

    def test_fs_is_shared(self):
        with fs_sharing():
            self._test_is_shared()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_is_shared_cuda(self):
        t = torch.randn(5, 5).cuda()
        self.assertTrue(t.is_shared())

    @unittest.skip('this test occasionally fails and deadlocks; see https://github.com/pytorch/pytorch/issues/5834')
    def test_backwards_fork(self):
        r"backwards() should succeed when called before and after a fork"
        call_backward()
        p = mp.Process(target=call_backward)
        p.start()
        p.join(1)
        self.assertFalse(p.is_alive())


if __name__ == '__main__':
    run_tests()
