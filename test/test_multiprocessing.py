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
from torch.autograd import Variable
from torch.nn import Parameter
from common import TestCase, run_tests


TEST_REPEATS = 30
HAS_SHM_FILES = os.path.isdir('/dev/shm')
TEST_CUDA_IPC = torch.cuda.is_available() and \
    sys.version_info[0] == 3 and \
    sys.platform != 'darwin'
TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1


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


def sum_tensors(inq, outq):
    with torch.cuda.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put((tensor.sum(), tensor.get_device(),
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


def autograd_sharing(queue, ready, master_modified):
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.range(1, 25).view(5, 5)
    expected_var[0, 0] = 1000
    is_ok = var.data.equal(expected_var)
    var.data[:] = torch.ones(5, 5)

    if var.grad is not None:
        is_ok &= var.grad.data.equal(torch.ones(5, 5) * 4)
        var.grad.data[:] = torch.ones(5, 5)

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
            available_fds = self._get_next_fds(10)
            self.test_case.assertLessEqual(
                available_fds[-1] - self.next_fds[-1], 5)
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

    def __init__(self, *args, **kwargs):
        super(TestMultiprocessing, self).__init__(*args, **kwargs)

    def _test_sharing(self, ctx=mp, type=torch.FloatTensor, repeat=1):
        def test_fill():
            x = torch.zeros(5, 5).type(type)
            q = ctx.Queue()
            e = ctx.Event()
            data = [x, x[:, 1]]
            q.put(data)
            p = ctx.Process(target=simple_fill, args=(q, e))
            lc.check_pid(p.pid)
            p.start()
            e.wait()
            self.assertTrue(data[0].eq(4).all())
            self.assertTrue(data[1].eq(4).all())
            p.join(1)
            self.assertFalse(p.is_alive())

        def test_receive():
            q = ctx.Queue()
            e = ctx.Event()
            p = ctx.Process(target=send_tensor, args=(q, e, type))
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
            for i in range(repeat):
                test_fill()
                test_receive()

    def _test_preserve_sharing(self, ctx=mp, repeat=1):
        def do_test():
            x = torch.randn(5, 5)
            data = [x.storage(), x.storage()[1:4], x, x[2], x[:, 1]]
            q = ctx.Queue()
            q.put(data)
            new_data = q.get()
            self.assertEqual(new_data, data, 0)
            storage_cdata = data[0]._cdata
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            for t in new_data[2:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)
            # TODO: enable after fixing #46
            # new_data[0].fill_(10)
            # self.assertEqual(new_data[1], new_data[0][1:4], 0)

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

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_sharing(self):
        self._test_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_preserve_sharing(self):
        self._test_preserve_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_fd_pool(self):
        self._test_pool(repeat=TEST_REPEATS)

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
            for i in range(TEST_REPEATS):
                queue_put()

    def test_inherit_tensor(self):
        class SubProcess(mp.Process):

            def __init__(self, tensor):
                super(SubProcess, self).__init__()
                self.tensor = tensor

            def run(self):
                self.tensor.add_(3)

        t = torch.zeros(5, 5)
        p = SubProcess(t.share_memory_())
        p.start()
        p.join()
        self.assertEqual(t, torch.ones(5, 5) * 3, 0)

    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    def test_cuda(self):
        torch.cuda.FloatTensor([1])  # initialize CUDA outside of leak checker
        self._test_sharing(mp.get_context('spawn'), torch.cuda.FloatTensor)

    @unittest.skipIf(not TEST_CUDA_IPC, 'CUDA IPC not available')
    @unittest.skipIf(not TEST_MULTIGPU, 'found only 1 GPU')
    def test_cuda_small_tensors(self):
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context('spawn')
        tensors = []
        for i in range(5):
            tensors += [torch.range(i * 5, (i * 5) + 4).cuda()]

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
            self.assertEqual(v, torch.range(i * 5, (i * 5) + 4).sum())
            self.assertEqual(device, 0)
            self.assertEqual(tensor_size, 5)
            self.assertEqual(storage_size, 5)

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

    def _test_autograd_sharing(self, var):
        ready = mp.Event()
        master_modified = mp.Event()
        queue = mp.Queue()
        p = mp.Process(target=autograd_sharing, args=(queue, ready, master_modified))
        p.start()
        queue.put(var)

        ready.wait()
        var.data[0, 0] = 1000
        if var.grad is not None:
            var.grad.data[:] = torch.ones(5, 5) * 4
        master_modified.set()

        worker_ok = queue.get()
        self.assertTrue(worker_ok)

        self.assertEqual(var.data, torch.ones(5, 5))
        if var.grad is not None:
            self.assertEqual(var.grad.data, torch.ones(5, 5))
        p.join()

    def test_variable_sharing(self):
        configs = [
            (True, False),
            (False, False),
            (False, True),
        ]
        for requires_grad, volatile in configs:
            var = Variable(torch.range(1, 25).view(5, 5),
                           requires_grad=requires_grad,
                           volatile=volatile)
            self._test_autograd_sharing(var)

    def test_parameter_sharing(self):
        param = Parameter(torch.range(1, 25).view(5, 5))
        self._test_autograd_sharing(param)

    def _test_is_shared(self):
        t = torch.randn(5, 5)
        self.assertFalse(t.is_shared())
        t.share_memory_()
        self.assertTrue(t.is_shared())

    @unittest.skipIf(platform == 'darwin', "file descriptor strategy is not supported on OS X")
    def test_is_shared(self):
        self._test_is_shared()

    def test_fs_is_shared(self):
        with fs_sharing():
            self._test_is_shared()

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA not available')
    def test_is_shared_cuda(self):
        t = torch.randn(5, 5).cuda()
        self.assertTrue(t.is_shared())


if __name__ == '__main__':
    run_tests()
