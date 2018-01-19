import math
import sys
import errno
import os
import ctypes
import signal
import torch
import time
import traceback
import unittest
import socket
from torch import multiprocessing
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.utils import ExceptionWrapper, QueueWrapper
from common import TestCase, run_tests, TEST_NUMPY, IS_WINDOWS
from common_nn import TEST_CUDA


if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


JOIN_TIMEOUT = 17.0 if IS_WINDOWS else 4.5


class TestDatasetRandomSplit(TestCase):
    def test_lengths_must_equal_datset_size(self):
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

    def test_splits_are_mutually_exclusive(self):
        data = [5, 2, 3, 4, 1, 6]
        splits = random_split(data, [2, 4])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)


class TestTensorDataset(TestCase):

    def test_len(self):
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])


class TestConcatDataset(TestCase):

    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())


# Stores the first encountered exception in .exception.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(multiprocessing.Process):

    def __init__(self, *args, **kwargs):
        super(ErrorTrackingProcess, self).__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        # Disable stderr printing from os level, and make workers not printing
        # to stderr.
        # Can't use sys.stderr.close, otherwise Python `raise` will error with
        # ValueError: I/O operation on closed file.
        os.close(sys.stderr.fileno())
        try:
            super(ErrorTrackingProcess, self).run()
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)

    # ESRCH means that os.kill can't finds alive proc
    def send_signal(self, signum, ignore_ESRCH=False):
        try:
            os.kill(self.pid, signum)
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


class ErrorDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class SegfaultDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return ctypes.string_at(0)

    def __len__(self):
        return self.size


class SleepDataset(Dataset):

    def __init__(self, size, sleep_sec):
        self.size = size
        self.sleep_sec = sleep_sec

    def __getitem__(self, idx):
        time.sleep(self.sleep_sec)
        return idx

    def __len__(self):
        return self.size


class SeedDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.initial_seed()

    def __len__(self):
        return self.size


# Inspired by https://stackoverflow.com/a/26703365
# This will ensure that each worker at least processes one data
class SynchronizedSeedDataset(Dataset):

    def __init__(self, size, num_workers):
        assert size >= num_workers
        self.count = multiprocessing.Value('i', 0, lock=True)
        self.barrier = multiprocessing.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def __getitem__(self, idx):
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()
        return torch.initial_seed()

    def __len__(self):
        return self.size


def _test_timeout():
    dataset = SleepDataset(10, 10)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1)
    _ = next(iter(dataloader))


def _test_segfault():
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    _ = next(iter(dataloader))


def _test_interrupt_retry(timeout=0):
    dataset = TensorDataset(torch.randn(1, 1), torch.randn(1, 1))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, timeout=timeout)
    dataloaderiter = iter(dataloader)

    # make SIGUSR1 interrupt
    def handler(signum, frame):
        pass
    signal.signal(signal.SIGUSR1, handler)

    # Replace iterator getter with a wrapper that reliably calls an
    # interruptable blocking recv syscall to simulate interruption during recv
    # in queue.get.
    # The used socket.recv call below in the replacing function is quite
    # dangerous because it blocks everything on Python side (likely by holding
    # GIL), including any Python signal handlers and the cleaning up in
    # dataloder.__del__ when this process exits. To prevent orphan worker child,
    # we manually terminate worker process here.
    # Conveniently, the worker has SIGTERM handler installed so SIGTERM from
    # loader process won't cause loader error.

    data_queue = dataloaderiter.data_queue
    data = data_queue.get()  # ensure that worker handlers are installed
    for w in dataloaderiter.workers:
        w.terminate()

    def interruptable_get(*args, **kwargs):
        if dataloaderiter.shutdown:
            return data
        # get and config timeout if the argument is present
        if timeout >= 0:
            if 'timeout' in kwargs:
                timeout_val = kwargs['timeout']
            elif len(args) > 1:
                timeout_val = args[1]  # first arg is `block`
            else:
                timeout_val = None
            socket.setdefaulttimeout(timeout_val)
        s = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        try:
            return s.recv(1024)
        except socket.timeout:
            raise queue.Empty
        finally:
            s.close()

    try:
        if isinstance(data_queue, QueueWrapper):
            data_queue.queue.get = interruptable_get
    except Exception:
        # in Python >= 3.5 or on Windows QueueWrapper is not a class but a no-op
        pass
    _ = next(dataloaderiter)


# test custom init function
def init_fn(worker_id):
    torch.manual_seed(12345)


class TestDataLoader(TestCase):

    def setUp(self):
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        for i, (sample, target) in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(sample, self.data[idx:idx + batch_size])
            self.assertEqual(target, self.labels[idx:idx + batch_size])
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = {i: 0 for i in range(self.data.size(0))}
        found_labels = {i: 0 for i in range(self.labels.size(0))}
        batch_size = loader.batch_size
        for i, (batch_samples, batch_targets) in enumerate(loader):
            for sample, target in zip(batch_samples, batch_targets):
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
            self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
            self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_error(self, loader):
        it = iter(loader)
        errors = 0
        while True:
            try:
                next(it)
            except NotImplementedError:
                errors += 1
            except StopIteration:
                self.assertEqual(errors,
                                 math.ceil(float(len(loader.dataset)) / loader.batch_size))
                return

    def test_sequential(self):
        self._test_sequential(DataLoader(self.dataset))

    def test_sequential_batch(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2))

    def test_growing_dataset(self):
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = DataLoader(dataset, shuffle=False)
        dataloader_shuffle = DataLoader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_sequential_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    def test_multiple_dataloaders(self):
        loader1_it = iter(DataLoader(self.dataset, num_workers=1))
        loader2_it = iter(DataLoader(self.dataset, num_workers=2))
        next(loader1_it)
        next(loader1_it)
        next(loader2_it)
        next(loader2_it)
        next(loader1_it)
        next(loader2_it)

    def test_segfault(self):
        p = ErrorTrackingProcess(target=_test_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            if IS_WINDOWS:
                self.assertIsInstance(p.exception, OSError)
                self.assertRegex(str(p.exception), r'access violation reading ')
            else:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(str(p.exception), r'DataLoader worker \(pid \d+\) is killed by signal: ')
        finally:
            p.terminate()

    def test_timeout(self):
        p = ErrorTrackingProcess(target=_test_timeout)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r'DataLoader timed out after \d+ seconds')
        finally:
            p.terminate()

    def test_worker_seed(self):
        num_workers = 6
        dataset = SynchronizedSeedDataset(num_workers, num_workers)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        seeds = set()
        for batch in dataloader:
            seeds.add(batch[0])
        self.assertEqual(len(seeds), num_workers)

    def test_worker_init_fn(self):
        dataset = SeedDataset(4)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2,
                                worker_init_fn=init_fn)
        for batch in dataloader:
            self.assertEqual(12345, batch[0])
            self.assertEqual(12345, batch[1])

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this test case for Windows")
    def test_interrupt_retry(self):
        # time.sleep doesn't seem to work on main process when running unittest
        # for all tests together. Use Process.join as a sleep function.
        p = ErrorTrackingProcess(target=_test_interrupt_retry)
        p.start()
        p.join(1)  # give it some time to start
        try:
            self.assertTrue(p.is_alive())
            for i in range(3):
                p.send_signal(signal.SIGUSR1)
                p.join(0.5)
                self.assertTrue(p.is_alive())
        except OSError as e:
            self.fail("DataLoader shouldn't fail due to interrupted syscall")
        try:
            self.assertTrue(p.is_alive())
            self.assertIsNone(p.exception)
        finally:
            p.terminate()

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this test case for Windows")
    def test_interrupt_retry_with_timeout(self):
        # time.sleep doesn't seem to work on main process when running unittest
        # for all tests together. Use Process.join as a sleep function.
        timeout = 2
        p = ErrorTrackingProcess(target=lambda: _test_interrupt_retry(timeout))
        p.start()
        p.join(1)  # give it some time to start
        try:
            for _ in range(5):
                if p.is_alive():
                    p.send_signal(signal.SIGUSR1)
                    p.join(0.5)
                else:
                    break
            p.join(JOIN_TIMEOUT)
        except OSError as e:
            self.fail("DataLoader shouldn't fail due to interrupted syscall")
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r'DataLoader timed out after {} seconds'.format(timeout))
        finally:
            p.terminate()

    def test_shuffle(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True))

    def test_sequential_workers(self):
        self._test_sequential(DataLoader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2, num_workers=4))

    def test_shuffle_workers(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4))

    def _test_batch_sampler(self, **kwargs):
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []
        for i in range(0, 100, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = DataLoader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 40)
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                self.assertEqual(len(input), 2)
                self.assertEqual(input, self.data[offset:offset + 2])
            else:
                offset = i * 5 // 2
                self.assertEqual(len(input), 3)
                self.assertEqual(input, self.data[offset:offset + 3])

    def test_batch_sampler(self):
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy(self):
        import numpy as np

        class TestDataset(torch.utils.data.Dataset):
            def __getitem__(self, i):
                return np.ones((2, 3, 4)) * i

            def __len__(self):
                return 1000

        loader = DataLoader(TestDataset(), batch_size=12)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.DoubleTensor)
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))

    def test_error(self):
        self._test_error(DataLoader(ErrorDataset(100), batch_size=2, shuffle=True))

    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_partial_workers(self):
        "check that workers exit even if the iterator is not exhausted"
        loader = iter(DataLoader(self.dataset, batch_size=2, num_workers=4, pin_memory=True))
        workers = loader.workers
        worker_manager_thread = loader.worker_manager_thread
        for i, sample in enumerate(loader):
            if i == 3:
                break
        del loader
        for w in workers:
            w.join(JOIN_TIMEOUT)
            self.assertFalse(w.is_alive(), 'subprocess not terminated')
            self.assertEqual(w.exitcode, 0)
        worker_manager_thread.join(JOIN_TIMEOUT)
        self.assertFalse(worker_manager_thread.is_alive())

    def test_len(self):
        def check_len(dl, expected):
            self.assertEqual(len(dl), expected)
            n = 0
            for sample in dl:
                n += 1
            self.assertEqual(n, expected)
        check_len(self.dataset, 100)
        check_len(DataLoader(self.dataset, batch_size=2), 50)
        check_len(DataLoader(self.dataset, batch_size=3), 34)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_scalars(self):
        import numpy as np

        class ScalarDataset(torch.utils.data.Dataset):
            def __init__(self, dtype):
                self.dtype = dtype

            def __getitem__(self, i):
                return self.dtype()

            def __len__(self):
                return 4

        dtypes = {
            np.float64: torch.DoubleTensor,
            np.float32: torch.FloatTensor,
            np.float16: torch.HalfTensor,
            np.int64: torch.LongTensor,
            np.int32: torch.IntTensor,
            np.int16: torch.ShortTensor,
            np.int8: torch.CharTensor,
            np.uint8: torch.ByteTensor,
        }
        for dt, tt in dtypes.items():
            dset = ScalarDataset(dt)
            loader = DataLoader(dset, batch_size=2)
            batch = next(iter(loader))
            self.assertIsInstance(batch, tt)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_colate_bad_numpy_types(self):
        import numpy as np

        # Should be a no-op
        arr = np.array(['a', 'b', 'c'])
        default_collate(arr)

        arr = np.array([[['a', 'b', 'c']]])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: default_collate(arr))


class StringDataset(Dataset):
    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)

    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


class TestStringDataLoader(TestCase):
    def setUp(self):
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for batch_ndx, (s, n) in enumerate(loader):
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.Tensor(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


class TestDictDataLoader(TestCase):
    def setUp(self):
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batch_size = loader.batch_size
        for i, sample in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(set(sample.keys()), {'a_tensor', 'another_dict'})
            self.assertEqual(set(sample['another_dict'].keys()), {'a_number'})

            t = sample['a_tensor']
            self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
            self.assertTrue((t[0] == idx).all())
            self.assertTrue((t[1] == idx + 1).all())

            n = sample['another_dict']['a_number']
            self.assertEqual(n.size(), torch.Size([batch_size]))
            self.assertEqual(n[0], idx)
            self.assertEqual(n[1], idx + 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())


if __name__ == '__main__':
    run_tests()
