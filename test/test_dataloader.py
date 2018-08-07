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
import subprocess
from torch import multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import default_collate, ExceptionWrapper, MANAGER_STATUS_CHECK_INTERVAL
from common import TestCase, run_tests, TEST_NUMPY, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN, skipIfRocm

# We cannot import TEST_CUDA from common_nn here, because if we do that,
# the TEST_CUDNN line from common_nn will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()

# We need spawn start method for test_manager_unclean_exit, but
# Python 2.7 doesn't allow it.
if sys.version_info[0] == 3:
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers RuntimeError.
    mp = mp.get_context(method='spawn')


JOIN_TIMEOUT = 17.0 if IS_WINDOWS else 6.5


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

    def test_single_tensor(self):
        t = torch.randn(5, 10)
        source = TensorDataset(t)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t[i], source[i][0])

    def test_many_tensors(self):
        t0 = torch.randn(5, 10, 2, 3, 4, 5)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10, 2, 5)
        t3 = torch.randn(5, 10, 3, 7)
        source = TensorDataset(t0, t1, t2, t3)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t0[i], source[i][0])
            self.assertEqual(t1[i], source[i][1])
            self.assertEqual(t2[i], source[i][2])
            self.assertEqual(t3[i], source[i][3])


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
class ErrorTrackingProcess(mp.Process):

    def __init__(self, *args, **kwargs):
        super(ErrorTrackingProcess, self).__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
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
        self.count = mp.Value('i', 0, lock=True)
        self.barrier = mp.Semaphore(0)
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

    def test_invalid_assign_after_init(self):
        dl = DataLoader(self.dataset)
        for attr in ('batch_size', 'sampler', 'drop_last'):
            def fn():
                setattr(dl, attr, {})

            self.assertRaises(ValueError, fn)

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
    @skipIfRocm
    def test_sequential_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @skipIfRocm
    def test_multiple_dataloaders(self):
        loader1_it = iter(DataLoader(self.dataset, num_workers=1))
        loader2_it = iter(DataLoader(self.dataset, num_workers=2))
        next(loader1_it)
        next(loader1_it)
        next(loader2_it)
        next(loader2_it)
        next(loader1_it)
        next(loader2_it)

    @unittest.skip("temporarily disable until flaky failures are fixed")
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

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    def test_batch_sampler(self):
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
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

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
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

    @staticmethod
    def _manager_process(dataset, worker_pids, manager_exit_event):
        loader = iter(DataLoader(dataset, batch_size=2, num_workers=4, pin_memory=True))
        workers = loader.workers
        for i in range(len(workers)):
            worker_pids[i] = int(workers[i].pid)
        for i, sample in enumerate(loader):
            if i == 3:
                break
        # Simulate a dirty exit of the manager process
        manager_exit_event.set()
        if IS_WINDOWS:
            os.system('taskkill /PID ' + str(os.getpid()) + ' /F')
        else:
            os.kill(os.getpid(), signal.SIGKILL)

    @staticmethod
    def _is_process_alive(pid, pname):
        # There is a chance of a terminated child process's pid being reused by a new unrelated process,
        # but since we are looping this check very frequently, we will know that the child process dies
        # before the new unrelated process starts.
        if IS_WINDOWS:
            command = 'tasklist | find "{}" /i'.format(pid)
        else:
            command = 'ps -p {} -o comm='.format(pid)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        output = output.decode('utf-8')
        return pname in output

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    @unittest.skipIf(sys.version_info[0] == 2,
                     "spawn start method is not supported in Python 2, \
                     but we need it for creating another process with CUDA")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
    def test_main_process_unclean_exit(self):
        '''There might be ConnectionResetError or leaked semaphore warning (due to dirty process exit), \
but they are all safe to ignore'''
        worker_pids = mp.Array('i', [0] * 4)

        manager_exit_event = mp.Event()
        p = mp.Process(target=TestDataLoader._manager_process,
                       args=(self.dataset, worker_pids, manager_exit_event))
        p.start()

        manager_exit_event.wait()

        exit_status = [False] * len(worker_pids)
        start_time = time.time()
        pname = 'python'
        while True:
            for i in range(len(worker_pids)):
                pid = worker_pids[i]
                if not exit_status[i]:
                    if not TestDataLoader._is_process_alive(pid, pname):
                        exit_status[i] = True
            if all(exit_status):
                break
            else:
                time.sleep(1)
                self.assertFalse(time.time() - start_time > MANAGER_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT,
                                 'subprocess not terminated')

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
    @skipIfRocm
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
    @skipIfRocm
    def test_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())


class TestWorkerQueueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.worker_id = None

    def worker_init_fn(self, worker_id):
        self.worker_id = worker_id

    def __getitem__(self, item):
        return self.worker_id, self.data[item]

    def __len__(self):
        return len(self.data)


class TestIndividualWorkerQueue(TestCase):
    def setUp(self):
        self.dataset = TestWorkerQueueDataset([i for i in range(128)])

    def _run_ind_worker_queue_test(self, batch_size, num_workers):
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            worker_init_fn=self.dataset.worker_init_fn
        )
        current_worker_idx = 0
        for i, (worker_ids, sample) in enumerate(loader):
            self.assertEqual(worker_ids.tolist(), [current_worker_idx] * batch_size)
            self.assertEqual(sample.tolist(), [j for j in range(i * batch_size, (i + 1) * batch_size)])
            current_worker_idx += 1
            if current_worker_idx == num_workers:
                current_worker_idx = 0

    @skipIfRocm
    def test_ind_worker_queue(self):
        for batch_size in (8, 16, 32, 64):
            for num_workers in range(1, 6):
                self._run_ind_worker_queue_test(batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    run_tests()
