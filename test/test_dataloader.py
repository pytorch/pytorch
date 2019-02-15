import math
import sys
import errno
import os
import ctypes
import signal
import torch
import gc
import time
import traceback
import unittest
import subprocess
import itertools
import warnings
from torch import multiprocessing as mp
from torch.utils.data import _utils, Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.data._utils import ExceptionWrapper, MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from common_utils import (TestCase, run_tests, TEST_NUMPY, IS_WINDOWS, IS_PPC,
                          IS_PYTORCH_CI, NO_MULTIPROCESSING_SPAWN, skipIfRocm,
                          load_tests)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    err_msg = ("psutil not found. Some critical data loader tests relying on it "
               "(e.g., TestDataLoader.test_proper_exit) will not run.")
    if IS_PYTORCH_CI:
        raise ImportError(err_msg)
    else:
        warnings.warn(err_msg)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA from common_cuda here, because if we do that,
# the TEST_CUDNN line from common_cuda will be executed multiple times
# as well during the execution of this test suite, and it will cause
# CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()

if not NO_MULTIPROCESSING_SPAWN:
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers RuntimeError.
    mp = mp.get_context(method='spawn')


JOIN_TIMEOUT = 17.0 if (IS_WINDOWS or IS_PPC) else 13.0


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
        # Disable polluting stderr with errors that are supposed to happen.
        sys.stderr = open(os.devnull, "w")
        try:
            super(ErrorTrackingProcess, self).run()
            self._cconn.send(None)
        except Exception:
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
        self.sleeped = False

    def __getitem__(self, idx):
        if not self.sleeped:
            time.sleep(self.sleep_sec)
            self.sleeped = True
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
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1)
    _ = next(iter(dataloader))


def _test_timeout_pin_memory():
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1, pin_memory=True)
    _ = next(iter(dataloader))


def disable_stderr(worker_id):
    r"""
    Avoids printing "ERROR: Unexpected segmentation fault encountered in worker."
    from workers. Since worker signal handler prints with low-level write(),
    this has to be done on OS level via dup.

    This is used as worker_init_fn for test_segfault.
    """
    sys.stderr.flush()  # flush library buffers that dup2 knows nothing about
    # Can't use a with-block because otherwise the fd will be closed when this
    # function ends.
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), sys.stderr.fileno())


def _test_segfault():
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=disable_stderr)
    _ = next(iter(dataloader))


class TestProperExitDataset(object):
    def __init__(self, size, error_event):
        self.size = size
        self.error_event = error_event

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.error_event is not None and self.error_event.is_set():
            raise RuntimeError('Worker error')
        return torch.tensor([idx])


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(use_workers, pin_memory, exit_method, hold_iter_reference,
                      loader_setup_event, tester_setup_event):
    num_workers = 2 if use_workers else 0

    if exit_method == 'worker_error' or exit_method == 'worker_kill':
        assert use_workers is True

    if exit_method == 'worker_error':
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    ds = TestProperExitDataset(12, worker_error_event)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    error_it = 2

    if use_workers:
        # 2 is the magical per-worker prefetch number...
        # FIXME: change this after the number becomes configurable.
        assert len(loader) > (error_it + 2 + 1) * num_workers

    it = iter(loader)
    if use_workers:
        workers = it.workers

    def kill_pid(pid):
        psutil_p = psutil.Process(pid)
        psutil_p.kill()
        psutil_p.wait(JOIN_TIMEOUT)
        assert not psutil_p.is_running()

    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                del it
            loader_setup_event.set()
            tester_setup_event.wait()
            # ensure that the workers are still alive
            if use_workers:
                for w in workers:
                    assert w.is_alive()
            if worker_error_event is not None:
                worker_error_event.set()

        if i == error_it:
            if exit_method == 'loader_error':
                raise RuntimeError('Loader error')
            elif exit_method == 'loader_kill':
                kill_pid(os.getpid())
            elif exit_method == 'worker_kill':
                kill_pid(workers[0].pid)

    if not hold_iter_reference:
        # Tries to trigger the __del__ clean-up rather than the automatic
        # exiting of daemonic children. Technically it should be automatically
        # triggered, but I don't want to rely on the implementation detail of
        # Python gc.
        gc.collect()


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
        if TEST_CUDA and not NO_MULTIPROCESSING_SPAWN:
            targets = (_test_timeout, _test_timeout_pin_memory)
        else:
            targets = (_test_timeout,)
        for target in targets:
            p = ErrorTrackingProcess(target=target)
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

    def test_RandomSampler(self):

        from collections import Counter
        from torch.utils.data import RandomSampler

        def sample_stat(sampler, num_samples):
            counts = Counter(sampler)
            count_repeated = sum(val > 1 for val in counts.values())
            return (count_repeated, min(counts.keys()), max(counts.keys()))

        # test sample with replacement
        n = len(self.dataset) + 1  # ensure at least one sample is drawn more than once
        sampler_with_replacement = RandomSampler(self.dataset, replacement=True, num_samples=n)
        count_repeated, minval, maxval = sample_stat(sampler_with_replacement, n)
        self.assertTrue(count_repeated > 0)
        self.assertTrue(minval >= 0)
        self.assertTrue(maxval < len(self.dataset))

        # test sample without replacement
        sampler_without_replacement = RandomSampler(self.dataset)
        count_repeated, minval, maxval = sample_stat(sampler_without_replacement, len(self.dataset))
        self.assertTrue(count_repeated == 0)
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)

        # raise error when replacement=False and num_samples is not None
        self.assertRaises(ValueError, lambda: RandomSampler(self.dataset, num_samples=len(self.dataset)))

        self.assertRaises(ValueError, lambda: RandomSampler(self.dataset, num_samples=0))

    def test_random_sampler_len_with_replacement(self):
        from torch.utils.data import RandomSampler
        # add 5 extra samples
        num_samples = len(self.dataset) + 5
        sampler = RandomSampler(self.dataset,
                                replacement=True,
                                num_samples=num_samples)
        # test len method
        self.assertEqual(num_samples, len(sampler))

        # test with iteration
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # test with dataloader, batch_size = 1
        batch_size = 1
        count_num_samples_in_data_loader = len(DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # test with dataloader, batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(int(math.ceil(float(num_samples) / batch_size)),
                         count_num_samples_in_data_loader)

    def test_duplicating_data_with_drop_last(self):

        from torch.utils.data.distributed import DistributedSampler

        num_processes = 4
        num_batches = 9
        data_set = torch.IntTensor(range(num_batches))
        scanned_data = torch.IntTensor([])
        for i in range(num_processes):
            s = DistributedSampler(data_set, num_processes, i)
            d_loader = DataLoader(data_set, batch_size=int(num_batches / num_processes), drop_last=True, sampler=s)
            for k, data in enumerate(d_loader):
                scanned_data = torch.cat((scanned_data, data), 0)

        self.assertEqual(scanned_data.size(), scanned_data.unique().size())

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
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

    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                     don't support multiprocessing with spawn start method")
    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    def test_partial_workers(self):
        r"""Check that workers exit even if the iterator is not exhausted."""
        if TEST_CUDA:
            pin_memory_configs = (True, False)
        else:
            pin_memory_configs = (False,)

        for pin_memory in pin_memory_configs:
            loader = iter(DataLoader(self.dataset, batch_size=2, num_workers=4, pin_memory=pin_memory))
            workers = loader.workers
            if pin_memory:
                pin_memory_thread = loader.pin_memory_thread
            for i, sample in enumerate(loader):
                if i == 10:
                    break
            assert i == 10
            del loader
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive(), 'subprocess not terminated')
            if pin_memory:
                pin_memory_thread.join(JOIN_TIMEOUT)
                self.assertFalse(pin_memory_thread.is_alive())

    @skipIfRocm
    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    def test_proper_exit(self):
        (r'''There might be ConnectionResetError or leaked semaphore warning '''
         r'''(due to dirty process exit), but they are all safe to ignore''')

        # TODO: test the case where the pin_memory_thread triggers an
        #       error/fatal signal. I haven't found out how to properly do that.

        for use_workers, pin_memory, hold_iter_reference in itertools.product([True, False], repeat=3):
            # `hold_iter_reference` specifies whether we hold a reference to the
            # iterator. This is interesting because Python3 error traces holds a
            # reference to the frames, which hold references to all the local
            # variables including the iterator, and then the iterator dtor may
            # not be called before process end. It is important to see that the
            # processes still exit in both cases.

            if pin_memory and (not TEST_CUDA or NO_MULTIPROCESSING_SPAWN):
                # Can't use CUDA without spawn
                continue

            # `exit_method` controls the way the loader process ends.
            #   - `*_kill` means that `*` is killed by OS.
            #   - `*_error` means that `*` raises an error.
            #   - `None` means that no error happens.
            # In all cases, all processes should end properly.
            if use_workers:
                exit_methods = [None, 'loader_error', 'loader_kill', 'worker_kill', 'worker_error']
            else:
                exit_methods = [None, 'loader_error', 'loader_kill']

            for exit_method in exit_methods:

                desc = []
                desc.append('use_workers={}'.format(use_workers))
                desc.append('pin_memory={}'.format(pin_memory))
                desc.append('hold_iter_reference={}'.format(hold_iter_reference))
                desc.append('exit_method={}'.format(exit_method))
                desc = 'test_proper_exit with ' + ', '.join(desc)

                # Event that the loader process uses to signal testing process
                # that various things are setup, including that the worker pids
                # are specified in `worker_pids` array.
                loader_setup_event = mp.Event()

                # Event that this process has finished setting up, and the
                # loader process can now proceed to trigger error events or
                # finish normally.
                tester_setup_event = mp.Event()

                loader_p = ErrorTrackingProcess(target=_test_proper_exit,
                                                args=(use_workers, pin_memory, exit_method,
                                                      hold_iter_reference, loader_setup_event,
                                                      tester_setup_event))
                loader_p.start()

                # Wait for loader process to set everything up, e.g., starting
                # workers.
                loader_setup_event.wait(timeout=JOIN_TIMEOUT)
                if not loader_setup_event.is_set():
                    fail_msg = desc + ': loader process failed to setup within given time'
                    if loader_p.exception is not None:
                        self.fail(fail_msg + ', and had exception {}'.format(loader_p.exception))
                    elif not loader_p.is_alive():
                        self.fail(fail_msg + ', and exited with code {} but had no exception'.format(loader_p.exitcode))
                    else:
                        self.fail(fail_msg + ', and is still alive.')

                worker_psutil_p = psutil.Process(loader_p.pid).children()

                tester_setup_event.set()

                try:
                    loader_p.join(JOIN_TIMEOUT + MP_STATUS_CHECK_INTERVAL)
                    if loader_p.is_alive():
                        fail_msg = desc + ': loader process did not terminate'
                        if loader_p.exception is not None:
                            self.fail(fail_msg + ', and had exception {}'.format(loader_p.exception))
                        else:
                            self.fail(fail_msg + ', and had no exception')
                    _, alive = psutil.wait_procs(worker_psutil_p, timeout=(MP_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT))
                    if len(alive) > 0:
                        self.fail(desc + ': worker process (pid(s) {}) did not terminate'.format(
                            ', '.join(str(p.pid) for p in alive)))
                    if exit_method is None:
                        self.assertEqual(loader_p.exitcode, 0)
                    else:
                        self.assertNotEqual(loader_p.exitcode, 0)
                        if exit_method == 'loader_error':
                            self.assertIsInstance(loader_p.exception, RuntimeError, desc)
                            self.assertIn('Loader error', str(loader_p.exception), desc)
                        elif exit_method == 'worker_kill':
                            self.assertIsInstance(loader_p.exception, RuntimeError, desc)
                            self.assertIn('DataLoader worker (pid', str(loader_p.exception), desc)
                        elif exit_method == 'worker_error':
                            self.assertIsInstance(loader_p.exception, RuntimeError, desc)
                            self.assertIn('Worker error', str(loader_p.exception), desc)
                finally:
                    loader_p.terminate()

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

    def test_default_collate_dtype(self):
        arr = [1, 2, -1]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.int64)

        arr = [1.1, 2.3, -0.9]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.float64)

        arr = [True, False]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.uint8)

        # Should be a no-op
        arr = ['a', 'b', 'c']
        self.assertEqual(arr, _utils.collate.default_collate(arr))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_bad_numpy_types(self):
        import numpy as np

        # Should be a no-op
        arr = np.array(['a', 'b', 'c'])
        self.assertEqual(arr, _utils.collate.default_collate(arr))

        arr = np.array([[['a', 'b', 'c']]])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: _utils.collate.default_collate(arr))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_shared_tensor(self):
        import numpy as np
        t_in = torch.zeros(1)
        n_in = np.zeros(1)

        self.assertEqual(t_in.is_shared(), False)

        self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), False)
        self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), False)

        old = _utils.collate._use_shared_memory
        try:
            _utils.collate._use_shared_memory = True
            self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), True)
            self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), True)
        finally:
            _utils.collate._use_shared_memory = old


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


class NamedTupleDataset(Dataset):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data', 'label'])
    Data = namedtuple('Data', ['positive', 'negative'])

    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return self.Batch(data=self.Data(positive=ndx, negative=-ndx),
                          label=str(ndx))


class TestNamedTupleDataLoader(TestCase):
    def setUp(self):
        self.dataset = NamedTupleDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_collate_and_pin_memory_with_namedtuple(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for batch in loader:
            self.assertIsInstance(batch, NamedTupleDataset.Batch)
            self.assertIsInstance(batch.data, NamedTupleDataset.Data)


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


class TestCustomPinFn(TestCase):
    def setUp(self):
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.dataset = TensorDataset(inps, tgts)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
    def test_custom_batch_pin(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_wrapper,
                            pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample.inp.is_pinned())
            self.assertTrue(sample.tgt.is_pinned())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @skipIfRocm
    def test_custom_batch_pin_worker(self):
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_wrapper,
                            pin_memory=True, num_workers=1)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample.inp.is_pinned())
            self.assertTrue(sample.tgt.is_pinned())


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

    def test_ind_worker_queue(self):
        for batch_size in (8, 16, 32, 64):
            for num_workers in range(1, 6):
                self._run_ind_worker_queue_test(batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    run_tests()
