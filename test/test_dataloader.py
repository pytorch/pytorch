# Owner(s): ["module: dataloader"]

import math
import sys
import errno
import os
import ctypes
import faulthandler
import torch
import gc
import time
import signal
import unittest
import itertools
import warnings
import tempfile
import torch.utils.data.datapipes as dp
from torch import multiprocessing as mp
from torch.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    IterDataPipe,
    Subset,
    TensorDataset,
    StackDataset,
    _utils
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from torch.utils.data.datapipes.iter import IterableWrapper
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_utils import (TestCase, run_tests, TEST_NUMPY, IS_WINDOWS, IS_JETSON,
                                                  IS_CI, NO_MULTIPROCESSING_SPAWN, skipIfRocm, slowTest,
                                                  load_tests, TEST_WITH_ASAN, TEST_WITH_TSAN, IS_SANDCASTLE,
                                                  IS_MACOS, TEST_CUDA, parametrize, skipIfNoDill)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
import functools
import operator


try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    err_msg = ("psutil not found. Some critical data loader tests relying on it "
               "(e.g., TestDataLoader.test_proper_exit) will not run.")
    if IS_CI:
        raise ImportError(err_msg) from None
    else:
        warnings.warn(err_msg)


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_CUDA:
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')

if not NO_MULTIPROCESSING_SPAWN:
    # We want to use `spawn` if able because some of our tests check that the
    # data loader terminiates gracefully. To prevent hanging in the testing
    # process, such data loaders are run in a separate subprocess.
    #
    # We also want to test the `pin_memory=True` configuration, thus `spawn` is
    # required to launch such processes and they initialize the CUDA context.
    #
    # Mixing different start method is a recipe for disaster (e.g., using a fork
    # `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
    # to avoid bugs.
    #
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers `RuntimeError`.
    mp = mp.get_context(method='spawn')


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


supported_multiprocessing_contexts = [None] + list(torch.multiprocessing.get_all_start_methods())


# collate_fn that returns the batch cloned; defined globally here for pickle purposes.
def _clone_collate(b):
    return [x.clone() for x in b]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestDatasetRandomSplit(TestCase):
    def test_lengths_must_equal_dataset_size(self):
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

        splits = random_split([1, 2, 3, 4, 5, 6], [0.5, 0.5])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 3)
        self.assertEqual(len(splits[1]), 3)

        # Odd size splits
        self.assertEqual(
            len(random_split(range(3), [0.5, 0.5], generator=torch.Generator().manual_seed(1))),
            2
        )

        # Odd sized round-robin splits
        splits = random_split(range(106), [0.1, 0.2, 0.3, 0.4],
                              generator=torch.Generator().manual_seed(1))
        self.assertEqual(len(splits[0]), 11)
        self.assertEqual(len(splits[1]), 22)
        self.assertEqual(len(splits[2]), 31)
        self.assertEqual(len(splits[3]), 42)


    def test_splits_are_mutually_exclusive(self):
        data = [5, 2, 3, 4, 1, 6]
        splits = random_split(data, [2, 4])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        splits = random_split(data, [0.33, 0.67])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        data = [1, 2, 3, 4]
        splits = random_split(data, [0.25, 0.75])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

    def test_splits_indexing_type(self):
        r"""Indices generated by random_split
          should be of integer type
        """
        class CustomDataset:
            def __init__(self, test_object, custom_list):
                self.data = custom_list
                self.test_object = test_object

            def __getitem__(self, key):
                self.test_object.assertEqual(type(key), int)
                return self.data[key]

            def __len__(self):
                return len(self.data)

        x = [1, 2, 3, 4, 5]
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [5])[0]
        data_loader = DataLoader(dataset)
        for batch in data_loader:
            pass

        # fractional splitting
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [1.0])[0]
        data_loader = DataLoader(dataset)
        for batch in data_loader:
            pass

    def test_splits_reproducibility(self):
        self.assertEqual(
            [list(x) for x in random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(1))],
            [[5, 6, 1], [2, 0, 8, 9, 3, 7, 4]],
        )
        self.assertEqual(
            random_split(range(100), [60, 40], generator=torch.Generator().manual_seed(42)),
            random_split(range(100), [60, 40], generator=torch.Generator().manual_seed(42)),
        )
        self.assertEqual(
            random_split(range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)),
            random_split(range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)),
        )
        self.assertEqual(
            random_split(range(100), [0.33, 0.33, 0.34], generator=torch.Generator().manual_seed(42)),
            random_split(range(100), [0.33, 0.33, 0.34], generator=torch.Generator().manual_seed(42)),
        )

    def test_incomplete_fractional_splits(self):
        with self.assertRaises(ValueError):
            # should raise since the sum of fractions is not 1
            random_split([1, 2, 3, 4], [0.1])

        with self.assertRaises(ValueError):
            # should raise since fraction > 1
            random_split([1, 2, 3, 4], [1.1])

    def test_splits_generator(self):
        # A random_split without a specific generator should affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5])
        b = torch.rand(10)
        self.assertNotEqual(a, b)

        # A random_split with a specific generator should not affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5], generator=torch.Generator().manual_seed(42))
        b = torch.rand(10)
        self.assertEqual(a, b)

    def test_slicing_of_subset_of_dataset(self):
        # Testing slicing a subset initialized with a dataset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_dataset[:], dataset[:])
        self.assertEqual(subset_of_dataset[1:2], dataset[1:2])
        self.assertEqual(subset_of_dataset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset from random split
        subset1, subset2 = random_split(dataset, [3, 2])
        self.assertEqual(subset1[:], dataset[subset1.indices[:]])
        self.assertEqual(subset1[0:2], dataset[subset1.indices[0:2]])
        self.assertEqual(subset1[0:-1:2], dataset[subset1.indices[0:-1:2]])

    def test_slicing_of_subset_of_subset(self):
        # Testing slicing a subset initialized with a subset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        subset_of_subset = Subset(subset_of_dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_subset[:], dataset[:])
        self.assertEqual(subset_of_subset[0:2], dataset[0:2])
        self.assertEqual(subset_of_subset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset of subset from random split
        subset1, subset2 = random_split(dataset, [4, 1])
        subset_of_subset1, subset_of_subset2 = random_split(subset1, [3, 1])
        idx = [subset1.indices[i] for i in subset_of_subset1.indices]
        self.assertEqual(subset_of_subset1[:], dataset[idx.copy()])
        self.assertEqual(subset_of_subset1[0:2], dataset[idx[0:2]])
        self.assertEqual(subset_of_subset1[0:-1:2], dataset[idx[0:-1:2]])


class CUDACountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i):
        return torch.as_tensor(i, device='cuda')

    def __len__(self):
        return self.n


class CountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i):
        return i

    def __len__(self):
        return self.n


class CountingIterableDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
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


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestStackDataset(TestCase):

    def test_empty(self):
        with self.assertRaisesRegex(ValueError, "At least one dataset should be passed"):
            StackDataset()

    def test_mixed(self):
        with self.assertRaisesRegex(ValueError, "Supported either"):
            StackDataset(TensorDataset(torch.randn(15, 10)), a=TensorDataset(torch.randn(10, 15)))

    def test_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(10, 15)))
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(a=TensorDataset(torch.randn(15, 10)), b=TensorDataset(torch.randn(10, 15)))

    def test_len(self):
        source = StackDataset(TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(15)))
        self.assertEqual(len(source), 15)
        source = StackDataset(TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)
        source = StackDataset(a=TensorDataset(torch.randn(15, 10)), b=TensorDataset(torch.randn(15)))
        self.assertEqual(len(source), 15)
        source = StackDataset(a=TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)

    def test_single(self):
        t = TensorDataset(torch.randn(15, 10))
        source = StackDataset(t)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
        source = StackDataset(a=t)
        for i in range(15):
            self.assertEqual(t[i], source[i]['a'])

    def test_getitem(self):
        t = TensorDataset(torch.randn(15, 10))
        l = TensorDataset(torch.randn(15, 5, 4))
        source = StackDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])
        source = StackDataset(a=t, b=l)
        for i in range(15):
            self.assertEqual(t[i], source[i]['a'])
            self.assertEqual(l[i], source[i]['b'])

    def test_getitems(self):
        class GetItemsDataset(Dataset):
            def __init__(self):
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items]

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i][0])
            self.assertEqual(l[i], batch[i][1])

        source = StackDataset(t=t, l=l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i]['t'])
            self.assertEqual(l[i], batch[i]['l'])

    def test_getitems_raises_index_error(self):
        class GetItemsDataset(Dataset):
            def __init__(self):
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items]

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaises(IndexError):
            source.__getitems__([0, 4])

    def test_getitems_value_error(self):
        class GetItemsDataset(Dataset):
            def __init__(self):
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items][:-1]  # return less

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaisesRegex(ValueError,
                                    "Nested dataset's output size mismatch. Expected 4, got 3"):
            source.__getitems__([0, 1, 2, 3])


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
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

    def test_iterable_dataset_err(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        it1 = CountingIterableDataset(5)
        it2 = CountingIterableDataset(10)

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([d1, it2, it1])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it2])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it1, d1])


# takes in dummy var so this can also be used as a `worker_init_fn`
def set_faulthander_if_available(_=None):
    faulthandler.enable(sys.__stderr__)
    if not IS_WINDOWS:
        # windows does not have faulthandler.register
        # chain=False prevents the default behavior of killing the process
        faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, chain=False)


set_faulthander_if_available()

# Process `pid` must have called `set_faulthander_if_available`
def print_traces_of_all_threads(pid):
    if not IS_WINDOWS:
        # use the custom signal if available
        os.kill(pid, signal.SIGUSR1)
    else:
        # otherwise we can still use the handler given by faulthandler.enable()
        # at the cost of killing the process.
        os.kill(pid, signal.SIGSEGV)

    # wait in parent process to give subprocess some time to print
    time.sleep(5)


# The following `ErrorTrackingProcess` stores the first encountered exception in
# its `.exception` attribute.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):

    # Why no *args?
    #   py2 doesn't support def fn(x, *args, key=val, **kwargs)
    # Setting disable_stderr=True may generate a lot of unrelated error outputs
    # but could be helpful for debugging.
    def __init__(self, disable_stderr=True, **kwargs):
        super().__init__(**kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.disable_stderr = disable_stderr

    def run(self):
        set_faulthander_if_available()
        if self.disable_stderr:
            # Disable polluting stderr with errors that are supposed to happen.
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            super().run()
            self._cconn.send(None)
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    def print_traces_of_all_threads(self):
        assert self.is_alive(), "can only use print_traces_of_all_threads if the process is alive"
        assert not self.disable_stderr, "do not disable stderr if you use print_traces_of_all_threads"
        # On platforms without `SIGUSR1`, `set_faulthander_if_available` sets
        # `faulthandler.enable()`, and `print_traces_of_all_threads` may kill
        # the process. So let's poll the exception first
        _ = self.exception
        print_traces_of_all_threads(self.pid)

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


class WorkerSpecificIterableDataset(IterableDataset):
    def __init__(self, sizes_for_all_workers):
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        return iter(range(self.sizes_for_all_workers[worker_info.id]))

    def __len__(self):
        return sum(self.sizes_for_all_workers)


# Inspired by https://stackoverflow.com/a/26703365
# If all workers will call `sync_once`, they will be blocked until all workers
# reach the call (i.e., acting like a barrier).
# This can be used to ensure that each worker at least processes one data.
class SynchronizedDataset(Dataset):

    def __init__(self, size, batch_size, num_workers):
        assert size >= num_workers * batch_size
        self.count = mp.Value('i', 0, lock=True)
        self.barrier = mp.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def sync_once(self):
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.size


class EmptyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, any):
        return torch.empty(0)


class SynchronizedSeedDataset(SynchronizedDataset):
    def __getitem__(self, idx):
        self.sync_once()
        return torch.initial_seed()


def _test_timeout(persistent_workers):
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1,
                            persistent_workers=persistent_workers)
    _ = next(iter(dataloader))


def _test_timeout_pin_memory(persistent_workers):
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, timeout=1, pin_memory=True,
                            persistent_workers=persistent_workers)
    _ = next(iter(dataloader))


def _test_large_sampler_indices(persistent_workers):
    # See
    #   test_large_sampler_indices
    #   https://github.com/pytorch/pytorch/issues/48666

    dataloader = torch.utils.data.DataLoader(
        EmptyTensorDataset(10000000),
        batch_size=40960,
        persistent_workers=persistent_workers,
        num_workers=1)

    it = iter(dataloader)

    for x in it:
        assert x.numel() == 0
        raise RuntimeError('My Error')


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
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())


def _test_segfault():
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=disable_stderr)
    _ = next(iter(dataloader))


def _test_no_segfault():
    dataset = [1, 2, 3]
    num_threads = torch.get_num_threads()
    if num_threads < 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(num_threads)
    mp_ctx = torch.multiprocessing.get_context(method='fork')
    dataloader = DataLoader(dataset, num_workers=1, worker_init_fn=disable_stderr,
                            multiprocessing_context=mp_ctx)
    _ = next(iter(dataloader))


class TestProperExitDataset(Dataset):
    def __init__(self, size, error_event):
        self.size = size
        self.error_event = error_event

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if self.error_event is not None and self.error_event.is_set() and \
                worker_info.id == worker_info.num_workers - 1:
            # only error in the last worker
            raise RuntimeError('Worker error')
        return torch.tensor([idx])


class TestProperExitIterableDataset(IterableDataset):
    def __init__(self, size, error_event):
        self.error_event = error_event
        self.size = size
        self.remaining = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.error_event is not None and self.error_event.is_set() and \
                worker_info.id == worker_info.num_workers - 1:
            # only error in the last worker
            raise RuntimeError('Worker error')
        self.remaining -= 1
        if self.remaining < 0:
            raise StopIteration
        return torch.tensor(-1000)


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(is_iterable_dataset, use_workers, pin_memory, exit_method,
                      hold_iter_reference, loader_setup_event, tester_setup_event,
                      persistent_workers):
    num_workers = 2 if use_workers else 0

    if exit_method == 'worker_error' or exit_method == 'worker_kill':
        assert use_workers is True

    if exit_method == 'worker_error':
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    if is_iterable_dataset:
        ds = TestProperExitIterableDataset(7, worker_error_event)
    else:
        ds = TestProperExitDataset(12, worker_error_event)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        worker_init_fn=set_faulthander_if_available,
                        persistent_workers=persistent_workers)

    error_it = 2

    if use_workers:
        # 2 is the magical per-worker prefetch number...
        # FIXME: change this after the number becomes configurable.
        if is_iterable_dataset:
            assert len(ds) * num_workers > (error_it + 2 + 1)
        else:
            assert len(loader) > (error_it + 2 + 1) * num_workers
    else:
        if is_iterable_dataset:
            assert len(ds) > error_it + 1
        else:
            assert len(loader) > error_it + 1

    it = iter(loader)
    if use_workers:
        workers = it._workers

    def kill_pid(pid):
        psutil_p = psutil.Process(pid)
        psutil_p.kill()
        psutil_p.wait(JOIN_TIMEOUT)
        assert not psutil_p.is_running()

    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                del it
                del loader
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
                kill_pid(workers[-1].pid)  # kill last worker

    if not hold_iter_reference:
        # Tries to trigger the __del__ clean-up rather than the automatic
        # exiting of daemonic children. Technically it should be automatically
        # triggered, but I don't want to rely on the implementation detail of
        # Python gc.
        gc.collect()


class TestWorkerInfoDataset(SynchronizedDataset):
    def __getitem__(self, idx):
        self.sync_once()
        return torch.tensor(self.value)


# Should be used as worker_init_fn with TestWorkerInfoDataset.
# See _test_get_worker_info below for usage.
def _test_worker_info_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_id == worker_info.id, "worker_init_fn and worker_info should have consistent id"
    assert worker_id < worker_info.num_workers, "worker_init_fn and worker_info should have valid id"
    assert worker_info.seed == torch.initial_seed(), "worker_init_fn and worker_info should have consistent seed"
    dataset = worker_info.dataset
    assert isinstance(dataset, TestWorkerInfoDataset), "worker_info should have correct dataset copy"
    assert not hasattr(dataset, 'value'), "worker_info should have correct dataset copy"
    # test that WorkerInfo attributes are read-only
    try:
        worker_info.id = 3999
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"
    try:
        worker_info.a = 3
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"
    for k in ['id', 'num_workers', 'seed', 'dataset']:
        assert f"{k}=" in repr(worker_info)
    dataset.value = [worker_id, os.getpid()]


def _test_get_worker_info():
    # get_worker_info returns None in main proc
    assert torch.utils.data.get_worker_info() is None
    num_workers = 2
    batch_size = 2
    dataset = TestWorkerInfoDataset(6, batch_size, num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=_test_worker_info_init_fn)
    it = iter(dataloader)
    data = []
    for d in it:
        data.append(d)  # noqa: PERF402
    worker_pids = [w.pid for w in it._workers]
    data = torch.cat(data, 0)
    for d in data:
        # each `d` is a [worker_id, worker_pid] pair, which is set in
        # _test_worker_info_init_fn
        assert d[1] == worker_pids[d[0]]
    # get_worker_info returns None in main proc after data loading
    assert torch.utils.data.get_worker_info() is None
    # main proc dataset was never assigned this attribute
    assert not hasattr(dataset, 'value')
    try:
        _ = dataset[0]
    except AttributeError:
        return
    raise RuntimeError('Expected AttributeError')


# test custom init function
def init_fn(worker_id):
    torch.manual_seed(12345)


# used with test_error_in_init
class ErrorIterableDataset(IterableDataset):
    def __iter__(self):
        raise RuntimeError("Error in __iter__")


# used with test_error_in_init
def error_worker_init_fn(_):
    raise RuntimeError("Error in worker_init_fn")


class BulkLoadingDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, indices):
        assert isinstance(indices, (list, tuple))
        return torch.as_tensor(indices)

    def __len__(self):
        return self.length


class BulkLoadingSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for x in torch.randperm(len(self.dataset)).split(self.batch_size):
            yield x.tolist()

    def __len__(self):
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))


class TestMultiEpochDataset(IterableDataset):
    def __init__(self, length):
        self.length = length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        worker_id = worker_info.id
        for idx in range(self.length // worker_info.num_workers):
            yield worker_id

    def __len__(self):
        return self.length


class CustomList(list):
    pass


class CustomDict(dict):
    pass


def row_processor(row):
    return np.add(row, 1)


def filter_len(row):
    return len(row) == 4


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223")
class TestDataLoader(TestCase):

    def setUp(self):
        super().setUp()
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)
        self.persistent_workers = False

    def _get_data_loader(self, dataset, **kwargs):
        persistent_workers = kwargs.get('persistent_workers', self.persistent_workers)
        if persistent_workers and kwargs.get('num_workers', 0) == 0:
            persistent_workers = False
        kwargs['persistent_workers'] = persistent_workers
        return DataLoader(dataset, **kwargs)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        if batch_size is None:
            for idx, (sample, target) in enumerate(loader):
                self.assertEqual(sample, self.data[idx])
                self.assertEqual(target, self.labels[idx])
            self.assertEqual(idx, len(self.dataset) - 1)
        else:
            for i, (sample, target) in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(sample, self.data[idx:idx + batch_size])
                self.assertEqual(target, self.labels[idx:idx + batch_size])
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = dict.fromkeys(range(self.data.size(0)), 0)
        found_labels = dict.fromkeys(range(self.labels.size(0)), 0)
        batch_size = loader.batch_size
        if batch_size is None:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                sample, target = (batch_samples, batch_targets)
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
                self.assertEqual(sum(found_data.values()), (i + 1))
                self.assertEqual(sum(found_labels.values()), (i + 1))
            self.assertEqual(i, (len(self.dataset) - 1))
        else:
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

    def test_error_in_init(self):
        for num_workers in [0, 2]:
            loader = self._get_data_loader(ErrorIterableDataset(), num_workers=num_workers)
            with self.assertRaisesRegex(RuntimeError, 'Error in __iter__'):
                list(iter(loader))

        loader = self._get_data_loader(self.dataset, num_workers=2, worker_init_fn=error_worker_init_fn)
        with self.assertRaisesRegex(RuntimeError, 'Error in worker_init_fn'):
            list(iter(loader))

    def test_typing(self):
        from typing import List
        # Make sure there is no TypeError

        class SomeDatasetClass(Dataset[List[torch.Tensor]]):
            pass

        def _create_dataloader(is_train: bool) -> DataLoader[List[torch.Tensor]]:
            pass

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # See NOTE [ DataLoader on Linux and open files limit ]
        import subprocess
        subprocess.check_output([sys.executable, '-c', """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
"""])

    def test_invalid_assign_after_init(self):
        dl = self._get_data_loader(self.dataset)
        for attr in ('batch_size', 'sampler', 'batch_sampler', 'drop_last', 'dataset'):
            def fn():
                setattr(dl, attr, {})

            self.assertRaises(ValueError, fn)

    def test_sequential_nonbatch(self):
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=None))

    def test_sequential_batch(self):
        self._test_sequential(self._get_data_loader(self.dataset))
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=2))

    def test_bulk_loading_nobatch(self):
        n = 35
        bs = 4
        ds = BulkLoadingDataset(n)
        sampler = BulkLoadingSampler(ds, batch_size=4)

        for num_workers in [0, 4]:
            dl = self._get_data_loader(ds, num_workers=num_workers, batch_size=None, sampler=sampler, pin_memory=TEST_CUDA)
            self.assertFalse(dl._auto_collation)
            samples = list(dl)
            self.assertEqual(samples[0].is_pinned(), TEST_CUDA)
            self.assertEqual(set(torch.cat(samples, 0).tolist()), set(range(n)))

    def test_growing_dataset(self):
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = self._get_data_loader(dataset, shuffle=False)
        dataloader_shuffle = self._get_data_loader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_sequential_pin_memory(self):
        loader = self._get_data_loader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(IS_JETSON, "Not working on Jetson")
    def test_multiple_dataloaders(self):
        for multiprocessing_context in supported_multiprocessing_contexts:
            loader1_it = iter(self._get_data_loader(self.dataset, num_workers=1))
            loader2_it = iter(self._get_data_loader(self.dataset, num_workers=2, multiprocessing_context=multiprocessing_context))
            next(loader1_it)
            next(loader1_it)
            next(loader2_it)
            next(loader2_it)
            next(loader1_it)
            next(loader2_it)
            del loader1_it
            del loader2_it

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

    # Tests if the child process forked by the DataLoader segfaults due to having more than 3 threads
    # in the parent process after at least one set_num_threads invocation in the parent process.
    # After forking, set_num_threads(1) in the child process entails handling some inherited data-structures
    # of the Caffe2 thread-pool of the parent process, culminating in a segfault.
    # Reference: https://github.com/pytorch/pytorch/issues/54752
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_no_segfault(self):
        p = ErrorTrackingProcess(target=_test_no_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            if p.exception:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(str(p.exception), r'DataLoader worker \(pid \d+\) is killed by signal: ')
                self.fail("Segfault occurred in worker process after fork")
        finally:
            p.terminate()

    def test_timeout(self):
        if TEST_CUDA and not NO_MULTIPROCESSING_SPAWN:
            # This test runs in a subprocess, which can only initialize CUDA with spawn.
            # _test_timeout_pin_memory with pin_memory=True initializes CUDA when the iterator is
            # constructed.
            targets = (_test_timeout, _test_timeout_pin_memory)
        else:
            targets = (_test_timeout,)
        for target in targets:
            p = ErrorTrackingProcess(target=target, args=(self.persistent_workers,))
            p.start()
            p.join(JOIN_TIMEOUT)
            try:
                self.assertFalse(p.is_alive())
                self.assertNotEqual(p.exitcode, 0)
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(str(p.exception), r'DataLoader timed out after \d+ seconds')
            finally:
                p.terminate()

    def test_large_sampler_indices(self):
        # Test that the data loader cleanly exit when the process errors
        #   1. having an reference to the iterator
        #   2. using a sampler that yields big elements s.t. _index_queues putters block
        #
        # More context: https://github.com/pytorch/pytorch/issues/48666

        p = ErrorTrackingProcess(target=_test_large_sampler_indices, args=(self.persistent_workers,))
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r'My Error')
        finally:
            p.terminate()

    def test_invalid_ctor_args_combinations(self):
        # general
        with self.assertRaisesRegex(ValueError, "num_workers option should be non-negative"):
            self._get_data_loader(self.dataset, num_workers=-1)
        with self.assertRaisesRegex(ValueError, "timeout option should be non-negative"):
            self._get_data_loader(self.dataset, timeout=-1)

        # disable auto-batching
        with self.assertRaisesRegex(ValueError,
                                    "batch_size=None option disables auto-batching and is mutually exclusive"):
            self._get_data_loader(self.dataset, batch_size=None, drop_last=True)

        valid_ctx = list(torch.multiprocessing.get_all_start_methods())[-1]
        with self.assertRaisesRegex(ValueError, r"multi-process loading \(num_workers > 0\), but got"):
            self._get_data_loader(self.dataset, num_workers=0, multiprocessing_context=valid_ctx)
        with self.assertRaisesRegex(ValueError, "should specify a valid start method in"):
            self._get_data_loader(self.dataset, num_workers=1, multiprocessing_context='bad')
        with self.assertRaisesRegex(TypeError, "multiprocessing_context option should be a valid context "):
            self._get_data_loader(self.dataset, num_workers=1, multiprocessing_context=object())

        # map-style
        sampler = torch.utils.data.SequentialSampler(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 3, False)
        with self.assertRaisesRegex(ValueError, "sampler option is mutually exclusive with shuffle"):
            self._get_data_loader(self.dataset, batch_size=11, sampler=sampler, shuffle=True)
        with self.assertRaisesRegex(ValueError, "sampler option is mutually exclusive with shuffle"):
            self._get_data_loader(self.dataset, batch_sampler=batch_sampler, sampler=sampler, shuffle=True)
        with self.assertRaisesRegex(ValueError, "sampler option is mutually exclusive with shuffle"):
            self._get_data_loader(self.dataset, batch_sampler=batch_sampler, sampler=sampler, shuffle=3)
        with self.assertRaisesRegex(ValueError, "batch_sampler option is mutually exclusive with"):
            self._get_data_loader(self.dataset, batch_size=11, batch_sampler=batch_sampler)
        with self.assertRaisesRegex(ValueError, "batch_sampler option is mutually exclusive with"):
            self._get_data_loader(self.dataset, shuffle=True, batch_sampler=batch_sampler)
        with self.assertRaisesRegex(ValueError, "batch_sampler option is mutually exclusive with"):
            self._get_data_loader(self.dataset, drop_last=True, batch_sampler=batch_sampler)
        with self.assertRaisesRegex(ValueError, "batch_sampler option is mutually exclusive with"):
            self._get_data_loader(self.dataset, drop_last=3, batch_sampler=batch_sampler)

        # iterable-style
        dataset = CountingIterableDataset(20)
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified shuffle"):
            self._get_data_loader(dataset, shuffle=True)
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified shuffle"):
            self._get_data_loader(dataset, shuffle=3)
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified sampler"):
            self._get_data_loader(dataset, sampler=torch.utils.data.SequentialSampler(dataset))
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified sampler"):
            self._get_data_loader(dataset, sampler=3)
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified batch_sampler"):
            self._get_data_loader(dataset, batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset), 3, False))
        with self.assertRaisesRegex(ValueError, "DataLoader with IterableDataset: expected unspecified batch_sampler"):
            self._get_data_loader(dataset, batch_sampler=3)

    def test_builtin_collection_conversion(self):
        for coll_ty in (list, tuple):
            for num_workers in (0, 1):
                # map-style dataset
                dataset = CountingDataset(20)
                # no auto-batching
                fetched = coll_ty(self._get_data_loader(dataset, batch_size=None, num_workers=num_workers))
                self.assertEqual(fetched, coll_ty(range(20)))
                # auto-batching
                fetched = coll_ty(self._get_data_loader(dataset, batch_size=2, num_workers=num_workers))
                self.assertEqual(fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2)))

                # iterable-style dataset
                dataset = CountingIterableDataset(20)
                # no auto-batching
                fetched = coll_ty(self._get_data_loader(dataset, batch_size=None, num_workers=num_workers))
                self.assertEqual(fetched, coll_ty(range(20)))
                # auto-batching
                # this IterableDataset isn't configured for each worker, so for
                # the equality test below to be valid, we cannot have more than 1 workers.
                assert num_workers in [0, 1], "invalid test"
                fetched = coll_ty(self._get_data_loader(dataset, batch_size=2, num_workers=num_workers))
                self.assertEqual(fetched, coll_ty(torch.tensor([i, i + 1]) for i in range(0, 20, 2)))

    def test_iterable_style_dataset(self):
        # [no auto-batching] single process loading
        dataset = CountingIterableDataset(20)
        dataloader = self._get_data_loader(dataset, batch_size=None)
        fetched = list(dataloader)
        self.assertEqual(len(fetched), 20)
        for i, d in enumerate(fetched):
            # non-batched should not convert ints into tensors
            self.assertIsInstance(d, int)
            self.assertEqual(d, i)
        # DataLoader should match len of the iterable-style dataset (if implemented)
        self.assertEqual(len(dataloader), len(dataset))

        # [no auto-batching] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(functools.reduce(operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []))
        assert len(sizes_for_all_workers) == num_workers, 'invalid test case'
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            dataloader = self._get_data_loader(dataset, num_workers=num_workers, batch_size=None,
                                               worker_init_fn=set_faulthander_if_available,
                                               prefetch_factor=prefetch_factor)
            dataloader_iter = iter(dataloader)
            fetched = sorted(dataloader_iter)
            for a, b in zip(fetched, expected):
                # non-batched should not convert ints into tensors
                self.assertIsInstance(a, int)
                self.assertEqual(a, b)
            # DataLoader should match len of the iterable-style dataset (if implemented)
            self.assertEqual(len(dataloader), len(dataset))
            # When loading more than len(dataset) data, after accessing len(dataloader),
            # we should get a warning. See NOTE [ IterableDataset and __len__ ].
            dataset = CountingIterableDataset(20)
            dataloader = self._get_data_loader(dataset, num_workers=num_workers,
                                               worker_init_fn=set_faulthander_if_available,
                                               prefetch_factor=prefetch_factor)
            it = iter(dataloader)
            for _ in range(40):
                self.assertNotWarn(lambda: next(it), "Should not warn before accessing len(dataloader)")
            self.assertEqual(len(dataloader), len(dataset))
            self.assertEqual(len(dataloader), 20)
            it = iter(dataloader)
            for _ in range(20):
                self.assertNotWarn(lambda: next(it), "Should not warn before exceeding length")
            for _ in range(3):
                with self.assertWarnsRegex(
                    UserWarning,
                    r"but [0-9]+ samples have been fetched\. For multiprocessing data-loading, this",
                        msg="Should always warn after exceeding length"):
                    next(it)
        # [no auto-batching] test that workers exit gracefully
        workers = dataloader_iter._workers
        del dataloader_iter
        del dataloader
        try:
            for w in workers:
                w.join(JOIN_TIMEOUT)
                self.assertFalse(w.is_alive())
                self.assertEqual(w.exitcode, 0)
        finally:
            for w in workers:
                w.terminate()

        # [auto-batching] single process loading
        dataset = CountingIterableDataset(20)
        fetched = list(self._get_data_loader(dataset, batch_size=7))
        self.assertEqual(len(fetched), 3)
        self.assertEqual(fetched[0].tolist(), list(range(7)))
        self.assertEqual(fetched[1].tolist(), list(range(7, 14)))
        self.assertEqual(fetched[2].tolist(), list(range(14, 20)))

        # [auto-batching] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(functools.reduce(operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []))
        assert len(sizes_for_all_workers) == num_workers, 'invalid test case'
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            # worker 0 should return 0 batches
            # worker 1 should return 1 batches
            # worker 2 should return 3 batches
            dataloader = self._get_data_loader(dataset, num_workers=num_workers, batch_size=7, prefetch_factor=prefetch_factor)
            dataloader_iter = iter(dataloader)
            fetched = list(dataloader_iter)
            self.assertEqual(len(fetched), 4)
            fetched = {tuple(t.tolist()) for t in fetched}
            self.assertEqual(fetched, {tuple(range(4)), tuple(range(7)), tuple(range(7, 14)), tuple(range(14, 20))})

            # [auto-batching] test that workers exit gracefully
            workers = dataloader_iter._workers
            del dataloader_iter
            del dataloader
            try:
                for w in workers:
                    w.join(JOIN_TIMEOUT)
                    self.assertFalse(w.is_alive())
                    self.assertEqual(w.exitcode, 0)
            finally:
                for w in workers:
                    w.terminate()
        # [auto-batching & drop_last] single process loading
        dataset = CountingIterableDataset(20)
        fetched = list(self._get_data_loader(dataset, batch_size=7, drop_last=True))
        self.assertEqual(len(fetched), 2)
        self.assertEqual(fetched[0].tolist(), list(range(7)))
        self.assertEqual(fetched[1].tolist(), list(range(7, 14)))

        # [auto-batching & drop_last] multiprocessing loading
        num_workers = 3
        sizes_for_all_workers = [0, 4, 20]
        expected = sorted(functools.reduce(operator.iadd, (list(range(s)) for s in sizes_for_all_workers), []))
        assert len(sizes_for_all_workers) == num_workers, 'invalid test case'
        for prefetch_factor in [2, 3, 4]:
            dataset = WorkerSpecificIterableDataset(sizes_for_all_workers)
            # worker 0 should return 0 batches
            # worker 1 should return 1 batches
            # worker 2 should return 3 batches
            dataloader = self._get_data_loader(dataset, num_workers=num_workers, batch_size=7, drop_last=True,
                                               worker_init_fn=set_faulthander_if_available,
                                               prefetch_factor=prefetch_factor)
            dataloader_iter = iter(dataloader)
            fetched = list(dataloader_iter)
            self.assertEqual(len(fetched), 2)
            fetched = {tuple(t.tolist()) for t in fetched}
            self.assertEqual(fetched, {tuple(range(7)), tuple(range(7, 14))})

            # [auto-batching & drop_last] test that workers exit gracefully
            workers = dataloader_iter._workers
            del dataloader_iter
            del dataloader
            try:
                for w in workers:
                    w.join(JOIN_TIMEOUT)
                    self.assertFalse(w.is_alive())
                    self.assertEqual(w.exitcode, 0)
            finally:
                for w in workers:
                    w.terminate()

    def test_chain_iterable_style_dataset(self):
        # chaining (concatenation)
        dataset1 = CountingIterableDataset(20)
        dataset2 = CountingIterableDataset(15)
        expected = list(range(20)) + list(range(15))
        for num_workers in [0, 1]:
            for chained_dataset in [dataset1 + dataset2, ChainDataset([dataset1, dataset2])]:
                fetched = list(self._get_data_loader(chained_dataset, num_workers=num_workers))
                self.assertEqual(len(fetched), len(expected))
                for e, d in zip(expected, fetched):
                    self.assertIsInstance(d, torch.Tensor)
                    self.assertEqual(e, d)

        with self.assertRaisesRegex(AssertionError, "ChainDataset only supports IterableDataset"):
            list(iter(dataset1 + self.dataset))

        with self.assertRaisesRegex(AssertionError, "ChainDataset only supports IterableDataset"):
            list(iter(ChainDataset([dataset1, self.dataset])))

    @unittest.skipIf(IS_MACOS, "Not working on macos")
    @unittest.skipIf(IS_MACOS or IS_JETSON, "Not working on macos or Jetson")
    @skipIfRocm  # https://github.com/pytorch/pytorch/issues/90940
    def test_multiprocessing_contexts(self):
        reference = [
            torch.arange(3),
            torch.arange(3, 6),
            torch.arange(6, 9),
            torch.arange(9, 11),
        ]
        counting_ds_n = 11
        dl_common_args = dict(num_workers=3, batch_size=3, pin_memory=(not TEST_CUDA))
        for ctx in supported_multiprocessing_contexts:
            # windows and jetson devices don't support sharing cuda tensor; ROCm does not yet fully support IPC
            if ctx in ['spawn', 'forkserver'] and TEST_CUDA and not IS_WINDOWS and not IS_JETSON:
                ds_cls = CUDACountingDataset
            else:
                ds_cls = CountingDataset
            self.assertEqual(
                reference, list(self._get_data_loader(ds_cls(counting_ds_n), multiprocessing_context=ctx, **dl_common_args)))
            if ctx is not None:
                # test ctx object
                ctx = mp.get_context(ctx)
                self.assertEqual(
                    reference, list(self._get_data_loader(ds_cls(counting_ds_n), multiprocessing_context=ctx, **dl_common_args)))

    def _test_multiprocessing_iterdatapipe(self, with_dill):
        # Testing to make sure that function from global scope (e.g. imported from library) can be serialized
        # and used with multiprocess DataLoader

        reference = [torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),
                     torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64)]
        datapipe: IterDataPipe = IterableWrapper([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
        datapipe = datapipe.map(row_processor)
        datapipe = datapipe.filter(lambda row: len(row) == 4) if with_dill else datapipe.filter(filter_len)

        dl_common_args = dict(num_workers=2, batch_size=2, shuffle=True, pin_memory=(not TEST_CUDA))
        for ctx in supported_multiprocessing_contexts:
            self.assertEqual(reference,
                             [t.type(torch.int64)
                              for t in self._get_data_loader(datapipe, multiprocessing_context=ctx, **dl_common_args)])
            if ctx is not None:
                # test ctx object
                ctx = mp.get_context(ctx)
                self.assertEqual(reference,
                                 [t.type(torch.int64)
                                  for t in
                                  self._get_data_loader(datapipe, multiprocessing_context=ctx, **dl_common_args)])

    @skipIfNoNumpy
    @unittest.skipIf(IS_JETSON, "Not working on Jetson")
    def test_multiprocessing_iterdatapipe(self):
        self._test_multiprocessing_iterdatapipe(with_dill=False)

    @unittest.expectedFailure
    @skipIfNoNumpy
    @unittest.skipIf(IS_JETSON, "Not working on Jetson")
    @skipIfNoDill
    def test_multiprocessing_iterdatapipe_with_dill(self):
        self._test_multiprocessing_iterdatapipe(with_dill=True)

    def test_worker_seed(self):
        num_workers = 6
        batch_size = 1
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)
        dataloader = self._get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        seeds = set()
        seeds.update(batch[0] for batch in dataloader)
        self.assertEqual(len(seeds), num_workers)

    def test_worker_seed_reproducibility(self):
        def get_dataloader():
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, generator=torch.Generator().manual_seed(42))

        num_workers = 6
        batch_size = 1
        dataset = SynchronizedSeedDataset(num_workers, batch_size, num_workers)
        self.assertEqual({int(batch) for batch in get_dataloader()}, {int(batch) for batch in get_dataloader()})

    def test_multi_epochs_reproducibility(self):
        num_workers = 2
        batch_size = 10
        num_epochs = 3

        dataset = TestMultiEpochDataset(batch_size * num_workers)
        dataloader = self._get_data_loader(dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)

        for ind in range(num_epochs):
            for batch_idx, sample in enumerate(dataloader):
                self.assertEqual(sample.tolist(), [batch_idx % num_workers] * batch_size)

    def test_worker_init_fn(self):
        dataset = SeedDataset(4)
        dataloader = self._get_data_loader(dataset, batch_size=2, num_workers=2,
                                           worker_init_fn=init_fn)
        for batch in dataloader:
            self.assertEqual(12345, batch[0])
            self.assertEqual(12345, batch[1])

    def test_get_worker_info(self):
        p = ErrorTrackingProcess(target=_test_get_worker_info)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertEqual(p.exitcode, 0)
        finally:
            p.terminate()

    def test_shuffle(self):
        self._test_shuffle(self._get_data_loader(self.dataset, shuffle=True))

    def test_shuffle_batch_none(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=None, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(self._get_data_loader(self.dataset, batch_size=2, shuffle=True))

    def test_shuffle_reproducibility(self):
        for fn in (
            lambda: DataLoader(self.dataset, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42)),
            lambda: DataLoader(self.dataset, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(42)),
        ):
            self.assertEqual(list(fn()), list(fn()))

    def test_sequential_workers(self):
        self._test_sequential(self._get_data_loader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=2, num_workers=4))

    def test_seqential_batch_workers_prefetch(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2, num_workers=4, prefetch_factor=3))

    def test_shuffle_workers(self):
        self._test_shuffle(self._get_data_loader(self.dataset, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers(self):
        self._test_shuffle(self._get_data_loader(self.dataset, batch_size=2, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers_prefetch(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, prefetch_factor=3))

    def test_random_sampler(self):

        from collections import Counter
        from torch.utils.data import RandomSampler

        def sample_stat(sampler, num_samples):
            counts = Counter(sampler)
            count_repeated = sum(val > 1 for val in counts.values())
            return (count_repeated, min(counts.keys()), max(counts.keys()), sum(counts.values()))

        # test sample with replacement
        n = len(self.dataset) + 1  # ensure at least one sample is drawn more than once
        sampler_with_replacement = RandomSampler(self.dataset, replacement=True, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(sampler_with_replacement, n)
        self.assertTrue(count_repeated > 0)
        self.assertTrue(minval >= 0)
        self.assertTrue(maxval < len(self.dataset))
        self.assertTrue(count_total == n)

        # test sample without replacement and without specified num_samples
        sampler_without_replacement = RandomSampler(self.dataset)
        count_repeated, minval, maxval, count_total = sample_stat(sampler_without_replacement, len(self.dataset))
        self.assertTrue(count_repeated == 0)
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == len(self.dataset))

        # test sample without replacement and with specified num_samples
        n = len(self.dataset) * 2
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(sampler_without_replacement, len(self.dataset))
        self.assertTrue(count_repeated == len(self.dataset))
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == n)

        n = len(self.dataset) - 1
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(sampler_without_replacement, len(self.dataset))
        self.assertTrue(count_repeated == 0)
        self.assertTrue(minval >= 0)
        self.assertTrue(maxval < len(self.dataset))
        self.assertTrue(count_total == n)

        n = len(self.dataset) + 1
        sampler_without_replacement = RandomSampler(self.dataset, num_samples=n)
        count_repeated, minval, maxval, count_total = sample_stat(sampler_without_replacement, len(self.dataset))
        self.assertTrue(count_repeated == 1)
        self.assertTrue(minval == 0)
        self.assertTrue(maxval == len(self.dataset) - 1)
        self.assertTrue(count_total == n)

        # raise error when replacement is non-boolean
        with self.assertRaisesRegex(TypeError, "replacement should be a boolean value, but got replacement=0"):
            RandomSampler(self.dataset, replacement=0)

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
        count_num_samples_in_data_loader = len(self._get_data_loader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # test with dataloader, batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(self._get_data_loader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(int(math.ceil(float(num_samples) / batch_size)),
                         count_num_samples_in_data_loader)

    def test_random_sampler_len_without_replacement(self):
        from torch.utils.data import RandomSampler
        # add 5 extra samples
        num_samples = len(self.dataset) + 5
        sampler = RandomSampler(self.dataset,
                                replacement=False,
                                num_samples=num_samples)
        # test len method
        self.assertEqual(num_samples, len(sampler))

        # test with iteration
        count_num_samples = sum(1 for _ in sampler)
        self.assertEqual(num_samples, count_num_samples)

        # test with dataloader, batch_size = 1
        batch_size = 1
        count_num_samples_in_data_loader = len(self._get_data_loader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(num_samples, count_num_samples_in_data_loader)

        # test with dataloader, batch_size = 6
        batch_size = 6
        count_num_samples_in_data_loader = len(self._get_data_loader(
            self.dataset, batch_size=batch_size, sampler=sampler))
        self.assertEqual(num_samples // batch_size + (num_samples % batch_size > 0),
                         count_num_samples_in_data_loader)

    def test_distributed_sampler_invalid_rank(self):
        from torch.utils.data.distributed import DistributedSampler
        dataset = torch.IntTensor(range(10))
        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, 3)

        with self.assertRaisesRegex(ValueError, "Invalid rank"):
            sampler = DistributedSampler(dataset, 3, -1)

    def test_duplicating_data_with_drop_last(self):

        from torch.utils.data.distributed import DistributedSampler

        num_processes = 4
        num_batches = 9
        data_set = torch.IntTensor(range(num_batches))
        scanned_data = torch.IntTensor([])
        for i in range(num_processes):
            s = DistributedSampler(data_set, num_processes, i)
            d_loader = self._get_data_loader(data_set, batch_size=int(num_batches / num_processes), drop_last=True, sampler=s)
            for data in d_loader:
                scanned_data = torch.cat((scanned_data, data), 0)

        self.assertEqual(scanned_data.size(), scanned_data.unique().size())

    def test_sampler_reproducibility(self):
        from torch.utils.data import RandomSampler, WeightedRandomSampler, SubsetRandomSampler

        weights = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
        for fn in (
            lambda: RandomSampler(self.dataset, num_samples=5, replacement=True, generator=torch.Generator().manual_seed(42)),
            lambda: RandomSampler(self.dataset, replacement=False, generator=torch.Generator().manual_seed(42)),
            lambda: WeightedRandomSampler(weights, num_samples=5, replacement=True, generator=torch.Generator().manual_seed(42)),
            lambda: WeightedRandomSampler(weights, num_samples=5, replacement=False, generator=torch.Generator().manual_seed(42)),
            lambda: SubsetRandomSampler(range(10), generator=torch.Generator().manual_seed(42)),
        ):
            self.assertEqual(list(fn()), list(fn()))

        for sampler in (
            RandomSampler(self.dataset, num_samples=5, replacement=True),
            RandomSampler(self.dataset, replacement=False),
            WeightedRandomSampler(weights, num_samples=5, replacement=True),
            WeightedRandomSampler(weights, num_samples=5, replacement=False),
            SubsetRandomSampler(range(10)),
        ):
            torch.manual_seed(0)
            l1 = list(sampler) + list(sampler)

            torch.manual_seed(0)
            l2 = list(sampler) + list(sampler)
            self.assertEqual(l1, l2)

            its = (iter(sampler), iter(sampler))
            ls = ([], [])
            for idx in range(len(sampler)):
                for i in range(2):
                    if idx == 0:
                        torch.manual_seed(0)
                    ls[i].append(next(its[i]))
            self.assertEqual(ls[0], ls[1])

    def _test_sampler(self, **kwargs):
        indices = range(2, 12)  # using a regular iterable
        dl = self._get_data_loader(self.dataset, sampler=indices, batch_size=2, **kwargs)
        self.assertEqual(len(dl), 5)
        for i, (input, _target) in enumerate(dl):
            self.assertEqual(len(input), 2)
            self.assertEqual(input, self.data[i * 2 + 2:i * 2 + 4])

    def test_sampler(self):
        self._test_sampler()
        self._test_sampler(num_workers=4)
        if not NO_MULTIPROCESSING_SPAWN:
            self._test_batch_sampler(num_workers=4, multiprocessing_context='spawn')

    def _test_batch_sampler(self, **kwargs):
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []  # using a regular iterable
        for i in range(0, 20, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = self._get_data_loader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 8)
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
        if not NO_MULTIPROCESSING_SPAWN:
            self._test_batch_sampler(num_workers=4, multiprocessing_context='spawn')

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = self._get_data_loader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
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

        loader = self._get_data_loader(TestDataset(), batch_size=12)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.DoubleTensor)
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_gen_state(self):
        from torch.utils.data._utils.worker import _generate_state
        # Using NumPy generated states as the reference to test `_generate_state`
        # having the same result.
        # Test case: ((worker_id, base_seed), expected_state)
        test_cases = [
            ((4, 13434589827475259383), (2884386318, 1088094898, 3523808998, 3860348662)),
            ((1, 15014285634777110771), (1934848465, 763213760, 2959016433, 179751970)),
            ((10, 978296274032934101), (1759791917, 3550927336, 1225977135, 1036538043)),
            ((12, 11868770762134256968), (3974661794, 3331131333, 3630387033, 2885815368)),
            ((9, 15378787925219019706), (3815056996, 3162224466, 2735102421, 3190253477)),
            ((5, 9055612723125076328), (3522565701, 3368424109, 959377806, 621878693)),
            ((15, 14617792358407278405), (3402479508, 1588702753, 1169536393, 3675067356)),
            ((9, 17363320784006640087), (957989458, 2518334477, 1421725660, 3086155459)),
            ((12, 480002904169484764), (2732851467, 1762620729, 4055801988, 1277640511)),
            ((15, 16803975943592702950), (3479415043, 4022359553, 295994005, 3358606349)),
            ((9, 11704776406047813044), (1968928009, 710113752, 2442656196, 1587420279)),
            ((10, 16357891985431864516), (1271733898, 4197047399, 3727213786, 2338547348)),
            ((2, 17423369006318065007), (544294336, 1911284083, 3299147734, 3231058347)),
            ((2, 2889492011444113593), (3721591783, 2595811276, 2212881745, 977682627)),
            ((0, 8979703111668486195), (4276723937, 2556068849, 2962827292, 233130238)),
            ((6, 6269787272229682235), (2548857855, 1216457374, 1012973562, 2999759647))
        ]

        for (worker_id, base_seed), exp in test_cases:
            self.assertEqual(exp, _generate_state(base_seed, worker_id))

    def test_error(self):
        self._test_error(self._get_data_loader(ErrorDataset(100), batch_size=2, shuffle=True))

    def test_error_workers(self):
        self._test_error(self._get_data_loader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(IS_WINDOWS, "FIXME: stuck test")
    def test_partial_workers(self):
        r"""Check that workers exit even if the iterator is not exhausted."""
        if TEST_CUDA:
            pin_memory_configs = (True, False)
        else:
            pin_memory_configs = (False,)

        for pin_memory in pin_memory_configs:
            loader = iter(self._get_data_loader(self.dataset, batch_size=2, num_workers=4, pin_memory=pin_memory))
            workers = loader._workers
            if pin_memory:
                pin_memory_thread = loader._pin_memory_thread
            for i, _ in enumerate(loader):
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

    # Takes 2.5min to finish, see https://github.com/pytorch/pytorch/issues/46065
    @skipIfRocm
    @unittest.skipIf(not HAS_PSUTIL, "psutil not found")
    @slowTest
    def test_proper_exit(self):
        (r'''There might be ConnectionResetError or leaked semaphore warning '''
         r'''(due to dirty process exit), but they are all safe to ignore''')

        # TODO: test the case where the pin_memory_thread triggers an
        #       error/fatal signal. I haven't found out how to properly do that.

        for is_iterable_dataset, use_workers, pin_memory, hold_iter_reference in \
                itertools.product([True, False], repeat=4):

            # `hold_iter_reference` specifies whether we hold a reference to the
            # iterator. This is interesting because Python3 error traces holds a
            # reference to the frames, which hold references to all the local
            # variables including the iterator, and then the iterator dtor may
            # not be called before process end. It is important to see that the
            # processes still exit in both cases.

            if pin_memory and (not TEST_CUDA or NO_MULTIPROCESSING_SPAWN or IS_WINDOWS):
                # This test runs in a subprocess, which can only initialize CUDA with spawn.
                # DataLoader with pin_memory=True initializes CUDA when its iterator is constructed.
                # For windows, pin_memory sometimes causes CUDA oom.
                continue

            # `exit_method` controls the way the loader process ends.
            #   - `*_kill` means that `*` is killed by OS.
            #   - `*_error` means that `*` raises an error.
            #   - `None` means that no error happens.
            # In all cases, all processes should end properly.
            if use_workers:
                # TODO: Fix test for 'loader_kill' that would cause running out of shared memory.
                # Killing loader process would prevent DataLoader iterator clean up all queues
                # and worker processes
                exit_methods = [None, 'loader_error', 'worker_error', 'worker_kill']
                persistent_workers = self.persistent_workers
            else:
                exit_methods = [None, 'loader_error', 'loader_kill']
                persistent_workers = False

            for exit_method in exit_methods:
                if exit_method == 'worker_kill':
                    # FIXME: This sometimes hangs. See #16608.
                    continue

                desc = []
                desc.append(f'is_iterable_dataset={is_iterable_dataset}')
                desc.append(f'use_workers={use_workers}')
                desc.append(f'pin_memory={pin_memory}')
                desc.append(f'hold_iter_reference={hold_iter_reference}')
                desc.append(f'exit_method={exit_method}')
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
                                                args=(is_iterable_dataset, use_workers, pin_memory,
                                                      exit_method, hold_iter_reference,
                                                      loader_setup_event, tester_setup_event,
                                                      persistent_workers),
                                                disable_stderr=False)
                loader_p.start()
                loader_psutil_p = psutil.Process(loader_p.pid)

                # Wait for loader process to set everything up, e.g., starting
                # workers.
                loader_setup_event.wait(timeout=JOIN_TIMEOUT)
                if not loader_setup_event.is_set():
                    fail_msg = desc + ': loader process failed to setup within given time'
                    if loader_p.exception is not None:
                        fail_msg += f', and had exception {loader_p.exception}'
                    elif not loader_p.is_alive():
                        fail_msg += f', and exited with code {loader_p.exitcode} but had no exception'
                    else:
                        fail_msg += ', and is still alive.'
                    if loader_p.is_alive():
                        # this may kill the process, needs to run after the above lines
                        loader_p.print_traces_of_all_threads()
                    self.fail(fail_msg)

                # We are certain that the workers have started now.
                worker_psutil_ps = loader_psutil_p.children()

                def fail(reason):
                    report_psutil_attrs = ['pid', 'name', 'cpu_times', 'io_counters',
                                           'memory_full_info', 'num_ctx_switches',
                                           'open_files', 'threads', 'status',
                                           'nice', 'ionice']
                    if reason is None:
                        err_msg = desc
                    else:
                        err_msg = f'{desc}: {reason}'
                    err_msg += '\nLoader info:\n\t'
                    if loader_psutil_p.is_running():
                        err_msg += str(loader_psutil_p.as_dict(attrs=report_psutil_attrs))
                        # this may kill the process, needs to run after the above line
                        loader_p.print_traces_of_all_threads()
                    else:
                        err_msg += f'exited with code {loader_p.exitcode}'
                    if use_workers:
                        err_msg += '\nWorker(s) info:'
                        for idx, worker_psutil_p in enumerate(worker_psutil_ps):
                            err_msg += f'\n\tWorker {idx}:\n\t\t'
                            if worker_psutil_p.is_running():
                                err_msg += str(worker_psutil_p.as_dict(attrs=report_psutil_attrs))
                                # this may kill the process, needs to run after the above line
                                print_traces_of_all_threads(worker_psutil_p.pid)
                            else:
                                err_msg += 'exited with unknown code'
                    self.fail(err_msg)

                tester_setup_event.set()

                try:
                    loader_p.join(JOIN_TIMEOUT + MP_STATUS_CHECK_INTERVAL)
                    if loader_p.is_alive():
                        fail_reason = 'loader process did not terminate'
                        if loader_p.exception is not None:
                            fail(fail_reason + f', and had exception {loader_p.exception}')
                        else:
                            fail(fail_reason + ', and had no exception')
                    _, alive = psutil.wait_procs(worker_psutil_ps, timeout=(MP_STATUS_CHECK_INTERVAL + JOIN_TIMEOUT))
                    if len(alive) > 0:
                        fail('worker process (pid(s) {}) did not terminate'.format(
                            ', '.join(str(p.pid) for p in alive)))
                    if exit_method is None:
                        if loader_p.exitcode != 0:
                            fail(f'loader process had nonzero exitcode {loader_p.exitcode}')
                    else:
                        if loader_p.exitcode == 0:
                            fail('loader process had zero exitcode')
                        if exit_method == 'loader_error':
                            if not isinstance(loader_p.exception, RuntimeError) or \
                                    'Loader error' not in str(loader_p.exception):
                                fail(f'loader process did not raise expected exception, but had {loader_p.exception}')
                        elif exit_method == 'worker_kill':
                            if isinstance(loader_p.exception, RuntimeError):
                                if 'DataLoader worker (pid' not in str(loader_p.exception):
                                    fail(f'loader process did not raise expected exception, but had {loader_p.exception}')
                            elif isinstance(loader_p.exception, ConnectionRefusedError):
                                # Sometimes, when the worker is being killed and is freeing its
                                # resources, the unpickling in loader process will be met an
                                # a `ConnectionRefusedError` as it can not open a socket to receive
                                # resource. In such cases, the worker may not have fully exited,
                                # and the loader can't know this via `is_alive` check or `SIGCHLD`
                                # handler. So we permit this as an allowed error as well.
                                # After all, we are happy as long as it terminates.
                                pass
                            else:
                                fail(f'loader process did not raise expected exception, but had {loader_p.exception}')
                        elif exit_method == 'worker_error':
                            if not isinstance(loader_p.exception, RuntimeError) or \
                                    'Worker error' not in str(loader_p.exception):
                                fail(f'loader process did not raise expected exception, but had {loader_p.exception}')
                finally:
                    loader_p.terminate()

    def test_len(self):
        def check_len(dl, expected):
            self.assertEqual(len(dl), expected)
            n = 0
            for _ in dl:
                n += 1
            self.assertEqual(n, expected)
        check_len(self.dataset, 100)
        check_len(self._get_data_loader(self.dataset, batch_size=2), 50)
        check_len(self._get_data_loader(self.dataset, batch_size=3), 34)

    def test_iterabledataset_len(self):
        class IterableDataset(torch.utils.data.IterableDataset):
            def __len__(self):
                return 10

            def __iter__(self):
                return iter(range(10))

        iterable_loader = DataLoader(IterableDataset(), batch_size=1)
        self.assertEqual(len(iterable_loader), 10)
        iterable_loader = DataLoader(IterableDataset(), batch_size=1, drop_last=True)
        self.assertEqual(len(iterable_loader), 10)

        iterable_loader = DataLoader(IterableDataset(), batch_size=2)
        self.assertEqual(len(iterable_loader), 5)
        iterable_loader = DataLoader(IterableDataset(), batch_size=2, drop_last=True)
        self.assertEqual(len(iterable_loader), 5)

        iterable_loader = DataLoader(IterableDataset(), batch_size=3)
        self.assertEqual(len(iterable_loader), 4)
        iterable_loader = DataLoader(IterableDataset(), batch_size=3, drop_last=True)
        self.assertEqual(len(iterable_loader), 3)

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
            loader = self._get_data_loader(dset, batch_size=2)
            batch = next(iter(loader))
            self.assertIsInstance(batch, tt)

    def test_default_convert_mapping_keep_type(self):
        data = CustomDict({"a": 1, "b": 2})
        converted = _utils.collate.default_convert(data)

        self.assertEqual(converted, data)

    def test_default_convert_sequence_keep_type(self):
        data = CustomList([1, 2, 3])
        converted = _utils.collate.default_convert(data)

        self.assertEqual(converted, data)

    def test_default_convert_sequence_dont_keep_type(self):
        data = range(2)
        converted = _utils.collate.default_convert(data)

        self.assertEqual(converted, [0, 1])

    def test_default_collate_dtype(self):
        arr = [1, 2, -1]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.int64)

        arr = [1.1, 2.3, -0.9]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr, dtype=torch.float64))

        arr = [True, False]
        collated = _utils.collate.default_collate(arr)
        self.assertEqual(collated, torch.tensor(arr))
        self.assertEqual(collated.dtype, torch.bool)

        # Should be a no-op
        arr = ['a', 'b', 'c']
        self.assertEqual(arr, _utils.collate.default_collate(arr))

    def test_default_collate_mapping_keep_type(self):
        batch = [CustomDict({"a": 1, "b": 2}), CustomDict({"a": 3, "b": 4})]
        collated = _utils.collate.default_collate(batch)

        expected = CustomDict({"a": torch.tensor([1, 3]), "b": torch.tensor([2, 4])})
        self.assertEqual(collated, expected)

    def test_default_collate_sequence_keep_type(self):
        batch = [CustomList([1, 2, 3]), CustomList([4, 5, 6])]
        collated = _utils.collate.default_collate(batch)

        expected = CustomList([
            torch.tensor([1, 4]),
            torch.tensor([2, 5]),
            torch.tensor([3, 6]),
        ])
        self.assertEqual(collated, expected)

    def test_default_collate_sequence_dont_keep_type(self):
        batch = [range(2), range(2)]
        collated = _utils.collate.default_collate(batch)

        self.assertEqual(collated, [torch.tensor([0, 0]), torch.tensor([1, 1])])

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
    def test_default_collate_numpy_memmap(self):
        import numpy as np

        with tempfile.TemporaryFile() as f:
            arr = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
            arr_memmap = np.memmap(f, dtype=arr.dtype, mode='w+', shape=arr.shape)
            arr_memmap[:] = arr[:]
            arr_new = np.memmap(f, dtype=arr.dtype, mode='r', shape=arr.shape)
            tensor = _utils.collate.default_collate(list(arr_new))

        self.assertTrue((tensor == tensor.new_tensor([[0, 1], [2, 3], [4, 5], [6, 7]])).all().item())

    def test_default_collate_bad_sequence_type(self):
        batch = [['X'], ['X', 'X']]
        self.assertRaises(RuntimeError, lambda: _utils.collate.default_collate(batch))
        self.assertRaises(RuntimeError, lambda: _utils.collate.default_collate(batch[::-1]))

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_collate_shared_tensor(self):
        import numpy as np
        t_in = torch.zeros(1)
        n_in = np.zeros(1)

        self.assertEqual(t_in.is_shared(), False)

        self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), False)
        self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), False)

        # FIXME: fix the following hack that makes `default_collate` believe
        #        that it is in a worker process (since it tests
        #        `get_worker_info() != None`), even though it is not.
        old = _utils.worker._worker_info
        try:
            _utils.worker._worker_info = 'x'
            self.assertEqual(_utils.collate.default_collate([t_in]).is_shared(), True)
            self.assertEqual(_utils.collate.default_collate([n_in]).is_shared(), True)
        finally:
            _utils.worker._worker_info = old

    def test_excessive_thread_creation_warning(self):
        with self.assertWarnsRegex(
            UserWarning,
                r"excessive worker creation might get DataLoader running slow or even freeze"):
            dataloader = DataLoader(self.dataset, batch_size=2, num_workers=1000)


class TestDataLoaderDeviceType(TestCase):
    @parametrize("context", [ctx for ctx in supported_multiprocessing_contexts if ctx is not None])
    def test_nested_tensor_multiprocessing(self, device, context):
        # The 'fork' multiprocessing context doesn't work for CUDA so skip it
        if 'cuda' in device and context == "fork":
            # TODO: Skip this better in a better way when the test framework allows
            return

        dataset = [torch.nested.nested_tensor([torch.randn(5)], device=device) for _ in range(10)]

        pin_memory_settings = [False]
        if device == 'cpu' and torch.cuda.is_available():
            pin_memory_settings.append(True)

        for pin_memory in pin_memory_settings:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                collate_fn=_clone_collate,
                pin_memory=pin_memory,
                multiprocessing_context=context,
            )

            for i, batch in enumerate(loader):
                self.assertEqual(batch[0], dataset[i])

        # Error case: default collate_fn doesn't currently support batches of nested tensors.
        # Following the current semantics, we'd need to stack them, which isn't possible atm.
        with self.assertRaisesRegex(
                RuntimeError, "not currently supported by the default collate_fn"):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                multiprocessing_context=context,
            )

            next(iter(loader))


class IntegrationTestDataLoaderDataPipe(TestCase):
    r"""
    Verify the behavior of a certain ``DataPipes`` with ``DataLoader``
    """

    def test_shuffler_iterdatapipe(self):
        r"""
        Verify ``IterDataPipe.shuffle`` is controlled by ``DataLoader``
        to generate different seeds deterministically per epoch.
        """
        exp = list(range(100))

        def _create_dp(buffer_size):
            input_ds = dp.iter.IterableWrapper(exp)
            return input_ds.shuffle(buffer_size=buffer_size).sharding_filter()

        for bs in (5, 20, 33):
            # Test Deterministic
            for num_workers, pw in itertools.product((0, 1, 2), (True, False)):
                if num_workers == 0 and pw:
                    continue

                shuffle_dp = _create_dp(bs)

                mp_ctx = "spawn" if num_workers > 0 else None
                dl = DataLoader(
                    shuffle_dp,
                    num_workers=num_workers,
                    shuffle=True,
                    multiprocessing_context=mp_ctx,
                    persistent_workers=pw
                )

                # No seed
                dl_res_ns = list(dl)
                self.assertEqual(sorted(dl_res_ns), exp)

                # Same seeds
                dl_res = []
                for epoch in range(2):
                    torch.manual_seed(123)
                    dl_res.append(list(dl))
                self.assertEqual(dl_res[0], dl_res[1])
                self.assertEqual(sorted(dl_res[0]), exp)

                # Different seeds
                torch.manual_seed(321)
                dl_res.append(list(dl))

                self.assertEqual(len(dl_res[0]), len(dl_res[2]))
                self.assertNotEqual(dl_res[0], dl_res[2])
                self.assertEqual(sorted(dl_res[0]), sorted(dl_res[2]))

                if dl._iterator is not None:
                    dl._iterator._shutdown_workers()
                    dl._iterator = None
                del dl


class StringDataset(Dataset):
    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)

    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestStringDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for (s, n) in loader:
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.empty(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestDictDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        for persistent_workers in (False, True):
            if persistent_workers:
                loader = DataLoader(self.dataset, batch_size=2, shuffle=False,
                                    persistent_workers=persistent_workers, num_workers=1)
            else:
                loader = DataLoader(self.dataset, batch_size=2, shuffle=False,
                                    persistent_workers=persistent_workers)
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
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pin_memory_device(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True, pin_memory_device='cuda')
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned(device='cuda'))
            self.assertTrue(sample['another_dict']['a_number'].is_pinned(device='cuda'))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pin_memory_with_only_device(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory_device='cuda')
        for sample in loader:
            self.assertFalse(sample['a_tensor'].is_pinned(device='cuda'))
            self.assertFalse(sample['another_dict']['a_number'].is_pinned(device='cuda'))

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = list(range(10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # The persistent workers always maintain the original
        # dataset through the dataloader lifetime
        # so the attributes will remain the same as the
        # first time the workers where spawned (dataloader iteration)
        assert self.start == 0
        return self.data[idx]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
@unittest.skipIf(
    TEST_WITH_ASAN, "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223")
class TestDataLoaderPersistentWorkers(TestDataLoader):

    def setUp(self):
        super().setUp()
        self.persistent_workers = True

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # See NOTE [ DataLoader on Linux and open files limit ]
        import subprocess
        subprocess.check_output([sys.executable, '-c', """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1, persistent_workers=True):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
"""])

    def test_dataset_not_reset(self):
        dataset = DummyDataset()
        pin_memory_configs = [False]
        if TEST_CUDA:
            pin_memory_configs.append(True)
        for pin_memory in pin_memory_configs:
            dataloader = self._get_data_loader(dataset, num_workers=2, pin_memory=pin_memory)
            dataset.start = 0
            for i in range(10):
                for x in dataloader:
                    pass
                # Changing the start value here doesn't have any effect in the dataset
                # cached by the workers. since they are not recreated between epochs
                # and can cache values safely
                dataset.start = i

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_early_exit(self):
        import subprocess
        proc = subprocess.check_output([sys.executable, '-c', """\
import torch
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

if __name__ == '__main__':
    dl = DataLoader(
        RandomDataset(64, (28, 28)),
        batch_size=16,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    for _ in dl:
        break
"""])


class NamedTupleDataset(Dataset):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data', 'label', 'random_tensor'])
    Data = namedtuple('Data', ['positive', 'negative'])

    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return self.Batch(data=self.Data(positive=ndx, negative=-ndx),
                          label=str(ndx), random_tensor=torch.randn(3))


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestNamedTupleDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = NamedTupleDataset()

    def test_dataloader_with_namedtuple(self):
        # auto-collation
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=TEST_CUDA)
        for batch in loader:
            self.assertIsInstance(batch, NamedTupleDataset.Batch)
            self.assertEqual(batch.random_tensor.is_pinned(), TEST_CUDA)
            self.assertIsInstance(batch.data, NamedTupleDataset.Data)
            self.assertIsInstance(batch.data.positive, torch.Tensor)
            self.assertEqual(batch.data.positive.is_pinned(), TEST_CUDA)
        # no auto-collation
        loader = DataLoader(self.dataset, batch_size=None, pin_memory=TEST_CUDA)
        for batch in loader:
            self.assertIsInstance(batch, NamedTupleDataset.Batch)
            self.assertEqual(batch.random_tensor.is_pinned(), TEST_CUDA)
            self.assertIsInstance(batch.data, NamedTupleDataset.Data)
            self.assertNotIsInstance(batch.data.positive, torch.Tensor)

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

    def is_pinned(self):
        return self.inp.is_pinned() and self.tgt.is_pinned()

# Workaround for https://github.com/pytorch/pytorch/issues/50661
# Classes from  `__main__` can not be correctly unpickled from spawned module
# See https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
self_module = __import__(os.path.splitext(os.path.basename(__file__))[0])

def collate_wrapper(batch):
    return self_module.SimpleCustomBatch(batch)


def collate_into_packed_sequence(batch):
    data = torch.stack([sample[0] for sample in batch], 1)
    t, b = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, enforce_sorted=False)


def collate_into_packed_sequence_batch_first(batch):
    data = torch.stack([sample[0] for sample in batch], 0)
    b, t = data.size()
    lengths = torch.randint(1, t, size=(b,), dtype=torch.int64)
    return torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
class TestCustomPinFn(TestCase):
    def setUp(self):
        super().setUp()
        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.dataset = TensorDataset(inps, tgts)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_custom_batch_pin(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (collate_into_packed_sequence_batch_first, torch.nn.utils.rnn.PackedSequence),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn,
                                pin_memory=True)
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_custom_batch_pin_worker(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (collate_into_packed_sequence_batch_first, torch.nn.utils.rnn.PackedSequence),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn,
                                pin_memory=True, num_workers=1)
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())


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


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)")
@unittest.skipIf(
    TEST_WITH_ASAN,
    "Flaky with ASAN, see https://github.com/pytorch/pytorch/issues/65727")
class TestIndividualWorkerQueue(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = TestWorkerQueueDataset(list(range(128)))

    def _run_ind_worker_queue_test(self, batch_size, num_workers):
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            timeout=5, worker_init_fn=self.dataset.worker_init_fn
        )
        current_worker_idx = 0
        for i, (worker_ids, sample) in enumerate(loader):
            self.assertEqual(worker_ids.tolist(), [current_worker_idx] * batch_size)
            self.assertEqual(sample.tolist(), list(range(i * batch_size, (i + 1) * batch_size)))
            current_worker_idx += 1
            if current_worker_idx == num_workers:
                current_worker_idx = 0

    def test_ind_worker_queue(self):
        max_num_workers = None
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_workers = len(os.sched_getaffinity(0))
            except Exception:
                pass
        if max_num_workers is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                # Use half number of CPUs
                max_num_workers = cpu_count // 2

        if max_num_workers is None:
            max_num_workers = 1

        for batch_size in (8, 16, 32, 64):
            for num_workers in range(0, min(6, max_num_workers)):
                self._run_ind_worker_queue_test(batch_size=batch_size, num_workers=num_workers + 1)


class SetAffinityDataset(IterableDataset):

    def __iter__(self):
        torch.randperm(1)
        after = os.sched_getaffinity(0)
        return iter(after)

@unittest.skipIf(
    not hasattr(os, 'sched_setaffinity'),
    "os.sched_setaffinity is not available")
class TestSetAffinity(TestCase):
    def test_set_affinity_in_worker_init(self):
        # Query the current affinity mask to avoid setting a disallowed one
        old_affinity = os.sched_getaffinity(0)
        if not old_affinity:
            self.skipTest("No affinity information")
        # Choose any
        expected_affinity = list(old_affinity)[-1]

        def worker_set_affinity(_):
            os.sched_setaffinity(0, [expected_affinity])


        dataset = SetAffinityDataset()

        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=2, worker_init_fn=worker_set_affinity)
        for sample in dataloader:
            self.assertEqual(sample, [expected_affinity])

class ConvDataset(Dataset):
    def __init__(self):
        self.x = torch.ones(1, 1, 24000)
        # Call convolution on parent process
        self[0]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.nn.functional.conv1d(self.x, torch.ones(1, 1, 2))


@unittest.skipIf(IS_WINDOWS, "Needs fork")
@unittest.skipIf(
    TEST_WITH_ASAN,
    "This test hangs when running with ASAN, see https://github.com/pytorch/pytorch/issues/75492")
class TestConvAfterFork(TestCase):
    # Tests crash reported in https://github.com/pytorch/pytorch/issues/53565
    def test_conv_after_fork(self):
        loader = DataLoader(ConvDataset(), num_workers=1)
        for x in loader:
            self.assertEqual(x.shape, (1, 1, 1, 23999))


instantiate_device_type_tests(TestDataLoaderDeviceType, globals())


if __name__ == '__main__':
    run_tests()
