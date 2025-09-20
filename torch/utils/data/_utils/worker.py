# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import copy
import os
import queue
import random
import threading
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch._utils import ExceptionWrapper

from . import HAS_NUMPY, IS_WINDOWS, MP_STATUS_CHECK_INTERVAL, signal_handling


if TYPE_CHECKING:
    from torch.utils.data import Dataset

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self) -> None:
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(
                SYNCHRONIZE, 0, self.manager_pid
            )

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = (
                    self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
                )
            return not self.manager_dead

else:

    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


_worker_info: Optional["WorkerInfo"] = None
_thread_local_worker_info = threading.local()


class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: "Dataset"
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(
                f"Cannot assign attributes to {self.__class__.__name__} objects"
            )
        return super().__setattr__(key, val)

    def __repr__(self):
        items = [f"{k}={getattr(self, k)}" for k in self.__keys]
        return f"{self.__class__.__name__}({', '.join(items)})"


def get_worker_info() -> Optional[WorkerInfo]:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process or thread.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process/thread. Note
      that this will be a different object in a different process/thread than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process/thread differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    # Try thread-local storage first, fall back to _worker_info
    thread_local_worker_info = getattr(_thread_local_worker_info, "worker_info", None)
    if thread_local_worker_info is not None:
        return thread_local_worker_info

    return _worker_info


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    """Dummy class used to signal the end of an IterableDataset"""

    worker_id: int


@dataclass(frozen=True)
class _ResumeIteration:
    """Dummy class used to resume the fetching when worker reuse is enabled"""

    seed: Optional[int] = None


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# It's MIT licensed, here is the copyright:

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This function generates an array of int32 as the seed for
# `numpy.random`, in order to prevent state collision due to same
# seed and algorithm for `numpy.random` and `random` modules.
# TODO: Implement `SeedSequence` like object for `torch.random`
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def deep_copy_transforms(dataset, worker_id=None):
    # Create a shallow copy of the dataset, then deep copy transforms
    thread_dataset = copy.copy(dataset)

    # Common transform attribute names to deep copy
    transform_attrs = ["transform", "target_transform"]

    try:
        for attr_name in transform_attrs:
            if hasattr(dataset, attr_name):
                original_transform = getattr(dataset, attr_name)
                if original_transform is not None:
                    copied_transform = copy.deepcopy(original_transform)
                    setattr(thread_dataset, attr_name, copied_transform)
    except Exception as e:
        import warnings

        worker_prefix = f"Thread {worker_id}: " if worker_id is not None else ""
        warnings.warn(
            f"{worker_prefix}Failed to deep copy transform attributes ({e}). "
            "Transforms may not be thread-safe. Consider implementing custom __deepcopy__ "
            "method for your transform objects.",
            RuntimeWarning,
        )
    return thread_dataset


def _base_worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_rng=None,
    is_process=True,
    watchdog_constructor=None,
    error_prefix="worker process",
):
    """
    Base worker loop with common functionality for both process and thread workers.
    """
    try:
        torch.set_num_threads(1)

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collation, collate_fn, drop_last
            )
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader {error_prefix} {worker_id}"
            )

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        iteration_end = False

        # Create watchdog to check if parent is alive
        watchdog = watchdog_constructor() if watchdog_constructor is not None else None

        # Main worker loop
        while watchdog is None or watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                # Note: DataPipe is not supported in thread mode
                if is_process:
                    from torch.utils.data import IterDataPipe
                    from torch.utils.data.graph_settings import apply_random_seed

                    if isinstance(dataset, IterDataPipe):
                        assert r.seed is not None
                        shared_rng.manual_seed(r.seed)
                        dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last
                )
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, index = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                except Exception as e:
                    if (
                        isinstance(e, StopIteration)
                        and dataset_kind == _DatasetKind.Iterable
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where=f"in DataLoader {error_prefix} {worker_id}"
                        )
            data_queue.put((idx, data))
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        # TODO: Does this make sense for thread ?
        pass

    # Process-specific cleanup
    if is_process and done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal had already happened
    # again.
    # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
    signal_handling._set_worker_signal_handlers()
    torch.multiprocessing._set_thread_name("pt_data_worker")

    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np

        np.random.seed(np_seed)

    from torch.utils.data import IterDataPipe
    from torch.utils.data.graph_settings import apply_random_seed

    shared_rng = torch.Generator()
    if isinstance(dataset, IterDataPipe):
        assert shared_seed is not None
        shared_rng.manual_seed(shared_seed)
        dataset = apply_random_seed(dataset, shared_rng)

    worker_info = WorkerInfo(
        id=worker_id,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
    )

    global _worker_info
    _worker_info = worker_info

    _base_worker_loop(
        dataset_kind=dataset_kind,
        dataset=dataset,
        index_queue=index_queue,
        data_queue=data_queue,
        done_event=done_event,
        auto_collation=auto_collation,
        collate_fn=collate_fn,
        drop_last=drop_last,
        seed=seed,
        init_fn=init_fn,
        worker_id=worker_id,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shared_rng=shared_rng,
        is_process=True,
        watchdog_constructor=ManagerWatchdog,
        error_prefix="worker process",
    )


def _thread_worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
):
    """
    Thread worker loop that uses the common base worker loop for threads.
    Sets up thread-local RNG state and creates deep copies of dataset/transforms
    to avoid race conditions and shared state issues.
    """

    # Set the thread name for better debugging
    threading.current_thread().name = f"DataLoader_thread_{worker_id}"

    # Thread-local RNG setup to avoid race conditions with global state
    seed = base_seed + worker_id

    thread_rng = threading.local()

    # Set up thread-local random generators
    thread_rng.random_state = random.Random(seed)
    thread_rng.torch_generator = torch.Generator()
    thread_rng.torch_generator.manual_seed(seed)

    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np

        thread_rng.numpy_generator = np.random.default_rng(np_seed)

    # Deep copy the dataset's transforms to avoid race conditions and shared state issues
    dataset = deep_copy_transforms(dataset, worker_id)

    worker_info = WorkerInfo(
        id=worker_id,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
        thread_rng=thread_rng,  # not set for process workers
    )

    _thread_local_worker_info.worker_info = worker_info

    # Use the common base worker loop with thread-specific settings
    _base_worker_loop(
        dataset_kind=dataset_kind,
        dataset=dataset,
        index_queue=index_queue,
        data_queue=data_queue,
        done_event=done_event,
        auto_collation=auto_collation,
        collate_fn=collate_fn,
        drop_last=drop_last,
        seed=seed,
        init_fn=init_fn,
        worker_id=worker_id,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        is_process=False,
        watchdog_constructor=None,  # No watchdog needed for threads
        error_prefix="worker thread",
    )
