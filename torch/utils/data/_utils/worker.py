# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

from __future__ import annotations

import os
import queue
import random
import threading
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch._utils import ExceptionWrapper

from . import (
    HAS_NUMPY,
    IS_WINDOWS,
    pin_memory as pin_memory_module,
    signal_handling,
    STATUS_CHECK_INTERVAL,
)


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

        def is_alive(self) -> bool:
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

        def is_alive(self) -> bool:
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


_worker_info: Optional[WorkerInfo] = None
_thread_local_worker_info = threading.local()


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Information about the current DataLoader worker process or thread.

    Attributes:
        id: The current worker id (0 to num_workers - 1)
        num_workers: Total number of workers
        seed: Random seed set for this worker
        dataset: Copy of the dataset object in this worker
        rng: Optional RNG state container. Defaults to None.
        worker_method: Optional worker method ("multiprocessing" or "thread"). Defaults to "multiprocessing".
    """

    id: int
    num_workers: int
    seed: int
    dataset: Dataset
    rng: Optional[_RNG] = None
    worker_method: Optional[str] = "multiprocessing"


def get_worker_info() -> WorkerInfo | None:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process or thread.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process or thread. Note
      that this will be a different object in a different process than the one
      in the main process. For thread, this is the same object as the one in the main process.
    * :attr:`worker_method`: the worker method being used. Either ``"multiprocessing"``
      for process-based workers or ``"thread"`` for thread-based workers.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process/thread differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    # There is no global worker_method flag because it is set in worker info,
    # so try thread-local storage first, fall back to _worker_info.
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

    seed: int | None = None


@dataclass(frozen=True, slots=True)
class _RNG:
    """Container for thread-local random number generator state.

    Used by thread workers to maintain separate RNG state per worker thread
    to avoid race conditions.

    Attributes:
        random_generator: Python random.Random generator for this thread
        torch_generator: PyTorch Generator for this thread
        numpy_generator: NumPy Generator for this thread (None if numpy not available)
    """

    random_generator: random.Random
    torch_generator: torch.Generator
    numpy_generator: Optional[object] = None


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


def _base_worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    init_fn,
    worker_id,
    shared_rng=None,
    worker_method="multiprocessing",
    watchdog_constructor=None,
    post_fetch_fn=None,
) -> None:
    """
    Base worker loop with common functionality for both process and thread workers.

    Args:
        worker_method: The worker method ("multiprocessing", "thread")
        post_fetch_fn: Optional callback to process data after fetching (e.g., pin_memory for thread workers)
    """
    try:
        torch.set_num_threads(1)

        from torch.utils.data import _DatasetKind

        init_exception = None

        error_prefix = (
            "worker process" if worker_method == "multiprocessing" else "worker thread"
        )
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
                r = index_queue.get(timeout=STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                # Note: DataPipe is not supported in thread mode
                if worker_method == "multiprocessing":
                    from torch.utils.data import IterDataPipe

                    if isinstance(dataset, IterDataPipe):
                        from torch.utils.data.graph_settings import apply_random_seed

                        if r.seed is None:
                            raise AssertionError(
                                "resume iteration seed is None for IterDataPipe"
                            )
                        if shared_rng is None:
                            raise AssertionError(
                                "shared_rng is None for IterDataPipe in multiprocessing mode"
                            )
                        shared_rng.manual_seed(r.seed)
                        dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last
                )
                continue
            elif r is None:
                # Received the final signal
                if not done_event.is_set() and not iteration_end:
                    raise AssertionError(
                        "Received final signal but neither done_event nor iteration_end is set"
                    )
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            task_id, data_index = r
            data: _IterableDatasetStopIteration | ExceptionWrapper
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(data_index)  # type: ignore[possibly-undefined]

                    # Apply post-fetch processing if provided (e.g., pin_memory for thread workers)
                    if post_fetch_fn is not None and not isinstance(
                        data, ExceptionWrapper
                    ):
                        try:
                            data = post_fetch_fn(data)
                        except Exception:
                            data = ExceptionWrapper(
                                where=f"in {post_fetch_fn.__name__} for DataLoader {error_prefix} {worker_id}"
                            )
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
            data_queue.put((task_id, data))
            del data, task_id, data_index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass


def _process_worker_loop(
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
    torch.multiprocessing._set_thread_name("pt_multiprocess_data_worker")

    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)

    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np

        np.random.seed(np_seed)

    from torch.utils.data import IterDataPipe

    shared_rng = torch.Generator()
    if isinstance(dataset, IterDataPipe):
        from torch.utils.data.graph_settings import apply_random_seed

        if shared_seed is None:
            raise AssertionError(
                "shared_seed must be provided for IterDataPipe workers"
            )
        shared_rng.manual_seed(shared_seed)
        dataset = apply_random_seed(dataset, shared_rng)

    global _worker_info
    _worker_info = WorkerInfo(
        id=worker_id,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
        worker_method="multiprocessing",
    )

    _base_worker_loop(
        dataset_kind=dataset_kind,
        dataset=dataset,
        index_queue=index_queue,
        data_queue=data_queue,
        done_event=done_event,
        auto_collation=auto_collation,
        collate_fn=collate_fn,
        drop_last=drop_last,
        init_fn=init_fn,
        worker_id=worker_id,
        shared_rng=shared_rng,
        worker_method="multiprocessing",
        watchdog_constructor=ManagerWatchdog,
    )

    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


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
    pin_memory=False,
):
    """
    Thread worker loop that uses the common base worker loop for threads.
    Sets up thread-local RNG state to avoid race conditions and shared state issues.
    """
    torch.multiprocessing._set_thread_name("pt_thread_data_worker")

    # Set the thread name for better debugging
    threading.current_thread().name = f"DataLoader_thread_{worker_id}"

    # Thread-local RNG setup to avoid race conditions with global state
    seed = base_seed + worker_id

    # Set up thread-local random generators
    random_generator = random.Random(seed)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    numpy_generator = None
    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np

        numpy_generator = np.random.default_rng(np_seed)

    rng = _RNG(
        random_generator=random_generator,
        torch_generator=torch_generator,
        numpy_generator=numpy_generator,
    )

    worker_info = WorkerInfo(
        id=worker_id,
        num_workers=num_workers,
        seed=seed,
        dataset=dataset,
        rng=rng,
        worker_method="thread",
    )

    _thread_local_worker_info.worker_info = worker_info

    # Create a post-fetch function for pin_memory if enabled
    post_fetch_fn = None
    if pin_memory:
        post_fetch_fn = pin_memory_module.pin_memory

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
        init_fn=init_fn,
        worker_id=worker_id,
        shared_rng=None,  # Not used for thread workers
        worker_method="thread",
        watchdog_constructor=None,  # No watchdog needed for threads
        post_fetch_fn=post_fetch_fn,
    )
