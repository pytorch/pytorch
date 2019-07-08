r"""Definition of the DataLoader and it's iterator _DataLoaderIter classes.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

import torch
import torch.multiprocessing as multiprocessing
from . import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler
from . import _utils
import threading
import itertools
from torch._six import queue


get_worker_info = _utils.worker.get_worker_info

# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate = _utils.collate.default_collate


class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None

    def __len__(self):
        # This has to be a TypeError, otherwise, since this is used in
        # `len(dataloader)`, `list(dataloader)` will fail.
        # see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        raise TypeError('Cannot determine the DataLoader length of a IterableDataset')


class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, :attr:`shuffle` must be ``False``.
        batch_sampler (Sampler, optional): like :attr:`sampler`, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. note:: ``len(dataloader)`` heuristic based on the length of the sampler used.
              When :attr:`dataset` is a subclass of :class:`~torch.utils.data.IterableDataset`,
              an infinite sampler is used, whose :meth:`__len__` is not
              implemented, because the actual length depends on both the
              iterable as well as multi-process loading configurations. So one
              should not query this method unless they work with a map-style
              dataset. See `Dataset Types`_ for more details on these two types
              of datasets.
    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        torch._C._log_api_usage_once("python.data_loader")
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if self.num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self.dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and `IterableDataset` ]
            #
            # `IterableDataset` does not supports custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            self.dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if shuffle or sampler is not None or drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with '
                                 'shuffle, sampler, and drop_last')

        if sampler is None:  # give default samplers
            if self.dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        return len(self._index_sampler)  # with iterable-style dataset, this will error


class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.dataset_kind = loader.dataset_kind
        self.auto_collation = loader._auto_collation
        self.drop_last = loader.drop_last
        self.index_sampler = loader._index_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.collate_fn = loader.collate_fn
        self.sampler_iter = iter(self.index_sampler)
        self.base_seed = torch.empty((), dtype=torch.int64).random_().item()

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self.sampler_iter)  # may raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self.timeout == 0
        assert self.num_workers == 0

        self.dataset_fetcher = _DatasetKind.create_fetcher(
            self.dataset_kind, self.dataset, self.auto_collation, self.collate_fn, self.drop_last)

    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
        if self.pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

    next = __next__  # Python 2 compatibility


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children. Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed. (Hooks freeing those
    #      resources are registered at importing the Python core libraries at
    #      the top of this file.) So in `__del__`, we check if
    #      `_utils.python_exit_status` is set or `None` (freed), and perform
    #      no-op if so.
    #
    #      Another problem with `__del__` is also related to the library cleanup
    #      calls. When a process ends, it shuts the all its daemonic children
    #      down with a SIGTERM (instead of joining them without a timeout).
    #      Simiarly for threads, but by a different mechanism. This fact,
    #      together with a few implementation details of multiprocessing, forces
    #      us to make workers daemonic. All of our problems arise when a
    #      DataLoader is used in a subprocess, and are caused by multiprocessing
    #      code which looks more or less like this:
    #
    #          try:
    #              your_function_using_a_dataloader()
    #          finally:
    #              multiprocessing.util._exit_function()
    #
    #      The joining/termination mentioned above happens inside
    #      `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #      throws, the stack trace stored in the exception will prevent the
    #      frame which uses `DataLoaderIter` to be freed. If the frame has any
    #      reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #      its  `__del__`, which starts the shutdown procedure, will not be
    #      called. That, in turn, means that workers aren't notified. Attempting
    #      to join in `_exit_function` will then result in a hang.
    #
    #      For context, `_exit_function` is also registered as an `atexit` call.
    #      So it is unclear to me (@ssnl) why this is needed in a finally block.
    #      The code dates back to 2008 and there is no comment on the original
    #      PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #      the finally block and the `atexit` registration) that explains this.
    #
    #      Another choice is to just shutdown workers with logic in 1 above
    #      whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           exits.
    #
    #           However, in case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever. Therefore,
    #           for both `worker_result_queue` (worker -> main process/pin_memory_thread)
    #           and each `index_queue` (main process -> worker), we use
    #           `q.cancel_join_thread()` in sender process before any `q.put` to
    #           prevent this automatic join.
    #
    #           Moreover, having all queues called `cancel_join_thread` makes
    #           implementing graceful shutdown logic in `__del__` much easier.
    #           It won't need to get from any queue, which would also need to be
    #           guarded by periodic status checks.
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           `pin_memory_thread`'s `data_queue` is a `queue.Queue` that does
    #           a blocking `put` if the queue is full. So there is no above
    #           problem, but we do need to wrap the `put` in a loop that breaks
    #           not only upon success, but also when the main process stops
    #           reading, i.e., is shutting down.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are neede is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `index_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timeing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self.num_workers > 0

        self.worker_init_fn = loader.worker_init_fn
        self.worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
        self.worker_result_queue = multiprocessing.Queue()
        self.worker_pids_set = False
        self.shutdown = False
        self.send_idx = 0  # idx of the next task to be sent to workers
        self.rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self.task_info = {}
        self.tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        self.workers_done_event = multiprocessing.Event()

        self.index_queues = []
        self.workers = []
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        self.workers_status = []
        for i in range(self.num_workers):
            index_queue = multiprocessing.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=_utils.worker._worker_loop,
                args=(self.dataset_kind, self.dataset, index_queue,
                      self.worker_result_queue, self.workers_done_event,
                      self.auto_collation, self.collate_fn, self.drop_last,
                      self.base_seed + i, self.worker_init_fn, i, self.num_workers))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self.workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self.index_queues.append(index_queue)
            self.workers.append(w)
            self.workers_status.append(True)

        if self.pin_memory:
            self.pin_memory_thread_done_event = threading.Event()
            self.data_queue = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self.worker_result_queue, self.data_queue,
                      torch.cuda.current_device(),
                      self.pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self.pin_memory_thread = pin_memory_thread
        else:
            self.data_queue = self.worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self.workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self.worker_pids_set = True

        # prime the prefetch loop
        for _ in range(2 * self.num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self.data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self.data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self.workers):
                if self.workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._shutdown_worker(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str))
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def _get_data(self):
        # Fetches data from `self.data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self.timeout > 0:
            success, data = self._try_get_data(self.timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        elif self.pin_memory:
            while self.pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def __next__(self):
        while True:
            # If the worker responsible for `self.rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self.rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self.rcvd_idx < self.send_idx:
                info = self.task_info[self.rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self.workers_status[worker_id]:  # has data or is still active
                    break
                del self.task_info[self.rcvd_idx]
                self.rcvd_idx += 1
            else:
                # no valid `self.rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self.rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self.task_info[self.rcvd_idx]) == 2:
                data = self.task_info.pop(self.rcvd_idx)[1]
                return self._process_data(data)

            assert not self.shutdown and self.tasks_outstanding > 0
            idx, data = self._get_data()
            self.tasks_outstanding -= 1

            if self.dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.task_info[idx] += (data,)
            else:
                del self.task_info[idx]
                return self._process_data(data)

    next = __next__  # Python 2 compatibility

    def _try_put_index(self):
        assert self.tasks_outstanding < 2 * self.num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self.num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self.worker_queue_idx_cycle)
            if self.workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self.index_queues[worker_queue_idx].put((self.send_idx, index))
        self.task_info[self.send_idx] = (worker_queue_idx,)
        self.tasks_outstanding += 1
        self.send_idx += 1

    def _process_data(self, data):
        self.rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, _utils.ExceptionWrapper):
            # make multiline KeyError msg readable by working around
            # a python bug https://bugs.python.org/issue2651
            if data.exc_type == KeyError and "\n" in data.exc_msg:
                raise Exception("KeyError:" + data.exc_msg)
            else:
                raise data.exc_type(data.exc_msg)
        return data

    def _shutdown_worker(self, worker_id):
        # Mark a worker as having finished its work and dead, e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_DataLoaderIter` is going to continue running.

        assert self.workers_status[worker_id]

        # Signal termination to that specific worker.
        q = self.index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.
        self.workers_status[worker_id] = False

    def _shutdown_workers(self):
        # Called when shutting down this `_DataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            try:
                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, 'pin_memory_thread'):
                    self.pin_memory_thread_done_event.set()
                    # Use hasattr in case error happens before we set the attribute.
                    self.pin_memory_thread.join()

                # Exit workers now.
                self.workers_done_event.set()
                for worker_id in range(self.num_workers):
                    if self.workers_status[worker_id]:
                        self._shutdown_worker(worker_id)
                for w in self.workers:
                    w.join()
                for q in self.index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self.worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self.worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()
