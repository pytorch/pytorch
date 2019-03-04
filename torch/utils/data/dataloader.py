r"""Definition of the DataLoader and it's iterator _DataLoaderIter classes.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

import torch
import torch.multiprocessing as multiprocessing
from . import IterableDataset, SequentialSampler, RandomSampler, BatchSampler
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


class _DataLoaderMode(object):
    Map = 0
    MapWithBatchedRead = 1
    Iterable = 2


class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    :class:`~torch.utils.data.DataLoader` supports two different styles of
    datasets:

    * Map-style datasets

      A map-style dataset is one that implements the ``__getitem__`` protocol,
      and represents a map from indices/keys to data samples.  E.g., such a
      dataset, when called ``dataset[idx]`` could read and the ``idx``-th image
      and its corresponding label from a folder on the disk.

      See :class:`~torch.utils.data.Dataset` for more details.

    * Iterable-style datasets

      An iterable-style dataset is one that implements the ``__iter__`` protocol,
      and represents an iterable over data samples.  E.g., such a dataset, when
      called ``iter(dataset)``, could return a stream of data reading from a
      database, a remote server, or even logs generated in real time.

      See :class:`~torch.utils.data.IterableDataset` for more details.

    :class:`~torch.utils.data.DataLoader` constructor receives a
    :attr:`dataset` object as its first argument. Based on the other provided
    arguments, a :class:`~torch.utils.data.DataLoader` operates in one of three
    following modes:

    * Batched loading from a map-style dataset (default).

      This is the most common case, and corresponds to fetching a minibatch of
      data and collate them into batched Tensors, i.e., Tensors with one dimension
      being the batch dimension (usually the first). Two combinations of configs
      that starts this mode are:

      * (default) Using :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and
        :attr:`drop_last` to specify the batch indices sampling behavior.

        With the default arguments, a :class:`~torch.utils.data.DataLoader`
        loads data as batches of size ``1`` with indices sampled with replacement.

      * Setting :attr:`batch_sampler` as a custom sampler returning a list of
        indices to the :attr:`dataset`at a time.

      At each time, :attr:`collate_fn` is used to collate the list of samples
      into batched Tensors. Users may use customized :attr:`collate_fn` to
      achieve custom batching, e.g., along a dimension other than the first.

      The behavior of this mode is roughly equivalent with::

          for indices in batch_sampler:
              yield collate_fn([dataset[i] for i in indices])

    * Loading individual elements from a map-style dataset.

    * Loading from an iterable-style dataset.


    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, :attr:`shuffle must be False.
        batch_sampler (Sampler, optional): like :attr:`sampler`, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        convert_fn (callable, optional): converts a sample to Tensor(s).
        collate_fn (callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
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

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use :func:`torch.initial_seed()` to access the PyTorch seed for
              each worker in :attr:`worker_init_fn`, and use it to set other
              seeds before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.

    .. note:: When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
              ``len(dataloader)`` returns ``len(dataset)``. But this may not be
              accurate because the actual iterator size also depends on :attr:`num_workers`
              and :attr:`worker_init_fn`. If ``num_workers > 0``, each worker
              gets a copy of the same iterable dataset object and can return
              duplicate data, unless the dataset copies and/or the workers are
              configured differently in :attr:`worker_init_fn`. See
              :class:`~torch.utils.data.IterableDataset` for more details and
              examples.

    .. note:: The default memory pinning logic only recognizes Tensors and maps and iterables
              containing Tensors.  By default, if the pinning logic sees a batch that is a custom type
              (which will occur if you have a :attr:`collate_fn` that returns a custom batch type),
              or if each element of your batch is a custom type, the pinning logic will not
              recognize them, and it will return that batch (or those elements)
              without pinning the memory.  To enable memory pinning for custom batch or data types,
              define a ``pin_memory`` method on your custom type(s). See the example below.

    Example::

        class SimpleCustomBatch:
            def __init__(self, data):
                transposed_data = list(zip(*data))
                self.inp = torch.stack(transposed_data[0], 0)
                self.tgt = torch.stack(transposed_data[1], 0)

            # custom memory pinning method on custom type
            def pin_memory(self):
                self.inp = self.inp.pin_memory()
                self.tgt = self.tgt.pin_memory()
                return self

        def collate_wrapper(batch):
            return SimpleCustomBatch(batch)

        inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        dataset = TensorDataset(inps, tgts)

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                            pin_memory=True)

        for batch_ndx, sample in enumerate(loader):
            print(sample.inp.is_pinned())
            print(sample.tgt.is_pinned())

    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 convert_fn=_utils.collate.default_convert,
                 collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.convert_fn = convert_fn
        self.collate_fn = collate_fn
        if isinstance(dataset, IterableDataset):
            # FIXME: check default for some args
            self.mode = _DataLoaderMode.Iterable
            self.sampler = None
        elif batch_size is None:
            # FIXME: check default for some args
            self.mode = _DataLoaderMode.Map
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            self.sampler = sampler
        else:
            # FIXME: check default for some args
            self.mode = _DataLoaderMode.MapWithBatchedRead

            self.batch_size = batch_size
            self.drop_last = drop_last

            if timeout < 0:
                raise ValueError('timeout option should be non-negative')

            if batch_sampler is not None:
                if batch_size > 1 or shuffle or sampler is not None or drop_last:
                    raise ValueError('batch_sampler option is mutually exclusive '
                                     'with batch_size, shuffle, sampler, and '
                                     'drop_last')
                self.batch_size = None
                self.drop_last = None

            if sampler is not None and shuffle:
                raise ValueError('sampler option is mutually exclusive with '
                                 'shuffle')

            if self.num_workers < 0:
                raise ValueError('num_workers option cannot be negative; '
                                 'use num_workers=0 to disable multiprocessing.')

            if batch_sampler is None:
                if sampler is None:
                    if shuffle:
                        sampler = RandomSampler(dataset)
                    else:
                        sampler = SequentialSampler(dataset)
                batch_sampler = BatchSampler(sampler, batch_size, drop_last)

            self.sampler = batch_sampler
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

    def __len__(self):
        if self.mode == _DataLoaderMode.Iterable:
            raise len(self.dataset)
        return len(self.sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.mode = loader.mode
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.convert_fn = loader.convert_fn
        self.collate_fn = loader.collate_fn

        if self.mode == _DataLoaderMode.Iterable:
            self.sampler_iter = None
        else:
            self.sampler_iter = iter(self.sampler)

        if self.mode == _DataLoaderMode.Iterable and self.num_workers == 0:
            self.dataset_iter = iter(self.dataset)
        else:
            self.dataset_iter = None

        self.base_seed = torch.empty((), dtype=torch.int64).random_().item()

    def __iter__(self):
        return self

    def _next_index(self):
        if self.mode == _DataLoaderMode.Iterable:
            return None
        else:
            return next(self.sampler_iter)  # may raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        if self.mode == _DataLoaderMode.Iterable:
            raise len(self.dataset)
        else:
            return len(self.sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self.timeout == 0
        assert self.num_workers == 0

    def __next__(self):
        if self.mode == _DataLoaderMode.Iterable:
            data = self.convert_fn(next(self.dataset_iter))  # may raise StopIteration
        else:
            index = self._next_index()  # may raise StopIteration
            if self.mode == _DataLoaderMode.Map:
                data = self.convert_fn(self.dataset[index])
            else:
                # mode == _DataLoaderMode.MapWithBatchedRead:
                data = self.collate_fn([self.dataset[i] for i in index])
        if self.pin_memory:
            data = _utils.pin_memory.pin_memory_data(data)
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
    #           `_DataLoaderiter.try_get_data()` for details.
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
    #           Note that this may leave corrupted data in the queue, but we
    #           don't care about the data anyways once we are shutting down.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `done_event`:
    #   A `multiprocessing.Event` shared among all processes and threads. This
    #   is used to signal the workers and pin_memory_thread that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # [worker processes]
    #   While loader process is alive:
    #     Get from index_queue.
    #       If got a `None`, exit.
    #       If get anything else,
    #          Check `done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data.
    #       If timed out,
    #          No matter `done_event` is set (still need to see `None`) or not,
    #          must continue to next iteration .
    #
    # [pin_memory_thread]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While True:
    #     Get from index_queue.
    #       If got a `None`, exit.
    #       If get anything else,
    #          Check `done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [main process]
    #   In the DataLoader Iter's `__del__`
    #     a. Set `done_event` (shared with `pin_memory_thread` and workers).
    #
    #        Note: from here on, the workers & `pin_memory_thread` may exit at
    #              any time after they receive `None`.
    #
    #     b. Exit `pin_memory_thread`
    #          i.   Put `None` in `worker_result_queue`.
    #          ii.  Join the `pin_memory_thread`.
    #
    #     c. Exit the workers.
    #          i.   Put `None` in each worker's `index_queue`.
    #          ii.  Join the workers.
    #
    #        NOTE: This has to be after (b) because it may leave corrupted data
    #              in `worker_result_queue`, which `pin_memory_thread` reads
    #              from.
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
        self.done_event = multiprocessing.Event()

        self.index_queues = []
        self.workers = []
        # A list of booleans representing whether each worker has exhausted
        # its iterable dataset object. Always contain all Trues if not using
        # an iterable dataset (i.e., if mode != Iterable).
        self.worker_status = []
        for i in range(self.num_workers):
            index_queue = multiprocessing.Queue()
            index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=_utils.worker._worker_loop,
                args=(self.mode, self.dataset, index_queue,
                      self.worker_result_queue, self.done_event,
                      self.convert_fn, self.collate_fn, self.base_seed + i,
                      self.worker_init_fn, i, self.num_workers))
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
            self.worker_status.append(True)

        if self.pin_memory:
            self.data_queue = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self.worker_result_queue, self.data_queue,
                      torch.cuda.current_device(), self.done_event))
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
            self.try_put_index()

    def try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
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
                if self.worker_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    _utils.signal_handling._remove_worker_pid(id(self), w.pid)
                    self.index_queues[worker_id].close()
                    self.worker_status[worker_id] = False
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str))
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def get_data(self):
        # Fetches data from `self.data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self.try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self.timeout > 0:
            success, data = self.try_get_data(self.timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        elif self.pin_memory:
            while self.pin_memory_thread.is_alive():
                success, data = self.try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self.try_get_data()
                if success:
                    return data

    def __next__(self):
        while True:
            # If the worker responsible for `self.rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # tries to advance `self.rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self.get_data()`
            # call and `IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self.rcvd_idx < self.send_idx:
                info = self.task_info[self.rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self.worker_status[worker_id]:  # has data or is still active
                    break
                del self.task_info[self.rcvd_idx]
                self.rcvd_idx += 1
            else:
                # no valid `self.rcvd_idx` is found (i.e., didn't break)
                self.shutdown_workers()
                raise StopIteration

            # Now `self.rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self.task_info[self.rcvd_idx]) == 2:
                data = self.task_info.pop(self.rcvd_idx)[1]
                return self.process_data(data)

            assert (not self.shutdown and self.tasks_outstanding > 0)
            idx, data = self.get_data()
            self.tasks_outstanding -= 1

            if self.mode == _DataLoaderMode.Iterable:
                # Check for IterableDatasetStopIteration
                if isinstance(data, _utils.worker.IterableDatasetStopIteration):
                    worker_id = data.worker_id
                    assert self.worker_status[worker_id]

                    # Signal termination to that specific worker that
                    # exhasuted its IterableDataset.
                    w = self.workers[worker_id]
                    _utils.signal_handling._remove_worker_pid(id(self), w.pid)
                    q = self.index_queues[worker_id]
                    q.put(None)
                    q.close()

                    # Note that we don't actually join the worker here
                    # because joining may be slow. Joinning is deferred to
                    # shutdown_workers, which it is called when all workers
                    # finish their IterableDataset copies or when this iterator
                    # is garbage collected.
                    self.worker_status[worker_id] = False
                    continue

            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.task_info[idx] += (data,)
            else:
                del self.task_info[idx]
                return self.process_data(data)

    next = __next__  # Python 2 compatibility

    def try_put_index(self):
        assert self.tasks_outstanding < 2 * self.num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self.num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self.worker_queue_idx_cycle)
            if self.worker_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return
        self.index_queues[worker_queue_idx].put((self.send_idx, index))
        self.task_info[self.send_idx] = (worker_queue_idx,)
        self.tasks_outstanding += 1
        self.send_idx += 1

    def process_data(self, data):
        self.rcvd_idx += 1
        self.try_put_index()
        if isinstance(data, _utils.ExceptionWrapper):
            raise data.exc_type(data.exc_msg)
        return data

    def shutdown_workers(self):
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
            # Removes pids from the C side data structure first so worker
            # termination afterwards won't trigger false positive error report.
            if self.worker_pids_set:
                _utils.signal_handling._remove_all_worker_pids(id(self))
                self.worker_pids_set = False

            self.done_event.set()

            # Exit `pin_memory_thread` first because exiting workers may leave
            # corrupted data in `worker_result_queue` which `pin_memory_thread`
            # reads from.
            if hasattr(self, 'pin_memory_thread'):
                # Use hasattr in case error happens before we set the attribute.
                # First time do `worker_result_queue.put` in this process.

                # `cancel_join_thread` in case that `pin_memory_thread` exited.
                self.worker_result_queue.cancel_join_thread()
                self.worker_result_queue.put(None)
                self.pin_memory_thread.join()
                # Indicate that no more data will be put on this queue by the
                # current process. This **must** be called after
                # `pin_memory_thread` is joined because that thread shares the
                # same pipe handles with this loader thread. If the handle is
                # closed, Py3 will error in this case, but Py2 will just time
                # out even if there is data in the queue.
                self.worker_result_queue.close()

            # Exit workers now.
            for i, q in enumerate(self.index_queues):
                if self.worker_status[i]:
                    q.put(None)
                    # Indicate that no more data will be put on this queue by
                    # the current process.
                    q.close()
                    self.worker_status[i] = False
            for w in self.workers:
                w.join()

    def __del__(self):
        self.shutdown_workers()
