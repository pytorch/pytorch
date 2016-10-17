import torch
import torch.multiprocessing as multiprocessing
from .sampler import SequentialSampler, RandomSampler
import collections
import sys
import traceback

class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _processBatch(dataset, indices, collate_fn):
    samples = [dataset[idx] for idx in indices]

    samples = collate_fn(samples)
    return samples


def _workerLoop(dataset, index_queue, data_queue, collate_fn):
    torch.set_num_threads(1)
    while True:
        batch_indices = index_queue.get()

        if batch_indices is None:
            break

        try:
            samples = _processBatch(dataset, batch_indices, collate_fn)
        except Exception:
            data_queue.put(ExceptionWrapper(sys.exc_info()))
        else:
            data_queue.put(samples)

# default collate function, puts each data field into a
# tensor with outer dimension batchSize
def default_collate(batch):
    if torch.is_tensor(batch[0]):
        return torch.cat([t.view(1, *t.size()) for t in batch], 0)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.Iterable):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, or lists; found {}"
                     .format(type(batch[0]))))

class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)

        if self.num_workers:
            self.index_queue = multiprocessing.Queue()
            self.data_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.joined = False

            self.workers = [
                multiprocessing.Process(
                    target=_workerLoop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for i in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True # ensure that the worker exits on process exit
                w.start()
                # prime the prefetch loop with exactly 1 batch per process
                # this ensures no deadlocks on the queues using the blocking queue API
                self._putBatch()

    def _nextBatch(self):
        batch = [next(self.sample_iter) for x in range(min(self.samples_remaining, self.batch_size))]
        self.samples_remaining -= len(batch)
        return batch

    def _putBatch(self):
        if self.samples_remaining > 0:
            self.index_queue.put(self._nextBatch())
            self.batches_outstanding += 1

    def next(self):
        if self.num_workers:
            # multi-process loading
            if self.batches_outstanding:
                assert(not self.joined)
                # maintain at most len(workers)+1 outstanding batches
                # to avoid deadlocks in the queues, using the blocking queue API
                # TODO: add and use non-blocking queue API
                self._putBatch()
                assert(self.batches_outstanding <= len(self.workers) + 1)
                self.batches_outstanding -= 1
                data = self.data_queue.get()

                if isinstance(data, ExceptionWrapper):
                    raise data.exc_type(data.exc_msg)
                else:
                    return data
            else:
                self._joinWorkers()
                raise StopIteration()
        else:
            # single-process loading
            if self.samples_remaining:
                return _processBatch(self.dataset, self._nextBatch(), self.collate_fn)
            else:
                raise StopIteration()

    __next__ = next

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _joinWorkers(self):
        self.joined = True
        if self.num_workers:
            [self.index_queue.put(None) for x in self.workers]
            [x.join() for x in self.workers]

    def __del__(self):
        self._joinWorkers()

class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, num_workers=0, collate_fn=default_collate):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.collate_fn  = collate_fn

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        elif not shuffle:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.sampler)
