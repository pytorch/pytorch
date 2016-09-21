import torch
import torch.multiprocessing as multiprocessing
from .sampler import SequentialSampler, RandomSampler

def _processBatch(dataset, indices, collate_fn):
    samples = []
    for idx in indices:
        sample = dataset[idx]
        samples.append(sample)

    samples = collate_fn(samples)
    return samples

def _workerLoop(dataset, index_queue, data_queue, collate_fn):
    while True:
        batch_indices = index_queue.get()
        if batch_indices == None:
            break
        samples = _processBatch(dataset, batch_indices, collate_fn)
        data_queue.put(samples)

# default collate function, puts each data field into a
# tensor with outer dimension batchSize
def default_collate(batch):
    if torch.isTensor(batch[0]):
        data = batch[0].new(len(batch), *batch[0].size())
        for i in range(len(batch)):
            data[i] = batch[i]
        return data
    else:
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

class DataLoader(object):

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, num_workers=0, collate_fn=default_collate):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        elif not shuffle:
            self.sampler = SequentialSampler(dataset)

        self.index_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        self.workers = [
            multiprocessing.Process(
                target=_workerLoop,
                args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
            for i in range(num_workers)]

        [x.start() for x in self.workers]
        
        # __del__ isn't guaranteed to be called on exit, so this avoids a hang
        import atexit
        atexit.register(self._cleanupWorkers)
    
    def _nextBatch(self):
        batch = [self.sample_iter.next() for x in range(min(self.samples_remaining, self.batch_size))]
        self.samples_remaining -= len(batch)
        return batch

    def _putBatch(self):
        if self.samples_remaining > 0:
            self.index_queue.put(self._nextBatch())
            self.batches_outstanding += 1

    def __iter__(self):

        self.samples_remaining = len(self.sampler)
        self.batches_outstanding = 0
        self.sample_iter = self.sampler.__iter__()

        if self.workers:
            # prime the prefetch loop
            for i in range(len(self.workers)):
                self._putBatch()

            while self.batches_outstanding:
                self._putBatch()
                self.batches_outstanding -= 1
                yield self.data_queue.get()

        else:
            while self.samples_remaining:
                yield _processBatch(self.dataset, self._nextBatch(), self.collate_fn)

    def __len__(self):
        return len(self.sampler)

    def _cleanupWorkers(self):
        if self.workers:
            [self.index_queue.put(None) for x in self.workers]
            [x.join() for x in self.workers]
            self.workers = None

    def __del__(self):
        self._cleanupWorkers()
