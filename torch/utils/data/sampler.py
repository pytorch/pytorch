import random
import torch


class RandomCycleIter:
    """Randomly iterate element in each cycle

    Example:
        >>> rand_cyc_iter = RandomCycleIter([1, 2, 3])
        >>> [next(rand_cyc_iter) for _ in range(10)]
        [2, 1, 3, 2, 3, 1, 1, 2, 3, 2]
    """
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__(self):
        return self

    def next(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
        return self.data_list[self.i]

def class_aware_sample_generator(cls_iter, data_iter_list, n):
    i = 0
    while i < n:
        yield next(data_iter_list[next(cls_iter)])
        i += 1


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    Arguments:
        weights (list)   : a list of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


class ClassAwareSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    
    Implemented Class-Aware Sampling: https://arxiv.org/abs/1512.05830
    Li Shen, Zhouchen Lin, Qingming Huang, Relay Backpropagation for Effective 
    Learning of Deep Convolutional Neural Networks, ECCV 2016
    By default num_samples equals to number of samples in the largest class 
    multiplied by num of classes such that all samples can be sampled.
    """

    def __init__(self, data_source, num_samples=0):
        self.data_source = data_source
        n_cls = len(data_source.classes)
        self.class_iter = RandomCycleIter(range(n_cls))
        cls_data_list = [list() for _ in range(n_cls)]
        for i, (_, label) in enumerate(data_source.imgs):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        if num_samples:
            self.num_samples = num_samples
        else:
            self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples)

    def __len__(self):
        return self.num_samples


class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
