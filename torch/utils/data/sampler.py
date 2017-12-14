import torch
import random
import numpy as np


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
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


class EnlargeLabelShufflingSampler(Sampler):
    """
        label shuffling technique aimed to deal with imbalanced class problem
        without replacement, manipulated by indices.
        All classes are enlarged to the same amount, so classes can be trained equally.
        argument:
        indices: indices of labels of the whole dataset
        """

    def __init__(self, indices):
        # mapping between label index and sorted label index
        sorted_labels = sorted(enumerate(indices), key=lambda x: x[1])
        count = 1
        count_of_each_label = []
        tmp = -1
        # get count of each label
        for (x, y) in sorted_labels:
            if y == tmp:
                count += 1
            else:
                if tmp != -1:
                    count_of_each_label.append(count)
                    count = 1
            tmp = y
        count_of_each_label.append(count)
        # get the largest count among all classes. used to enlarge every class to the same amount
        largest = int(np.amax(count_of_each_label))
        self.count_of_each_label = count_of_each_label
        self.enlarged_index = []

        # preidx used for find the mapping beginning of arg "sorted_labels"
        preidx = 0
        for x in range(len(self.count_of_each_label)):
            idxes = np.remainder(torch.randperm(largest).numpy(), self.count_of_each_label[x]) + preidx
            for y in idxes:
                self.enlarged_index.append(sorted_labels[y][0])
            preidx += int(self.count_of_each_label[x])

    def __iter__(self):
        random.shuffle(self.enlarged_index)
        return iter(self.enlarged_index)

    def __len__(self):
        return np.amax(self.count_of_each_label) * len(self.count_of_each_label)


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
