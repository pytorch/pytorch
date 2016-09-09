import torch
from .sampler import SequentialSampler, RandomSampler


class Dataset(object):

    def __init__(self, data_source, batch_size=1, shuffle=False,
            input_type=None, target_type=None, sampler=None):
        self.source = data_source
        self.batch_size = batch_size
        self._input_type = input_type
        self._target_type = target_type
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(data_source)
        elif not shuffle:
            self.sampler = SequentialSampler(data_source)

    def __iter__(self):
        output_idx = 0
        sample_buffer = None
        target_buffer = None
        for idx in self.sampler:
            sample, target = self.source[idx]
            sample_buffer = sample_buffer or self._new_buffer(self._input_type, sample)
            target_buffer = target_buffer or self._new_buffer(self._target_type, target)

            sample_buffer[output_idx] = sample
            target_buffer[output_idx] = target

            output_idx += 1
            if output_idx == self.batch_size:
                output_idx = 0
                yield sample_buffer, target_buffer

    def __len__(self):
        return len(self.sampler)

    def _new_buffer(self, default_type, elem):
        if default_type is None:
            default_type = type(elem)
        return default_type(self.batch_size, *elem.size())

    def input_type(self, new_type):
        self._input_type = new_type

    def target_type(self, new_type):
        self._target_type = new_type

