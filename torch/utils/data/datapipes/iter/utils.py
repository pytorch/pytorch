from torch.utils.data import IterDataPipe


class IterableAsDataPipeIterDataPipe(IterDataPipe):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        for data in self.iterable:
            yield data
