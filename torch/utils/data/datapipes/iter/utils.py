from torch.utils.data import IterDataPipe


class IterableWrapperIterDataPipe(IterDataPipe):
    r""":class:`IterableWrapperIterDataPipe`.

    Iterable datapipe that wraps an iterable object.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
    """
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        for data in self.iterable:
            yield data

    def __len__(self):
        return len(self.iterable)
