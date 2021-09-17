import copy
import warnings
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
        try:
            source_data = copy.deepcopy(self.iterable)
        # For the case that data cannot be deep-copied,
        # all in-place operations will affect iterable variable.
        # When this DataPipe is iterated second time, it will
        # yield modified items.
        except TypeError:
            warnings.warn(
                "The input iterable can not be deepcopied, "
                "please be aware of in-place modification would affect source data"
            )
            source_data = self.iterable
        for data in source_data:
            yield data

    def __len__(self):
        return len(self.iterable)
