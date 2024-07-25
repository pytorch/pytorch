# mypy: allow-untyped-defs
import copy
import warnings

from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = ["IterableWrapperIterDataPipe"]


class IterableWrapperIterDataPipe(IterDataPipe):
    r"""
    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, iterable, deepcopy=True):
        self.iterable = iterable
        self.deepcopy = deepcopy

    def __iter__(self):
        source_data = self.iterable
        if self.deepcopy:
            try:
                source_data = copy.deepcopy(self.iterable)
            # For the case that data cannot be deep-copied,
            # all in-place operations will affect iterable variable.
            # When this DataPipe is iterated second time, it will
            # yield modified items.
            except TypeError:
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data."
                )
        yield from source_data

    def __len__(self):
        return len(self.iterable)
