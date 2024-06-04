# mypy: allow-untyped-defs
import copy
import warnings
from torch.utils.data.datapipes.datapipe import MapDataPipe

__all__ = ["SequenceWrapperMapDataPipe", ]


class SequenceWrapperMapDataPipe(MapDataPipe):
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> dp = SequenceWrapper({'a': 100, 'b': 200, 'c': 300, 'd': 400})
        >>> dp['a']
        100
    """

    def __init__(self, sequence, deepcopy=True):
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)
            except TypeError:
                warnings.warn(
                    "The input sequence can not be deepcopied, "
                    "please be aware of in-place modification would affect source data"
                )
                self.sequence = sequence
        else:
            self.sequence = sequence

    def __getitem__(self, index):
        return self.sequence[index]

    def __len__(self):
        return len(self.sequence)
