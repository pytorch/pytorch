import copy
import warnings
from torch.utils.data import MapDataPipe


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
