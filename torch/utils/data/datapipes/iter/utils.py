import copy
import warnings
from torch.utils.data import IterDataPipe


class IterableWrapperIterDataPipe(IterDataPipe):
    r""":class:`IterableWrapperIterDataPipe`.

    Iterable datapipe that wraps an iterable object.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in iter().

    .. note::
      If `deepcopy` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    .. note:
      DataLoader always materialize iterable objects when performing serialization (e.g. when __getstate__) is called.
    """
    def __init__(self, iterable, deepcopy=True):
        self.iterable = iterable
        self.deepcopy = deepcopy
        self.state_counter = 0
        self.iter = None
        self.been_reset = False

    def __iter__(self):
        if self.been_reset is True:
            self.been_reset = False
            return self.iter
        elif self.iter is None:
            self._create_iterator()
            return self.iter
        else:
            raise RuntimeError(f"Only one iterator can exist for each {type(self).__name__} at a time.")

    def _create_iterator(self) -> None:
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
                    "The input iterable cannot be deep copied,"
                    "please be aware of in-place modification would affect source data."
                )
        self.iter = iter(source_data)

    def __next__(self):
        if self.iter is None:
            self._create_iterator()
        self.state_counter += 1
        return next(self.iter)

    def reset(self) -> None:
        self.iter = None
        self._create_iterator()
        self.been_reset = True

    def __len__(self):
        return len(self.iterable)
