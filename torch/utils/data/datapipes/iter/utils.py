import copy
import pickle
import warnings
from torch.utils.data import IterDataPipe

from typing import Callable


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

    def __iter__(self):
        if self.iter is None:
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
            self.iter = iter(source_data)
            return self.iter
        else:
            # TODO: Raise a warning about only one iter may exist at a time?
            #       Then, allow user to reset it with a separate method
            return self.iter

    def __next__(self):
        if self.iter is None:
            self.iter = self.__iter__()
        self.state_counter += 1
        return next(self.iter)

    def reset(self):
        self.iter = None
        return self.__iter__()

    def save_snapshot(self):
        return self.state_counter

    def restore_snapshot(self, target_count=None):  # This should be called after __setstate__
        if target_count is None:
            target_count = self.state_counter
        self.state_counter = 0
        while self.state_counter < target_count:
            next(self)

    def __getstate__(self, preprocess_iterable_fn: Callable = list):
        """
        Args:
            preprocess_iterable_fn: in cases where `self.iterable` is not serializable, this function
                will be called to preprocess `self.iterable`. By default, the `list` function
                is called in an attempt to materialize the iterable.
        """
        if self.iter is not None:
            raise Exception(f"{type(self).__name___} is only serializable before it has been iterated upon.")
        try:
            pickle.dumps(self.iterable)
            iterable_to_pickle = self.iterable
        except TypeError:
            iterable_to_pickle = preprocess_iterable_fn(self.iterable)
        state = self.__dict__.copy()
        del state['iter']
        state['iterable'] = iterable_to_pickle
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.iter = None
        self.iter = self.__iter__()

    def __len__(self):
        return len(self.iterable)
