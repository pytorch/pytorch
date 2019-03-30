r"""Utility classes & functions for data loading. Code in this folder is mostly
used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""

import sys
import traceback
import atexit
import itertools


IS_WINDOWS = sys.platform == "win32"


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""


python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""


def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True

atexit.register(_set_python_exit_flag)


from .. import Sampler


class ConstantSampler(Sampler):
    r"""A sampler that always return the same value :attr:`value` infinitely or
    for :attr:`length` times. This is like ``itertools.repeat(value, length)``
    but is a subclass of :class:`torch.utils.data.Sampler`.

    .. note:: This is used as the dummy sampler when the dataset is an
    :class:`torch.utils.data.IterableDataset`.

    Arguments:
        value: the value that is always returned
        length: the length of this sampler. If ``None`` (default), this sampler
                is infinite.
    """

    def __init__(self, value, length=None):
        self.value = value
        self.length = length
        self.iterable = itertools.repeat(value, length)

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        if self.length is None:
            raise NotImplementedError
        return self.length


from . import worker, signal_handling, pin_memory, collate  # noqa: F401
