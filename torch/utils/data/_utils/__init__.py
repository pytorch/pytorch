r"""Utility classes & functions for data loading. Code in this folder is mostly
used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""

import sys
import atexit

# old private location of the ExceptionWrapper that some users rely on:
from torch._utils import ExceptionWrapper


IS_WINDOWS = sys.platform == "win32"


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

DATAPIPE_SHARED_SEED = "_dl_shared_seed"
r"""The key to share the same seed for shuffle DataPipe across distributed processes"""

DATAPIPE_SHARED_SEED_COUNTER = "_dl_shared_seed_recv_cnt"
r"""The key to count the number of distributed processes that have received the shared seed"""

DATAPIPE_SHARED_SEED_DEFAULT_TIMEOUT = 30 * 60
r"""Timeout (in seconds) sending the shared seed from Rank 0 and sending
    the signal of the shared seed received from other Ranks.
    It uses the same default timeout for the distributed process group"""

DATAPIPE_SHARED_SEED_CHECK_INTERVAL = 0.01
r"""Interval to check if each rank has received the shared seed"""


try:
    import numpy
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True

atexit.register(_set_python_exit_flag)


from . import worker, signal_handling, pin_memory, collate, fetch
