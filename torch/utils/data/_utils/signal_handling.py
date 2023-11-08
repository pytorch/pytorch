r"""Signal handling for multiprocessing data loading.

NOTE [ Signal handling in multiprocessing data loading ]

In cases like DataLoader, if a worker process dies due to bus error/segfault
or just hang, the main process will hang waiting for data. This is difficult
to avoid on PyTorch side as it can be caused by limited shm, or other
libraries users call in the workers. In this file and `DataLoader.cpp`, we make
our best effort to provide some error message to users when such unfortunate
events happen.

When a _BaseDataLoaderIter starts worker processes, their pids are registered in a
defined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]
via `_set_worker_pids`.

When an error happens in a worker process, the main process received a SIGCHLD,
and Python will eventually call the handler registered below
(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`
call checks all registered worker pids and raise proper error message to
prevent main process from hanging waiting for data from worker.

Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,
`_set_worker_signal_handlers` is called to register critical signal handlers
(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an error
message to stderr before triggering the default handler. So a message will also
be printed from the worker process when it is killed by such signals.

See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning of
this signal handling design and other mechanism we implement to make our
multiprocessing data loading robust to errors.
"""

import signal
import threading
from . import IS_WINDOWS

# Some of the following imported functions are not used in this file, but are to
# be used `_utils.signal_handling.XXXXX`.
from torch._C import _set_worker_pids, _remove_worker_pids  # noqa: F401
from torch._C import _error_if_any_worker_fails, _set_worker_signal_handlers  # noqa: F401

_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler():
    # Windows doesn't support SIGCHLD handler
    if IS_WINDOWS:
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):  # type: ignore[attr-defined]
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        # This doesn't catch default handler, but SIGCHLD default handler is a
        # no-op.
        previous_handler = None

    def handler(signum, frame):
        # This following call uses `waitid` with WNOHANG from C side. Therefore,
        # Python can still get and update the process status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            assert callable(previous_handler)
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True
