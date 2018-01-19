import sys
import traceback

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


class ExceptionWrapper(object):
    r"Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


# Queue get and put methods may fail if the underlying syscalls are interrupted
# (EINTR). So we need to wrap them to retry automatically if needed.
#
# Syscalls are automatically retried upon encountering EINTR since Python 3.5
# https://www.python.org/dev/peps/pep-0475/
# EINTR is not available on Windows.
if sys.platform == 'win32' or sys.version_info >= (3, 5):
    def QueueWrapper(queue):
        return queue
else:
    import time
    import errno

    if sys.version_info >= (3, 3):
        time_fn = time.perf_counter
    else:
        time_fn = time.time

    class QueueWrapper(object):
        r"""Wraps a queue object that conforms to following interface:

            .get()  or  .get(timeout=timeout)
            .put(value)
        """

        def __init__(self, queue):
            self.queue = queue

        def get(self, timeout=None):
            while True:
                try:
                    if timeout is None:
                        return self.queue.get()
                    else:
                        t0 = time_fn()
                        return self.queue.get(timeout=timeout)
                except IOError as e:
                    if e.errno != errno.EINTR:
                        raise
                    if timeout is not None:
                        timeout -= time_fn() - t0
                        if timeout <= 0:
                            raise queue.Empty

        def put(self, val):
            while True:
                try:
                    return self.queue.put(val)
                except IOError as e:
                    if e.errno != errno.EINTR:
                        raise

        def __getattr__(self, attr):
            return getattr(self.queue, attr)
