from typing import Optional
import multiprocessing
import multiprocessing.connection
import signal
import time
import sys
import warnings

from . import _prctl_pr_set_pdeathsig


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.error_index = error_index
        self.pid = pid


class ProcessRaisedException(ProcessException):
    """
    Exception is thrown when the process failed due to exception
    raised by the code.
    """

    def __init__(
            self,
            msg: str,
            error_index: int,
            error_pid: int,
    ):
        super().__init__(msg, error_index, error_pid)


class ProcessExitedException(ProcessException):
    """
    Exception is thrown when the process failed due to signal
    or exited with a specific code.
    """
    __slots__ = ["exit_code"]

    def __init__(
            self, msg: str, error_index: int, error_pid: int,
            exit_code: int, signal_name: Optional[str] = None
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name


def _wrap(fn, i, args, error_queue):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


# Multiprocessing contexts are introduced at Python 3.4
_supports_context = sys.version_info >= (3, 4)


def _python_version_check():
    if not _supports_context:
        raise RuntimeError("Requires python 3.4 or higher to use "
                           "torch.multiprocessing.spawn and "
                           "torch.multiprocessing.ProcessContext helper "
                           "to launch multiple processes. If you are using "
                           "this for distributed training and have a lower "
                           "version of python, please use "
                           "torch.distributed.launch instead.")


class ProcessContext:
    def __init__(self, processes, error_queues):
        _python_version_check()
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }
        self.process_errors = {}

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def _try_populate_process_errors(self):
        for idx, queue in enumerate(self.error_queues):
            if not queue.empty():
                self.process_errors[idx] = queue.get()

    def _get_deadline(self, timeout):
        if timeout:
            return time.monotonic() + timeout
        else:
            return sys.maxsize

    def _busy_join(self, process, timeout=None):
        """
        Python multiprocessing.queues use pipes to communicate between processes.
        If the writer process writes long message, the pipe can hang forever if the
        reader process does not start reading it. In order to prevent this, the
        method follows: check_data(), join(period) pattern in comparison to:
        join(timeout), check_data().  The second pattern will deadlock the
        writer thread.
        """
        deadline = self._get_deadline(timeout)
        period = 1  # one second
        while True:
            self._try_populate_process_errors()
            process.join(period)
            if deadline - time.monotonic() < 0:
                break

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Arguments:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        deadline = self._get_deadline(timeout)
        period = 1  # one second
        ready = []
        process_errors = {}
        while True:
            self._try_populate_process_errors()
            # Wait for any process to fail or all of them to succeed.
            period_ready = multiprocessing.connection.wait(
                self.sentinels.keys(),
                timeout=period,
            )
            ready += period_ready
            if len(ready) == len(self.processes):
                # All processes finished
                break
            for process in self.processes:
                # At least one process got error
                if process.exitcode != 0:
                    break
            if deadline - time.monotonic() < 0:
                # timeout finished
                break

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            self._busy_join(process)
            if process.exitcode != 0:
                error_index = index
                break

        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0

        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process_errors.update(self._try_retrieve_errors())
                process.terminate()
            self._busy_join(process)

        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )

        original_trace = process_errors[error_index]
        # original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues):
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super(SpawnContext, self).__init__(self, processes, error_queues)

    pass


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    _python_version_check()
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Arguments:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (string): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    if start_method != 'spawn':
        msg = ('This method only supports start_method=spawn (got: %s).\n'
               'To use a different start_method use:\n\t\t'
               ' torch.multiprocessing.start_process(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
