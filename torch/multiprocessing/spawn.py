
from typing import Optional
import logging
import mmap
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import warnings

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)

RANK_STATUSES_FILE_NAME = "rank_statuses"


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


class ProcessContext:
    def __init__(self, processes, error_queues, envs):
        self.error_queues = error_queues
        self.processes = processes
        self.envs = envs
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
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
                process.terminate()
            process.join()

        def _get_status(b: int) -> str:
            if b == 0x00:
                return "Not stuck"
            elif b == 0x01:
                return "Stuck"
            elif b == 0x02:
                return "Timed out"
            elif b == 0xFF:
                return "Not set"
            else:
                return "Unknown"

        def _rank_statuses() -> str:
            msg = "\n"
            if self.envs:
                msg += "failed rank = %s\n" % self.envs.get(error_index, {}).get("RANK")
            else:
                msg += "failed local rank = %s\n" % error_index
            msg += "Rank statuses were:\n"
            try:
                with \
                    open(RANK_STATUSES_FILE_NAME, "r+b") as fd, \
                        mmap.mmap(fd.fileno(), len(self.processes), access=mmap.ACCESS_READ, offset=0) as mm:
                    for rank in range(len(self.processes)):
                        status = _get_status(mm[rank])
                        if self.envs:
                            msg += "(rank = %s, status = %s)\n" % (self.envs.get(rank, {}).get("RANK"), status)
                        else:
                            msg += "(local rank = %s, status = %s)\n" % (rank, status)
            except Exception:
                logger.exception("_rank_statuses exception")
            return msg

        def _remove_rank_statuses_file():
            try:
                os.remove(RANK_STATUSES_FILE_NAME)
            except Exception:
                logger.exception("_remove_rank_statuses_file exception")

        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        rank_statuses = _rank_statuses()
        # This is only necessary to make CI happy.
        # The CI asserts for working directory being not dirty.
        _remove_rank_statuses_file()
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s\n%s" %
                    (error_index, name, rank_statuses),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d\n%s" %
                    (error_index, exitcode, rank_statuses),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        msg += rank_statuses
        raise ProcessRaisedException(msg, error_index, failed_process.pid)


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues):
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super(SpawnContext, self).__init__(processes, error_queues, None)
    pass


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn', envs=None):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []

    def _create_rank_statuses_file():
        try:
            # create initial file
            with open(RANK_STATUSES_FILE_NAME, "w+b") as fd:
                fd.truncate()
                for i in range(nprocs):
                    fd.write(b'\xFF')
        except Exception:
            logger.exception("_create_rank_statuses_file exception")

    _create_rank_statuses_file()
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

    context = ProcessContext(processes, error_queues, envs)
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

    Args:
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
               ' torch.multiprocessing.start_processes(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
