import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from threading import Thread
from time import monotonic, sleep
from typing import Any, IO

from torch._inductor.compile_worker import watchdog


_IN_TOPLEVEL_PROCESS = True


def popen_pass_fds(fds: Sequence[int]) -> tuple[list[int], dict[str, Any]]:
    """
    Popen kwargs that pass the pipe fds `fds` to the child, and the values to give
    it on its command line (see fd_from_parent). Windows has no pass_fds, so the
    child inherits just these OS handles instead.
    """
    if sys.platform != "win32":
        return list(fds), {"pass_fds": tuple(fds)}
    import msvcrt

    handles = [msvcrt.get_osfhandle(fd) for fd in fds]
    for handle in handles:
        os.set_handle_inheritable(handle, True)
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.lpAttributeList = {"handle_list": handles}
    return handles, {"startupinfo": startupinfo, "close_fds": True}


def fd_from_parent(value: int) -> int:
    if sys.platform != "win32":
        return value
    import msvcrt

    os.set_handle_inheritable(value, False)
    return msvcrt.open_osfhandle(value, 0)


def wait_readable_win32(pipe: IO[bytes], timeout: float) -> bool:
    """Wait for data or EOF on a pipe; Windows select() only supports sockets."""
    import _winapi
    import msvcrt

    handle = msvcrt.get_osfhandle(pipe.fileno())
    deadline = monotonic() + timeout
    while True:
        try:
            available, _ = _winapi.PeekNamedPipe(handle, 0)
        except OSError:
            return True  # broken pipe: a read returns EOF
        if available:
            return True
        if monotonic() >= deadline:
            return False
        sleep(0.001)


def parent_process_gone(orig_ppid: int) -> bool:
    if sys.platform != "win32":
        return orig_ppid != os.getppid()
    # Windows does not reparent orphans, and a venv python.exe launcher is the
    # real parent there, so check whether the original process still runs.
    import _winapi

    try:
        handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, orig_ppid)
    except OSError:
        return True
    try:
        return _winapi.WaitForSingleObject(handle, 0) != _winapi.WAIT_TIMEOUT
    finally:
        _winapi.CloseHandle(handle)


def in_toplevel_process() -> bool:
    global _IN_TOPLEVEL_PROCESS
    return _IN_TOPLEVEL_PROCESS


# If this process dies abnormally (e.g. segfault)
# it will not shut down the workers. Instead,
# the workers will have their parent reassigned to the
# init process. This launches a separate thread to
# watch for the worker getting reassigned,
# and cleans it up in this case.
#
# This function cannot be an inner function since otherwise mp_context="spawn" would
# not work for ProcessPoolExecutor since inner functions cannot be pickled.
def _async_compile_initializer(orig_ppid: int, close_fds: tuple[int, ...] = ()) -> None:
    # In fork mode a worker inherits the sidecar's entire fd table, including the
    # sidecar<->parent pipes. The worker must not keep those open: holding the
    # result pipe's write end would stop the parent from ever seeing EOF when the
    # sidecar dies. The caller only passes these fds under fork; under spawn the
    # worker doesn't inherit them and the same integers would name unrelated fds
    # the fresh interpreter reused, so closing them there would be silent
    # corruption (the OSError guard wouldn't catch it) rather than a no-op.
    for fd in close_fds:
        try:
            os.close(fd)
        except OSError:
            pass

    import torch._C

    def run() -> None:
        while True:
            sleep(60)
            if parent_process_gone(orig_ppid):
                os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

    global _watchdog_thread, _original_parent
    _original_parent = orig_ppid
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
    # Ignore Ctrl-C (i.e. SIGINT) sent to pool workers to avoid meaningless log spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Install a crash handler to print out the stacktrace for SEGV
    torch._C._initCrashHandler()

    # Set a bit to distinguish async_compile subprocesses from the toplevel process.
    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False

    # Claim a heartbeat slot so the sidecar watchdog can report this worker's
    # compile phase (no-op unless the sidecar allocated the shared buffer).
    watchdog.init_worker_slot()


_watchdog_thread: Thread | None = None
_original_parent: int | None = None


def has_parent_changed() -> bool:
    return _original_parent is None or parent_process_gone(_original_parent)
