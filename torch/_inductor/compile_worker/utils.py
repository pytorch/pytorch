import os
import signal
from threading import Thread
from time import sleep

from torch._inductor.compile_worker import watchdog


_IN_TOPLEVEL_PROCESS = True


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
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)

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
    return _original_parent != os.getppid()
