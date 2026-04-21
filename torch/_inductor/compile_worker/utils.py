import os
import signal
from threading import Thread
from time import sleep


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
def _async_compile_initializer(orig_ppid: int) -> None:
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

    # Opt-in SIGUSR1 stack dumper for CI hang diagnosis; parent process (the
    # pytest test subprocess) enables this via test/conftest.py and propagates
    # the env var. On SIGUSR1 we dump all Python threads to stderr.
    if os.environ.get("PYTORCH_FAULT_HANDLER") == "1":
        import faulthandler
        import sys

        faulthandler.enable(file=sys.stderr, all_threads=True)
        if hasattr(faulthandler, "register"):
            faulthandler.register(
                signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False
            )

    # Set a bit to distinguish async_compile subprocesses from the toplevel process.
    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False


_watchdog_thread: Thread | None = None
_original_parent: int | None = None


def has_parent_changed() -> bool:
    return _original_parent != os.getppid()
