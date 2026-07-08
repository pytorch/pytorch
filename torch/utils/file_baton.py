# mypy: allow-untyped-defs
import os
import time
import warnings


class FileBaton:
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1, warn_after_seconds=None,
                 timeout_seconds=None) -> None:
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            warn_after_seconds: The seconds to wait before showing
                lock file path to warn existing lock file.
            timeout_seconds: If not ``None``, the maximum number of seconds
                ``wait()`` will block before raising ``RuntimeError``. This
                guards against a permanent deadlock when the process that
                acquired the lock was killed (e.g. by OOM, ``SIGKILL``, or a
                cancelled job) and never released the lock file. ``None``
                (the default) preserves the previous behavior of waiting
                indefinitely.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None
        self.warn_after_seconds = warn_after_seconds
        self.timeout_seconds = timeout_seconds

    def try_acquire(self) -> bool | None:
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self) -> None:
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        has_warned = False

        start_time = time.time()
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

            elapsed = time.time() - start_time

            if self.timeout_seconds is not None and elapsed > self.timeout_seconds:
                raise RuntimeError(
                    f'Timed out after {self.timeout_seconds:g} seconds waiting for '
                    f'lock file "{self.lock_file_path}". The process that acquired the '
                    f'lock may have been killed (e.g. by the OOM killer, SIGKILL, or a '
                    f'cancelled job) without releasing it, leaving a stale lock. Delete '
                    f'the lock file and retry.')

            if self.warn_after_seconds is not None:
                if elapsed > self.warn_after_seconds and not has_warned:
                    warnings.warn(f'Waited on lock file "{self.lock_file_path}" for '
                                  f'{self.warn_after_seconds} seconds.', stacklevel=2)
                    has_warned = True

    def release(self) -> None:
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
