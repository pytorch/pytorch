# mypy: allow-untyped-defs
import os
import time


class FileBaton:
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1):
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_acquire(self):
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

    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
