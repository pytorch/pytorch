import os
import tempfile
import time


class FileBaton:
    '''A primitive, file-based synchronization utility.'''

    def __init__(self, lock_file_path, wait_seconds=0.1):
        '''
        Creates a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periorically sleep (spin) when
                calling ``wait()``.
        '''
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_acquire(self):
        '''
        Tries to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        '''
        # See `open(2)` man page for why we don't use `O_EXCL`.
        # TL;DR: this is more portable for different file systems.
        with tempfile.NamedTemporaryFile(
                mode='w',
                # Use the same directory so we decrease chances of
                # cross-device linking (which would fail).
                dir=os.path.dirname(self.lock_file_path),
        ) as tmp_f:
            try:
                os.link(tmp_f.name, self.lock_file_path)
                self.fd = os.open(self.lock_file_path, os.O_RDONLY)
                return True
            except FileExistsError:
                return os.stat(tmp_f.name).st_nlink == 2

    def wait(self):
        '''
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        '''
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        '''Releases the baton and removes its file.'''
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
