from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import time

if sys.version < '3.3':
    # Note(jiayq): in Python 2, FileExistsError is not defined and the
    # error manifests it as OSError.
    FileExistsError = OSError


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
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self):
        '''
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        '''
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        '''Releaes the baton and removes its file.'''
        os.close(self.fd)
        os.remove(self.lock_file_path)
