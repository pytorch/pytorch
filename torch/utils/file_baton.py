import os
import time
import psutil


class FileBaton:
    '''A primitive, file-based synchronization utility.'''
    _FILE_CONTENT_HEADER = b"PID:"

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
        self._try_clear_old_lock_file()
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(self.fd, self._FILE_CONTENT_HEADER + str.encode(str(os.getpid())))
            os.fsync(self.fd)
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
            self._try_clear_old_lock_file()
            time.sleep(self.wait_seconds)

    def release(self):
        '''Releases the baton and removes its file.'''
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)

    def _try_clear_old_lock_file(self):
        '''
        Try to detect lingering lock files and delete them.
        This function is NOT guaranteed to always detect them and only aims at solving
        the most basic case: another process died while holding the baton and was not
        able to remove the file.

        This function must NEVER clear the lock while there is a change a valid owner for
        the lock exists. In particular, it is important to keep in mind that this function
        might race with any step of the lock acquisition step.
        '''

        if os.path.exists(self.lock_file_path):
            remove = False
            try:
                with open(self.lock_file_path, 'rb') as f:
                    data = f.read()
                    if data.startswith(self._FILE_CONTENT_HEADER):
                        pid = data[len(self._FILE_CONTENT_HEADER):]
                        # If the file content is not valid, assume that it is still being written
                        # into to avoid any race while the main process is writing into it.
                        try:
                            pid = int(pid.decode())
                        except:
                            pass
                        else:
                            # Check if that process is running
                            # if any step of getting information about the process (like it died)
                            # in between the calls, just do nothing.
                            try:
                                if psutil.pid_exists(pid):
                                    p = psutil.Process(pid)
                                    if p.status() == psutil.STATUS_ZOMBIE:
                                        # Steal the lock from zombie processes
                                        remove = True
                                else:
                                    # The process died, it is ok to unlock the file
                                    remove = True
                            except:
                                pass
            finally:
                if remove:
                    os.remove(self.lock_file_path)
