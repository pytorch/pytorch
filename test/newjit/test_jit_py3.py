import sys
import torch
from common_utils import TestCase
from contextlib import contextmanager

WINDOWS = sys.platform == 'win32'

class TestScriptPy3(TestCase):
    @contextmanager
    def capture_stdout(self):
        # No idea how to capture stdout from C++ on Windows
        if WINDOWS:
            yield ['']
            return
        import os
        import fcntl
        import errno
        sys.stdout.flush()
        stdout_fd = os.dup(1)
        r, w = os.pipe()
        try:
            # Override stdout with r - dup is guaranteed to return the lowest free fd
            os.close(1)
            os.dup(w)

            captured_stdout = ['']
            yield captured_stdout
            sys.stdout.flush()  # Make sure that Python hasn't buffered anything

            # Do the ugly dance to read all the data that was written into the pipe
            fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK)
            total_stdout = ''
            while True:
                try:
                    total_stdout += os.read(r, 1000).decode('ascii')
                except OSError as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    break
            captured_stdout[0] = total_stdout
        finally:
            # Revert the change, and clean up all fds
            os.close(1)
            os.dup(stdout_fd)
            os.close(stdout_fd)
            os.close(r)
            os.close(w)

    def test_joined_str(self):
        def func(x):
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}") # noqa E999
            print(f"format blank")
            hi = 'hi'
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = torch.arange(4., requires_grad=True)
        # TODO: Add support for f-strings in string parser frontend
        # self.checkScript(func, [x], optimize=True, capture_output=True)

        with self.capture_stdout() as captured:
            out = func(x)

        scripted = torch.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        self.assertAlmostEqual(out, out_script)
        self.assertEqual(captured, captured_script)
