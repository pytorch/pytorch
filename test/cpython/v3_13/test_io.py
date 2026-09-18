# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_io.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

"""Unit tests for the io module."""

# Tests of io are scattered over the test suite:
# * test_bufio - tests file buffering
# * test_memoryio - tests BytesIO and StringIO
# * test_fileio - tests FileIO
# * test_file - tests the file interface
# * test_io - tests everything else in the io module
# * test_univnewlines - tests universal newline support
# * test_largefile - tests operations on a file greater than 2**32 bytes
#     (only enabled with -ulargefile)

################################################################################
# ATTENTION TEST WRITERS!!!
################################################################################
# When writing tests for io, it's important to test both the C and Python
# implementations. This is usually done by writing a base test that refers to
# the type it is testing as an attribute. Then it provides custom subclasses to
# test both implementations. This file has lots of examples.
################################################################################

import abc
import array
import errno
import locale
import os
import pickle
import random
import signal
import sys
import textwrap
import threading
import time
import unittest
import warnings
import weakref
from collections import deque, UserList
from itertools import cycle, count
from test import support
from test.support.script_helper import (
    assert_python_ok, assert_python_failure, run_python_until_end)
from test.support import (
    import_helper, is_apple, os_helper, threading_helper, warnings_helper,
)
from test.support.os_helper import FakePath

import codecs
import io  # C implementation of io
import _pyio as pyio # Python implementation of io

try:
    import ctypes
except ImportError:
    def byteslike(*pos, **kw):
        return array.array("b", bytes(*pos, **kw))
else:
    def byteslike(*pos, **kw):
        """Create a bytes-like object having no string or sequence methods"""
        data = bytes(*pos, **kw)
        obj = EmptyStruct()
        ctypes.resize(obj, len(data))
        memoryview(obj).cast("B")[:] = data
        return obj
    class EmptyStruct(ctypes.Structure):
        pass


def _default_chunk_size():
    """Get the default TextIOWrapper chunk size"""
    with open(__file__, "r", encoding="latin-1") as f:
        return f._CHUNK_SIZE

requires_alarm = unittest.skipUnless(
    hasattr(signal, "alarm"), "test requires signal.alarm()"
)


class BadIndex:
    def __index__(self):
        1/0

class MockRawIOWithoutRead:
    """A RawIO implementation without read(), so as to exercise the default
    RawIO.read() which calls readinto()."""

    def __init__(self, read_stack=()):
        self._read_stack = list(read_stack)
        self._write_stack = []
        self._reads = 0
        self._extraneous_reads = 0

    def write(self, b):
        self._write_stack.append(bytes(b))
        return len(b)

    def writable(self):
        return True

    def fileno(self):
        return 42

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence):
        return 0   # wrong but we gotta return something

    def tell(self):
        return 0   # same comment as above

    def readinto(self, buf):
        self._reads += 1
        max_len = len(buf)
        try:
            data = self._read_stack[0]
        except IndexError:
            self._extraneous_reads += 1
            return 0
        if data is None:
            del self._read_stack[0]
            return None
        n = len(data)
        if len(data) <= max_len:
            del self._read_stack[0]
            buf[:n] = data
            return n
        else:
            buf[:] = data[:max_len]
            self._read_stack[0] = data[max_len:]
            return max_len

    def truncate(self, pos=None):
        return pos

class CMockRawIOWithoutRead(MockRawIOWithoutRead, io.RawIOBase):
    pass

class PyMockRawIOWithoutRead(MockRawIOWithoutRead, pyio.RawIOBase):
    pass


class MockRawIO(MockRawIOWithoutRead):

    def read(self, n=None):
        self._reads += 1
        try:
            return self._read_stack.pop(0)
        except:
            self._extraneous_reads += 1
            return b""

class CMockRawIO(MockRawIO, io.RawIOBase):
    pass

class PyMockRawIO(MockRawIO, pyio.RawIOBase):
    pass


class MisbehavedRawIO(MockRawIO):
    def write(self, b):
        return super().write(b) * 2

    def read(self, n=None):
        return super().read(n) * 2

    def seek(self, pos, whence):
        return -123

    def tell(self):
        return -456

    def readinto(self, buf):
        super().readinto(buf)
        return len(buf) * 5

class CMisbehavedRawIO(MisbehavedRawIO, io.RawIOBase):
    pass

class PyMisbehavedRawIO(MisbehavedRawIO, pyio.RawIOBase):
    pass


class SlowFlushRawIO(MockRawIO):
    def __init__(self):
        super().__init__()
        self.in_flush = threading.Event()

    def flush(self):
        self.in_flush.set()
        time.sleep(0.25)

class CSlowFlushRawIO(SlowFlushRawIO, io.RawIOBase):
    pass

class PySlowFlushRawIO(SlowFlushRawIO, pyio.RawIOBase):
    pass


class CloseFailureIO(MockRawIO):
    closed = 0

    def close(self):
        if not self.closed:
            self.closed = 1
            raise OSError

class CCloseFailureIO(CloseFailureIO, io.RawIOBase):
    pass

class PyCloseFailureIO(CloseFailureIO, pyio.RawIOBase):
    pass


class MockFileIO:

    def __init__(self, data):
        self.read_history = []
        super().__init__(data)

    def read(self, n=None):
        res = super().read(n)
        self.read_history.append(None if res is None else len(res))
        return res

    def readinto(self, b):
        res = super().readinto(b)
        self.read_history.append(res)
        return res

class CMockFileIO(MockFileIO, io.BytesIO):
    pass

class PyMockFileIO(MockFileIO, pyio.BytesIO):
    pass


class MockUnseekableIO:
    def seekable(self):
        return False

    def seek(self, *args):
        raise self.UnsupportedOperation("not seekable")

    def tell(self, *args):
        raise self.UnsupportedOperation("not seekable")

    def truncate(self, *args):
        raise self.UnsupportedOperation("not seekable")

class CMockUnseekableIO(MockUnseekableIO, io.BytesIO):
    UnsupportedOperation = io.UnsupportedOperation

class PyMockUnseekableIO(MockUnseekableIO, pyio.BytesIO):
    UnsupportedOperation = pyio.UnsupportedOperation


class MockCharPseudoDevFileIO(MockFileIO):
    # GH-95782
    # ftruncate() does not work on these special files (and CPython then raises
    # appropriate exceptions), so truncate() does not have to be accounted for
    # here.
    def __init__(self, data):
        super().__init__(data)

    def seek(self, *args):
        return 0

    def tell(self, *args):
        return 0

class CMockCharPseudoDevFileIO(MockCharPseudoDevFileIO, io.BytesIO):
    pass

class PyMockCharPseudoDevFileIO(MockCharPseudoDevFileIO, pyio.BytesIO):
    pass


class MockNonBlockWriterIO:

    def __init__(self):
        self._write_stack = []
        self._blocker_char = None

    def pop_written(self):
        s = b"".join(self._write_stack)
        self._write_stack[:] = []
        return s

    def block_on(self, char):
        """Block when a given char is encountered."""
        self._blocker_char = char

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        # naive implementation, enough for tests
        return 0

    def writable(self):
        return True

    def write(self, b):
        b = bytes(b)
        n = -1
        if self._blocker_char:
            try:
                n = b.index(self._blocker_char)
            except ValueError:
                pass
            else:
                if n > 0:
                    # write data up to the first blocker
                    self._write_stack.append(b[:n])
                    return n
                else:
                    # cancel blocker and indicate would block
                    self._blocker_char = None
                    return None
        self._write_stack.append(b)
        return len(b)

class CMockNonBlockWriterIO(MockNonBlockWriterIO, io.RawIOBase):
    BlockingIOError = io.BlockingIOError

class PyMockNonBlockWriterIO(MockNonBlockWriterIO, pyio.RawIOBase):
    BlockingIOError = pyio.BlockingIOError


class IOTest(CPythonTestCase):

    def setUp(self):
        os_helper.unlink(os_helper.TESTFN)

    def tearDown(self):
        os_helper.unlink(os_helper.TESTFN)

    def write_ops(self, f):
        self.assertEqual(f.write(b"blah."), 5)
        f.truncate(0)
        self.assertEqual(f.tell(), 5)
        f.seek(0)

        self.assertEqual(f.write(b"blah."), 5)
        self.assertEqual(f.seek(0), 0)
        self.assertEqual(f.write(b"Hello."), 6)
        self.assertEqual(f.tell(), 6)
        self.assertEqual(f.seek(-1, 1), 5)
        self.assertEqual(f.tell(), 5)
        buffer = bytearray(b" world\n\n\n")
        self.assertEqual(f.write(buffer), 9)
        buffer[:] = b"*" * 9  # Overwrite our copy of the data
        self.assertEqual(f.seek(0), 0)
        self.assertEqual(f.write(b"h"), 1)
        self.assertEqual(f.seek(-1, 2), 13)
        self.assertEqual(f.tell(), 13)

        self.assertEqual(f.truncate(12), 12)
        self.assertEqual(f.tell(), 13)
        self.assertRaises(TypeError, f.seek, 0.0)

    def read_ops(self, f, buffered=False):
        data = f.read(5)
        self.assertEqual(data, b"hello")
        data = byteslike(data)
        self.assertEqual(f.readinto(data), 5)
        self.assertEqual(bytes(data), b" worl")
        data = bytearray(5)
        self.assertEqual(f.readinto(data), 2)
        self.assertEqual(len(data), 5)
        self.assertEqual(data[:2], b"d\n")
        self.assertEqual(f.seek(0), 0)
        self.assertEqual(f.read(20), b"hello world\n")
        self.assertEqual(f.read(1), b"")
        self.assertEqual(f.readinto(byteslike(b"x")), 0)
        self.assertEqual(f.seek(-6, 2), 6)
        self.assertEqual(f.read(5), b"world")
        self.assertEqual(f.read(0), b"")
        self.assertEqual(f.readinto(byteslike()), 0)
        self.assertEqual(f.seek(-6, 1), 5)
        self.assertEqual(f.read(5), b" worl")
        self.assertEqual(f.tell(), 10)
        self.assertRaises(TypeError, f.seek, 0.0)
        if buffered:
            f.seek(0)
            self.assertEqual(f.read(), b"hello world\n")
            f.seek(6)
            self.assertEqual(f.read(), b"world\n")
            self.assertEqual(f.read(), b"")
            f.seek(0)
            data = byteslike(5)
            self.assertEqual(f.readinto1(data), 5)
            self.assertEqual(bytes(data), b"hello")

    LARGE = 2**31

    def large_file_ops(self, f):
        assert f.readable()
        assert f.writable()
        try:
            self.assertEqual(f.seek(self.LARGE), self.LARGE)
        except (OverflowError, ValueError):
            self.skipTest("no largefile support")
        self.assertEqual(f.tell(), self.LARGE)
        self.assertEqual(f.write(b"xxx"), 3)
        self.assertEqual(f.tell(), self.LARGE + 3)
        self.assertEqual(f.seek(-1, 1), self.LARGE + 2)
        self.assertEqual(f.truncate(), self.LARGE + 2)
        self.assertEqual(f.tell(), self.LARGE + 2)
        self.assertEqual(f.seek(0, 2), self.LARGE + 2)
        self.assertEqual(f.truncate(self.LARGE + 1), self.LARGE + 1)
        self.assertEqual(f.tell(), self.LARGE + 2)
        self.assertEqual(f.seek(0, 2), self.LARGE + 1)
        self.assertEqual(f.seek(-1, 2), self.LARGE)
        self.assertEqual(f.read(2), b"x")

    def test_invalid_operations(self):
        # Try writing on a file opened in read mode and vice-versa.
        exc = self.UnsupportedOperation
        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as fp:
            self.assertRaises(exc, fp.read)
            self.assertRaises(exc, fp.readline)
        with self.open(os_helper.TESTFN, "wb") as fp:
            self.assertRaises(exc, fp.read)
            self.assertRaises(exc, fp.readline)
        with self.open(os_helper.TESTFN, "wb", buffering=0) as fp:
            self.assertRaises(exc, fp.read)
            self.assertRaises(exc, fp.readline)
        with self.open(os_helper.TESTFN, "rb", buffering=0) as fp:
            self.assertRaises(exc, fp.write, b"blah")
            self.assertRaises(exc, fp.writelines, [b"blah\n"])
        with self.open(os_helper.TESTFN, "rb") as fp:
            self.assertRaises(exc, fp.write, b"blah")
            self.assertRaises(exc, fp.writelines, [b"blah\n"])
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as fp:
            self.assertRaises(exc, fp.write, "blah")
            self.assertRaises(exc, fp.writelines, ["blah\n"])
            # Non-zero seeking from current or end pos
            self.assertRaises(exc, fp.seek, 1, self.SEEK_CUR)
            self.assertRaises(exc, fp.seek, -1, self.SEEK_END)

    @unittest.skipIf(
        support.is_emscripten, "fstat() of a pipe fd is not supported"
    )
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_optional_abilities(self):
        # Test for OSError when optional APIs are not supported
        # The purpose of this test is to try fileno(), reading, writing and
        # seeking operations with various objects that indicate they do not
        # support these operations.

        def pipe_reader():
            [r, w] = os.pipe()
            os.close(w)  # So that read() is harmless
            return self.FileIO(r, "r")

        def pipe_writer():
            [r, w] = os.pipe()
            self.addCleanup(os.close, r)
            # Guarantee that we can write into the pipe without blocking
            thread = threading.Thread(target=os.read, args=(r, 100))
            thread.start()
            self.addCleanup(thread.join)
            return self.FileIO(w, "w")

        def buffered_reader():
            return self.BufferedReader(self.MockUnseekableIO())

        def buffered_writer():
            return self.BufferedWriter(self.MockUnseekableIO())

        def buffered_random():
            return self.BufferedRandom(self.BytesIO())

        def buffered_rw_pair():
            return self.BufferedRWPair(self.MockUnseekableIO(),
                self.MockUnseekableIO())

        def text_reader():
            class UnseekableReader(self.MockUnseekableIO):
                writable = self.BufferedIOBase.writable
                write = self.BufferedIOBase.write
            return self.TextIOWrapper(UnseekableReader(), "ascii")

        def text_writer():
            class UnseekableWriter(self.MockUnseekableIO):
                readable = self.BufferedIOBase.readable
                read = self.BufferedIOBase.read
            return self.TextIOWrapper(UnseekableWriter(), "ascii")

        tests = (
            (pipe_reader, "fr"), (pipe_writer, "fw"),
            (buffered_reader, "r"), (buffered_writer, "w"),
            (buffered_random, "rws"), (buffered_rw_pair, "rw"),
            (text_reader, "r"), (text_writer, "w"),
            (self.BytesIO, "rws"), (self.StringIO, "rws"),
        )
        for [test, abilities] in tests:
            with self.subTest(test), test() as obj:
                readable = "r" in abilities
                self.assertEqual(obj.readable(), readable)
                writable = "w" in abilities
                self.assertEqual(obj.writable(), writable)

                if isinstance(obj, self.TextIOBase):
                    data = "3"
                elif isinstance(obj, (self.BufferedIOBase, self.RawIOBase)):
                    data = b"3"
                else:
                    self.fail("Unknown base class")

                if "f" in abilities:
                    obj.fileno()
                else:
                    self.assertRaises(OSError, obj.fileno)

                if readable:
                    obj.read(1)
                    obj.read()
                else:
                    self.assertRaises(OSError, obj.read, 1)
                    self.assertRaises(OSError, obj.read)

                if writable:
                    obj.write(data)
                else:
                    self.assertRaises(OSError, obj.write, data)

                if sys.platform.startswith("win") and test in (
                        pipe_reader, pipe_writer):
                    # Pipes seem to appear as seekable on Windows
                    continue
                seekable = "s" in abilities
                self.assertEqual(obj.seekable(), seekable)

                if seekable:
                    obj.tell()
                    obj.seek(0)
                else:
                    self.assertRaises(OSError, obj.tell)
                    self.assertRaises(OSError, obj.seek, 0)

                if writable and seekable:
                    obj.truncate()
                    obj.truncate(0)
                else:
                    self.assertRaises(OSError, obj.truncate)
                    self.assertRaises(OSError, obj.truncate, 0)

    def test_open_handles_NUL_chars(self):
        fn_with_NUL = 'foo\0bar'
        self.assertRaises(ValueError, self.open, fn_with_NUL, 'w', encoding="utf-8")

        bytes_fn = bytes(fn_with_NUL, 'ascii')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertRaises(ValueError, self.open, bytes_fn, 'w', encoding="utf-8")

    def test_raw_file_io(self):
        with self.open(os_helper.TESTFN, "wb", buffering=0) as f:
            self.assertEqual(f.readable(), False)
            self.assertEqual(f.writable(), True)
            self.assertEqual(f.seekable(), True)
            self.write_ops(f)
        with self.open(os_helper.TESTFN, "rb", buffering=0) as f:
            self.assertEqual(f.readable(), True)
            self.assertEqual(f.writable(), False)
            self.assertEqual(f.seekable(), True)
            self.read_ops(f)

    def test_buffered_file_io(self):
        with self.open(os_helper.TESTFN, "wb") as f:
            self.assertEqual(f.readable(), False)
            self.assertEqual(f.writable(), True)
            self.assertEqual(f.seekable(), True)
            self.write_ops(f)
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.readable(), True)
            self.assertEqual(f.writable(), False)
            self.assertEqual(f.seekable(), True)
            self.read_ops(f, True)

    def test_readline(self):
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(b"abc\ndef\nxyzzy\nfoo\x00bar\nanother line")
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.readline(), b"abc\n")
            self.assertEqual(f.readline(10), b"def\n")
            self.assertEqual(f.readline(2), b"xy")
            self.assertEqual(f.readline(4), b"zzy\n")
            self.assertEqual(f.readline(), b"foo\x00bar\n")
            self.assertEqual(f.readline(None), b"another line")
            self.assertRaises(TypeError, f.readline, 5.3)
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as f:
            self.assertRaises(TypeError, f.readline, 5.3)

    def test_readline_nonsizeable(self):
        # Issue #30061
        # Crash when readline() returns an object without __len__
        class R(self.IOBase):
            def readline(self):
                return None
        self.assertRaises((TypeError, StopIteration), next, R())

    def test_next_nonsizeable(self):
        # Issue #30061
        # Crash when __next__() returns an object without __len__
        class R(self.IOBase):
            def __next__(self):
                return None
        self.assertRaises(TypeError, R().readlines, 1)

    def test_raw_bytes_io(self):
        f = self.BytesIO()
        self.write_ops(f)
        data = f.getvalue()
        self.assertEqual(data, b"hello world\n")
        f = self.BytesIO(data)
        self.read_ops(f, True)

    def test_large_file_ops(self):
        # On Windows and Apple platforms this test consumes large resources; It
        # takes a long time to build the >2 GiB file and takes >2 GiB of disk
        # space therefore the resource must be enabled to run this test.
        if sys.platform[:3] == 'win' or is_apple:
            support.requires(
                'largefile',
                'test requires %s bytes and a long time to run' % self.LARGE)
        with self.open(os_helper.TESTFN, "w+b", 0) as f:
            self.large_file_ops(f)
        with self.open(os_helper.TESTFN, "w+b") as f:
            self.large_file_ops(f)

    def test_with_open(self):
        for bufsize in (0, 100):
            with self.open(os_helper.TESTFN, "wb", bufsize) as f:
                f.write(b"xxx")
            self.assertEqual(f.closed, True)
            try:
                with self.open(os_helper.TESTFN, "wb", bufsize) as f:
                    1/0
            except ZeroDivisionError:
                self.assertEqual(f.closed, True)
            else:
                self.fail("1/0 didn't raise an exception")

    # issue 5008
    def test_append_mode_tell(self):
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(b"xxx")
        with self.open(os_helper.TESTFN, "ab", buffering=0) as f:
            self.assertEqual(f.tell(), 3)
        with self.open(os_helper.TESTFN, "ab") as f:
            self.assertEqual(f.tell(), 3)
        with self.open(os_helper.TESTFN, "a", encoding="utf-8") as f:
            self.assertGreater(f.tell(), 0)

    def test_destructor(self):
        record = []
        class MyFileIO(self.FileIO):
            def __del__(self):
                record.append(1)
                try:
                    f = super().__del__
                except AttributeError:
                    pass
                else:
                    f()
            def close(self):
                record.append(2)
                super().close()
            def flush(self):
                record.append(3)
                super().flush()
        with warnings_helper.check_warnings(('', ResourceWarning)):
            f = MyFileIO(os_helper.TESTFN, "wb")
            f.write(b"xxx")
            del f
            support.gc_collect()
            self.assertEqual(record, [1, 2, 3])
            with self.open(os_helper.TESTFN, "rb") as f:
                self.assertEqual(f.read(), b"xxx")

    def _check_base_destructor(self, base):
        record = []
        class MyIO(base):
            def __init__(self):
                # This exercises the availability of attributes on object
                # destruction.
                # (in the C version, close() is called by the tp_dealloc
                # function, not by __del__)
                self.on_del = 1
                self.on_close = 2
                self.on_flush = 3
            def __del__(self):
                record.append(self.on_del)
                try:
                    f = super().__del__
                except AttributeError:
                    pass
                else:
                    f()
            def close(self):
                record.append(self.on_close)
                super().close()
            def flush(self):
                record.append(self.on_flush)
                super().flush()
        f = MyIO()
        del f
        support.gc_collect()
        self.assertEqual(record, [1, 2, 3])

    def test_IOBase_destructor(self):
        self._check_base_destructor(self.IOBase)

    def test_RawIOBase_destructor(self):
        self._check_base_destructor(self.RawIOBase)

    def test_BufferedIOBase_destructor(self):
        self._check_base_destructor(self.BufferedIOBase)

    def test_TextIOBase_destructor(self):
        self._check_base_destructor(self.TextIOBase)

    def test_close_flushes(self):
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(b"xxx")
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"xxx")

    def test_array_writes(self):
        a = array.array('i', range(10))
        n = len(a.tobytes())
        def check(f):
            with f:
                self.assertEqual(f.write(a), n)
                f.writelines((a,))
        check(self.BytesIO())
        check(self.FileIO(os_helper.TESTFN, "w"))
        check(self.BufferedWriter(self.MockRawIO()))
        check(self.BufferedRandom(self.MockRawIO()))
        check(self.BufferedRWPair(self.MockRawIO(), self.MockRawIO()))

    def test_closefd(self):
        self.assertRaises(ValueError, self.open, os_helper.TESTFN, 'w',
                          encoding="utf-8", closefd=False)

    def test_read_closed(self):
        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as f:
            f.write("egg\n")
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as f:
            file = self.open(f.fileno(), "r", encoding="utf-8", closefd=False)
            self.assertEqual(file.read(), "egg\n")
            file.seek(0)
            file.close()
            self.assertRaises(ValueError, file.read)
        with self.open(os_helper.TESTFN, "rb") as f:
            file = self.open(f.fileno(), "rb", closefd=False)
            self.assertEqual(file.read()[:3], b"egg")
            file.close()
            self.assertRaises(ValueError, file.readinto, bytearray(1))

    def test_no_closefd_with_filename(self):
        # can't use closefd in combination with a file name
        self.assertRaises(ValueError, self.open, os_helper.TESTFN, "r",
                          encoding="utf-8", closefd=False)

    def test_closefd_attr(self):
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(b"egg\n")
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as f:
            self.assertEqual(f.buffer.raw.closefd, True)
            file = self.open(f.fileno(), "r", encoding="utf-8", closefd=False)
            self.assertEqual(file.buffer.raw.closefd, False)

    def test_garbage_collection(self):
        # FileIO objects are collected, and collecting them flushes
        # all data to disk.
        with warnings_helper.check_warnings(('', ResourceWarning)):
            f = self.FileIO(os_helper.TESTFN, "wb")
            f.write(b"abcxxx")
            f.f = f
            wr = weakref.ref(f)
            del f
            support.gc_collect()
        self.assertIsNone(wr(), wr)
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"abcxxx")

    def test_unbounded_file(self):
        # Issue #1174606: reading from an unbounded stream such as /dev/zero.
        zero = "/dev/zero"
        if not os.path.exists(zero):
            self.skipTest("{0} does not exist".format(zero))
        if sys.maxsize > 0x7FFFFFFF:
            self.skipTest("test can only run in a 32-bit address space")
        if support.real_max_memuse < support._2G:
            self.skipTest("test requires at least 2 GiB of memory")
        with self.open(zero, "rb", buffering=0) as f:
            self.assertRaises(OverflowError, f.read)
        with self.open(zero, "rb") as f:
            self.assertRaises(OverflowError, f.read)
        with self.open(zero, "r") as f:
            self.assertRaises(OverflowError, f.read)

    def check_flush_error_on_close(self, *args, **kwargs):
        # Test that the file is closed despite failed flush
        # and that flush() is called before file closed.
        f = self.open(*args, **kwargs)
        closed = []
        def bad_flush():
            closed[:] = [f.closed]
            raise OSError()
        f.flush = bad_flush
        self.assertRaises(OSError, f.close) # exception not swallowed
        self.assertTrue(f.closed)
        self.assertTrue(closed)      # flush() called
        self.assertFalse(closed[0])  # flush() called before file closed
        f.flush = lambda: None  # break reference loop

    def test_flush_error_on_close(self):
        # raw file
        # Issue #5700: io.FileIO calls flush() after file closed
        self.check_flush_error_on_close(os_helper.TESTFN, 'wb', buffering=0)
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'wb', buffering=0)
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'wb', buffering=0, closefd=False)
        os.close(fd)
        # buffered io
        self.check_flush_error_on_close(os_helper.TESTFN, 'wb')
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'wb')
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'wb', closefd=False)
        os.close(fd)
        # text io
        self.check_flush_error_on_close(os_helper.TESTFN, 'w', encoding="utf-8")
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'w', encoding="utf-8")
        fd = os.open(os_helper.TESTFN, os.O_WRONLY|os.O_CREAT)
        self.check_flush_error_on_close(fd, 'w', encoding="utf-8", closefd=False)
        os.close(fd)

    def test_multi_close(self):
        f = self.open(os_helper.TESTFN, "wb", buffering=0)
        f.close()
        f.close()
        f.close()
        self.assertRaises(ValueError, f.flush)

    def test_RawIOBase_read(self):
        # Exercise the default limited RawIOBase.read(n) implementation (which
        # calls readinto() internally).
        rawio = self.MockRawIOWithoutRead((b"abc", b"d", None, b"efg", None))
        self.assertEqual(rawio.read(2), b"ab")
        self.assertEqual(rawio.read(2), b"c")
        self.assertEqual(rawio.read(2), b"d")
        self.assertEqual(rawio.read(2), None)
        self.assertEqual(rawio.read(2), b"ef")
        self.assertEqual(rawio.read(2), b"g")
        self.assertEqual(rawio.read(2), None)
        self.assertEqual(rawio.read(2), b"")

    def test_types_have_dict(self):
        test = (
            self.IOBase(),
            self.RawIOBase(),
            self.TextIOBase(),
            self.StringIO(),
            self.BytesIO()
        )
        for obj in test:
            self.assertTrue(hasattr(obj, "__dict__"))

    def test_opener(self):
        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as f:
            f.write("egg\n")
        fd = os.open(os_helper.TESTFN, os.O_RDONLY)
        def opener(path, flags):
            return fd
        with self.open("non-existent", "r", encoding="utf-8", opener=opener) as f:
            self.assertEqual(f.read(), "egg\n")

    def test_bad_opener_negative_1(self):
        # Issue #27066.
        def badopener(fname, flags):
            return -1
        with self.assertRaises(ValueError) as cm:
            self.open('non-existent', 'r', opener=badopener)
        self.assertEqual(str(cm.exception), 'opener returned -1')

    def test_bad_opener_other_negative(self):
        # Issue #27066.
        def badopener(fname, flags):
            return -2
        with self.assertRaises(ValueError) as cm:
            self.open('non-existent', 'r', opener=badopener)
        self.assertEqual(str(cm.exception), 'opener returned -2')

    def test_opener_invalid_fd(self):
        # Check that OSError is raised with error code EBADF if the
        # opener returns an invalid file descriptor (see gh-82212).
        fd = os_helper.make_bad_fd()
        with self.assertRaises(OSError) as cm:
            self.open('foo', opener=lambda name, flags: fd)
        self.assertEqual(cm.exception.errno, errno.EBADF)

    def test_fileio_closefd(self):
        # Issue #4841
        with self.open(__file__, 'rb') as f1, \
             self.open(__file__, 'rb') as f2:
            fileio = self.FileIO(f1.fileno(), closefd=False)
            # .__init__() must not close f1
            fileio.__init__(f2.fileno(), closefd=False)
            f1.readline()
            # .close() must not close f2
            fileio.close()
            f2.readline()

    def test_nonbuffered_textio(self):
        with warnings_helper.check_no_resource_warning(self):
            with self.assertRaises(ValueError):
                self.open(os_helper.TESTFN, 'w', encoding="utf-8", buffering=0)

    def test_invalid_newline(self):
        with warnings_helper.check_no_resource_warning(self):
            with self.assertRaises(ValueError):
                self.open(os_helper.TESTFN, 'w', encoding="utf-8", newline='invalid')

    def test_buffered_readinto_mixin(self):
        # Test the implementation provided by BufferedIOBase
        class Stream(self.BufferedIOBase):
            def read(self, size):
                return b"12345"
            read1 = read
        stream = Stream()
        for method in ("readinto", "readinto1"):
            with self.subTest(method):
                buffer = byteslike(5)
                self.assertEqual(getattr(stream, method)(buffer), 5)
                self.assertEqual(bytes(buffer), b"12345")

    def test_fspath_support(self):
        def check_path_succeeds(path):
            with self.open(path, "w", encoding="utf-8") as f:
                f.write("egg\n")

            with self.open(path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "egg\n")

        check_path_succeeds(FakePath(os_helper.TESTFN))
        check_path_succeeds(FakePath(os.fsencode(os_helper.TESTFN)))

        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as f:
            bad_path = FakePath(f.fileno())
            with self.assertRaises(TypeError):
                self.open(bad_path, 'w', encoding="utf-8")

        bad_path = FakePath(None)
        with self.assertRaises(TypeError):
            self.open(bad_path, 'w', encoding="utf-8")

        bad_path = FakePath(FloatingPointError)
        with self.assertRaises(FloatingPointError):
            self.open(bad_path, 'w', encoding="utf-8")

        # ensure that refcounting is correct with some error conditions
        with self.assertRaisesRegex(ValueError, 'read/write/append mode'):
            self.open(FakePath(os_helper.TESTFN), 'rwxa', encoding="utf-8")

    def test_RawIOBase_readall(self):
        # Exercise the default unlimited RawIOBase.read() and readall()
        # implementations.
        rawio = self.MockRawIOWithoutRead((b"abc", b"d", b"efg"))
        self.assertEqual(rawio.read(), b"abcdefg")
        rawio = self.MockRawIOWithoutRead((b"abc", b"d", b"efg"))
        self.assertEqual(rawio.readall(), b"abcdefg")

    def test_BufferedIOBase_readinto(self):
        # Exercise the default BufferedIOBase.readinto() and readinto1()
        # implementations (which call read() or read1() internally).
        class Reader(self.BufferedIOBase):
            def __init__(self, avail):
                self.avail = avail
            def read(self, size):
                result = self.avail[:size]
                self.avail = self.avail[size:]
                return result
            def read1(self, size):
                """Returns no more than 5 bytes at once"""
                return self.read(min(size, 5))
        tests = (
            # (test method, total data available, read buffer size, expected
            #     read size)
            ("readinto", 10, 5, 5),
            ("readinto", 10, 6, 6),  # More than read1() can return
            ("readinto", 5, 6, 5),  # Buffer larger than total available
            ("readinto", 6, 7, 6),
            ("readinto", 10, 0, 0),  # Empty buffer
            ("readinto1", 10, 5, 5),  # Result limited to single read1() call
            ("readinto1", 10, 6, 5),  # Buffer larger than read1() can return
            ("readinto1", 5, 6, 5),  # Buffer larger than total available
            ("readinto1", 6, 7, 5),
            ("readinto1", 10, 0, 0),  # Empty buffer
        )
        UNUSED_BYTE = 0x81
        for test in tests:
            with self.subTest(test):
                method, avail, request, result = test
                reader = Reader(bytes(range(avail)))
                buffer = bytearray((UNUSED_BYTE,) * request)
                method = getattr(reader, method)
                self.assertEqual(method(buffer), result)
                self.assertEqual(len(buffer), request)
                self.assertSequenceEqual(buffer[:result], range(result))
                unused = (UNUSED_BYTE,) * (request - result)
                self.assertSequenceEqual(buffer[result:], unused)
                self.assertEqual(len(reader.avail), avail - result)

    def test_close_assert(self):
        class R(self.IOBase):
            def __setattr__(self, name, value):
                pass
            def flush(self):
                raise OSError()
        f = R()
        # This would cause an assertion failure.
        self.assertRaises(OSError, f.close)

        # Silence destructor error
        R.flush = lambda self: None

    @threading_helper.requires_working_threading()
    def test_write_readline_races(self):
        # gh-134908: Concurrent iteration over a file caused races
        thread_count = 2
        write_count = 100
        read_count = 100

        def writer(file, barrier):
            barrier.wait()
            for _ in range(write_count):
                file.write("x")

        def reader(file, barrier):
            barrier.wait()
            for _ in range(read_count):
                for line in file:
                    self.assertEqual(line, "")

        with self.open(os_helper.TESTFN, "w+") as f:
            barrier = threading.Barrier(thread_count + 1)
            reader = threading.Thread(target=reader, args=(f, barrier))
            writers = [threading.Thread(target=writer, args=(f, barrier))
                       for _ in range(thread_count)]
            with threading_helper.catch_threading_exception() as cm:
                with threading_helper.start_threads(writers + [reader]):
                    pass
                self.assertIsNone(cm.exc_type)

        self.assertEqual(os.stat(os_helper.TESTFN).st_size,
                         write_count * thread_count)


class CIOTest(IOTest):

    def test_IOBase_finalize(self):
        # Issue #12149: segmentation fault on _PyIOBase_finalize when both a
        # class which inherits IOBase and an object of this class are caught
        # in a reference cycle and close() is already in the method cache.
        class MyIO(self.IOBase):
            def close(self):
                pass

        # create an instance to populate the method cache
        MyIO()
        obj = MyIO()
        obj.obj = obj
        wr = weakref.ref(obj)
        del MyIO
        del obj
        support.gc_collect()
        self.assertIsNone(wr(), wr)

@support.cpython_only
class TestIOCTypes(__TestCase):
    def setUp(self):
        super().setUp()
        _io = import_helper.import_module("_io")
        self.types = [
            _io.BufferedRWPair,
            _io.BufferedRandom,
            _io.BufferedReader,
            _io.BufferedWriter,
            _io.BytesIO,
            _io.FileIO,
            _io.IncrementalNewlineDecoder,
            _io.StringIO,
            _io.TextIOWrapper,
            _io._BufferedIOBase,
            _io._BytesIOBuffer,
            _io._IOBase,
            _io._RawIOBase,
            _io._TextIOBase,
        ]
        if sys.platform == "win32":
            self.types.append(_io._WindowsConsoleIO)
        self._io = _io

    def test_immutable_types(self):
        for tp in self.types:
            with self.subTest(tp=tp):
                with self.assertRaisesRegex(TypeError, "immutable"):
                    tp.foo = "bar"

    def test_class_hierarchy(self):
        def check_subs(types, base):
            for tp in types:
                with self.subTest(tp=tp, base=base):
                    self.assertTrue(issubclass(tp, base))

        def recursive_check(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_check(v)
                elif isinstance(v, set):
                    check_subs(v, k)
                else:
                    self.fail("corrupt test dataset")

        _io = self._io
        hierarchy = {
            _io._IOBase: {
                _io._BufferedIOBase: {
                    _io.BufferedRWPair,
                    _io.BufferedRandom,
                    _io.BufferedReader,
                    _io.BufferedWriter,
                    _io.BytesIO,
                },
                _io._RawIOBase: {
                    _io.FileIO,
                },
                _io._TextIOBase: {
                    _io.StringIO,
                    _io.TextIOWrapper,
                },
            },
        }
        if sys.platform == "win32":
            hierarchy[_io._IOBase][_io._RawIOBase].add(_io._WindowsConsoleIO)

        recursive_check(hierarchy)

    def test_subclassing(self):
        _io = self._io
        dataset = {k: True for k in self.types}
        dataset[_io._BytesIOBuffer] = False

        for tp, is_basetype in dataset.items():
            with self.subTest(tp=tp, is_basetype=is_basetype):
                name = f"{tp.__name__}_subclass"
                bases = (tp,)
                if is_basetype:
                    _ = type(name, bases, {})
                else:
                    msg = "not an acceptable base type"
                    with self.assertRaisesRegex(TypeError, msg):
                        _ = type(name, bases, {})

    def test_disallow_instantiation(self):
        _io = self._io
        support.check_disallow_instantiation(self, _io._BytesIOBuffer)

    def test_stringio_setstate(self):
        # gh-127182: Calling __setstate__() with invalid arguments must not crash
        obj = self._io.StringIO()
        with self.assertRaisesRegex(
            TypeError,
            'initial_value must be str or None, not int',
        ):
            obj.__setstate__((1, '', 0, {}))

        obj.__setstate__((None, '', 0, {}))  # should not crash
        self.assertEqual(obj.getvalue(), '')

        obj.__setstate__(('', '', 0, {}))
        self.assertEqual(obj.getvalue(), '')

class PyIOTest(IOTest):
    pass


@support.cpython_only
class APIMismatchTest(__TestCase):

    def test_RawIOBase_io_in_pyio_match(self):
        """Test that pyio RawIOBase class has all c RawIOBase methods"""
        mismatch = support.detect_api_mismatch(pyio.RawIOBase, io.RawIOBase,
                                               ignore=('__weakref__', '__static_attributes__'))
        self.assertEqual(mismatch, set(), msg='Python RawIOBase does not have all C RawIOBase methods')

    def test_RawIOBase_pyio_in_io_match(self):
        """Test that c RawIOBase class has all pyio RawIOBase methods"""
        mismatch = support.detect_api_mismatch(io.RawIOBase, pyio.RawIOBase)
        self.assertEqual(mismatch, set(), msg='C RawIOBase does not have all Python RawIOBase methods')


class CommonBufferedTests:
    # Tests common to BufferedReader, BufferedWriter and BufferedRandom

    def test_detach(self):
        raw = self.MockRawIO()
        buf = self.tp(raw)
        self.assertIs(buf.detach(), raw)
        self.assertRaises(ValueError, buf.detach)

        repr(buf)  # Should still work

    def test_fileno(self):
        rawio = self.MockRawIO()
        bufio = self.tp(rawio)

        self.assertEqual(42, bufio.fileno())

    def test_invalid_args(self):
        rawio = self.MockRawIO()
        bufio = self.tp(rawio)
        # Invalid whence
        self.assertRaises(ValueError, bufio.seek, 0, -1)
        self.assertRaises(ValueError, bufio.seek, 0, 9)

    def test_override_destructor(self):
        tp = self.tp
        record = []
        class MyBufferedIO(tp):
            def __del__(self):
                record.append(1)
                try:
                    f = super().__del__
                except AttributeError:
                    pass
                else:
                    f()
            def close(self):
                record.append(2)
                super().close()
            def flush(self):
                record.append(3)
                super().flush()
        rawio = self.MockRawIO()
        bufio = MyBufferedIO(rawio)
        del bufio
        support.gc_collect()
        self.assertEqual(record, [1, 2, 3])

    def test_context_manager(self):
        # Test usability as a context manager
        rawio = self.MockRawIO()
        bufio = self.tp(rawio)
        def _with():
            with bufio:
                pass
        _with()
        # bufio should now be closed, and using it a second time should raise
        # a ValueError.
        self.assertRaises(ValueError, _with)

    def test_error_through_destructor(self):
        # Test that the exception state is not modified by a destructor,
        # even if close() fails.
        rawio = self.CloseFailureIO()
        with support.catch_unraisable_exception() as cm:
            with self.assertRaises(AttributeError):
                self.tp(rawio).xyzzy

            self.assertEqual(cm.unraisable.exc_type, OSError)

    def test_repr(self):
        raw = self.MockRawIO()
        b = self.tp(raw)
        clsname = r"(%s\.)?%s" % (self.tp.__module__, self.tp.__qualname__)
        self.assertRegex(repr(b), "<%s>" % clsname)
        raw.name = "dummy"
        self.assertRegex(repr(b), "<%s name='dummy'>" % clsname)
        raw.name = b"dummy"
        self.assertRegex(repr(b), "<%s name=b'dummy'>" % clsname)

    def test_recursive_repr(self):
        # Issue #25455
        raw = self.MockRawIO()
        b = self.tp(raw)
        with support.swap_attr(raw, 'name', b), support.infinite_recursion(25):
            with self.assertRaises(RuntimeError):
                repr(b)  # Should not crash

    def test_flush_error_on_close(self):
        # Test that buffered file is closed despite failed flush
        # and that flush() is called before file closed.
        raw = self.MockRawIO()
        closed = []
        def bad_flush():
            closed[:] = [b.closed, raw.closed]
            raise OSError()
        raw.flush = bad_flush
        b = self.tp(raw)
        self.assertRaises(OSError, b.close) # exception not swallowed
        self.assertTrue(b.closed)
        self.assertTrue(raw.closed)
        self.assertTrue(closed)      # flush() called
        self.assertFalse(closed[0])  # flush() called before file closed
        self.assertFalse(closed[1])
        raw.flush = lambda: None  # break reference loop

    def test_close_error_on_close(self):
        raw = self.MockRawIO()
        def bad_flush():
            raise OSError('flush')
        def bad_close():
            raise OSError('close')
        raw.close = bad_close
        b = self.tp(raw)
        b.flush = bad_flush
        with self.assertRaises(OSError) as err: # exception not swallowed
            b.close()
        self.assertEqual(err.exception.args, ('close',))
        self.assertIsInstance(err.exception.__context__, OSError)
        self.assertEqual(err.exception.__context__.args, ('flush',))
        self.assertFalse(b.closed)

        # Silence destructor error
        raw.close = lambda: None
        b.flush = lambda: None

    def test_nonnormalized_close_error_on_close(self):
        # Issue #21677
        raw = self.MockRawIO()
        def bad_flush():
            raise non_existing_flush
        def bad_close():
            raise non_existing_close
        raw.close = bad_close
        b = self.tp(raw)
        b.flush = bad_flush
        with self.assertRaises(NameError) as err: # exception not swallowed
            b.close()
        self.assertIn('non_existing_close', str(err.exception))
        self.assertIsInstance(err.exception.__context__, NameError)
        self.assertIn('non_existing_flush', str(err.exception.__context__))
        self.assertFalse(b.closed)

        # Silence destructor error
        b.flush = lambda: None
        raw.close = lambda: None

    def test_multi_close(self):
        raw = self.MockRawIO()
        b = self.tp(raw)
        b.close()
        b.close()
        b.close()
        self.assertRaises(ValueError, b.flush)

    def test_unseekable(self):
        bufio = self.tp(self.MockUnseekableIO(b"A" * 10))
        self.assertRaises(self.UnsupportedOperation, bufio.tell)
        self.assertRaises(self.UnsupportedOperation, bufio.seek, 0)

    def test_readonly_attributes(self):
        raw = self.MockRawIO()
        buf = self.tp(raw)
        x = self.MockRawIO()
        with self.assertRaises(AttributeError):
            buf.raw = x

    def test_pickling_subclass(self):
        global MyBufferedIO
        class MyBufferedIO(self.tp):
            def __init__(self, raw, tag):
                super().__init__(raw)
                self.tag = tag
            def __getstate__(self):
                return self.tag, self.raw.getvalue()
            def __setstate__(slf, state):
                tag, value = state
                slf.__init__(self.BytesIO(value), tag)

        raw = self.BytesIO(b'data')
        buf = MyBufferedIO(raw, tag='ham')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=proto):
                pickled = pickle.dumps(buf, proto)
                newbuf = pickle.loads(pickled)
                self.assertEqual(newbuf.raw.getvalue(), b'data')
                self.assertEqual(newbuf.tag, 'ham')
        del MyBufferedIO


class SizeofTest:

    @support.cpython_only
    def test_sizeof(self):
        bufsize1 = 4096
        bufsize2 = 8192
        rawio = self.MockRawIO()
        bufio = self.tp(rawio, buffer_size=bufsize1)
        size = sys.getsizeof(bufio) - bufsize1
        rawio = self.MockRawIO()
        bufio = self.tp(rawio, buffer_size=bufsize2)
        self.assertEqual(sys.getsizeof(bufio), size + bufsize2)

    @support.cpython_only
    def test_buffer_freeing(self) :
        bufsize = 4096
        rawio = self.MockRawIO()
        bufio = self.tp(rawio, buffer_size=bufsize)
        size = sys.getsizeof(bufio) - bufsize
        bufio.close()
        self.assertEqual(sys.getsizeof(bufio), size)

class BufferedReaderTest(CPythonTestCase, CommonBufferedTests):
    read_mode = "rb"

    def test_constructor(self):
        rawio = self.MockRawIO([b"abc"])
        bufio = self.tp(rawio)
        bufio.__init__(rawio)
        bufio.__init__(rawio, buffer_size=1024)
        bufio.__init__(rawio, buffer_size=16)
        self.assertEqual(b"abc", bufio.read())
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=0)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-16)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-1)
        rawio = self.MockRawIO([b"abc"])
        bufio.__init__(rawio)
        self.assertEqual(b"abc", bufio.read())

    def test_uninitialized(self):
        bufio = self.tp.__new__(self.tp)
        del bufio
        bufio = self.tp.__new__(self.tp)
        self.assertRaisesRegex((ValueError, AttributeError),
                               'uninitialized|has no attribute',
                               bufio.read, 0)
        bufio.__init__(self.MockRawIO())
        self.assertEqual(bufio.read(0), b'')

    def test_read(self):
        for arg in (None, 7):
            rawio = self.MockRawIO((b"abc", b"d", b"efg"))
            bufio = self.tp(rawio)
            self.assertEqual(b"abcdefg", bufio.read(arg))
        # Invalid args
        self.assertRaises(ValueError, bufio.read, -2)

    def test_read1(self):
        rawio = self.MockRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)
        self.assertEqual(b"a", bufio.read(1))
        self.assertEqual(b"b", bufio.read1(1))
        self.assertEqual(rawio._reads, 1)
        self.assertEqual(b"", bufio.read1(0))
        self.assertEqual(b"c", bufio.read1(100))
        self.assertEqual(rawio._reads, 1)
        self.assertEqual(b"d", bufio.read1(100))
        self.assertEqual(rawio._reads, 2)
        self.assertEqual(b"efg", bufio.read1(100))
        self.assertEqual(rawio._reads, 3)
        self.assertEqual(b"", bufio.read1(100))
        self.assertEqual(rawio._reads, 4)

    def test_read1_arbitrary(self):
        rawio = self.MockRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)
        self.assertEqual(b"a", bufio.read(1))
        self.assertEqual(b"bc", bufio.read1())
        self.assertEqual(b"d", bufio.read1())
        self.assertEqual(b"efg", bufio.read1(-1))
        self.assertEqual(rawio._reads, 3)
        self.assertEqual(b"", bufio.read1())
        self.assertEqual(rawio._reads, 4)

    def test_readinto(self):
        rawio = self.MockRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)
        b = bytearray(2)
        self.assertEqual(bufio.readinto(b), 2)
        self.assertEqual(b, b"ab")
        self.assertEqual(bufio.readinto(b), 2)
        self.assertEqual(b, b"cd")
        self.assertEqual(bufio.readinto(b), 2)
        self.assertEqual(b, b"ef")
        self.assertEqual(bufio.readinto(b), 1)
        self.assertEqual(b, b"gf")
        self.assertEqual(bufio.readinto(b), 0)
        self.assertEqual(b, b"gf")
        rawio = self.MockRawIO((b"abc", None))
        bufio = self.tp(rawio)
        self.assertEqual(bufio.readinto(b), 2)
        self.assertEqual(b, b"ab")
        self.assertEqual(bufio.readinto(b), 1)
        self.assertEqual(b, b"cb")

    def test_readinto1(self):
        buffer_size = 10
        rawio = self.MockRawIO((b"abc", b"de", b"fgh", b"jkl"))
        bufio = self.tp(rawio, buffer_size=buffer_size)
        b = bytearray(2)
        self.assertEqual(bufio.peek(3), b'abc')
        self.assertEqual(rawio._reads, 1)
        self.assertEqual(bufio.readinto1(b), 2)
        self.assertEqual(b, b"ab")
        self.assertEqual(rawio._reads, 1)
        self.assertEqual(bufio.readinto1(b), 1)
        self.assertEqual(b[:1], b"c")
        self.assertEqual(rawio._reads, 1)
        self.assertEqual(bufio.readinto1(b), 2)
        self.assertEqual(b, b"de")
        self.assertEqual(rawio._reads, 2)
        b = bytearray(2*buffer_size)
        self.assertEqual(bufio.peek(3), b'fgh')
        self.assertEqual(rawio._reads, 3)
        self.assertEqual(bufio.readinto1(b), 6)
        self.assertEqual(b[:6], b"fghjkl")
        self.assertEqual(rawio._reads, 4)

    def test_readinto_array(self):
        buffer_size = 60
        data = b"a" * 26
        rawio = self.MockRawIO((data,))
        bufio = self.tp(rawio, buffer_size=buffer_size)

        # Create an array with element size > 1 byte
        b = array.array('i', b'x' * 32)
        assert len(b) != 16

        # Read into it. We should get as many *bytes* as we can fit into b
        # (which is more than the number of elements)
        n = bufio.readinto(b)
        self.assertGreater(n, len(b))

        # Check that old contents of b are preserved
        bm = memoryview(b).cast('B')
        self.assertLess(n, len(bm))
        self.assertEqual(bm[:n], data[:n])
        self.assertEqual(bm[n:], b'x' * (len(bm[n:])))

    def test_readinto1_array(self):
        buffer_size = 60
        data = b"a" * 26
        rawio = self.MockRawIO((data,))
        bufio = self.tp(rawio, buffer_size=buffer_size)

        # Create an array with element size > 1 byte
        b = array.array('i', b'x' * 32)
        assert len(b) != 16

        # Read into it. We should get as many *bytes* as we can fit into b
        # (which is more than the number of elements)
        n = bufio.readinto1(b)
        self.assertGreater(n, len(b))

        # Check that old contents of b are preserved
        bm = memoryview(b).cast('B')
        self.assertLess(n, len(bm))
        self.assertEqual(bm[:n], data[:n])
        self.assertEqual(bm[n:], b'x' * (len(bm[n:])))

    def test_readlines(self):
        def bufio():
            rawio = self.MockRawIO((b"abc\n", b"d\n", b"ef"))
            return self.tp(rawio)
        self.assertEqual(bufio().readlines(), [b"abc\n", b"d\n", b"ef"])
        self.assertEqual(bufio().readlines(5), [b"abc\n", b"d\n"])
        self.assertEqual(bufio().readlines(None), [b"abc\n", b"d\n", b"ef"])

    def test_buffering(self):
        data = b"abcdefghi"
        dlen = len(data)

        tests = [
            [ 100, [ 3, 1, 4, 8 ], [ dlen, 0 ] ],
            [ 100, [ 3, 3, 3],     [ dlen ]    ],
            [   4, [ 1, 2, 4, 2 ], [ 4, 4, 1 ] ],
        ]

        for bufsize, buf_read_sizes, raw_read_sizes in tests:
            rawio = self.MockFileIO(data)
            bufio = self.tp(rawio, buffer_size=bufsize)
            pos = 0
            for nbytes in buf_read_sizes:
                self.assertEqual(bufio.read(nbytes), data[pos:pos+nbytes])
                pos += nbytes
            # this is mildly implementation-dependent
            self.assertEqual(rawio.read_history, raw_read_sizes)

    def test_read_non_blocking(self):
        # Inject some None's in there to simulate EWOULDBLOCK
        rawio = self.MockRawIO((b"abc", b"d", None, b"efg", None, None, None))
        bufio = self.tp(rawio)
        self.assertEqual(b"abcd", bufio.read(6))
        self.assertEqual(b"e", bufio.read(1))
        self.assertEqual(b"fg", bufio.read())
        self.assertEqual(b"", bufio.peek(1))
        self.assertIsNone(bufio.read())
        self.assertEqual(b"", bufio.read())

        rawio = self.MockRawIO((b"a", None, None))
        self.assertEqual(b"a", rawio.readall())
        self.assertIsNone(rawio.readall())

    def test_read_past_eof(self):
        rawio = self.MockRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)

        self.assertEqual(b"abcdefg", bufio.read(9000))

    def test_read_all(self):
        rawio = self.MockRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)

        self.assertEqual(b"abcdefg", bufio.read())

    @threading_helper.requires_working_threading()
    @support.requires_resource('cpu')
    def test_threads(self):
        try:
            # Write out many bytes with exactly the same number of 0's,
            # 1's... 255's. This will help us check that concurrent reading
            # doesn't duplicate or forget contents.
            N = 1000
            l = list(range(256)) * N
            random.shuffle(l)
            s = bytes(bytearray(l))
            with self.open(os_helper.TESTFN, "wb") as f:
                f.write(s)
            with self.open(os_helper.TESTFN, self.read_mode, buffering=0) as raw:
                bufio = self.tp(raw, 8)
                errors = []
                results = []
                def f():
                    try:
                        # Intra-buffer read then buffer-flushing read
                        for n in cycle([1, 19]):
                            s = bufio.read(n)
                            if not s:
                                break
                            # list.append() is atomic
                            results.append(s)
                    except Exception as e:
                        errors.append(e)
                        raise
                threads = [threading.Thread(target=f) for x in range(20)]
                with threading_helper.start_threads(threads):
                    time.sleep(0.02) # yield
                self.assertFalse(errors,
                    "the following exceptions were caught: %r" % errors)
                s = b''.join(results)
                for i in range(256):
                    c = bytes(bytearray([i]))
                    self.assertEqual(s.count(c), N)
        finally:
            os_helper.unlink(os_helper.TESTFN)

    def test_unseekable(self):
        bufio = self.tp(self.MockUnseekableIO(b"A" * 10))
        self.assertRaises(self.UnsupportedOperation, bufio.tell)
        self.assertRaises(self.UnsupportedOperation, bufio.seek, 0)
        bufio.read(1)
        self.assertRaises(self.UnsupportedOperation, bufio.seek, 0)
        self.assertRaises(self.UnsupportedOperation, bufio.tell)

    def test_misbehaved_io(self):
        rawio = self.MisbehavedRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)
        self.assertRaises(OSError, bufio.seek, 0)
        self.assertRaises(OSError, bufio.tell)

        # Silence destructor error
        bufio.close = lambda: None

    def test_no_extraneous_read(self):
        # Issue #9550; when the raw IO object has satisfied the read request,
        # we should not issue any additional reads, otherwise it may block
        # (e.g. socket).
        bufsize = 16
        for n in (2, bufsize - 1, bufsize, bufsize + 1, bufsize * 2):
            rawio = self.MockRawIO([b"x" * n])
            bufio = self.tp(rawio, bufsize)
            self.assertEqual(bufio.read(n), b"x" * n)
            # Simple case: one raw read is enough to satisfy the request.
            self.assertEqual(rawio._extraneous_reads, 0,
                             "failed for {}: {} != 0".format(n, rawio._extraneous_reads))
            # A more complex case where two raw reads are needed to satisfy
            # the request.
            rawio = self.MockRawIO([b"x" * (n - 1), b"x"])
            bufio = self.tp(rawio, bufsize)
            self.assertEqual(bufio.read(n), b"x" * n)
            self.assertEqual(rawio._extraneous_reads, 0,
                             "failed for {}: {} != 0".format(n, rawio._extraneous_reads))

    def test_read_on_closed(self):
        # Issue #23796
        b = self.BufferedReader(self.BytesIO(b"12"))
        b.read(1)
        b.close()
        with self.subTest('peek'):
            self.assertRaises(ValueError, b.peek)
        with self.subTest('read1'):
            self.assertRaises(ValueError, b.read1, 1)
        with self.subTest('read'):
            self.assertRaises(ValueError, b.read)
        with self.subTest('readinto'):
            self.assertRaises(ValueError, b.readinto, bytearray())
        with self.subTest('readinto1'):
            self.assertRaises(ValueError, b.readinto1, bytearray())
        with self.subTest('flush'):
            self.assertRaises(ValueError, b.flush)
        with self.subTest('truncate'):
            self.assertRaises(ValueError, b.truncate)
        with self.subTest('seek'):
            self.assertRaises(ValueError, b.seek, 0)

    def test_truncate_on_read_only(self):
        rawio = self.MockFileIO(b"abc")
        bufio = self.tp(rawio)
        self.assertFalse(bufio.writable())
        self.assertRaises(self.UnsupportedOperation, bufio.truncate)
        self.assertRaises(self.UnsupportedOperation, bufio.truncate, 0)

    def test_tell_character_device_file(self):
        # GH-95782
        # For the (former) bug in BufferedIO to manifest, the wrapped IO obj
        # must be able to produce at least 2 bytes.
        raw = self.MockCharPseudoDevFileIO(b"12")
        buf = self.tp(raw)
        self.assertEqual(buf.tell(), 0)
        self.assertEqual(buf.read(1), b"1")
        self.assertEqual(buf.tell(), 0)

    def test_seek_character_device_file(self):
        raw = self.MockCharPseudoDevFileIO(b"12")
        buf = self.tp(raw)
        self.assertEqual(buf.seek(0, io.SEEK_CUR), 0)
        self.assertEqual(buf.seek(1, io.SEEK_SET), 0)
        self.assertEqual(buf.seek(0, io.SEEK_CUR), 0)
        self.assertEqual(buf.read(1), b"1")

        # In the C implementation, tell() sets the BufferedIO's abs_pos to 0,
        # which means that the next seek() could return a negative offset if it
        # does not sanity-check:
        self.assertEqual(buf.tell(), 0)
        self.assertEqual(buf.seek(0, io.SEEK_CUR), 0)


class CBufferedReaderTest(BufferedReaderTest, SizeofTest):
    tp = io.BufferedReader

    def test_initialization(self):
        rawio = self.MockRawIO([b"abc"])
        bufio = self.tp(rawio)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=0)
        self.assertRaises(ValueError, bufio.read)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-16)
        self.assertRaises(ValueError, bufio.read)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-1)
        self.assertRaises(ValueError, bufio.read)

    def test_misbehaved_io_read(self):
        rawio = self.MisbehavedRawIO((b"abc", b"d", b"efg"))
        bufio = self.tp(rawio)
        # _pyio.BufferedReader seems to implement reading different, so that
        # checking this is not so easy.
        self.assertRaises(OSError, bufio.read, 10)

    def test_garbage_collection(self):
        # C BufferedReader objects are collected.
        # The Python version has __del__, so it ends into gc.garbage instead
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with warnings_helper.check_warnings(('', ResourceWarning)):
            rawio = self.FileIO(os_helper.TESTFN, "w+b")
            f = self.tp(rawio)
            f.f = f
            wr = weakref.ref(f)
            del f
            support.gc_collect()
        self.assertIsNone(wr(), wr)

    def test_args_error(self):
        # Issue #17275
        with self.assertRaisesRegex(TypeError, "BufferedReader"):
            self.tp(self.BytesIO(), 1024, 1024, 1024)

    def test_bad_readinto_value(self):
        rawio = self.tp(self.BytesIO(b"12"))
        rawio.readinto = lambda buf: -1
        bufio = self.tp(rawio)
        with self.assertRaises(OSError) as cm:
            bufio.readline()
        self.assertIsNone(cm.exception.__cause__)

    def test_bad_readinto_type(self):
        rawio = self.tp(self.BytesIO(b"12"))
        rawio.readinto = lambda buf: b''
        bufio = self.tp(rawio)
        with self.assertRaises(OSError) as cm:
            bufio.readline()
        self.assertIsInstance(cm.exception.__cause__, TypeError)


class PyBufferedReaderTest(BufferedReaderTest):
    tp = pyio.BufferedReader


class BufferedWriterTest(CommonBufferedTests):
    write_mode = "wb"

    def test_constructor(self):
        rawio = self.MockRawIO()
        bufio = self.tp(rawio)
        bufio.__init__(rawio)
        bufio.__init__(rawio, buffer_size=1024)
        bufio.__init__(rawio, buffer_size=16)
        self.assertEqual(3, bufio.write(b"abc"))
        bufio.flush()
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=0)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-16)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-1)
        bufio.__init__(rawio)
        self.assertEqual(3, bufio.write(b"ghi"))
        bufio.flush()
        self.assertEqual(b"".join(rawio._write_stack), b"abcghi")

    def test_uninitialized(self):
        bufio = self.tp.__new__(self.tp)
        del bufio
        bufio = self.tp.__new__(self.tp)
        self.assertRaisesRegex((ValueError, AttributeError),
                               'uninitialized|has no attribute',
                               bufio.write, b'')
        bufio.__init__(self.MockRawIO())
        self.assertEqual(bufio.write(b''), 0)

    def test_detach_flush(self):
        raw = self.MockRawIO()
        buf = self.tp(raw)
        buf.write(b"howdy!")
        self.assertFalse(raw._write_stack)
        buf.detach()
        self.assertEqual(raw._write_stack, [b"howdy!"])

    def test_write(self):
        # Write to the buffered IO but don't overflow the buffer.
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        bufio.write(b"abc")
        self.assertFalse(writer._write_stack)
        buffer = bytearray(b"def")
        bufio.write(buffer)
        buffer[:] = b"***"  # Overwrite our copy of the data
        bufio.flush()
        self.assertEqual(b"".join(writer._write_stack), b"abcdef")

    def test_write_overflow(self):
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        contents = b"abcdefghijklmnop"
        for n in range(0, len(contents), 3):
            bufio.write(contents[n:n+3])
        flushed = b"".join(writer._write_stack)
        # At least (total - 8) bytes were implicitly flushed, perhaps more
        # depending on the implementation.
        self.assertTrue(flushed.startswith(contents[:-8]), flushed)

    def check_writes(self, intermediate_func):
        # Lots of writes, test the flushed output is as expected.
        contents = bytes(range(256)) * 1000
        n = 0
        writer = self.MockRawIO()
        bufio = self.tp(writer, 13)
        # Generator of write sizes: repeat each N 15 times then proceed to N+1
        def gen_sizes():
            for size in count(1):
                for i in range(15):
                    yield size
        sizes = gen_sizes()
        while n < len(contents):
            size = min(next(sizes), len(contents) - n)
            self.assertEqual(bufio.write(contents[n:n+size]), size)
            intermediate_func(bufio)
            n += size
        bufio.flush()
        self.assertEqual(contents, b"".join(writer._write_stack))

    def test_writes(self):
        self.check_writes(lambda bufio: None)

    def test_writes_and_flushes(self):
        self.check_writes(lambda bufio: bufio.flush())

    def test_writes_and_seeks(self):
        def _seekabs(bufio):
            pos = bufio.tell()
            bufio.seek(pos + 1, 0)
            bufio.seek(pos - 1, 0)
            bufio.seek(pos, 0)
        self.check_writes(_seekabs)
        def _seekrel(bufio):
            pos = bufio.seek(0, 1)
            bufio.seek(+1, 1)
            bufio.seek(-1, 1)
            bufio.seek(pos, 0)
        self.check_writes(_seekrel)

    def test_writes_and_truncates(self):
        self.check_writes(lambda bufio: bufio.truncate(bufio.tell()))

    def test_write_non_blocking(self):
        raw = self.MockNonBlockWriterIO()
        bufio = self.tp(raw, 8)

        self.assertEqual(bufio.write(b"abcd"), 4)
        self.assertEqual(bufio.write(b"efghi"), 5)
        # 1 byte will be written, the rest will be buffered
        raw.block_on(b"k")
        self.assertEqual(bufio.write(b"jklmn"), 5)

        # 8 bytes will be written, 8 will be buffered and the rest will be lost
        raw.block_on(b"0")
        try:
            bufio.write(b"opqrwxyz0123456789")
        except self.BlockingIOError as e:
            written = e.characters_written
        else:
            self.fail("BlockingIOError should have been raised")
        self.assertEqual(written, 16)
        self.assertEqual(raw.pop_written(),
            b"abcdefghijklmnopqrwxyz")

        self.assertEqual(bufio.write(b"ABCDEFGHI"), 9)
        s = raw.pop_written()
        # Previously buffered bytes were flushed
        self.assertTrue(s.startswith(b"01234567A"), s)

    def test_write_and_rewind(self):
        raw = self.BytesIO()
        bufio = self.tp(raw, 4)
        self.assertEqual(bufio.write(b"abcdef"), 6)
        self.assertEqual(bufio.tell(), 6)
        bufio.seek(0, 0)
        self.assertEqual(bufio.write(b"XY"), 2)
        bufio.seek(6, 0)
        self.assertEqual(raw.getvalue(), b"XYcdef")
        self.assertEqual(bufio.write(b"123456"), 6)
        bufio.flush()
        self.assertEqual(raw.getvalue(), b"XYcdef123456")

    def test_flush(self):
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        bufio.write(b"abc")
        bufio.flush()
        self.assertEqual(b"abc", writer._write_stack[0])

    def test_writelines(self):
        l = [b'ab', b'cd', b'ef']
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        bufio.writelines(l)
        bufio.flush()
        self.assertEqual(b''.join(writer._write_stack), b'abcdef')

    def test_writelines_userlist(self):
        l = UserList([b'ab', b'cd', b'ef'])
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        bufio.writelines(l)
        bufio.flush()
        self.assertEqual(b''.join(writer._write_stack), b'abcdef')

    def test_writelines_error(self):
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        self.assertRaises(TypeError, bufio.writelines, [1, 2, 3])
        self.assertRaises(TypeError, bufio.writelines, None)
        self.assertRaises(TypeError, bufio.writelines, 'abc')

    def test_destructor(self):
        writer = self.MockRawIO()
        bufio = self.tp(writer, 8)
        bufio.write(b"abc")
        del bufio
        support.gc_collect()
        self.assertEqual(b"abc", writer._write_stack[0])

    def test_truncate(self):
        # Truncate implicitly flushes the buffer.
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with self.open(os_helper.TESTFN, self.write_mode, buffering=0) as raw:
            bufio = self.tp(raw, 8)
            bufio.write(b"abcdef")
            self.assertEqual(bufio.truncate(3), 3)
            self.assertEqual(bufio.tell(), 6)
        with self.open(os_helper.TESTFN, "rb", buffering=0) as f:
            self.assertEqual(f.read(), b"abc")

    def test_truncate_after_write(self):
        # Ensure that truncate preserves the file position after
        # writes longer than the buffer size.
        # Issue: https://bugs.python.org/issue32228
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with self.open(os_helper.TESTFN, "wb") as f:
            # Fill with some buffer
            f.write(b'\x00' * 10000)
        buffer_sizes = [8192, 4096, 200]
        for buffer_size in buffer_sizes:
            with self.open(os_helper.TESTFN, "r+b", buffering=buffer_size) as f:
                f.write(b'\x00' * (buffer_size + 1))
                # After write write_pos and write_end are set to 0
                f.read(1)
                # read operation makes sure that pos != raw_pos
                f.truncate()
                self.assertEqual(f.tell(), buffer_size + 2)

    @threading_helper.requires_working_threading()
    @support.requires_resource('cpu')
    def test_threads(self):
        try:
            # Write out many bytes from many threads and test they were
            # all flushed.
            N = 1000
            contents = bytes(range(256)) * N
            sizes = cycle([1, 19])
            n = 0
            queue = deque()
            while n < len(contents):
                size = next(sizes)
                queue.append(contents[n:n+size])
                n += size
            del contents
            # We use a real file object because it allows us to
            # exercise situations where the GIL is released before
            # writing the buffer to the raw streams. This is in addition
            # to concurrency issues due to switching threads in the middle
            # of Python code.
            with self.open(os_helper.TESTFN, self.write_mode, buffering=0) as raw:
                bufio = self.tp(raw, 8)
                errors = []
                def f():
                    try:
                        while True:
                            try:
                                s = queue.popleft()
                            except IndexError:
                                return
                            bufio.write(s)
                    except Exception as e:
                        errors.append(e)
                        raise
                threads = [threading.Thread(target=f) for x in range(20)]
                with threading_helper.start_threads(threads):
                    time.sleep(0.02) # yield
                self.assertFalse(errors,
                    "the following exceptions were caught: %r" % errors)
                bufio.close()
            with self.open(os_helper.TESTFN, "rb") as f:
                s = f.read()
            for i in range(256):
                self.assertEqual(s.count(bytes([i])), N)
        finally:
            os_helper.unlink(os_helper.TESTFN)

    def test_misbehaved_io(self):
        rawio = self.MisbehavedRawIO()
        bufio = self.tp(rawio, 5)
        self.assertRaises(OSError, bufio.seek, 0)
        self.assertRaises(OSError, bufio.tell)
        self.assertRaises(OSError, bufio.write, b"abcdef")

        # Silence destructor error
        bufio.close = lambda: None

    def test_max_buffer_size_removal(self):
        with self.assertRaises(TypeError):
            self.tp(self.MockRawIO(), 8, 12)

    def test_write_error_on_close(self):
        raw = self.MockRawIO()
        def bad_write(b):
            raise OSError()
        raw.write = bad_write
        b = self.tp(raw)
        b.write(b'spam')
        self.assertRaises(OSError, b.close) # exception not swallowed
        self.assertTrue(b.closed)

    @threading_helper.requires_working_threading()
    def test_slow_close_from_thread(self):
        # Issue #31976
        rawio = self.SlowFlushRawIO()
        bufio = self.tp(rawio, 8)
        t = threading.Thread(target=bufio.close)
        t.start()
        rawio.in_flush.wait()
        self.assertRaises(ValueError, bufio.write, b'spam')
        self.assertTrue(bufio.closed)
        t.join()



class CBufferedWriterTest(BufferedWriterTest, SizeofTest, __TestCase):
    tp = io.BufferedWriter

    def test_initialization(self):
        rawio = self.MockRawIO()
        bufio = self.tp(rawio)
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=0)
        self.assertRaises(ValueError, bufio.write, b"def")
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-16)
        self.assertRaises(ValueError, bufio.write, b"def")
        self.assertRaises(ValueError, bufio.__init__, rawio, buffer_size=-1)
        self.assertRaises(ValueError, bufio.write, b"def")

    def test_garbage_collection(self):
        # C BufferedWriter objects are collected, and collecting them flushes
        # all data to disk.
        # The Python version has __del__, so it ends into gc.garbage instead
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        with warnings_helper.check_warnings(('', ResourceWarning)):
            rawio = self.FileIO(os_helper.TESTFN, "w+b")
            f = self.tp(rawio)
            f.write(b"123xxx")
            f.x = f
            wr = weakref.ref(f)
            del f
            support.gc_collect()
        self.assertIsNone(wr(), wr)
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"123xxx")

    def test_args_error(self):
        # Issue #17275
        with self.assertRaisesRegex(TypeError, "BufferedWriter"):
            self.tp(self.BytesIO(), 1024, 1024, 1024)


class PyBufferedWriterTest(BufferedWriterTest, __TestCase):
    tp = pyio.BufferedWriter

class BufferedRWPairTest:

    def test_constructor(self):
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertFalse(pair.closed)

    def test_uninitialized(self):
        pair = self.tp.__new__(self.tp)
        del pair
        pair = self.tp.__new__(self.tp)
        self.assertRaisesRegex((ValueError, AttributeError),
                               'uninitialized|has no attribute',
                               pair.read, 0)
        self.assertRaisesRegex((ValueError, AttributeError),
                               'uninitialized|has no attribute',
                               pair.write, b'')
        pair.__init__(self.MockRawIO(), self.MockRawIO())
        self.assertEqual(pair.read(0), b'')
        self.assertEqual(pair.write(b''), 0)

    def test_detach(self):
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertRaises(self.UnsupportedOperation, pair.detach)

    def test_constructor_max_buffer_size_removal(self):
        with self.assertRaises(TypeError):
            self.tp(self.MockRawIO(), self.MockRawIO(), 8, 12)

    def test_constructor_with_not_readable(self):
        class NotReadable(MockRawIO):
            def readable(self):
                return False

        self.assertRaises(OSError, self.tp, NotReadable(), self.MockRawIO())

    def test_constructor_with_not_writeable(self):
        class NotWriteable(MockRawIO):
            def writable(self):
                return False

        self.assertRaises(OSError, self.tp, self.MockRawIO(), NotWriteable())

    def test_read(self):
        pair = self.tp(self.BytesIO(b"abcdef"), self.MockRawIO())

        self.assertEqual(pair.read(3), b"abc")
        self.assertEqual(pair.read(1), b"d")
        self.assertEqual(pair.read(), b"ef")
        pair = self.tp(self.BytesIO(b"abc"), self.MockRawIO())
        self.assertEqual(pair.read(None), b"abc")

    def test_readlines(self):
        pair = lambda: self.tp(self.BytesIO(b"abc\ndef\nh"), self.MockRawIO())
        self.assertEqual(pair().readlines(), [b"abc\n", b"def\n", b"h"])
        self.assertEqual(pair().readlines(), [b"abc\n", b"def\n", b"h"])
        self.assertEqual(pair().readlines(5), [b"abc\n", b"def\n"])

    def test_read1(self):
        # .read1() is delegated to the underlying reader object, so this test
        # can be shallow.
        pair = self.tp(self.BytesIO(b"abcdef"), self.MockRawIO())

        self.assertEqual(pair.read1(3), b"abc")
        self.assertEqual(pair.read1(), b"def")

    def test_readinto(self):
        for method in ("readinto", "readinto1"):
            with self.subTest(method):
                pair = self.tp(self.BytesIO(b"abcdef"), self.MockRawIO())

                data = byteslike(b'\0' * 5)
                self.assertEqual(getattr(pair, method)(data), 5)
                self.assertEqual(bytes(data), b"abcde")

    def test_write(self):
        w = self.MockRawIO()
        pair = self.tp(self.MockRawIO(), w)

        pair.write(b"abc")
        pair.flush()
        buffer = bytearray(b"def")
        pair.write(buffer)
        buffer[:] = b"***"  # Overwrite our copy of the data
        pair.flush()
        self.assertEqual(w._write_stack, [b"abc", b"def"])

    def test_peek(self):
        pair = self.tp(self.BytesIO(b"abcdef"), self.MockRawIO())

        self.assertTrue(pair.peek(3).startswith(b"abc"))
        self.assertEqual(pair.read(3), b"abc")

    def test_readable(self):
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertTrue(pair.readable())

    def test_writeable(self):
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertTrue(pair.writable())

    def test_seekable(self):
        # BufferedRWPairs are never seekable, even if their readers and writers
        # are.
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertFalse(pair.seekable())

    # .flush() is delegated to the underlying writer object and has been
    # tested in the test_write method.

    def test_close_and_closed(self):
        pair = self.tp(self.MockRawIO(), self.MockRawIO())
        self.assertFalse(pair.closed)
        pair.close()
        self.assertTrue(pair.closed)

    def test_reader_close_error_on_close(self):
        def reader_close():
            reader_non_existing
        reader = self.MockRawIO()
        reader.close = reader_close
        writer = self.MockRawIO()
        pair = self.tp(reader, writer)
        with self.assertRaises(NameError) as err:
            pair.close()
        self.assertIn('reader_non_existing', str(err.exception))
        self.assertTrue(pair.closed)
        self.assertFalse(reader.closed)
        self.assertTrue(writer.closed)

        # Silence destructor error
        reader.close = lambda: None

    def test_writer_close_error_on_close(self):
        def writer_close():
            writer_non_existing
        reader = self.MockRawIO()
        writer = self.MockRawIO()
        writer.close = writer_close
        pair = self.tp(reader, writer)
        with self.assertRaises(NameError) as err:
            pair.close()
        self.assertIn('writer_non_existing', str(err.exception))
        self.assertFalse(pair.closed)
        self.assertTrue(reader.closed)
        self.assertFalse(writer.closed)

        # Silence destructor error
        writer.close = lambda: None
        writer = None

        # Ignore BufferedWriter (of the BufferedRWPair) unraisable exception
        with support.catch_unraisable_exception():
            # Ignore BufferedRWPair unraisable exception
            with support.catch_unraisable_exception():
                pair = None
                support.gc_collect()
            support.gc_collect()

    def test_reader_writer_close_error_on_close(self):
        def reader_close():
            reader_non_existing
        def writer_close():
            writer_non_existing
        reader = self.MockRawIO()
        reader.close = reader_close
        writer = self.MockRawIO()
        writer.close = writer_close
        pair = self.tp(reader, writer)
        with self.assertRaises(NameError) as err:
            pair.close()
        self.assertIn('reader_non_existing', str(err.exception))
        self.assertIsInstance(err.exception.__context__, NameError)
        self.assertIn('writer_non_existing', str(err.exception.__context__))
        self.assertFalse(pair.closed)
        self.assertFalse(reader.closed)
        self.assertFalse(writer.closed)

        # Silence destructor error
        reader.close = lambda: None
        writer.close = lambda: None

    def test_isatty(self):
        class SelectableIsAtty(MockRawIO):
            def __init__(self, isatty):
                MockRawIO.__init__(self)
                self._isatty = isatty

            def isatty(self):
                return self._isatty

        pair = self.tp(SelectableIsAtty(False), SelectableIsAtty(False))
        self.assertFalse(pair.isatty())

        pair = self.tp(SelectableIsAtty(True), SelectableIsAtty(False))
        self.assertTrue(pair.isatty())

        pair = self.tp(SelectableIsAtty(False), SelectableIsAtty(True))
        self.assertTrue(pair.isatty())

        pair = self.tp(SelectableIsAtty(True), SelectableIsAtty(True))
        self.assertTrue(pair.isatty())

    def test_weakref_clearing(self):
        brw = self.tp(self.MockRawIO(), self.MockRawIO())
        ref = weakref.ref(brw)
        brw = None
        ref = None # Shouldn't segfault.

class CBufferedRWPairTest(BufferedRWPairTest, __TestCase):
    tp = io.BufferedRWPair

class PyBufferedRWPairTest(BufferedRWPairTest, __TestCase):
    tp = pyio.BufferedRWPair


class BufferedRandomTest(BufferedReaderTest, BufferedWriterTest):
    read_mode = "rb+"
    write_mode = "wb+"

    def test_constructor(self):
        BufferedReaderTest.test_constructor(self)
        BufferedWriterTest.test_constructor(self)

    def test_uninitialized(self):
        BufferedReaderTest.test_uninitialized(self)
        BufferedWriterTest.test_uninitialized(self)

    def test_read_and_write(self):
        raw = self.MockRawIO((b"asdf", b"ghjk"))
        rw = self.tp(raw, 8)

        self.assertEqual(b"as", rw.read(2))
        rw.write(b"ddd")
        rw.write(b"eee")
        self.assertFalse(raw._write_stack) # Buffer writes
        self.assertEqual(b"ghjk", rw.read())
        self.assertEqual(b"dddeee", raw._write_stack[0])

    def test_seek_and_tell(self):
        raw = self.BytesIO(b"asdfghjkl")
        rw = self.tp(raw)

        self.assertEqual(b"as", rw.read(2))
        self.assertEqual(2, rw.tell())
        rw.seek(0, 0)
        self.assertEqual(b"asdf", rw.read(4))

        rw.write(b"123f")
        rw.seek(0, 0)
        self.assertEqual(b"asdf123fl", rw.read())
        self.assertEqual(9, rw.tell())
        rw.seek(-4, 2)
        self.assertEqual(5, rw.tell())
        rw.seek(2, 1)
        self.assertEqual(7, rw.tell())
        self.assertEqual(b"fl", rw.read(11))
        rw.flush()
        self.assertEqual(b"asdf123fl", raw.getvalue())

        self.assertRaises(TypeError, rw.seek, 0.0)

    def check_flush_and_read(self, read_func):
        raw = self.BytesIO(b"abcdefghi")
        bufio = self.tp(raw)

        self.assertEqual(b"ab", read_func(bufio, 2))
        bufio.write(b"12")
        self.assertEqual(b"ef", read_func(bufio, 2))
        self.assertEqual(6, bufio.tell())
        bufio.flush()
        self.assertEqual(6, bufio.tell())
        self.assertEqual(b"ghi", read_func(bufio))
        raw.seek(0, 0)
        raw.write(b"XYZ")
        # flush() resets the read buffer
        bufio.flush()
        bufio.seek(0, 0)
        self.assertEqual(b"XYZ", read_func(bufio, 3))

    def test_flush_and_read(self):
        self.check_flush_and_read(lambda bufio, *args: bufio.read(*args))

    def test_flush_and_readinto(self):
        def _readinto(bufio, n=-1):
            b = bytearray(n if n >= 0 else 9999)
            n = bufio.readinto(b)
            return bytes(b[:n])
        self.check_flush_and_read(_readinto)

    def test_flush_and_peek(self):
        def _peek(bufio, n=-1):
            # This relies on the fact that the buffer can contain the whole
            # raw stream, otherwise peek() can return less.
            b = bufio.peek(n)
            if n != -1:
                b = b[:n]
            bufio.seek(len(b), 1)
            return b
        self.check_flush_and_read(_peek)

    def test_flush_and_write(self):
        raw = self.BytesIO(b"abcdefghi")
        bufio = self.tp(raw)

        bufio.write(b"123")
        bufio.flush()
        bufio.write(b"45")
        bufio.flush()
        bufio.seek(0, 0)
        self.assertEqual(b"12345fghi", raw.getvalue())
        self.assertEqual(b"12345fghi", bufio.read())

    def test_threads(self):
        BufferedReaderTest.test_threads(self)
        BufferedWriterTest.test_threads(self)

    def test_writes_and_peek(self):
        def _peek(bufio):
            bufio.peek(1)
        self.check_writes(_peek)
        def _peek(bufio):
            pos = bufio.tell()
            bufio.seek(-1, 1)
            bufio.peek(1)
            bufio.seek(pos, 0)
        self.check_writes(_peek)

    def test_writes_and_reads(self):
        def _read(bufio):
            bufio.seek(-1, 1)
            bufio.read(1)
        self.check_writes(_read)

    def test_writes_and_read1s(self):
        def _read1(bufio):
            bufio.seek(-1, 1)
            bufio.read1(1)
        self.check_writes(_read1)

    def test_writes_and_readintos(self):
        def _read(bufio):
            bufio.seek(-1, 1)
            bufio.readinto(bytearray(1))
        self.check_writes(_read)

    def test_write_after_readahead(self):
        # Issue #6629: writing after the buffer was filled by readahead should
        # first rewind the raw stream.
        for overwrite_size in [1, 5]:
            raw = self.BytesIO(b"A" * 10)
            bufio = self.tp(raw, 4)
            # Trigger readahead
            self.assertEqual(bufio.read(1), b"A")
            self.assertEqual(bufio.tell(), 1)
            # Overwriting should rewind the raw stream if it needs so
            bufio.write(b"B" * overwrite_size)
            self.assertEqual(bufio.tell(), overwrite_size + 1)
            # If the write size was smaller than the buffer size, flush() and
            # check that rewind happens.
            bufio.flush()
            self.assertEqual(bufio.tell(), overwrite_size + 1)
            s = raw.getvalue()
            self.assertEqual(s,
                b"A" + b"B" * overwrite_size + b"A" * (9 - overwrite_size))

    def test_write_rewind_write(self):
        # Various combinations of reading / writing / seeking backwards / writing again
        def mutate(bufio, pos1, pos2):
            assert pos2 >= pos1
            # Fill the buffer
            bufio.seek(pos1)
            bufio.read(pos2 - pos1)
            bufio.write(b'\x02')
            # This writes earlier than the previous write, but still inside
            # the buffer.
            bufio.seek(pos1)
            bufio.write(b'\x01')

        b = b"\x80\x81\x82\x83\x84"
        for i in range(0, len(b)):
            for j in range(i, len(b)):
                raw = self.BytesIO(b)
                bufio = self.tp(raw, 100)
                mutate(bufio, i, j)
                bufio.flush()
                expected = bytearray(b)
                expected[j] = 2
                expected[i] = 1
                self.assertEqual(raw.getvalue(), expected,
                                 "failed result for i=%d, j=%d" % (i, j))

    def test_truncate_after_read_or_write(self):
        raw = self.BytesIO(b"A" * 10)
        bufio = self.tp(raw, 100)
        self.assertEqual(bufio.read(2), b"AA") # the read buffer gets filled
        self.assertEqual(bufio.truncate(), 2)
        self.assertEqual(bufio.write(b"BB"), 2) # the write buffer increases
        self.assertEqual(bufio.truncate(), 4)

    def test_misbehaved_io(self):
        BufferedReaderTest.test_misbehaved_io(self)
        BufferedWriterTest.test_misbehaved_io(self)

    def test_interleaved_read_write(self):
        # Test for issue #12213
        with self.BytesIO(b'abcdefgh') as raw:
            with self.tp(raw, 100) as f:
                f.write(b"1")
                self.assertEqual(f.read(1), b'b')
                f.write(b'2')
                self.assertEqual(f.read1(1), b'd')
                f.write(b'3')
                buf = bytearray(1)
                f.readinto(buf)
                self.assertEqual(buf, b'f')
                f.write(b'4')
                self.assertEqual(f.peek(1), b'h')
                f.flush()
                self.assertEqual(raw.getvalue(), b'1b2d3f4h')

        with self.BytesIO(b'abc') as raw:
            with self.tp(raw, 100) as f:
                self.assertEqual(f.read(1), b'a')
                f.write(b"2")
                self.assertEqual(f.read(1), b'c')
                f.flush()
                self.assertEqual(raw.getvalue(), b'a2c')

    def test_read1_after_write(self):
        with self.BytesIO(b'abcdef') as raw:
            with self.tp(raw, 3) as f:
                f.write(b"1")
                self.assertEqual(f.read1(1), b'b')
                f.flush()
                self.assertEqual(raw.getvalue(), b'1bcdef')
        with self.BytesIO(b'abcdef') as raw:
            with self.tp(raw, 3) as f:
                f.write(b"1")
                self.assertEqual(f.read1(), b'bcd')
                f.flush()
                self.assertEqual(raw.getvalue(), b'1bcdef')
        with self.BytesIO(b'abcdef') as raw:
            with self.tp(raw, 3) as f:
                f.write(b"1")
                # XXX: read(100) returns different numbers of bytes
                # in Python and C implementations.
                self.assertEqual(f.read1(100)[:3], b'bcd')
                f.flush()
                self.assertEqual(raw.getvalue(), b'1bcdef')

    def test_interleaved_readline_write(self):
        with self.BytesIO(b'ab\ncdef\ng\n') as raw:
            with self.tp(raw) as f:
                f.write(b'1')
                self.assertEqual(f.readline(), b'b\n')
                f.write(b'2')
                self.assertEqual(f.readline(), b'def\n')
                f.write(b'3')
                self.assertEqual(f.readline(), b'\n')
                f.flush()
                self.assertEqual(raw.getvalue(), b'1b\n2def\n3\n')

    # You can't construct a BufferedRandom over a non-seekable stream.
    test_unseekable = None

    # writable() returns True, so there's no point to test it over
    # a writable stream.
    test_truncate_on_read_only = None


class CBufferedRandomTest(BufferedRandomTest, SizeofTest):
    tp = io.BufferedRandom

    def test_garbage_collection(self):
        CBufferedReaderTest.test_garbage_collection(self)
        CBufferedWriterTest.test_garbage_collection(self)

    def test_args_error(self):
        # Issue #17275
        with self.assertRaisesRegex(TypeError, "BufferedRandom"):
            self.tp(self.BytesIO(), 1024, 1024, 1024)


class PyBufferedRandomTest(BufferedRandomTest):
    tp = pyio.BufferedRandom


# To fully exercise seek/tell, the StatefulIncrementalDecoder has these
# properties:
#   - A single output character can correspond to many bytes of input.
#   - The number of input bytes to complete the character can be
#     undetermined until the last input byte is received.
#   - The number of input bytes can vary depending on previous input.
#   - A single input byte can correspond to many characters of output.
#   - The number of output characters can be undetermined until the
#     last input byte is received.
#   - The number of output characters can vary depending on previous input.

class StatefulIncrementalDecoder(codecs.IncrementalDecoder):
    """
    For testing seek/tell behavior with a stateful, buffering decoder.

    Input is a sequence of words.  Words may be fixed-length (length set
    by input) or variable-length (period-terminated).  In variable-length
    mode, extra periods are ignored.  Possible words are:
      - 'i' followed by a number sets the input length, I (maximum 99).
        When I is set to 0, words are space-terminated.
      - 'o' followed by a number sets the output length, O (maximum 99).
      - Any other word is converted into a word followed by a period on
        the output.  The output word consists of the input word truncated
        or padded out with hyphens to make its length equal to O.  If O
        is 0, the word is output verbatim without truncating or padding.
    I and O are initially set to 1.  When I changes, any buffered input is
    re-scanned according to the new I.  EOF also terminates the last word.
    """

    def __init__(self, errors='strict'):
        codecs.IncrementalDecoder.__init__(self, errors)
        self.reset()

    def __repr__(self):
        return '<SID %x>' % id(self)

    def reset(self):
        self.i = 1
        self.o = 1
        self.buffer = bytearray()

    def getstate(self):
        i, o = self.i ^ 1, self.o ^ 1 # so that flags = 0 after reset()
        return bytes(self.buffer), i*100 + o

    def setstate(self, state):
        buffer, io = state
        self.buffer = bytearray(buffer)
        i, o = divmod(io, 100)
        self.i, self.o = i ^ 1, o ^ 1

    def decode(self, input, final=False):
        output = ''
        for b in input:
            if self.i == 0: # variable-length, terminated with period
                if b == ord('.'):
                    if self.buffer:
                        output += self.process_word()
                else:
                    self.buffer.append(b)
            else: # fixed-length, terminate after self.i bytes
                self.buffer.append(b)
                if len(self.buffer) == self.i:
                    output += self.process_word()
        if final and self.buffer: # EOF terminates the last word
            output += self.process_word()
        return output

    def process_word(self):
        output = ''
        if self.buffer[0] == ord('i'):
            self.i = min(99, int(self.buffer[1:] or 0)) # set input length
        elif self.buffer[0] == ord('o'):
            self.o = min(99, int(self.buffer[1:] or 0)) # set output length
        else:
            output = self.buffer.decode('ascii')
            if len(output) < self.o:
                output += '-'*self.o # pad out with hyphens
            if self.o:
                output = output[:self.o] # truncate to output length
            output += '.'
        self.buffer = bytearray()
        return output

    codecEnabled = False


# bpo-41919: This method is separated from StatefulIncrementalDecoder to avoid a resource leak
# when registering codecs and cleanup functions.
def lookupTestDecoder(name):
    if StatefulIncrementalDecoder.codecEnabled and name == 'test_decoder':
        latin1 = codecs.lookup('latin-1')
        return codecs.CodecInfo(
            name='test_decoder', encode=latin1.encode, decode=None,
            incrementalencoder=None,
            streamreader=None, streamwriter=None,
            incrementaldecoder=StatefulIncrementalDecoder)


class StatefulIncrementalDecoderTest(__TestCase):
    """
    Make sure the StatefulIncrementalDecoder actually works.
    """

    test_cases = [
        # I=1, O=1 (fixed-length input == fixed-length output)
        (b'abcd', False, 'a.b.c.d.'),
        # I=0, O=0 (variable-length input, variable-length output)
        (b'oiabcd', True, 'abcd.'),
        # I=0, O=0 (should ignore extra periods)
        (b'oi...abcd...', True, 'abcd.'),
        # I=0, O=6 (variable-length input, fixed-length output)
        (b'i.o6.x.xyz.toolongtofit.', False, 'x-----.xyz---.toolon.'),
        # I=2, O=6 (fixed-length input < fixed-length output)
        (b'i.i2.o6xyz', True, 'xy----.z-----.'),
        # I=6, O=3 (fixed-length input > fixed-length output)
        (b'i.o3.i6.abcdefghijklmnop', True, 'abc.ghi.mno.'),
        # I=0, then 3; O=29, then 15 (with longer output)
        (b'i.o29.a.b.cde.o15.abcdefghijabcdefghij.i3.a.b.c.d.ei00k.l.m', True,
         'a----------------------------.' +
         'b----------------------------.' +
         'cde--------------------------.' +
         'abcdefghijabcde.' +
         'a.b------------.' +
         '.c.------------.' +
         'd.e------------.' +
         'k--------------.' +
         'l--------------.' +
         'm--------------.')
    ]

    def test_decoder(self):
        # Try a few one-shot test cases.
        for input, eof, output in self.test_cases:
            d = StatefulIncrementalDecoder()
            self.assertEqual(d.decode(input, eof), output)

        # Also test an unfinished decode, followed by forcing EOF.
        d = StatefulIncrementalDecoder()
        self.assertEqual(d.decode(b'oiabcd'), '')
        self.assertEqual(d.decode(b'', 1), 'abcd.')

class TextIOWrapperTest:

    def setUp(self):
        self.testdata = b"AAA\r\nBBB\rCCC\r\nDDD\nEEE\r\n"
        self.normalized = b"AAA\nBBB\nCCC\nDDD\nEEE\n".decode("ascii")
        os_helper.unlink(os_helper.TESTFN)
        codecs.register(lookupTestDecoder)
        self.addCleanup(codecs.unregister, lookupTestDecoder)

    def tearDown(self):
        os_helper.unlink(os_helper.TESTFN)

    def test_constructor(self):
        r = self.BytesIO(b"\xc3\xa9\n\n")
        b = self.BufferedReader(r, 1000)
        t = self.TextIOWrapper(b, encoding="utf-8")
        t.__init__(b, encoding="latin-1", newline="\r\n")
        self.assertEqual(t.encoding, "latin-1")
        self.assertEqual(t.line_buffering, False)
        t.__init__(b, encoding="utf-8", line_buffering=True)
        self.assertEqual(t.encoding, "utf-8")
        self.assertEqual(t.line_buffering, True)
        self.assertEqual("\xe9\n", t.readline())
        invalid_type = TypeError if self.is_C else ValueError
        with self.assertRaises(invalid_type):
            t.__init__(b, encoding=42)
        with self.assertRaises(UnicodeEncodeError):
            t.__init__(b, encoding='\udcfe')
        with self.assertRaises(ValueError):
            t.__init__(b, encoding='utf-8\0')
        with self.assertRaises(invalid_type):
            t.__init__(b, encoding="utf-8", errors=42)
        if support.Py_DEBUG or sys.flags.dev_mode or self.is_C:
            with self.assertRaises(UnicodeEncodeError):
                t.__init__(b, encoding="utf-8", errors='\udcfe')
        if support.Py_DEBUG or sys.flags.dev_mode or self.is_C:
            with self.assertRaises(ValueError):
                t.__init__(b, encoding="utf-8", errors='replace\0')
        with self.assertRaises(TypeError):
            t.__init__(b, encoding="utf-8", newline=42)
        with self.assertRaises(ValueError):
            t.__init__(b, encoding="utf-8", newline='\udcfe')
        with self.assertRaises(ValueError):
            t.__init__(b, encoding="utf-8", newline='\n\0')
        with self.assertRaises(ValueError):
            t.__init__(b, encoding="utf-8", newline='xyzzy')

    def test_uninitialized(self):
        t = self.TextIOWrapper.__new__(self.TextIOWrapper)
        del t
        t = self.TextIOWrapper.__new__(self.TextIOWrapper)
        self.assertRaises(Exception, repr, t)
        self.assertRaisesRegex((ValueError, AttributeError),
                               'uninitialized|has no attribute',
                               t.read, 0)
        t.__init__(self.MockRawIO(), encoding="utf-8")
        self.assertEqual(t.read(0), '')

    def test_non_text_encoding_codecs_are_rejected(self):
        # Ensure the constructor complains if passed a codec that isn't
        # marked as a text encoding
        # http://bugs.python.org/issue20404
        r = self.BytesIO()
        b = self.BufferedWriter(r)
        with self.assertRaisesRegex(LookupError, "is not a text encoding"):
            self.TextIOWrapper(b, encoding="hex")

    def test_detach(self):
        r = self.BytesIO()
        b = self.BufferedWriter(r)
        t = self.TextIOWrapper(b, encoding="ascii")
        self.assertIs(t.detach(), b)

        t = self.TextIOWrapper(b, encoding="ascii")
        t.write("howdy")
        self.assertFalse(r.getvalue())
        t.detach()
        self.assertEqual(r.getvalue(), b"howdy")
        self.assertRaises(ValueError, t.detach)

        # Operations independent of the detached stream should still work
        repr(t)
        self.assertEqual(t.encoding, "ascii")
        self.assertEqual(t.errors, "strict")
        self.assertFalse(t.line_buffering)
        self.assertFalse(t.write_through)

    def test_repr(self):
        raw = self.BytesIO("hello".encode("utf-8"))
        b = self.BufferedReader(raw)
        t = self.TextIOWrapper(b, encoding="utf-8")
        modname = self.TextIOWrapper.__module__
        self.assertRegex(repr(t),
                         r"<(%s\.)?TextIOWrapper encoding='utf-8'>" % modname)
        raw.name = "dummy"
        self.assertRegex(repr(t),
                         r"<(%s\.)?TextIOWrapper name='dummy' encoding='utf-8'>" % modname)
        t.mode = "r"
        self.assertRegex(repr(t),
                         r"<(%s\.)?TextIOWrapper name='dummy' mode='r' encoding='utf-8'>" % modname)
        raw.name = b"dummy"
        self.assertRegex(repr(t),
                         r"<(%s\.)?TextIOWrapper name=b'dummy' mode='r' encoding='utf-8'>" % modname)

        t.buffer.detach()
        repr(t)  # Should not raise an exception

    def test_recursive_repr(self):
        # Issue #25455
        raw = self.BytesIO()
        t = self.TextIOWrapper(raw, encoding="utf-8")
        with support.swap_attr(raw, 'name', t), support.infinite_recursion(25):
            with self.assertRaises(RuntimeError):
                repr(t)  # Should not crash

    def test_subclass_repr(self):
        class TestSubclass(self.TextIOWrapper):
            pass

        f = TestSubclass(self.StringIO())
        self.assertIn(TestSubclass.__name__, repr(f))

    def test_line_buffering(self):
        r = self.BytesIO()
        b = self.BufferedWriter(r, 1000)
        t = self.TextIOWrapper(b, encoding="utf-8", newline="\n", line_buffering=True)
        t.write("X")
        self.assertEqual(r.getvalue(), b"")  # No flush happened
        t.write("Y\nZ")
        self.assertEqual(r.getvalue(), b"XY\nZ")  # All got flushed
        t.write("A\rB")
        self.assertEqual(r.getvalue(), b"XY\nZA\rB")

    def test_reconfigure_line_buffering(self):
        r = self.BytesIO()
        b = self.BufferedWriter(r, 1000)
        t = self.TextIOWrapper(b, encoding="utf-8", newline="\n", line_buffering=False)
        t.write("AB\nC")
        self.assertEqual(r.getvalue(), b"")

        t.reconfigure(line_buffering=True)   # implicit flush
        self.assertEqual(r.getvalue(), b"AB\nC")
        t.write("DEF\nG")
        self.assertEqual(r.getvalue(), b"AB\nCDEF\nG")
        t.write("H")
        self.assertEqual(r.getvalue(), b"AB\nCDEF\nG")
        t.reconfigure(line_buffering=False)   # implicit flush
        self.assertEqual(r.getvalue(), b"AB\nCDEF\nGH")
        t.write("IJ")
        self.assertEqual(r.getvalue(), b"AB\nCDEF\nGH")

        # Keeping default value
        t.reconfigure()
        t.reconfigure(line_buffering=None)
        self.assertEqual(t.line_buffering, False)
        t.reconfigure(line_buffering=True)
        t.reconfigure()
        t.reconfigure(line_buffering=None)
        self.assertEqual(t.line_buffering, True)

    @unittest.skipIf(sys.flags.utf8_mode, "utf-8 mode is enabled")
    def test_default_encoding(self):
        old_environ = dict(os.environ)
        try:
            # try to get a user preferred encoding different than the current
            # locale encoding to check that TextIOWrapper() uses the current
            # locale encoding and not the user preferred encoding
            for key in ('LC_ALL', 'LANG', 'LC_CTYPE'):
                if key in os.environ:
                    del os.environ[key]

            current_locale_encoding = locale.getencoding()
            b = self.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EncodingWarning)
                t = self.TextIOWrapper(b)
            self.assertEqual(t.encoding, current_locale_encoding)
        finally:
            os.environ.clear()
            os.environ.update(old_environ)

    def test_encoding(self):
        # Check the encoding attribute is always set, and valid
        b = self.BytesIO()
        t = self.TextIOWrapper(b, encoding="utf-8")
        self.assertEqual(t.encoding, "utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EncodingWarning)
            t = self.TextIOWrapper(b)
        self.assertIsNotNone(t.encoding)
        codecs.lookup(t.encoding)

    def test_encoding_errors_reading(self):
        # (1) default
        b = self.BytesIO(b"abc\n\xff\n")
        t = self.TextIOWrapper(b, encoding="ascii")
        self.assertRaises(UnicodeError, t.read)
        # (2) explicit strict
        b = self.BytesIO(b"abc\n\xff\n")
        t = self.TextIOWrapper(b, encoding="ascii", errors="strict")
        self.assertRaises(UnicodeError, t.read)
        # (3) ignore
        b = self.BytesIO(b"abc\n\xff\n")
        t = self.TextIOWrapper(b, encoding="ascii", errors="ignore")
        self.assertEqual(t.read(), "abc\n\n")
        # (4) replace
        b = self.BytesIO(b"abc\n\xff\n")
        t = self.TextIOWrapper(b, encoding="ascii", errors="replace")
        self.assertEqual(t.read(), "abc\n\ufffd\n")

    def test_encoding_errors_writing(self):
        # (1) default
        b = self.BytesIO()
        t = self.TextIOWrapper(b, encoding="ascii")
        self.assertRaises(UnicodeError, t.write, "\xff")
        # (2) explicit strict
        b = self.BytesIO()
        t = self.TextIOWrapper(b, encoding="ascii", errors="strict")
        self.assertRaises(UnicodeError, t.write, "\xff")
        # (3) ignore
        b = self.BytesIO()
        t = self.TextIOWrapper(b, encoding="ascii", errors="ignore",
                             newline="\n")
        t.write("abc\xffdef\n")
        t.flush()
        self.assertEqual(b.getvalue(), b"abcdef\n")
        # (4) replace
        b = self.BytesIO()
        t = self.TextIOWrapper(b, encoding="ascii", errors="replace",
                             newline="\n")
        t.write("abc\xffdef\n")
        t.flush()
        self.assertEqual(b.getvalue(), b"abc?def\n")

    def test_newlines(self):
        input_lines = [ "unix\n", "windows\r\n", "os9\r", "last\n", "nonl" ]

        tests = [
            [ None, [ 'unix\n', 'windows\n', 'os9\n', 'last\n', 'nonl' ] ],
            [ '', input_lines ],
            [ '\n', [ "unix\n", "windows\r\n", "os9\rlast\n", "nonl" ] ],
            [ '\r\n', [ "unix\nwindows\r\n", "os9\rlast\nnonl" ] ],
            [ '\r', [ "unix\nwindows\r", "\nos9\r", "last\nnonl" ] ],
        ]
        encodings = (
            'utf-8', 'latin-1',
            'utf-16', 'utf-16-le', 'utf-16-be',
            'utf-32', 'utf-32-le', 'utf-32-be',
        )

        # Try a range of buffer sizes to test the case where \r is the last
        # character in TextIOWrapper._pending_line.
        for encoding in encodings:
            # XXX: str.encode() should return bytes
            data = bytes(''.join(input_lines).encode(encoding))
            for do_reads in (False, True):
                for bufsize in range(1, 10):
                    for newline, exp_lines in tests:
                        bufio = self.BufferedReader(self.BytesIO(data), bufsize)
                        textio = self.TextIOWrapper(bufio, newline=newline,
                                                  encoding=encoding)
                        if do_reads:
                            got_lines = []
                            while True:
                                c2 = textio.read(2)
                                if c2 == '':
                                    break
                                self.assertEqual(len(c2), 2)
                                got_lines.append(c2 + textio.readline())
                        else:
                            got_lines = list(textio)

                        for got_line, exp_line in zip(got_lines, exp_lines):
                            self.assertEqual(got_line, exp_line)
                        self.assertEqual(len(got_lines), len(exp_lines))

    def test_newlines_input(self):
        testdata = b"AAA\nBB\x00B\nCCC\rDDD\rEEE\r\nFFF\r\nGGG"
        normalized = testdata.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        for newline, expected in [
            (None, normalized.decode("ascii").splitlines(keepends=True)),
            ("", testdata.decode("ascii").splitlines(keepends=True)),
            ("\n", ["AAA\n", "BB\x00B\n", "CCC\rDDD\rEEE\r\n", "FFF\r\n", "GGG"]),
            ("\r\n", ["AAA\nBB\x00B\nCCC\rDDD\rEEE\r\n", "FFF\r\n", "GGG"]),
            ("\r",  ["AAA\nBB\x00B\nCCC\r", "DDD\r", "EEE\r", "\nFFF\r", "\nGGG"]),
            ]:
            buf = self.BytesIO(testdata)
            txt = self.TextIOWrapper(buf, encoding="ascii", newline=newline)
            self.assertEqual(txt.readlines(), expected)
            txt.seek(0)
            self.assertEqual(txt.read(), "".join(expected))

    def test_newlines_output(self):
        testdict = {
            "": b"AAA\nBBB\nCCC\nX\rY\r\nZ",
            "\n": b"AAA\nBBB\nCCC\nX\rY\r\nZ",
            "\r": b"AAA\rBBB\rCCC\rX\rY\r\rZ",
            "\r\n": b"AAA\r\nBBB\r\nCCC\r\nX\rY\r\r\nZ",
            }
        tests = [(None, testdict[os.linesep])] + sorted(testdict.items())
        for newline, expected in tests:
            buf = self.BytesIO()
            txt = self.TextIOWrapper(buf, encoding="ascii", newline=newline)
            txt.write("AAA\nB")
            txt.write("BB\nCCC\n")
            txt.write("X\rY\r\nZ")
            txt.flush()
            self.assertEqual(buf.closed, False)
            self.assertEqual(buf.getvalue(), expected)

    def test_destructor(self):
        l = []
        base = self.BytesIO
        class MyBytesIO(base):
            def close(self):
                l.append(self.getvalue())
                base.close(self)
        b = MyBytesIO()
        t = self.TextIOWrapper(b, encoding="ascii")
        t.write("abc")
        del t
        support.gc_collect()
        self.assertEqual([b"abc"], l)

    def test_override_destructor(self):
        record = []
        class MyTextIO(self.TextIOWrapper):
            def __del__(self):
                record.append(1)
                try:
                    f = super().__del__
                except AttributeError:
                    pass
                else:
                    f()
            def close(self):
                record.append(2)
                super().close()
            def flush(self):
                record.append(3)
                super().flush()
        b = self.BytesIO()
        t = MyTextIO(b, encoding="ascii")
        del t
        support.gc_collect()
        self.assertEqual(record, [1, 2, 3])

    def test_error_through_destructor(self):
        # Test that the exception state is not modified by a destructor,
        # even if close() fails.
        rawio = self.CloseFailureIO()
        with support.catch_unraisable_exception() as cm:
            with self.assertRaises(AttributeError):
                self.TextIOWrapper(rawio, encoding="utf-8").xyzzy

            self.assertEqual(cm.unraisable.exc_type, OSError)

    # Systematic tests of the text I/O API

    def test_basic_io(self):
        for chunksize in (1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65):
            for enc in "ascii", "latin-1", "utf-8" :# , "utf-16-be", "utf-16-le":
                f = self.open(os_helper.TESTFN, "w+", encoding=enc)
                f._CHUNK_SIZE = chunksize
                self.assertEqual(f.write("abc"), 3)
                f.close()
                f = self.open(os_helper.TESTFN, "r+", encoding=enc)
                f._CHUNK_SIZE = chunksize
                self.assertEqual(f.tell(), 0)
                self.assertEqual(f.read(), "abc")
                cookie = f.tell()
                self.assertEqual(f.seek(0), 0)
                self.assertEqual(f.read(None), "abc")
                f.seek(0)
                self.assertEqual(f.read(2), "ab")
                self.assertEqual(f.read(1), "c")
                self.assertEqual(f.read(1), "")
                self.assertEqual(f.read(), "")
                self.assertEqual(f.tell(), cookie)
                self.assertEqual(f.seek(0), 0)
                self.assertEqual(f.seek(0, 2), cookie)
                self.assertEqual(f.write("def"), 3)
                self.assertEqual(f.seek(cookie), cookie)
                self.assertEqual(f.read(), "def")
                if enc.startswith("utf"):
                    self.multi_line_test(f, enc)
                f.close()

    def multi_line_test(self, f, enc):
        f.seek(0)
        f.truncate()
        sample = "s\xff\u0fff\uffff"
        wlines = []
        for size in (0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 62, 63, 64, 65, 1000):
            chars = []
            for i in range(size):
                chars.append(sample[i % len(sample)])
            line = "".join(chars) + "\n"
            wlines.append((f.tell(), line))
            f.write(line)
        f.seek(0)
        rlines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            rlines.append((pos, line))
        self.assertEqual(rlines, wlines)

    def test_telling(self):
        f = self.open(os_helper.TESTFN, "w+", encoding="utf-8")
        p0 = f.tell()
        f.write("\xff\n")
        p1 = f.tell()
        f.write("\xff\n")
        p2 = f.tell()
        f.seek(0)
        self.assertEqual(f.tell(), p0)
        self.assertEqual(f.readline(), "\xff\n")
        self.assertEqual(f.tell(), p1)
        self.assertEqual(f.readline(), "\xff\n")
        self.assertEqual(f.tell(), p2)
        f.seek(0)
        for line in f:
            self.assertEqual(line, "\xff\n")
            self.assertRaises(OSError, f.tell)
        self.assertEqual(f.tell(), p2)
        f.close()

    def test_seeking(self):
        chunk_size = _default_chunk_size()
        prefix_size = chunk_size - 2
        u_prefix = "a" * prefix_size
        prefix = bytes(u_prefix.encode("utf-8"))
        self.assertEqual(len(u_prefix), len(prefix))
        u_suffix = "\u8888\n"
        suffix = bytes(u_suffix.encode("utf-8"))
        line = prefix + suffix
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(line*2)
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as f:
            s = f.read(prefix_size)
            self.assertEqual(s, str(prefix, "ascii"))
            self.assertEqual(f.tell(), prefix_size)
            self.assertEqual(f.readline(), u_suffix)

    def test_seeking_too(self):
        # Regression test for a specific bug
        data = b'\xe0\xbf\xbf\n'
        with self.open(os_helper.TESTFN, "wb") as f:
            f.write(data)
        with self.open(os_helper.TESTFN, "r", encoding="utf-8") as f:
            f._CHUNK_SIZE  # Just test that it exists
            f._CHUNK_SIZE = 2
            f.readline()
            f.tell()

    def test_seek_and_tell(self):
        #Test seek/tell using the StatefulIncrementalDecoder.
        # Make test faster by doing smaller seeks
        CHUNK_SIZE = 128

        def test_seek_and_tell_with_data(data, min_pos=0):
            """Tell/seek to various points within a data stream and ensure
            that the decoded data returned by read() is consistent."""
            f = self.open(os_helper.TESTFN, 'wb')
            f.write(data)
            f.close()
            f = self.open(os_helper.TESTFN, encoding='test_decoder')
            f._CHUNK_SIZE = CHUNK_SIZE
            decoded = f.read()
            f.close()

            for i in range(min_pos, len(decoded) + 1): # seek positions
                for j in [1, 5, len(decoded) - i]: # read lengths
                    f = self.open(os_helper.TESTFN, encoding='test_decoder')
                    self.assertEqual(f.read(i), decoded[:i])
                    cookie = f.tell()
                    self.assertEqual(f.read(j), decoded[i:i + j])
                    f.seek(cookie)
                    self.assertEqual(f.read(), decoded[i:])
                    f.close()

        # Enable the test decoder.
        StatefulIncrementalDecoder.codecEnabled = 1

        # Run the tests.
        try:
            # Try each test case.
            for input, _, _ in StatefulIncrementalDecoderTest.test_cases:
                test_seek_and_tell_with_data(input)

            # Position each test case so that it crosses a chunk boundary.
            for input, _, _ in StatefulIncrementalDecoderTest.test_cases:
                offset = CHUNK_SIZE - len(input)//2
                prefix = b'.'*offset
                # Don't bother seeking into the prefix (takes too long).
                min_pos = offset*2
                test_seek_and_tell_with_data(prefix + input, min_pos)

        # Ensure our test decoder won't interfere with subsequent tests.
        finally:
            StatefulIncrementalDecoder.codecEnabled = 0

    def test_multibyte_seek_and_tell(self):
        f = self.open(os_helper.TESTFN, "w", encoding="euc_jp")
        f.write("AB\n\u3046\u3048\n")
        f.close()

        f = self.open(os_helper.TESTFN, "r", encoding="euc_jp")
        self.assertEqual(f.readline(), "AB\n")
        p0 = f.tell()
        self.assertEqual(f.readline(), "\u3046\u3048\n")
        p1 = f.tell()
        f.seek(p0)
        self.assertEqual(f.readline(), "\u3046\u3048\n")
        self.assertEqual(f.tell(), p1)
        f.close()

    def test_seek_with_encoder_state(self):
        f = self.open(os_helper.TESTFN, "w", encoding="euc_jis_2004")
        f.write("\u00e6\u0300")
        p0 = f.tell()
        f.write("\u00e6")
        f.seek(p0)
        f.write("\u0300")
        f.close()

        f = self.open(os_helper.TESTFN, "r", encoding="euc_jis_2004")
        self.assertEqual(f.readline(), "\u00e6\u0300\u0300")
        f.close()

    def test_encoded_writes(self):
        data = "1234567890"
        tests = ("utf-16",
                 "utf-16-le",
                 "utf-16-be",
                 "utf-32",
                 "utf-32-le",
                 "utf-32-be")
        for encoding in tests:
            buf = self.BytesIO()
            f = self.TextIOWrapper(buf, encoding=encoding)
            # Check if the BOM is written only once (see issue1753).
            f.write(data)
            f.write(data)
            f.seek(0)
            self.assertEqual(f.read(), data * 2)
            f.seek(0)
            self.assertEqual(f.read(), data * 2)
            self.assertEqual(buf.getvalue(), (data * 2).encode(encoding))

    def test_unreadable(self):
        class UnReadable(self.BytesIO):
            def readable(self):
                return False
        txt = self.TextIOWrapper(UnReadable(), encoding="utf-8")
        self.assertRaises(OSError, txt.read)

    def test_read_one_by_one(self):
        txt = self.TextIOWrapper(self.BytesIO(b"AA\r\nBB"), encoding="utf-8")
        reads = ""
        while True:
            c = txt.read(1)
            if not c:
                break
            reads += c
        self.assertEqual(reads, "AA\nBB")

    def test_readlines(self):
        txt = self.TextIOWrapper(self.BytesIO(b"AA\nBB\nCC"), encoding="utf-8")
        self.assertEqual(txt.readlines(), ["AA\n", "BB\n", "CC"])
        txt.seek(0)
        self.assertEqual(txt.readlines(None), ["AA\n", "BB\n", "CC"])
        txt.seek(0)
        self.assertEqual(txt.readlines(5), ["AA\n", "BB\n"])

    # read in amounts equal to TextIOWrapper._CHUNK_SIZE which is 128.
    def test_read_by_chunk(self):
        # make sure "\r\n" straddles 128 char boundary.
        txt = self.TextIOWrapper(self.BytesIO(b"A" * 127 + b"\r\nB"), encoding="utf-8")
        reads = ""
        while True:
            c = txt.read(128)
            if not c:
                break
            reads += c
        self.assertEqual(reads, "A"*127+"\nB")

    def test_writelines(self):
        l = ['ab', 'cd', 'ef']
        buf = self.BytesIO()
        txt = self.TextIOWrapper(buf, encoding="utf-8")
        txt.writelines(l)
        txt.flush()
        self.assertEqual(buf.getvalue(), b'abcdef')

    def test_writelines_userlist(self):
        l = UserList(['ab', 'cd', 'ef'])
        buf = self.BytesIO()
        txt = self.TextIOWrapper(buf, encoding="utf-8")
        txt.writelines(l)
        txt.flush()
        self.assertEqual(buf.getvalue(), b'abcdef')

    def test_writelines_error(self):
        txt = self.TextIOWrapper(self.BytesIO(), encoding="utf-8")
        self.assertRaises(TypeError, txt.writelines, [1, 2, 3])
        self.assertRaises(TypeError, txt.writelines, None)
        self.assertRaises(TypeError, txt.writelines, b'abc')

    def test_issue1395_1(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")

        # read one char at a time
        reads = ""
        while True:
            c = txt.read(1)
            if not c:
                break
            reads += c
        self.assertEqual(reads, self.normalized)

    def test_issue1395_2(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        txt._CHUNK_SIZE = 4

        reads = ""
        while True:
            c = txt.read(4)
            if not c:
                break
            reads += c
        self.assertEqual(reads, self.normalized)

    def test_issue1395_3(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        txt._CHUNK_SIZE = 4

        reads = txt.read(4)
        reads += txt.read(4)
        reads += txt.readline()
        reads += txt.readline()
        reads += txt.readline()
        self.assertEqual(reads, self.normalized)

    def test_issue1395_4(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        txt._CHUNK_SIZE = 4

        reads = txt.read(4)
        reads += txt.read()
        self.assertEqual(reads, self.normalized)

    def test_issue1395_5(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        txt._CHUNK_SIZE = 4

        reads = txt.read(4)
        pos = txt.tell()
        txt.seek(0)
        txt.seek(pos)
        self.assertEqual(txt.read(4), "BBB\n")

    def test_issue2282(self):
        buffer = self.BytesIO(self.testdata)
        txt = self.TextIOWrapper(buffer, encoding="ascii")

        self.assertEqual(buffer.seekable(), txt.seekable())

    def test_append_bom(self):
        # The BOM is not written again when appending to a non-empty file
        filename = os_helper.TESTFN
        for charset in ('utf-8-sig', 'utf-16', 'utf-32'):
            with self.open(filename, 'w', encoding=charset) as f:
                f.write('aaa')
                pos = f.tell()
            with self.open(filename, 'rb') as f:
                self.assertEqual(f.read(), 'aaa'.encode(charset))

            with self.open(filename, 'a', encoding=charset) as f:
                f.write('xxx')
            with self.open(filename, 'rb') as f:
                self.assertEqual(f.read(), 'aaaxxx'.encode(charset))

    def test_seek_bom(self):
        # Same test, but when seeking manually
        filename = os_helper.TESTFN
        for charset in ('utf-8-sig', 'utf-16', 'utf-32'):
            with self.open(filename, 'w', encoding=charset) as f:
                f.write('aaa')
                pos = f.tell()
            with self.open(filename, 'r+', encoding=charset) as f:
                f.seek(pos)
                f.write('zzz')
                f.seek(0)
                f.write('bbb')
            with self.open(filename, 'rb') as f:
                self.assertEqual(f.read(), 'bbbzzz'.encode(charset))

    def test_seek_append_bom(self):
        # Same test, but first seek to the start and then to the end
        filename = os_helper.TESTFN
        for charset in ('utf-8-sig', 'utf-16', 'utf-32'):
            with self.open(filename, 'w', encoding=charset) as f:
                f.write('aaa')
            with self.open(filename, 'a', encoding=charset) as f:
                f.seek(0)
                f.seek(0, self.SEEK_END)
                f.write('xxx')
            with self.open(filename, 'rb') as f:
                self.assertEqual(f.read(), 'aaaxxx'.encode(charset))

    def test_errors_property(self):
        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as f:
            self.assertEqual(f.errors, "strict")
        with self.open(os_helper.TESTFN, "w", encoding="utf-8", errors="replace") as f:
            self.assertEqual(f.errors, "replace")

    @support.no_tracing
    @threading_helper.requires_working_threading()
    def test_threads_write(self):
        # Issue6750: concurrent writes could duplicate data
        event = threading.Event()
        with self.open(os_helper.TESTFN, "w", encoding="utf-8", buffering=1) as f:
            def run(n):
                text = "Thread%03d\n" % n
                event.wait()
                f.write(text)
            threads = [threading.Thread(target=run, args=(x,))
                       for x in range(20)]
            with threading_helper.start_threads(threads, event.set):
                time.sleep(0.02)
        with self.open(os_helper.TESTFN, encoding="utf-8") as f:
            content = f.read()
            for n in range(20):
                self.assertEqual(content.count("Thread%03d\n" % n), 1)

    def test_flush_error_on_close(self):
        # Test that text file is closed despite failed flush
        # and that flush() is called before file closed.
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        closed = []
        def bad_flush():
            closed[:] = [txt.closed, txt.buffer.closed]
            raise OSError()
        txt.flush = bad_flush
        self.assertRaises(OSError, txt.close) # exception not swallowed
        self.assertTrue(txt.closed)
        self.assertTrue(txt.buffer.closed)
        self.assertTrue(closed)      # flush() called
        self.assertFalse(closed[0])  # flush() called before file closed
        self.assertFalse(closed[1])
        txt.flush = lambda: None  # break reference loop

    def test_close_error_on_close(self):
        buffer = self.BytesIO(self.testdata)
        def bad_flush():
            raise OSError('flush')
        def bad_close():
            raise OSError('close')
        buffer.close = bad_close
        txt = self.TextIOWrapper(buffer, encoding="ascii")
        txt.flush = bad_flush
        with self.assertRaises(OSError) as err: # exception not swallowed
            txt.close()
        self.assertEqual(err.exception.args, ('close',))
        self.assertIsInstance(err.exception.__context__, OSError)
        self.assertEqual(err.exception.__context__.args, ('flush',))
        self.assertFalse(txt.closed)

        # Silence destructor error
        buffer.close = lambda: None
        txt.flush = lambda: None

    def test_nonnormalized_close_error_on_close(self):
        # Issue #21677
        buffer = self.BytesIO(self.testdata)
        def bad_flush():
            raise non_existing_flush
        def bad_close():
            raise non_existing_close
        buffer.close = bad_close
        txt = self.TextIOWrapper(buffer, encoding="ascii")
        txt.flush = bad_flush
        with self.assertRaises(NameError) as err: # exception not swallowed
            txt.close()
        self.assertIn('non_existing_close', str(err.exception))
        self.assertIsInstance(err.exception.__context__, NameError)
        self.assertIn('non_existing_flush', str(err.exception.__context__))
        self.assertFalse(txt.closed)

        # Silence destructor error
        buffer.close = lambda: None
        txt.flush = lambda: None

    def test_multi_close(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        txt.close()
        txt.close()
        txt.close()
        self.assertRaises(ValueError, txt.flush)

    def test_unseekable(self):
        txt = self.TextIOWrapper(self.MockUnseekableIO(self.testdata), encoding="utf-8")
        self.assertRaises(self.UnsupportedOperation, txt.tell)
        self.assertRaises(self.UnsupportedOperation, txt.seek, 0)

    def test_readonly_attributes(self):
        txt = self.TextIOWrapper(self.BytesIO(self.testdata), encoding="ascii")
        buf = self.BytesIO(self.testdata)
        with self.assertRaises(AttributeError):
            txt.buffer = buf

    def test_rawio(self):
        # Issue #12591: TextIOWrapper must work with raw I/O objects, so
        # that subprocess.Popen() can have the required unbuffered
        # semantics with universal_newlines=True.
        raw = self.MockRawIO([b'abc', b'def', b'ghi\njkl\nopq\n'])
        txt = self.TextIOWrapper(raw, encoding='ascii', newline='\n')
        # Reads
        self.assertEqual(txt.read(4), 'abcd')
        self.assertEqual(txt.readline(), 'efghi\n')
        self.assertEqual(list(txt), ['jkl\n', 'opq\n'])

    def test_rawio_write_through(self):
        # Issue #12591: with write_through=True, writes don't need a flush
        raw = self.MockRawIO([b'abc', b'def', b'ghi\njkl\nopq\n'])
        txt = self.TextIOWrapper(raw, encoding='ascii', newline='\n',
                                 write_through=True)
        txt.write('1')
        txt.write('23\n4')
        txt.write('5')
        self.assertEqual(b''.join(raw._write_stack), b'123\n45')

    def test_bufio_write_through(self):
        # Issue #21396: write_through=True doesn't force a flush()
        # on the underlying binary buffered object.
        flush_called, write_called = [], []
        class BufferedWriter(self.BufferedWriter):
            def flush(self, *args, **kwargs):
                flush_called.append(True)
                return super().flush(*args, **kwargs)
            def write(self, *args, **kwargs):
                write_called.append(True)
                return super().write(*args, **kwargs)

        rawio = self.BytesIO()
        data = b"a"
        bufio = BufferedWriter(rawio, len(data)*2)
        textio = self.TextIOWrapper(bufio, encoding='ascii',
                                    write_through=True)
        # write to the buffered io but don't overflow the buffer
        text = data.decode('ascii')
        textio.write(text)

        # buffer.flush is not called with write_through=True
        self.assertFalse(flush_called)
        # buffer.write *is* called with write_through=True
        self.assertTrue(write_called)
        self.assertEqual(rawio.getvalue(), b"") # no flush

        write_called = [] # reset
        textio.write(text * 10) # total content is larger than bufio buffer
        self.assertTrue(write_called)
        self.assertEqual(rawio.getvalue(), data * 11) # all flushed

    def test_reconfigure_write_through(self):
        raw = self.MockRawIO([])
        t = self.TextIOWrapper(raw, encoding='ascii', newline='\n')
        t.write('1')
        t.reconfigure(write_through=True)  # implied flush
        self.assertEqual(t.write_through, True)
        self.assertEqual(b''.join(raw._write_stack), b'1')
        t.write('23')
        self.assertEqual(b''.join(raw._write_stack), b'123')
        t.reconfigure(write_through=False)
        self.assertEqual(t.write_through, False)
        t.write('45')
        t.flush()
        self.assertEqual(b''.join(raw._write_stack), b'12345')
        # Keeping default value
        t.reconfigure()
        t.reconfigure(write_through=None)
        self.assertEqual(t.write_through, False)
        t.reconfigure(write_through=True)
        t.reconfigure()
        t.reconfigure(write_through=None)
        self.assertEqual(t.write_through, True)

    def test_read_nonbytes(self):
        # Issue #17106
        # Crash when underlying read() returns non-bytes
        t = self.TextIOWrapper(self.StringIO('a'), encoding="utf-8")
        self.assertRaises(TypeError, t.read, 1)
        t = self.TextIOWrapper(self.StringIO('a'), encoding="utf-8")
        self.assertRaises(TypeError, t.readline)
        t = self.TextIOWrapper(self.StringIO('a'), encoding="utf-8")
        self.assertRaises(TypeError, t.read)

    def test_illegal_encoder(self):
        # Issue 31271: Calling write() while the return value of encoder's
        # encode() is invalid shouldn't cause an assertion failure.
        rot13 = codecs.lookup("rot13")
        with support.swap_attr(rot13, '_is_text_encoding', True):
            t = self.TextIOWrapper(self.BytesIO(b'foo'), encoding="rot13")
        self.assertRaises(TypeError, t.write, 'bar')

    def test_illegal_decoder(self):
        # Issue #17106
        # Bypass the early encoding check added in issue 20404
        def _make_illegal_wrapper():
            quopri = codecs.lookup("quopri")
            quopri._is_text_encoding = True
            try:
                t = self.TextIOWrapper(self.BytesIO(b'aaaaaa'),
                                       newline='\n', encoding="quopri")
            finally:
                quopri._is_text_encoding = False
            return t
        # Crash when decoder returns non-string
        t = _make_illegal_wrapper()
        self.assertRaises(TypeError, t.read, 1)
        t = _make_illegal_wrapper()
        self.assertRaises(TypeError, t.readline)
        t = _make_illegal_wrapper()
        self.assertRaises(TypeError, t.read)

        # Issue 31243: calling read() while the return value of decoder's
        # getstate() is invalid should neither crash the interpreter nor
        # raise a SystemError.
        def _make_very_illegal_wrapper(getstate_ret_val):
            class BadDecoder:
                def getstate(self):
                    return getstate_ret_val
            def _get_bad_decoder(dummy):
                return BadDecoder()
            quopri = codecs.lookup("quopri")
            with support.swap_attr(quopri, 'incrementaldecoder',
                                   _get_bad_decoder):
                return _make_illegal_wrapper()
        t = _make_very_illegal_wrapper(42)
        self.assertRaises(TypeError, t.read, 42)
        t = _make_very_illegal_wrapper(())
        self.assertRaises(TypeError, t.read, 42)
        t = _make_very_illegal_wrapper((1, 2))
        self.assertRaises(TypeError, t.read, 42)

    def _check_create_at_shutdown(self, **kwargs):
        # Issue #20037: creating a TextIOWrapper at shutdown
        # shouldn't crash the interpreter.
        iomod = self.io.__name__
        code = """if 1:
            import codecs
            import {iomod} as io

            # Avoid looking up codecs at shutdown
            codecs.lookup('utf-8')

            class C:
                def __del__(self):
                    io.TextIOWrapper(io.BytesIO(), **{kwargs})
                    print("ok")
            c = C()
            """.format(iomod=iomod, kwargs=kwargs)
        return assert_python_ok("-c", code)

    def test_create_at_shutdown_without_encoding(self):
        rc, out, err = self._check_create_at_shutdown()
        if err:
            # Can error out with a RuntimeError if the module state
            # isn't found.
            self.assertIn(self.shutdown_error, err.decode())
        else:
            self.assertEqual("ok", out.decode().strip())

    def test_create_at_shutdown_with_encoding(self):
        rc, out, err = self._check_create_at_shutdown(encoding='utf-8',
                                                      errors='strict')
        self.assertFalse(err)
        self.assertEqual("ok", out.decode().strip())

    def test_read_byteslike(self):
        r = MemviewBytesIO(b'Just some random string\n')
        t = self.TextIOWrapper(r, 'utf-8')

        # TextIOwrapper will not read the full string, because
        # we truncate it to a multiple of the native int size
        # so that we can construct a more complex memoryview.
        bytes_val =  _to_memoryview(r.getvalue()).tobytes()

        self.assertEqual(t.read(200), bytes_val.decode('utf-8'))

    def test_issue22849(self):
        class F(object):
            def readable(self): return True
            def writable(self): return True
            def seekable(self): return True

        for i in range(10):
            try:
                self.TextIOWrapper(F(), encoding='utf-8')
            except Exception:
                pass

        F.tell = lambda x: 0
        t = self.TextIOWrapper(F(), encoding='utf-8')

    def test_reconfigure_locale(self):
        wrapper = self.TextIOWrapper(self.BytesIO(b"test"))
        wrapper.reconfigure(encoding="locale")

    def test_reconfigure_encoding_read(self):
        # latin1 -> utf8
        # (latin1 can decode utf-8 encoded string)
        data = 'abc\xe9\n'.encode('latin1') + 'd\xe9f\n'.encode('utf8')
        raw = self.BytesIO(data)
        txt = self.TextIOWrapper(raw, encoding='latin1', newline='\n')
        self.assertEqual(txt.readline(), 'abc\xe9\n')
        with self.assertRaises(self.UnsupportedOperation):
            txt.reconfigure(encoding='utf-8')
        with self.assertRaises(self.UnsupportedOperation):
            txt.reconfigure(newline=None)

    def test_reconfigure_write_fromascii(self):
        # ascii has a specific encodefunc in the C implementation,
        # but utf-8-sig has not. Make sure that we get rid of the
        # cached encodefunc when we switch encoders.
        raw = self.BytesIO()
        txt = self.TextIOWrapper(raw, encoding='ascii', newline='\n')
        txt.write('foo\n')
        txt.reconfigure(encoding='utf-8-sig')
        txt.write('\xe9\n')
        txt.flush()
        self.assertEqual(raw.getvalue(), b'foo\n\xc3\xa9\n')

    def test_reconfigure_write(self):
        # latin -> utf8
        raw = self.BytesIO()
        txt = self.TextIOWrapper(raw, encoding='latin1', newline='\n')
        txt.write('abc\xe9\n')
        txt.reconfigure(encoding='utf-8')
        self.assertEqual(raw.getvalue(), b'abc\xe9\n')
        txt.write('d\xe9f\n')
        txt.flush()
        self.assertEqual(raw.getvalue(), b'abc\xe9\nd\xc3\xa9f\n')

        # ascii -> utf-8-sig: ensure that no BOM is written in the middle of
        # the file
        raw = self.BytesIO()
        txt = self.TextIOWrapper(raw, encoding='ascii', newline='\n')
        txt.write('abc\n')
        txt.reconfigure(encoding='utf-8-sig')
        txt.write('d\xe9f\n')
        txt.flush()
        self.assertEqual(raw.getvalue(), b'abc\nd\xc3\xa9f\n')

    def test_reconfigure_write_non_seekable(self):
        raw = self.BytesIO()
        raw.seekable = lambda: False
        raw.seek = None
        txt = self.TextIOWrapper(raw, encoding='ascii', newline='\n')
        txt.write('abc\n')
        txt.reconfigure(encoding='utf-8-sig')
        txt.write('d\xe9f\n')
        txt.flush()

        # If the raw stream is not seekable, there'll be a BOM
        self.assertEqual(raw.getvalue(),  b'abc\n\xef\xbb\xbfd\xc3\xa9f\n')

    def test_reconfigure_defaults(self):
        txt = self.TextIOWrapper(self.BytesIO(), 'ascii', 'replace', '\n')
        txt.reconfigure(encoding=None)
        self.assertEqual(txt.encoding, 'ascii')
        self.assertEqual(txt.errors, 'replace')
        txt.write('LF\n')

        txt.reconfigure(newline='\r\n')
        self.assertEqual(txt.encoding, 'ascii')
        self.assertEqual(txt.errors, 'replace')

        txt.reconfigure(errors='ignore')
        self.assertEqual(txt.encoding, 'ascii')
        self.assertEqual(txt.errors, 'ignore')
        txt.write('CRLF\n')

        txt.reconfigure(encoding='utf-8', newline=None)
        self.assertEqual(txt.errors, 'strict')
        txt.seek(0)
        self.assertEqual(txt.read(), 'LF\nCRLF\n')

        self.assertEqual(txt.detach().getvalue(), b'LF\nCRLF\r\n')

    def test_reconfigure_errors(self):
        txt = self.TextIOWrapper(self.BytesIO(), 'ascii', 'replace', '\r')
        with self.assertRaises(TypeError):  # there was a crash
            txt.reconfigure(encoding=42)
        if self.is_C:
            with self.assertRaises(UnicodeEncodeError):
                txt.reconfigure(encoding='\udcfe')
            with self.assertRaises(LookupError):
                txt.reconfigure(encoding='locale\0')
        # TODO: txt.reconfigure(encoding='utf-8\0')
        # TODO: txt.reconfigure(encoding='nonexisting')
        with self.assertRaises(TypeError):
            txt.reconfigure(errors=42)
        if self.is_C:
            with self.assertRaises(UnicodeEncodeError):
                txt.reconfigure(errors='\udcfe')
        # TODO: txt.reconfigure(errors='ignore\0')
        # TODO: txt.reconfigure(errors='nonexisting')
        with self.assertRaises(TypeError):
            txt.reconfigure(newline=42)
        with self.assertRaises(ValueError):
            txt.reconfigure(newline='\udcfe')
        with self.assertRaises(ValueError):
            txt.reconfigure(newline='xyz')
        if not self.is_C:
            # TODO: Should fail in C too.
            with self.assertRaises(ValueError):
                txt.reconfigure(newline='\n\0')
        if self.is_C:
            # TODO: Use __bool__(), not __index__().
            with self.assertRaises(ZeroDivisionError):
                txt.reconfigure(line_buffering=BadIndex())
            with self.assertRaises(OverflowError):
                txt.reconfigure(line_buffering=2**1000)
            with self.assertRaises(ZeroDivisionError):
                txt.reconfigure(write_through=BadIndex())
            with self.assertRaises(OverflowError):
                txt.reconfigure(write_through=2**1000)
            with self.assertRaises(ZeroDivisionError):  # there was a crash
                txt.reconfigure(line_buffering=BadIndex(),
                                write_through=BadIndex())
        self.assertEqual(txt.encoding, 'ascii')
        self.assertEqual(txt.errors, 'replace')
        self.assertIs(txt.line_buffering, False)
        self.assertIs(txt.write_through, False)

        txt.reconfigure(encoding='latin1', errors='ignore', newline='\r\n',
                        line_buffering=True, write_through=True)
        self.assertEqual(txt.encoding, 'latin1')
        self.assertEqual(txt.errors, 'ignore')
        self.assertIs(txt.line_buffering, True)
        self.assertIs(txt.write_through, True)

    def test_reconfigure_newline(self):
        raw = self.BytesIO(b'CR\rEOF')
        txt = self.TextIOWrapper(raw, 'ascii', newline='\n')
        txt.reconfigure(newline=None)
        self.assertEqual(txt.readline(), 'CR\n')
        raw = self.BytesIO(b'CR\rEOF')
        txt = self.TextIOWrapper(raw, 'ascii', newline='\n')
        txt.reconfigure(newline='')
        self.assertEqual(txt.readline(), 'CR\r')
        raw = self.BytesIO(b'CR\rLF\nEOF')
        txt = self.TextIOWrapper(raw, 'ascii', newline='\r')
        txt.reconfigure(newline='\n')
        self.assertEqual(txt.readline(), 'CR\rLF\n')
        raw = self.BytesIO(b'LF\nCR\rEOF')
        txt = self.TextIOWrapper(raw, 'ascii', newline='\n')
        txt.reconfigure(newline='\r')
        self.assertEqual(txt.readline(), 'LF\nCR\r')
        raw = self.BytesIO(b'CR\rCRLF\r\nEOF')
        txt = self.TextIOWrapper(raw, 'ascii', newline='\r')
        txt.reconfigure(newline='\r\n')
        self.assertEqual(txt.readline(), 'CR\rCRLF\r\n')

        txt = self.TextIOWrapper(self.BytesIO(), 'ascii', newline='\r')
        txt.reconfigure(newline=None)
        txt.write('linesep\n')
        txt.reconfigure(newline='')
        txt.write('LF\n')
        txt.reconfigure(newline='\n')
        txt.write('LF\n')
        txt.reconfigure(newline='\r')
        txt.write('CR\n')
        txt.reconfigure(newline='\r\n')
        txt.write('CRLF\n')
        expected = 'linesep' + os.linesep + 'LF\nLF\nCR\rCRLF\r\n'
        self.assertEqual(txt.detach().getvalue().decode('ascii'), expected)

    def test_issue25862(self):
        # Assertion failures occurred in tell() after read() and write().
        t = self.TextIOWrapper(self.BytesIO(b'test'), encoding='ascii')
        t.read(1)
        t.read()
        t.tell()
        t = self.TextIOWrapper(self.BytesIO(b'test'), encoding='ascii')
        t.read(1)
        t.write('x')
        t.tell()

    def test_issue35928(self):
        p = self.BufferedRWPair(self.BytesIO(b'foo\nbar\n'), self.BytesIO())
        f = self.TextIOWrapper(p)
        res = f.readline()
        self.assertEqual(res, 'foo\n')
        f.write(res)
        self.assertEqual(res + f.readline(), 'foo\nbar\n')

    def test_pickling_subclass(self):
        global MyTextIO
        class MyTextIO(self.TextIOWrapper):
            def __init__(self, raw, tag):
                super().__init__(raw)
                self.tag = tag
            def __getstate__(self):
                return self.tag, self.buffer.getvalue()
            def __setstate__(slf, state):
                tag, value = state
                slf.__init__(self.BytesIO(value), tag)

        raw = self.BytesIO(b'data')
        txt = MyTextIO(raw, 'ham')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=proto):
                pickled = pickle.dumps(txt, proto)
                newtxt = pickle.loads(pickled)
                self.assertEqual(newtxt.buffer.getvalue(), b'data')
                self.assertEqual(newtxt.tag, 'ham')
        del MyTextIO


class MemviewBytesIO(io.BytesIO):
    '''A BytesIO object whose read method returns memoryviews
       rather than bytes'''

    def read1(self, len_):
        return _to_memoryview(super().read1(len_))

    def read(self, len_):
        return _to_memoryview(super().read(len_))

def _to_memoryview(buf):
    '''Convert bytes-object *buf* to a non-trivial memoryview'''

    arr = array.array('i')
    idx = len(buf) - len(buf) % arr.itemsize
    arr.frombytes(buf[:idx])
    return memoryview(arr)


class CTextIOWrapperTest(TextIOWrapperTest, __TestCase):
    io = io
    shutdown_error = "LookupError: unknown encoding: ascii"

    def test_initialization(self):
        r = self.BytesIO(b"\xc3\xa9\n\n")
        b = self.BufferedReader(r, 1000)
        t = self.TextIOWrapper(b, encoding="utf-8")
        self.assertRaises(ValueError, t.__init__, b, encoding="utf-8", newline='xyzzy')
        self.assertRaises(ValueError, t.read)

        t = self.TextIOWrapper.__new__(self.TextIOWrapper)
        self.assertRaises(Exception, repr, t)

    def test_garbage_collection(self):
        # C TextIOWrapper objects are collected, and collecting them flushes
        # all data to disk.
        # The Python version has __del__, so it ends in gc.garbage instead.
        with warnings_helper.check_warnings(('', ResourceWarning)):
            rawio = self.FileIO(os_helper.TESTFN, "wb")
            b = self.BufferedWriter(rawio)
            t = self.TextIOWrapper(b, encoding="ascii")
            t.write("456def")
            t.x = t
            wr = weakref.ref(t)
            del t
            support.gc_collect()
        self.assertIsNone(wr(), wr)
        with self.open(os_helper.TESTFN, "rb") as f:
            self.assertEqual(f.read(), b"456def")

    def test_rwpair_cleared_before_textio(self):
        # Issue 13070: TextIOWrapper's finalization would crash when called
        # after the reference to the underlying BufferedRWPair's writer got
        # cleared by the GC.
        for i in range(1000):
            b1 = self.BufferedRWPair(self.MockRawIO(), self.MockRawIO())
            t1 = self.TextIOWrapper(b1, encoding="ascii")
            b2 = self.BufferedRWPair(self.MockRawIO(), self.MockRawIO())
            t2 = self.TextIOWrapper(b2, encoding="ascii")
            # circular references
            t1.buddy = t2
            t2.buddy = t1
        support.gc_collect()

    def test_del__CHUNK_SIZE_SystemError(self):
        t = self.TextIOWrapper(self.BytesIO(), encoding='ascii')
        with self.assertRaises(AttributeError):
            del t._CHUNK_SIZE

    def test_internal_buffer_size(self):
        # bpo-43260: TextIOWrapper's internal buffer should not store
        # data larger than chunk size.
        chunk_size = 8192  # default chunk size, updated later

        class MockIO(self.MockRawIO):
            def write(self, data):
                if len(data) > chunk_size:
                    raise RuntimeError
                return super().write(data)

        buf = MockIO()
        t = self.TextIOWrapper(buf, encoding="ascii")
        chunk_size = t._CHUNK_SIZE
        t.write("abc")
        t.write("def")
        # default chunk size is 8192 bytes so t don't write data to buf.
        self.assertEqual([], buf._write_stack)

        with self.assertRaises(RuntimeError):
            t.write("x"*(chunk_size+1))

        self.assertEqual([b"abcdef"], buf._write_stack)
        t.write("ghi")
        t.write("x"*chunk_size)
        self.assertEqual([b"abcdef", b"ghi", b"x"*chunk_size], buf._write_stack)

    def test_issue119506(self):
        chunk_size = 8192

        class MockIO(self.MockRawIO):
            written = False
            def write(self, data):
                if not self.written:
                    self.written = True
                    t.write("middle")
                return super().write(data)

        buf = MockIO()
        t = self.TextIOWrapper(buf)
        t.write("abc")
        t.write("def")
        # writing data which size >= chunk_size cause flushing buffer before write.
        t.write("g" * chunk_size)
        t.flush()

        self.assertEqual([b"abcdef", b"middle", b"g"*chunk_size],
                         buf._write_stack)


class PyTextIOWrapperTest(TextIOWrapperTest, __TestCase):
    io = pyio
    shutdown_error = "LookupError: unknown encoding: ascii"


class IncrementalNewlineDecoderTest:

    def check_newline_decoding_utf8(self, decoder):
        # UTF-8 specific tests for a newline decoder
        def _check_decode(b, s, **kwargs):
            # We exercise getstate() / setstate() as well as decode()
            state = decoder.getstate()
            self.assertEqual(decoder.decode(b, **kwargs), s)
            decoder.setstate(state)
            self.assertEqual(decoder.decode(b, **kwargs), s)

        _check_decode(b'\xe8\xa2\x88', "\u8888")

        _check_decode(b'\xe8', "")
        _check_decode(b'\xa2', "")
        _check_decode(b'\x88', "\u8888")

        _check_decode(b'\xe8', "")
        _check_decode(b'\xa2', "")
        _check_decode(b'\x88', "\u8888")

        _check_decode(b'\xe8', "")
        self.assertRaises(UnicodeDecodeError, decoder.decode, b'', final=True)

        decoder.reset()
        _check_decode(b'\n', "\n")
        _check_decode(b'\r', "")
        _check_decode(b'', "\n", final=True)
        _check_decode(b'\r', "\n", final=True)

        _check_decode(b'\r', "")
        _check_decode(b'a', "\na")

        _check_decode(b'\r\r\n', "\n\n")
        _check_decode(b'\r', "")
        _check_decode(b'\r', "\n")
        _check_decode(b'\na', "\na")

        _check_decode(b'\xe8\xa2\x88\r\n', "\u8888\n")
        _check_decode(b'\xe8\xa2\x88', "\u8888")
        _check_decode(b'\n', "\n")
        _check_decode(b'\xe8\xa2\x88\r', "\u8888")
        _check_decode(b'\n', "\n")

    def check_newline_decoding(self, decoder, encoding):
        result = []
        if encoding is not None:
            encoder = codecs.getincrementalencoder(encoding)()
            def _decode_bytewise(s):
                # Decode one byte at a time
                for b in encoder.encode(s):
                    result.append(decoder.decode(bytes([b])))
        else:
            encoder = None
            def _decode_bytewise(s):
                # Decode one char at a time
                for c in s:
                    result.append(decoder.decode(c))
        self.assertEqual(decoder.newlines, None)
        _decode_bytewise("abc\n\r")
        self.assertEqual(decoder.newlines, '\n')
        _decode_bytewise("\nabc")
        self.assertEqual(decoder.newlines, ('\n', '\r\n'))
        _decode_bytewise("abc\r")
        self.assertEqual(decoder.newlines, ('\n', '\r\n'))
        _decode_bytewise("abc")
        self.assertEqual(decoder.newlines, ('\r', '\n', '\r\n'))
        _decode_bytewise("abc\r")
        self.assertEqual("".join(result), "abc\n\nabcabc\nabcabc")
        decoder.reset()
        input = "abc"
        if encoder is not None:
            encoder.reset()
            input = encoder.encode(input)
        self.assertEqual(decoder.decode(input), "abc")
        self.assertEqual(decoder.newlines, None)

    def test_newline_decoder(self):
        encodings = (
            # None meaning the IncrementalNewlineDecoder takes unicode input
            # rather than bytes input
            None, 'utf-8', 'latin-1',
            'utf-16', 'utf-16-le', 'utf-16-be',
            'utf-32', 'utf-32-le', 'utf-32-be',
        )
        for enc in encodings:
            decoder = enc and codecs.getincrementaldecoder(enc)()
            decoder = self.IncrementalNewlineDecoder(decoder, translate=True)
            self.check_newline_decoding(decoder, enc)
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder = self.IncrementalNewlineDecoder(decoder, translate=True)
        self.check_newline_decoding_utf8(decoder)
        self.assertRaises(TypeError, decoder.setstate, 42)

    def test_newline_bytes(self):
        # Issue 5433: Excessive optimization in IncrementalNewlineDecoder
        def _check(dec):
            self.assertEqual(dec.newlines, None)
            self.assertEqual(dec.decode("\u0D00"), "\u0D00")
            self.assertEqual(dec.newlines, None)
            self.assertEqual(dec.decode("\u0A00"), "\u0A00")
            self.assertEqual(dec.newlines, None)
        dec = self.IncrementalNewlineDecoder(None, translate=False)
        _check(dec)
        dec = self.IncrementalNewlineDecoder(None, translate=True)
        _check(dec)

    def test_translate(self):
        # issue 35062
        for translate in (-2, -1, 1, 2):
            decoder = codecs.getincrementaldecoder("utf-8")()
            decoder = self.IncrementalNewlineDecoder(decoder, translate)
            self.check_newline_decoding_utf8(decoder)
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder = self.IncrementalNewlineDecoder(decoder, translate=0)
        self.assertEqual(decoder.decode(b"\r\r\n"), "\r\r\n")

class CIncrementalNewlineDecoderTest(IncrementalNewlineDecoderTest, __TestCase):
    @support.cpython_only
    def test_uninitialized(self):
        uninitialized = self.IncrementalNewlineDecoder.__new__(
            self.IncrementalNewlineDecoder)
        self.assertRaises(ValueError, uninitialized.decode, b'bar')
        self.assertRaises(ValueError, uninitialized.getstate)
        self.assertRaises(ValueError, uninitialized.setstate, (b'foo', 0))
        self.assertRaises(ValueError, uninitialized.reset)


class PyIncrementalNewlineDecoderTest(IncrementalNewlineDecoderTest, __TestCase):
    pass


# XXX Tests for open()

class MiscIOTest:

    # for test__all__, actual values are set in subclasses
    name_of_module = None
    extra_exported = ()
    not_exported = ()

    def tearDown(self):
        os_helper.unlink(os_helper.TESTFN)

    def test___all__(self):
        support.check__all__(self, self.io, self.name_of_module,
                             extra=self.extra_exported,
                             not_exported=self.not_exported)

    def test_attributes(self):
        f = self.open(os_helper.TESTFN, "wb", buffering=0)
        self.assertEqual(f.mode, "wb")
        f.close()

        f = self.open(os_helper.TESTFN, "w+", encoding="utf-8")
        self.assertEqual(f.mode,            "w+")
        self.assertEqual(f.buffer.mode,     "rb+") # Does it really matter?
        self.assertEqual(f.buffer.raw.mode, "rb+")

        g = self.open(f.fileno(), "wb", closefd=False)
        self.assertEqual(g.mode,     "wb")
        self.assertEqual(g.raw.mode, "wb")
        self.assertEqual(g.name,     f.fileno())
        self.assertEqual(g.raw.name, f.fileno())
        f.close()
        g.close()

    def test_removed_u_mode(self):
        # bpo-37330: The "U" mode has been removed in Python 3.11
        for mode in ("U", "rU", "r+U"):
            with self.assertRaises(ValueError) as cm:
                self.open(os_helper.TESTFN, mode)
            self.assertIn('invalid mode', str(cm.exception))

    @unittest.skipIf(
        support.is_emscripten, "fstat() of a pipe fd is not supported"
    )
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_open_pipe_with_append(self):
        # bpo-27805: Ignore ESPIPE from lseek() in open().
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        f = self.open(w, 'a', encoding="utf-8")
        self.addCleanup(f.close)
        # Check that the file is marked non-seekable. On Windows, however, lseek
        # somehow succeeds on pipes.
        if sys.platform != 'win32':
            self.assertFalse(f.seekable())

    def test_io_after_close(self):
        for kwargs in [
                {"mode": "w"},
                {"mode": "wb"},
                {"mode": "w", "buffering": 1},
                {"mode": "w", "buffering": 2},
                {"mode": "wb", "buffering": 0},
                {"mode": "r"},
                {"mode": "rb"},
                {"mode": "r", "buffering": 1},
                {"mode": "r", "buffering": 2},
                {"mode": "rb", "buffering": 0},
                {"mode": "w+"},
                {"mode": "w+b"},
                {"mode": "w+", "buffering": 1},
                {"mode": "w+", "buffering": 2},
                {"mode": "w+b", "buffering": 0},
            ]:
            if "b" not in kwargs["mode"]:
                kwargs["encoding"] = "utf-8"
            f = self.open(os_helper.TESTFN, **kwargs)
            f.close()
            self.assertRaises(ValueError, f.flush)
            self.assertRaises(ValueError, f.fileno)
            self.assertRaises(ValueError, f.isatty)
            self.assertRaises(ValueError, f.__iter__)
            if hasattr(f, "peek"):
                self.assertRaises(ValueError, f.peek, 1)
            self.assertRaises(ValueError, f.read)
            if hasattr(f, "read1"):
                self.assertRaises(ValueError, f.read1, 1024)
                self.assertRaises(ValueError, f.read1)
            if hasattr(f, "readall"):
                self.assertRaises(ValueError, f.readall)
            if hasattr(f, "readinto"):
                self.assertRaises(ValueError, f.readinto, bytearray(1024))
            if hasattr(f, "readinto1"):
                self.assertRaises(ValueError, f.readinto1, bytearray(1024))
            self.assertRaises(ValueError, f.readline)
            self.assertRaises(ValueError, f.readlines)
            self.assertRaises(ValueError, f.readlines, 1)
            self.assertRaises(ValueError, f.seek, 0)
            self.assertRaises(ValueError, f.tell)
            self.assertRaises(ValueError, f.truncate)
            self.assertRaises(ValueError, f.write,
                              b"" if "b" in kwargs['mode'] else "")
            self.assertRaises(ValueError, f.writelines, [])
            self.assertRaises(ValueError, next, f)

    def test_blockingioerror(self):
        # Various BlockingIOError issues
        class C(str):
            pass
        c = C("")
        b = self.BlockingIOError(1, c)
        c.b = b
        b.c = c
        wr = weakref.ref(c)
        del c, b
        support.gc_collect()
        self.assertIsNone(wr(), wr)

    def test_abcs(self):
        # Test the visible base classes are ABCs.
        self.assertIsInstance(self.IOBase, abc.ABCMeta)
        self.assertIsInstance(self.RawIOBase, abc.ABCMeta)
        self.assertIsInstance(self.BufferedIOBase, abc.ABCMeta)
        self.assertIsInstance(self.TextIOBase, abc.ABCMeta)

    def _check_abc_inheritance(self, abcmodule):
        with self.open(os_helper.TESTFN, "wb", buffering=0) as f:
            self.assertIsInstance(f, abcmodule.IOBase)
            self.assertIsInstance(f, abcmodule.RawIOBase)
            self.assertNotIsInstance(f, abcmodule.BufferedIOBase)
            self.assertNotIsInstance(f, abcmodule.TextIOBase)
        with self.open(os_helper.TESTFN, "wb") as f:
            self.assertIsInstance(f, abcmodule.IOBase)
            self.assertNotIsInstance(f, abcmodule.RawIOBase)
            self.assertIsInstance(f, abcmodule.BufferedIOBase)
            self.assertNotIsInstance(f, abcmodule.TextIOBase)
        with self.open(os_helper.TESTFN, "w", encoding="utf-8") as f:
            self.assertIsInstance(f, abcmodule.IOBase)
            self.assertNotIsInstance(f, abcmodule.RawIOBase)
            self.assertNotIsInstance(f, abcmodule.BufferedIOBase)
            self.assertIsInstance(f, abcmodule.TextIOBase)

    def test_abc_inheritance(self):
        # Test implementations inherit from their respective ABCs
        self._check_abc_inheritance(self)

    def test_abc_inheritance_official(self):
        # Test implementations inherit from the official ABCs of the
        # baseline "io" module.
        self._check_abc_inheritance(io)

    def _check_warn_on_dealloc(self, *args, **kwargs):
        f = self.open(*args, **kwargs)
        r = repr(f)
        with self.assertWarns(ResourceWarning) as cm:
            f = None
            support.gc_collect()
        self.assertIn(r, str(cm.warning.args[0]))

    def test_warn_on_dealloc(self):
        self._check_warn_on_dealloc(os_helper.TESTFN, "wb", buffering=0)
        self._check_warn_on_dealloc(os_helper.TESTFN, "wb")
        self._check_warn_on_dealloc(os_helper.TESTFN, "w", encoding="utf-8")

    def _check_warn_on_dealloc_fd(self, *args, **kwargs):
        fds = []
        def cleanup_fds():
            for fd in fds:
                try:
                    os.close(fd)
                except OSError as e:
                    if e.errno != errno.EBADF:
                        raise
        self.addCleanup(cleanup_fds)
        r, w = os.pipe()
        fds += r, w
        self._check_warn_on_dealloc(r, *args, **kwargs)
        # When using closefd=False, there's no warning
        r, w = os.pipe()
        fds += r, w
        with warnings_helper.check_no_resource_warning(self):
            self.open(r, *args, closefd=False, **kwargs)

    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_warn_on_dealloc_fd(self):
        self._check_warn_on_dealloc_fd("rb", buffering=0)
        self._check_warn_on_dealloc_fd("rb")
        self._check_warn_on_dealloc_fd("r", encoding="utf-8")


    def test_pickling(self):
        # Pickling file objects is forbidden
        msg = "cannot pickle"
        for kwargs in [
                {"mode": "w"},
                {"mode": "wb"},
                {"mode": "wb", "buffering": 0},
                {"mode": "r"},
                {"mode": "rb"},
                {"mode": "rb", "buffering": 0},
                {"mode": "w+"},
                {"mode": "w+b"},
                {"mode": "w+b", "buffering": 0},
            ]:
            if "b" not in kwargs["mode"]:
                kwargs["encoding"] = "utf-8"
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(protocol=protocol, kwargs=kwargs):
                    with self.open(os_helper.TESTFN, **kwargs) as f:
                        with self.assertRaisesRegex(TypeError, msg):
                            pickle.dumps(f, protocol)

    @unittest.skipIf(
        support.is_emscripten, "fstat() of a pipe fd is not supported"
    )
    def test_nonblock_pipe_write_bigbuf(self):
        self._test_nonblock_pipe_write(16*1024)

    @unittest.skipIf(
        support.is_emscripten, "fstat() of a pipe fd is not supported"
    )
    def test_nonblock_pipe_write_smallbuf(self):
        self._test_nonblock_pipe_write(1024)

    @unittest.skipUnless(hasattr(os, 'set_blocking'),
                         'os.set_blocking() required for this test')
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def _test_nonblock_pipe_write(self, bufsize):
        sent = []
        received = []
        r, w = os.pipe()
        os.set_blocking(r, False)
        os.set_blocking(w, False)

        # To exercise all code paths in the C implementation we need
        # to play with buffer sizes.  For instance, if we choose a
        # buffer size less than or equal to _PIPE_BUF (4096 on Linux)
        # then we will never get a partial write of the buffer.
        rf = self.open(r, mode='rb', closefd=True, buffering=bufsize)
        wf = self.open(w, mode='wb', closefd=True, buffering=bufsize)

        with rf, wf:
            for N in 9999, 73, 7574:
                try:
                    i = 0
                    while True:
                        msg = bytes([i % 26 + 97]) * N
                        sent.append(msg)
                        wf.write(msg)
                        i += 1

                except self.BlockingIOError as e:
                    self.assertEqual(e.args[0], errno.EAGAIN)
                    self.assertEqual(e.args[2], e.characters_written)
                    sent[-1] = sent[-1][:e.characters_written]
                    received.append(rf.read())
                    msg = b'BLOCKED'
                    wf.write(msg)
                    sent.append(msg)

            while True:
                try:
                    wf.flush()
                    break
                except self.BlockingIOError as e:
                    self.assertEqual(e.args[0], errno.EAGAIN)
                    self.assertEqual(e.args[2], e.characters_written)
                    self.assertEqual(e.characters_written, 0)
                    received.append(rf.read())

            received += iter(rf.read, None)

        sent, received = b''.join(sent), b''.join(received)
        self.assertEqual(sent, received)
        self.assertTrue(wf.closed)
        self.assertTrue(rf.closed)

    def test_create_fail(self):
        # 'x' mode fails if file is existing
        with self.open(os_helper.TESTFN, 'w', encoding="utf-8"):
            pass
        self.assertRaises(FileExistsError, self.open, os_helper.TESTFN, 'x', encoding="utf-8")

    def test_create_writes(self):
        # 'x' mode opens for writing
        with self.open(os_helper.TESTFN, 'xb') as f:
            f.write(b"spam")
        with self.open(os_helper.TESTFN, 'rb') as f:
            self.assertEqual(b"spam", f.read())

    def test_open_allargs(self):
        # there used to be a buffer overflow in the parser for rawmode
        self.assertRaises(ValueError, self.open, os_helper.TESTFN, 'rwax+', encoding="utf-8")

    def test_check_encoding_errors(self):
        # bpo-37388: open() and TextIOWrapper must check encoding and errors
        # arguments in dev mode
        mod = self.io.__name__
        filename = __file__
        invalid = 'Boom, Shaka Laka, Boom!'
        code = textwrap.dedent(f'''
            import sys
            from {mod} import open, TextIOWrapper

            try:
                open({filename!r}, encoding={invalid!r})
            except LookupError:
                pass
            else:
                sys.exit(21)

            try:
                open({filename!r}, errors={invalid!r})
            except LookupError:
                pass
            else:
                sys.exit(22)

            fp = open({filename!r}, "rb")
            with fp:
                try:
                    TextIOWrapper(fp, encoding={invalid!r})
                except LookupError:
                    pass
                else:
                    sys.exit(23)

                try:
                    TextIOWrapper(fp, errors={invalid!r})
                except LookupError:
                    pass
                else:
                    sys.exit(24)

            sys.exit(10)
        ''')
        proc = assert_python_failure('-X', 'dev', '-c', code)
        self.assertEqual(proc.rc, 10, proc)

    def test_check_encoding_warning(self):
        # PEP 597: Raise warning when encoding is not specified
        # and sys.flags.warn_default_encoding is set.
        mod = self.io.__name__
        filename = __file__
        code = textwrap.dedent(f'''\
            import sys
            from {mod} import open, TextIOWrapper
            import pathlib

            with open({filename!r}) as f:           # line 5
                pass

            pathlib.Path({filename!r}).read_text()  # line 8
        ''')
        proc = assert_python_ok('-X', 'warn_default_encoding', '-c', code)
        warnings = proc.err.splitlines()
        self.assertEqual(len(warnings), 2)
        self.assertTrue(
            warnings[0].startswith(b"<string>:5: EncodingWarning: "))
        self.assertTrue(
            warnings[1].startswith(b"<string>:8: EncodingWarning: "))

    def test_text_encoding(self):
        # PEP 597, bpo-47000. io.text_encoding() returns "locale" or "utf-8"
        # based on sys.flags.utf8_mode
        code = "import io; print(io.text_encoding(None))"

        proc = assert_python_ok('-X', 'utf8=0', '-c', code)
        self.assertEqual(b"locale", proc.out.strip())

        proc = assert_python_ok('-X', 'utf8=1', '-c', code)
        self.assertEqual(b"utf-8", proc.out.strip())


class CMiscIOTest(MiscIOTest, __TestCase):
    io = io
    name_of_module = "io", "_io"
    extra_exported = "BlockingIOError",

    def test_readinto_buffer_overflow(self):
        # Issue #18025
        class BadReader(self.io.BufferedIOBase):
            def read(self, n=-1):
                return b'x' * 10**6
        bufio = BadReader()
        b = bytearray(2)
        self.assertRaises(ValueError, bufio.readinto, b)

    def check_daemon_threads_shutdown_deadlock(self, stream_name):
        # Issue #23309: deadlocks at shutdown should be avoided when a
        # daemon thread and the main thread both write to a file.
        code = """if 1:
            import sys
            import time
            import threading
            from test.support import SuppressCrashReport

            file = sys.{stream_name}

            def run():
                while True:
                    file.write('.')
                    file.flush()

            crash = SuppressCrashReport()
            crash.__enter__()
            # don't call __exit__(): the crash occurs at Python shutdown

            thread = threading.Thread(target=run)
            thread.daemon = True
            thread.start()

            time.sleep(0.5)
            file.write('!')
            file.flush()
            """.format_map(locals())
        res, _ = run_python_until_end("-c", code)
        err = res.err.decode()
        if res.rc != 0:
            # Failure: should be a fatal error
            pattern = (r"Fatal Python error: _enter_buffered_busy: "
                       r"could not acquire lock "
                       r"for <(_io\.)?BufferedWriter name='<{stream_name}>'> "
                       r"at interpreter shutdown, possibly due to "
                       r"daemon threads".format_map(locals()))
            self.assertRegex(err, pattern)
        else:
            self.assertFalse(err.strip('.!'))

    @threading_helper.requires_working_threading()
    @support.requires_resource('walltime')
    def test_daemon_threads_shutdown_stdout_deadlock(self):
        self.check_daemon_threads_shutdown_deadlock('stdout')

    @threading_helper.requires_working_threading()
    @support.requires_resource('walltime')
    def test_daemon_threads_shutdown_stderr_deadlock(self):
        self.check_daemon_threads_shutdown_deadlock('stderr')


class PyMiscIOTest(MiscIOTest, __TestCase):
    io = pyio
    name_of_module = "_pyio", "io"
    extra_exported = "BlockingIOError", "open_code",
    not_exported = "valid_seek_flags",


@unittest.skipIf(os.name == 'nt', 'POSIX signals required for this test.')
class SignalsTest:

    def setUp(self):
        self.oldalrm = signal.signal(signal.SIGALRM, self.alarm_interrupt)

    def tearDown(self):
        signal.signal(signal.SIGALRM, self.oldalrm)

    def alarm_interrupt(self, sig, frame):
        1/0

    def check_interrupted_write(self, item, bytes, **fdopen_kwargs):
        """Check that a partial write, when it gets interrupted, properly
        invokes the signal handler, and bubbles up the exception raised
        in the latter."""

        # XXX This test has three flaws that appear when objects are
        # XXX not reference counted.

        # - if wio.write() happens to trigger a garbage collection,
        #   the signal exception may be raised when some __del__
        #   method is running; it will not reach the assertRaises()
        #   call.

        # - more subtle, if the wio object is not destroyed at once
        #   and survives this function, the next opened file is likely
        #   to have the same fileno (since the file descriptor was
        #   actively closed).  When wio.__del__ is finally called, it
        #   will close the other's test file...  To trigger this with
        #   CPython, try adding "global wio" in this function.

        # - This happens only for streams created by the _pyio module,
        #   because a wio.close() that fails still consider that the
        #   file needs to be closed again.  You can try adding an
        #   "assert wio.closed" at the end of the function.

        # Fortunately, a little gc.collect() seems to be enough to
        # work around all these issues.
        support.gc_collect()  # For PyPy or other GCs.

        read_results = []
        def _read():
            s = os.read(r, 1)
            read_results.append(s)

        t = threading.Thread(target=_read)
        t.daemon = True
        r, w = os.pipe()
        fdopen_kwargs["closefd"] = False
        large_data = item * (support.PIPE_MAX_SIZE // len(item) + 1)
        try:
            wio = self.io.open(w, **fdopen_kwargs)
            if hasattr(signal, 'pthread_sigmask'):
                # create the thread with SIGALRM signal blocked
                signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGALRM])
                t.start()
                signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGALRM])
            else:
                t.start()

            # Fill the pipe enough that the write will be blocking.
            # It will be interrupted by the timer armed above.  Since the
            # other thread has read one byte, the low-level write will
            # return with a successful (partial) result rather than an EINTR.
            # The buffered IO layer must check for pending signal
            # handlers, which in this case will invoke alarm_interrupt().
            signal.alarm(1)
            try:
                self.assertRaises(ZeroDivisionError, wio.write, large_data)
            finally:
                signal.alarm(0)
                t.join()
            # We got one byte, get another one and check that it isn't a
            # repeat of the first one.
            read_results.append(os.read(r, 1))
            self.assertEqual(read_results, [bytes[0:1], bytes[1:2]])
        finally:
            os.close(w)
            os.close(r)
            # This is deliberate. If we didn't close the file descriptor
            # before closing wio, wio would try to flush its internal
            # buffer, and block again.
            try:
                wio.close()
            except OSError as e:
                if e.errno != errno.EBADF:
                    raise

    @requires_alarm
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_interrupted_write_unbuffered(self):
        self.check_interrupted_write(b"xy", b"xy", mode="wb", buffering=0)

    @requires_alarm
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_interrupted_write_buffered(self):
        self.check_interrupted_write(b"xy", b"xy", mode="wb")

    @requires_alarm
    @unittest.skipUnless(hasattr(os, "pipe"), "requires os.pipe()")
    def test_interrupted_write_text(self):
        self.check_interrupted_write("xy", b"xy", mode="w", encoding="ascii")

    @support.no_tracing
    def check_reentrant_write(self, data, **fdopen_kwargs):
        def on_alarm(*args):
            # Will be called reentrantly from the same thread
            wio.write(data)
            1/0
        signal.signal(signal.SIGALRM, on_alarm)
        r, w = os.pipe()
        wio = self.io.open(w, **fdopen_kwargs)
        try:
            signal.alarm(1)
            # Either the reentrant call to wio.write() fails with RuntimeError,
            # or the signal handler raises ZeroDivisionError.
            with self.assertRaises((ZeroDivisionError, RuntimeError)) as cm:
                while 1:
                    for i in range(100):
                        wio.write(data)
                        wio.flush()
                    # Make sure the buffer doesn't fill up and block further writes
                    os.read(r, len(data) * 100)
            exc = cm.exception
            if isinstance(exc, RuntimeError):
                self.assertTrue(str(exc).startswith("reentrant call"), str(exc))
        finally:
            signal.alarm(0)
            wio.close()
            os.close(r)

    @requires_alarm
    def test_reentrant_write_buffered(self):
        self.check_reentrant_write(b"xy", mode="wb")

    @requires_alarm
    def test_reentrant_write_text(self):
        self.check_reentrant_write("xy", mode="w", encoding="ascii")

    def check_interrupted_read_retry(self, decode, **fdopen_kwargs):
        """Check that a buffered read, when it gets interrupted (either
        returning a partial result or EINTR), properly invokes the signal
        handler and retries if the latter returned successfully."""
        r, w = os.pipe()
        fdopen_kwargs["closefd"] = False
        def alarm_handler(sig, frame):
            os.write(w, b"bar")
        signal.signal(signal.SIGALRM, alarm_handler)
        try:
            rio = self.io.open(r, **fdopen_kwargs)
            os.write(w, b"foo")
            signal.alarm(1)
            # Expected behaviour:
            # - first raw read() returns partial b"foo"
            # - second raw read() returns EINTR
            # - third raw read() returns b"bar"
            self.assertEqual(decode(rio.read(6)), "foobar")
        finally:
            signal.alarm(0)
            rio.close()
            os.close(w)
            os.close(r)

    @requires_alarm
    @support.requires_resource('walltime')
    def test_interrupted_read_retry_buffered(self):
        self.check_interrupted_read_retry(lambda x: x.decode('latin1'),
                                          mode="rb")

    @requires_alarm
    @support.requires_resource('walltime')
    def test_interrupted_read_retry_text(self):
        self.check_interrupted_read_retry(lambda x: x,
                                          mode="r", encoding="latin1")

    def check_interrupted_write_retry(self, item, **fdopen_kwargs):
        """Check that a buffered write, when it gets interrupted (either
        returning a partial result or EINTR), properly invokes the signal
        handler and retries if the latter returned successfully."""
        select = import_helper.import_module("select")

        # A quantity that exceeds the buffer size of an anonymous pipe's
        # write end.
        N = support.PIPE_MAX_SIZE
        r, w = os.pipe()
        fdopen_kwargs["closefd"] = False

        # We need a separate thread to read from the pipe and allow the
        # write() to finish.  This thread is started after the SIGALRM is
        # received (forcing a first EINTR in write()).
        read_results = []
        write_finished = False
        error = None
        def _read():
            try:
                while not write_finished:
                    while r in select.select([r], [], [], 1.0)[0]:
                        s = os.read(r, 1024)
                        read_results.append(s)
            except BaseException as exc:
                nonlocal error
                error = exc
        t = threading.Thread(target=_read)
        t.daemon = True
        def alarm1(sig, frame):
            signal.signal(signal.SIGALRM, alarm2)
            signal.alarm(1)
        def alarm2(sig, frame):
            t.start()

        large_data = item * N
        signal.signal(signal.SIGALRM, alarm1)
        try:
            wio = self.io.open(w, **fdopen_kwargs)
            signal.alarm(1)
            # Expected behaviour:
            # - first raw write() is partial (because of the limited pipe buffer
            #   and the first alarm)
            # - second raw write() returns EINTR (because of the second alarm)
            # - subsequent write()s are successful (either partial or complete)
            written = wio.write(large_data)
            self.assertEqual(N, written)

            wio.flush()
            write_finished = True
            t.join()

            self.assertIsNone(error)
            self.assertEqual(N, sum(len(x) for x in read_results))
        finally:
            signal.alarm(0)
            write_finished = True
            os.close(w)
            os.close(r)
            # This is deliberate. If we didn't close the file descriptor
            # before closing wio, wio would try to flush its internal
            # buffer, and could block (in case of failure).
            try:
                wio.close()
            except OSError as e:
                if e.errno != errno.EBADF:
                    raise

    @requires_alarm
    @support.requires_resource('walltime')
    def test_interrupted_write_retry_buffered(self):
        self.check_interrupted_write_retry(b"x", mode="wb")

    @requires_alarm
    @support.requires_resource('walltime')
    def test_interrupted_write_retry_text(self):
        self.check_interrupted_write_retry("x", mode="w", encoding="latin1")


class CSignalsTest(SignalsTest, __TestCase):
    io = io

class PySignalsTest(SignalsTest, __TestCase):
    io = pyio

    # Handling reentrancy issues would slow down _pyio even more, so the
    # tests are disabled.
    test_reentrant_write_buffered = None
    test_reentrant_write_text = None


def load_tests(loader, tests, pattern):
    tests = (CIOTest, PyIOTest, APIMismatchTest,
             CBufferedReaderTest, PyBufferedReaderTest,
             CBufferedWriterTest, PyBufferedWriterTest,
             CBufferedRWPairTest, PyBufferedRWPairTest,
             CBufferedRandomTest, PyBufferedRandomTest,
             StatefulIncrementalDecoderTest,
             CIncrementalNewlineDecoderTest, PyIncrementalNewlineDecoderTest,
             CTextIOWrapperTest, PyTextIOWrapperTest,
             CMiscIOTest, PyMiscIOTest,
             CSignalsTest, PySignalsTest, TestIOCTypes,
             )

    # Put the namespaces of the IO module we are testing and some useful mock
    # classes in the __dict__ of each test.
    mocks = (MockRawIO, MisbehavedRawIO, MockFileIO, CloseFailureIO,
             MockNonBlockWriterIO, MockUnseekableIO, MockRawIOWithoutRead,
             SlowFlushRawIO, MockCharPseudoDevFileIO)
    all_members = io.__all__
    c_io_ns = {name : getattr(io, name) for name in all_members}
    py_io_ns = {name : getattr(pyio, name) for name in all_members}
    globs = globals()
    c_io_ns.update((x.__name__, globs["C" + x.__name__]) for x in mocks)
    py_io_ns.update((x.__name__, globs["Py" + x.__name__]) for x in mocks)
    for test in tests:
        if test.__name__.startswith("C"):
            for name, obj in c_io_ns.items():
                setattr(test, name, obj)
            test.is_C = True
        elif test.__name__.startswith("Py"):
            for name, obj in py_io_ns.items():
                setattr(test, name, obj)
            test.is_C = False

    suite = loader.suiteClass()
    for test in tests:
        suite.addTest(loader.loadTestsFromTestCase(test))
    return suite


if __name__ == "__main__":
    run_tests()
