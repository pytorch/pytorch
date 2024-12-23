#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for psutil.Process class."""

import collections
import errno
import getpass
import itertools
import os
import signal
import socket
import stat
import string
import subprocess
import sys
import textwrap
import time
import types

import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import open_text
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import redirect_stderr
from psutil._compat import super
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_THREADS
from psutil.tests import MACOS_11PLUS
from psutil.tests import PYPY
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import QEMU_USER
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import copyload_shared_lib
from psutil.tests import create_c_exe
from psutil.tests import create_py_exe
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import pytest
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import skip_on_not_implemented
from psutil.tests import wait_for_pid


# ===================================================================
# --- psutil.Process class tests
# ===================================================================


class TestProcess(PsutilTestCase):
    """Tests for psutil.Process class."""

    def spawn_psproc(self, *args, **kwargs):
        sproc = self.spawn_testproc(*args, **kwargs)
        try:
            return psutil.Process(sproc.pid)
        except psutil.NoSuchProcess:
            self.assertPidGone(sproc.pid)
            raise

    # ---

    def test_pid(self):
        p = psutil.Process()
        assert p.pid == os.getpid()
        with pytest.raises(AttributeError):
            p.pid = 33

    def test_kill(self):
        p = self.spawn_psproc()
        p.kill()
        code = p.wait()
        if WINDOWS:
            assert code == signal.SIGTERM
        else:
            assert code == -signal.SIGKILL
        self.assertProcessGone(p)

    def test_terminate(self):
        p = self.spawn_psproc()
        p.terminate()
        code = p.wait()
        if WINDOWS:
            assert code == signal.SIGTERM
        else:
            assert code == -signal.SIGTERM
        self.assertProcessGone(p)

    def test_send_signal(self):
        sig = signal.SIGKILL if POSIX else signal.SIGTERM
        p = self.spawn_psproc()
        p.send_signal(sig)
        code = p.wait()
        if WINDOWS:
            assert code == sig
        else:
            assert code == -sig
        self.assertProcessGone(p)

    @pytest.mark.skipif(not POSIX, reason="not POSIX")
    def test_send_signal_mocked(self):
        sig = signal.SIGTERM
        p = self.spawn_psproc()
        with mock.patch(
            'psutil.os.kill', side_effect=OSError(errno.ESRCH, "")
        ):
            with pytest.raises(psutil.NoSuchProcess):
                p.send_signal(sig)

        p = self.spawn_psproc()
        with mock.patch(
            'psutil.os.kill', side_effect=OSError(errno.EPERM, "")
        ):
            with pytest.raises(psutil.AccessDenied):
                p.send_signal(sig)

    def test_wait_exited(self):
        # Test waitpid() + WIFEXITED -> WEXITSTATUS.
        # normal return, same as exit(0)
        cmd = [PYTHON_EXE, "-c", "pass"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        assert code == 0
        self.assertProcessGone(p)
        # exit(1), implicit in case of error
        cmd = [PYTHON_EXE, "-c", "1 / 0"]
        p = self.spawn_psproc(cmd, stderr=subprocess.PIPE)
        code = p.wait()
        assert code == 1
        self.assertProcessGone(p)
        # via sys.exit()
        cmd = [PYTHON_EXE, "-c", "import sys; sys.exit(5);"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        assert code == 5
        self.assertProcessGone(p)
        # via os._exit()
        cmd = [PYTHON_EXE, "-c", "import os; os._exit(5);"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        assert code == 5
        self.assertProcessGone(p)

    @pytest.mark.skipif(NETBSD, reason="fails on NETBSD")
    def test_wait_stopped(self):
        p = self.spawn_psproc()
        if POSIX:
            # Test waitpid() + WIFSTOPPED and WIFCONTINUED.
            # Note: if a process is stopped it ignores SIGTERM.
            p.send_signal(signal.SIGSTOP)
            with pytest.raises(psutil.TimeoutExpired):
                p.wait(timeout=0.001)
            p.send_signal(signal.SIGCONT)
            with pytest.raises(psutil.TimeoutExpired):
                p.wait(timeout=0.001)
            p.send_signal(signal.SIGTERM)
            assert p.wait() == -signal.SIGTERM
            assert p.wait() == -signal.SIGTERM
        else:
            p.suspend()
            with pytest.raises(psutil.TimeoutExpired):
                p.wait(timeout=0.001)
            p.resume()
            with pytest.raises(psutil.TimeoutExpired):
                p.wait(timeout=0.001)
            p.terminate()
            assert p.wait() == signal.SIGTERM
            assert p.wait() == signal.SIGTERM

    def test_wait_non_children(self):
        # Test wait() against a process which is not our direct
        # child.
        child, grandchild = self.spawn_children_pair()
        with pytest.raises(psutil.TimeoutExpired):
            child.wait(0.01)
        with pytest.raises(psutil.TimeoutExpired):
            grandchild.wait(0.01)
        # We also terminate the direct child otherwise the
        # grandchild will hang until the parent is gone.
        child.terminate()
        grandchild.terminate()
        child_ret = child.wait()
        grandchild_ret = grandchild.wait()
        if POSIX:
            assert child_ret == -signal.SIGTERM
            # For processes which are not our children we're supposed
            # to get None.
            assert grandchild_ret is None
        else:
            assert child_ret == signal.SIGTERM
            assert child_ret == signal.SIGTERM

    def test_wait_timeout(self):
        p = self.spawn_psproc()
        p.name()
        with pytest.raises(psutil.TimeoutExpired):
            p.wait(0.01)
        with pytest.raises(psutil.TimeoutExpired):
            p.wait(0)
        with pytest.raises(ValueError):
            p.wait(-1)

    def test_wait_timeout_nonblocking(self):
        p = self.spawn_psproc()
        with pytest.raises(psutil.TimeoutExpired):
            p.wait(0)
        p.kill()
        stop_at = time.time() + GLOBAL_TIMEOUT
        while time.time() < stop_at:
            try:
                code = p.wait(0)
                break
            except psutil.TimeoutExpired:
                pass
        else:
            raise self.fail('timeout')
        if POSIX:
            assert code == -signal.SIGKILL
        else:
            assert code == signal.SIGTERM
        self.assertProcessGone(p)

    def test_cpu_percent(self):
        p = psutil.Process()
        p.cpu_percent(interval=0.001)
        p.cpu_percent(interval=0.001)
        for _ in range(100):
            percent = p.cpu_percent(interval=None)
            assert isinstance(percent, float)
            assert percent >= 0.0
        with pytest.raises(ValueError):
            p.cpu_percent(interval=-1)

    def test_cpu_percent_numcpus_none(self):
        # See: https://github.com/giampaolo/psutil/issues/1087
        with mock.patch('psutil.cpu_count', return_value=None) as m:
            psutil.Process().cpu_percent()
            assert m.called

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_cpu_times(self):
        times = psutil.Process().cpu_times()
        assert times.user >= 0.0, times
        assert times.system >= 0.0, times
        assert times.children_user >= 0.0, times
        assert times.children_system >= 0.0, times
        if LINUX:
            assert times.iowait >= 0.0, times
        # make sure returned values can be pretty printed with strftime
        for name in times._fields:
            time.strftime("%H:%M:%S", time.localtime(getattr(times, name)))

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_cpu_times_2(self):
        user_time, kernel_time = psutil.Process().cpu_times()[:2]
        utime, ktime = os.times()[:2]

        # Use os.times()[:2] as base values to compare our results
        # using a tolerance  of +/- 0.1 seconds.
        # It will fail if the difference between the values is > 0.1s.
        if (max([user_time, utime]) - min([user_time, utime])) > 0.1:
            raise self.fail("expected: %s, found: %s" % (utime, user_time))

        if (max([kernel_time, ktime]) - min([kernel_time, ktime])) > 0.1:
            raise self.fail("expected: %s, found: %s" % (ktime, kernel_time))

    @pytest.mark.skipif(not HAS_PROC_CPU_NUM, reason="not supported")
    def test_cpu_num(self):
        p = psutil.Process()
        num = p.cpu_num()
        assert num >= 0
        if psutil.cpu_count() == 1:
            assert num == 0
        assert p.cpu_num() in range(psutil.cpu_count())

    def test_create_time(self):
        p = self.spawn_psproc()
        now = time.time()
        create_time = p.create_time()

        # Use time.time() as base value to compare our result using a
        # tolerance of +/- 1 second.
        # It will fail if the difference between the values is > 2s.
        difference = abs(create_time - now)
        if difference > 2:
            raise self.fail(
                "expected: %s, found: %s, difference: %s"
                % (now, create_time, difference)
            )

        # make sure returned value can be pretty printed with strftime
        time.strftime("%Y %m %d %H:%M:%S", time.localtime(p.create_time()))

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_terminal(self):
        terminal = psutil.Process().terminal()
        if terminal is not None:
            try:
                tty = os.path.realpath(sh('tty'))
            except RuntimeError:
                # Note: happens if pytest is run without the `-s` opt.
                raise pytest.skip("can't rely on `tty` CLI")
            else:
                assert terminal == tty

    @pytest.mark.skipif(not HAS_PROC_IO_COUNTERS, reason="not supported")
    @skip_on_not_implemented(only_if=LINUX)
    def test_io_counters(self):
        p = psutil.Process()
        # test reads
        io1 = p.io_counters()
        with open(PYTHON_EXE, 'rb') as f:
            f.read()
        io2 = p.io_counters()
        if not BSD and not AIX:
            assert io2.read_count > io1.read_count
            assert io2.write_count == io1.write_count
            if LINUX:
                assert io2.read_chars > io1.read_chars
                assert io2.write_chars == io1.write_chars
        else:
            assert io2.read_bytes >= io1.read_bytes
            assert io2.write_bytes >= io1.write_bytes

        # test writes
        io1 = p.io_counters()
        with open(self.get_testfn(), 'wb') as f:
            if PY3:
                f.write(bytes("x" * 1000000, 'ascii'))
            else:
                f.write("x" * 1000000)
        io2 = p.io_counters()
        assert io2.write_count >= io1.write_count
        assert io2.write_bytes >= io1.write_bytes
        assert io2.read_count >= io1.read_count
        assert io2.read_bytes >= io1.read_bytes
        if LINUX:
            assert io2.write_chars > io1.write_chars
            assert io2.read_chars >= io1.read_chars

        # sanity check
        for i in range(len(io2)):
            if BSD and i >= 2:
                # On BSD read_bytes and write_bytes are always set to -1.
                continue
            assert io2[i] >= 0
            assert io2[i] >= 0

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    @pytest.mark.skipif(not LINUX, reason="linux only")
    def test_ionice_linux(self):
        def cleanup(init):
            ioclass, value = init
            if ioclass == psutil.IOPRIO_CLASS_NONE:
                value = 0
            p.ionice(ioclass, value)

        p = psutil.Process()
        if not CI_TESTING:
            assert p.ionice()[0] == psutil.IOPRIO_CLASS_NONE
        assert psutil.IOPRIO_CLASS_NONE == 0
        assert psutil.IOPRIO_CLASS_RT == 1  # high
        assert psutil.IOPRIO_CLASS_BE == 2  # normal
        assert psutil.IOPRIO_CLASS_IDLE == 3  # low
        init = p.ionice()
        self.addCleanup(cleanup, init)

        # low
        p.ionice(psutil.IOPRIO_CLASS_IDLE)
        assert tuple(p.ionice()) == (psutil.IOPRIO_CLASS_IDLE, 0)
        with pytest.raises(ValueError):  # accepts no value
            p.ionice(psutil.IOPRIO_CLASS_IDLE, value=7)
        # normal
        p.ionice(psutil.IOPRIO_CLASS_BE)
        assert tuple(p.ionice()) == (psutil.IOPRIO_CLASS_BE, 0)
        p.ionice(psutil.IOPRIO_CLASS_BE, value=7)
        assert tuple(p.ionice()) == (psutil.IOPRIO_CLASS_BE, 7)
        with pytest.raises(ValueError):
            p.ionice(psutil.IOPRIO_CLASS_BE, value=8)
        try:
            p.ionice(psutil.IOPRIO_CLASS_RT, value=7)
        except psutil.AccessDenied:
            pass
        # errs
        with pytest.raises(ValueError, match="ioclass accepts no value"):
            p.ionice(psutil.IOPRIO_CLASS_NONE, 1)
        with pytest.raises(ValueError, match="ioclass accepts no value"):
            p.ionice(psutil.IOPRIO_CLASS_IDLE, 1)
        with pytest.raises(
            ValueError, match="'ioclass' argument must be specified"
        ):
            p.ionice(value=1)

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    @pytest.mark.skipif(
        not WINDOWS, reason="not supported on this win version"
    )
    def test_ionice_win(self):
        p = psutil.Process()
        if not CI_TESTING:
            assert p.ionice() == psutil.IOPRIO_NORMAL
        init = p.ionice()
        self.addCleanup(p.ionice, init)

        # base
        p.ionice(psutil.IOPRIO_VERYLOW)
        assert p.ionice() == psutil.IOPRIO_VERYLOW
        p.ionice(psutil.IOPRIO_LOW)
        assert p.ionice() == psutil.IOPRIO_LOW
        try:
            p.ionice(psutil.IOPRIO_HIGH)
        except psutil.AccessDenied:
            pass
        else:
            assert p.ionice() == psutil.IOPRIO_HIGH
        # errs
        with pytest.raises(
            TypeError, match="value argument not accepted on Windows"
        ):
            p.ionice(psutil.IOPRIO_NORMAL, value=1)
        with pytest.raises(ValueError, match="is not a valid priority"):
            p.ionice(psutil.IOPRIO_HIGH + 1)

    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_get(self):
        import resource

        p = psutil.Process(os.getpid())
        names = [x for x in dir(psutil) if x.startswith('RLIMIT')]
        assert names, names
        for name in names:
            value = getattr(psutil, name)
            assert value >= 0
            if name in dir(resource):
                assert value == getattr(resource, name)
                # XXX - On PyPy RLIMIT_INFINITY returned by
                # resource.getrlimit() is reported as a very big long
                # number instead of -1. It looks like a bug with PyPy.
                if PYPY:
                    continue
                assert p.rlimit(value) == resource.getrlimit(value)
            else:
                ret = p.rlimit(value)
                assert len(ret) == 2
                assert ret[0] >= -1
                assert ret[1] >= -1

    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_set(self):
        p = self.spawn_psproc()
        p.rlimit(psutil.RLIMIT_NOFILE, (5, 5))
        assert p.rlimit(psutil.RLIMIT_NOFILE) == (5, 5)
        # If pid is 0 prlimit() applies to the calling process and
        # we don't want that.
        if LINUX:
            with pytest.raises(ValueError, match="can't use prlimit"):
                psutil._psplatform.Process(0).rlimit(0)
        with pytest.raises(ValueError):
            p.rlimit(psutil.RLIMIT_NOFILE, (5, 5, 5))

    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        try:
            p.rlimit(psutil.RLIMIT_FSIZE, (1024, hard))
            with open(testfn, "wb") as f:
                f.write(b"X" * 1024)
            # write() or flush() doesn't always cause the exception
            # but close() will.
            with pytest.raises(IOError) as exc:
                with open(testfn, "wb") as f:
                    f.write(b"X" * 1025)
            assert (exc.value.errno if PY3 else exc.value[0]) == errno.EFBIG
        finally:
            p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))
            assert p.rlimit(psutil.RLIMIT_FSIZE) == (soft, hard)

    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_infinity(self):
        # First set a limit, then re-set it by specifying INFINITY
        # and assume we overridden the previous limit.
        p = psutil.Process()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        try:
            p.rlimit(psutil.RLIMIT_FSIZE, (1024, hard))
            p.rlimit(psutil.RLIMIT_FSIZE, (psutil.RLIM_INFINITY, hard))
            with open(self.get_testfn(), "wb") as f:
                f.write(b"X" * 2048)
        finally:
            p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))
            assert p.rlimit(psutil.RLIMIT_FSIZE) == (soft, hard)

    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_infinity_value(self):
        # RLIMIT_FSIZE should be RLIM_INFINITY, which will be a really
        # big number on a platform with large file support.  On these
        # platforms we need to test that the get/setrlimit functions
        # properly convert the number to a C long long and that the
        # conversion doesn't raise an error.
        p = psutil.Process()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        assert hard == psutil.RLIM_INFINITY
        p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))

    def test_num_threads(self):
        # on certain platforms such as Linux we might test for exact
        # thread number, since we always have with 1 thread per process,
        # but this does not apply across all platforms (MACOS, Windows)
        p = psutil.Process()
        if OPENBSD:
            try:
                step1 = p.num_threads()
            except psutil.AccessDenied:
                raise pytest.skip("on OpenBSD this requires root access")
        else:
            step1 = p.num_threads()

        with ThreadTask():
            step2 = p.num_threads()
            assert step2 == step1 + 1

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_num_handles(self):
        # a better test is done later into test/_windows.py
        p = psutil.Process()
        assert p.num_handles() > 0

    @pytest.mark.skipif(not HAS_THREADS, reason="not supported")
    def test_threads(self):
        p = psutil.Process()
        if OPENBSD:
            try:
                step1 = p.threads()
            except psutil.AccessDenied:
                raise pytest.skip("on OpenBSD this requires root access")
        else:
            step1 = p.threads()

        with ThreadTask():
            step2 = p.threads()
            assert len(step2) == len(step1) + 1
            athread = step2[0]
            # test named tuple
            assert athread.id == athread[0]
            assert athread.user_time == athread[1]
            assert athread.system_time == athread[2]

    @retry_on_failure()
    @skip_on_access_denied(only_if=MACOS)
    @pytest.mark.skipif(not HAS_THREADS, reason="not supported")
    def test_threads_2(self):
        p = self.spawn_psproc()
        if OPENBSD:
            try:
                p.threads()
            except psutil.AccessDenied:
                raise pytest.skip("on OpenBSD this requires root access")
        assert (
            abs(p.cpu_times().user - sum([x.user_time for x in p.threads()]))
            < 0.1
        )
        assert (
            abs(
                p.cpu_times().system
                - sum([x.system_time for x in p.threads()])
            )
            < 0.1
        )

    @retry_on_failure()
    def test_memory_info(self):
        p = psutil.Process()

        # step 1 - get a base value to compare our results
        rss1, vms1 = p.memory_info()[:2]
        percent1 = p.memory_percent()
        assert rss1 > 0
        assert vms1 > 0

        # step 2 - allocate some memory
        memarr = [None] * 1500000

        rss2, vms2 = p.memory_info()[:2]
        percent2 = p.memory_percent()

        # step 3 - make sure that the memory usage bumped up
        assert rss2 > rss1
        assert vms2 >= vms1  # vms might be equal
        assert percent2 > percent1
        del memarr

        if WINDOWS:
            mem = p.memory_info()
            assert mem.rss == mem.wset
            assert mem.vms == mem.pagefile

        mem = p.memory_info()
        for name in mem._fields:
            assert getattr(mem, name) >= 0

    def test_memory_full_info(self):
        p = psutil.Process()
        total = psutil.virtual_memory().total
        mem = p.memory_full_info()
        for name in mem._fields:
            value = getattr(mem, name)
            assert value >= 0
            if name == 'vms' and OSX or LINUX:
                continue
            assert value <= total
        if LINUX or WINDOWS or MACOS:
            assert mem.uss >= 0
        if LINUX:
            assert mem.pss >= 0
            assert mem.swap >= 0

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    def test_memory_maps(self):
        p = psutil.Process()
        maps = p.memory_maps()
        assert len(maps) == len(set(maps))
        ext_maps = p.memory_maps(grouped=False)

        for nt in maps:
            if not nt.path.startswith('['):
                if QEMU_USER and "/bin/qemu-" in nt.path:
                    continue
                assert os.path.isabs(nt.path), nt.path
                if POSIX:
                    try:
                        assert os.path.exists(nt.path) or os.path.islink(
                            nt.path
                        ), nt.path
                    except AssertionError:
                        if not LINUX:
                            raise
                        else:
                            # https://github.com/giampaolo/psutil/issues/759
                            with open_text('/proc/self/smaps') as f:
                                data = f.read()
                            if "%s (deleted)" % nt.path not in data:
                                raise
                else:
                    # XXX - On Windows we have this strange behavior with
                    # 64 bit dlls: they are visible via explorer but cannot
                    # be accessed via os.stat() (wtf?).
                    if '64' not in os.path.basename(nt.path):
                        try:
                            st = os.stat(nt.path)
                        except FileNotFoundError:
                            pass
                        else:
                            assert stat.S_ISREG(st.st_mode), nt.path
        for nt in ext_maps:
            for fname in nt._fields:
                value = getattr(nt, fname)
                if fname == 'path':
                    continue
                if fname in ('addr', 'perms'):
                    assert value, value
                else:
                    assert isinstance(value, (int, long))
                    assert value >= 0, value

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    def test_memory_maps_lists_lib(self):
        # Make sure a newly loaded shared lib is listed.
        p = psutil.Process()
        with copyload_shared_lib() as path:

            def normpath(p):
                return os.path.realpath(os.path.normcase(p))

            libpaths = [normpath(x.path) for x in p.memory_maps()]
            assert normpath(path) in libpaths

    def test_memory_percent(self):
        p = psutil.Process()
        p.memory_percent()
        with pytest.raises(ValueError):
            p.memory_percent(memtype="?!?")
        if LINUX or MACOS or WINDOWS:
            p.memory_percent(memtype='uss')

    def test_is_running(self):
        p = self.spawn_psproc()
        assert p.is_running()
        assert p.is_running()
        p.kill()
        p.wait()
        assert not p.is_running()
        assert not p.is_running()

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_exe(self):
        p = self.spawn_psproc()
        exe = p.exe()
        try:
            assert exe == PYTHON_EXE
        except AssertionError:
            if WINDOWS and len(exe) == len(PYTHON_EXE):
                # on Windows we don't care about case sensitivity
                normcase = os.path.normcase
                assert normcase(exe) == normcase(PYTHON_EXE)
            else:
                # certain platforms such as BSD are more accurate returning:
                # "/usr/local/bin/python2.7"
                # ...instead of:
                # "/usr/local/bin/python"
                # We do not want to consider this difference in accuracy
                # an error.
                ver = "%s.%s" % (sys.version_info[0], sys.version_info[1])
                try:
                    assert exe.replace(ver, '') == PYTHON_EXE.replace(ver, '')
                except AssertionError:
                    # Typically MACOS. Really not sure what to do here.
                    pass

        out = sh([exe, "-c", "import os; print('hey')"])
        assert out == 'hey'

    def test_cmdline(self):
        cmdline = [
            PYTHON_EXE,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        p = self.spawn_psproc(cmdline)

        if NETBSD and p.cmdline() == []:
            # https://github.com/giampaolo/psutil/issues/2250
            raise pytest.skip("OPENBSD: returned EBUSY")

        # XXX - most of the times the underlying sysctl() call on Net
        # and Open BSD returns a truncated string.
        # Also /proc/pid/cmdline behaves the same so it looks
        # like this is a kernel bug.
        # XXX - AIX truncates long arguments in /proc/pid/cmdline
        if NETBSD or OPENBSD or AIX:
            assert p.cmdline()[0] == PYTHON_EXE
        else:
            if MACOS and CI_TESTING:
                pyexe = p.cmdline()[0]
                if pyexe != PYTHON_EXE:
                    assert ' '.join(p.cmdline()[1:]) == ' '.join(cmdline[1:])
                    return
            if QEMU_USER:
                assert ' '.join(p.cmdline()[2:]) == ' '.join(cmdline)
                return
            assert ' '.join(p.cmdline()) == ' '.join(cmdline)

    @pytest.mark.skipif(PYPY, reason="broken on PYPY")
    def test_long_cmdline(self):
        cmdline = [PYTHON_EXE]
        cmdline.extend(["-v"] * 50)
        cmdline.extend(
            ["-c", "import time; [time.sleep(0.1) for x in range(100)]"]
        )
        p = self.spawn_psproc(cmdline)
        if OPENBSD:
            # XXX: for some reason the test process may turn into a
            # zombie (don't know why).
            try:
                assert p.cmdline() == cmdline
            except psutil.ZombieProcess:
                raise pytest.skip("OPENBSD: process turned into zombie")
        elif QEMU_USER:
            assert p.cmdline()[2:] == cmdline
        else:
            ret = p.cmdline()
            if NETBSD and ret == []:
                # https://github.com/giampaolo/psutil/issues/2250
                raise pytest.skip("OPENBSD: returned EBUSY")
            assert ret == cmdline

    def test_name(self):
        p = self.spawn_psproc()
        name = p.name().lower()
        pyexe = os.path.basename(os.path.realpath(sys.executable)).lower()
        assert pyexe.startswith(name), (pyexe, name)

    @pytest.mark.skipif(PYPY or QEMU_USER, reason="unreliable on PYPY")
    @pytest.mark.skipif(QEMU_USER, reason="unreliable on QEMU user")
    def test_long_name(self):
        pyexe = create_py_exe(self.get_testfn(suffix=string.digits * 2))
        cmdline = [
            pyexe,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        p = self.spawn_psproc(cmdline)
        if OPENBSD:
            # XXX: for some reason the test process may turn into a
            # zombie (don't know why). Because the name() is long, all
            # UNIX kernels truncate it to 15 chars, so internally psutil
            # tries to guess the full name() from the cmdline(). But the
            # cmdline() of a zombie on OpenBSD fails (internally), so we
            # just compare the first 15 chars. Full explanation:
            # https://github.com/giampaolo/psutil/issues/2239
            try:
                assert p.name() == os.path.basename(pyexe)
            except AssertionError:
                if p.status() == psutil.STATUS_ZOMBIE:
                    assert os.path.basename(pyexe).startswith(p.name())
                else:
                    raise
        else:
            assert p.name() == os.path.basename(pyexe)

    # XXX
    @pytest.mark.skipif(SUNOS, reason="broken on SUNOS")
    @pytest.mark.skipif(AIX, reason="broken on AIX")
    @pytest.mark.skipif(PYPY, reason="broken on PYPY")
    @pytest.mark.skipif(QEMU_USER, reason="broken on QEMU user")
    def test_prog_w_funky_name(self):
        # Test that name(), exe() and cmdline() correctly handle programs
        # with funky chars such as spaces and ")", see:
        # https://github.com/giampaolo/psutil/issues/628
        pyexe = create_py_exe(self.get_testfn(suffix='foo bar )'))
        cmdline = [
            pyexe,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        p = self.spawn_psproc(cmdline)
        assert p.cmdline() == cmdline
        assert p.name() == os.path.basename(pyexe)
        assert os.path.normcase(p.exe()) == os.path.normcase(pyexe)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_uids(self):
        p = psutil.Process()
        real, effective, _saved = p.uids()
        # os.getuid() refers to "real" uid
        assert real == os.getuid()
        # os.geteuid() refers to "effective" uid
        assert effective == os.geteuid()
        # No such thing as os.getsuid() ("saved" uid), but starting
        # from python 2.7 we have os.getresuid() which returns all
        # of them.
        if hasattr(os, "getresuid"):
            assert os.getresuid() == p.uids()

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_gids(self):
        p = psutil.Process()
        real, effective, _saved = p.gids()
        # os.getuid() refers to "real" uid
        assert real == os.getgid()
        # os.geteuid() refers to "effective" uid
        assert effective == os.getegid()
        # No such thing as os.getsgid() ("saved" gid), but starting
        # from python 2.7 we have os.getresgid() which returns all
        # of them.
        if hasattr(os, "getresuid"):
            assert os.getresgid() == p.gids()

    def test_nice(self):
        def cleanup(init):
            try:
                p.nice(init)
            except psutil.AccessDenied:
                pass

        p = psutil.Process()
        with pytest.raises(TypeError):
            p.nice("str")
        init = p.nice()
        self.addCleanup(cleanup, init)

        if WINDOWS:
            highest_prio = None
            for prio in [
                psutil.IDLE_PRIORITY_CLASS,
                psutil.BELOW_NORMAL_PRIORITY_CLASS,
                psutil.NORMAL_PRIORITY_CLASS,
                psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                psutil.HIGH_PRIORITY_CLASS,
                psutil.REALTIME_PRIORITY_CLASS,
            ]:
                with self.subTest(prio=prio):
                    try:
                        p.nice(prio)
                    except psutil.AccessDenied:
                        pass
                    else:
                        new_prio = p.nice()
                        # The OS may limit our maximum priority,
                        # even if the function succeeds. For higher
                        # priorities, we match either the expected
                        # value or the highest so far.
                        if prio in (
                            psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                            psutil.HIGH_PRIORITY_CLASS,
                            psutil.REALTIME_PRIORITY_CLASS,
                        ):
                            if new_prio == prio or highest_prio is None:
                                highest_prio = prio
                                assert new_prio == highest_prio
                        else:
                            assert new_prio == prio
        else:
            try:
                if hasattr(os, "getpriority"):
                    assert (
                        os.getpriority(os.PRIO_PROCESS, os.getpid())
                        == p.nice()
                    )
                p.nice(1)
                assert p.nice() == 1
                if hasattr(os, "getpriority"):
                    assert (
                        os.getpriority(os.PRIO_PROCESS, os.getpid())
                        == p.nice()
                    )
                # XXX - going back to previous nice value raises
                # AccessDenied on MACOS
                if not MACOS:
                    p.nice(0)
                    assert p.nice() == 0
            except psutil.AccessDenied:
                pass

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_status(self):
        p = psutil.Process()
        assert p.status() == psutil.STATUS_RUNNING

    def test_username(self):
        p = self.spawn_psproc()
        username = p.username()
        if WINDOWS:
            domain, username = username.split('\\')
            getpass_user = getpass.getuser()
            if getpass_user.endswith('$'):
                # When running as a service account (most likely to be
                # NetworkService), these user name calculations don't produce
                # the same result, causing the test to fail.
                raise pytest.skip('running as service account')
            assert username == getpass_user
            if 'USERDOMAIN' in os.environ:
                assert domain == os.environ['USERDOMAIN']
        else:
            assert username == getpass.getuser()

    def test_cwd(self):
        p = self.spawn_psproc()
        assert p.cwd() == os.getcwd()

    def test_cwd_2(self):
        cmd = [
            PYTHON_EXE,
            "-c",
            (
                "import os, time; os.chdir('..'); [time.sleep(0.1) for x in"
                " range(100)]"
            ),
        ]
        p = self.spawn_psproc(cmd)
        call_until(lambda: p.cwd() == os.path.dirname(os.getcwd()))

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity(self):
        p = psutil.Process()
        initial = p.cpu_affinity()
        assert initial, initial
        self.addCleanup(p.cpu_affinity, initial)

        if hasattr(os, "sched_getaffinity"):
            assert initial == list(os.sched_getaffinity(p.pid))
        assert len(initial) == len(set(initial))

        all_cpus = list(range(len(psutil.cpu_percent(percpu=True))))
        for n in all_cpus:
            p.cpu_affinity([n])
            assert p.cpu_affinity() == [n]
            if hasattr(os, "sched_getaffinity"):
                assert p.cpu_affinity() == list(os.sched_getaffinity(p.pid))
            # also test num_cpu()
            if hasattr(p, "num_cpu"):
                assert p.cpu_affinity()[0] == p.num_cpu()

        # [] is an alias for "all eligible CPUs"; on Linux this may
        # not be equal to all available CPUs, see:
        # https://github.com/giampaolo/psutil/issues/956
        p.cpu_affinity([])
        if LINUX:
            assert p.cpu_affinity() == p._proc._get_eligible_cpus()
        else:
            assert p.cpu_affinity() == all_cpus
        if hasattr(os, "sched_getaffinity"):
            assert p.cpu_affinity() == list(os.sched_getaffinity(p.pid))

        with pytest.raises(TypeError):
            p.cpu_affinity(1)
        p.cpu_affinity(initial)
        # it should work with all iterables, not only lists
        p.cpu_affinity(set(all_cpus))
        p.cpu_affinity(tuple(all_cpus))

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity_errs(self):
        p = self.spawn_psproc()
        invalid_cpu = [len(psutil.cpu_times(percpu=True)) + 10]
        with pytest.raises(ValueError):
            p.cpu_affinity(invalid_cpu)
        with pytest.raises(ValueError):
            p.cpu_affinity(range(10000, 11000))
        with pytest.raises(TypeError):
            p.cpu_affinity([0, "1"])
        with pytest.raises(ValueError):
            p.cpu_affinity([0, -1])

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity_all_combinations(self):
        p = psutil.Process()
        initial = p.cpu_affinity()
        assert initial, initial
        self.addCleanup(p.cpu_affinity, initial)

        # All possible CPU set combinations.
        if len(initial) > 12:
            initial = initial[:12]  # ...otherwise it will take forever
        combos = []
        for i in range(len(initial) + 1):
            for subset in itertools.combinations(initial, i):
                if subset:
                    combos.append(list(subset))

        for combo in combos:
            p.cpu_affinity(combo)
            assert sorted(p.cpu_affinity()) == sorted(combo)

    # TODO: #595
    @pytest.mark.skipif(BSD, reason="broken on BSD")
    # can't find any process file on Appveyor
    @pytest.mark.skipif(APPVEYOR, reason="unreliable on APPVEYOR")
    def test_open_files(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        files = p.open_files()
        assert testfn not in files
        with open(testfn, 'wb') as f:
            f.write(b'x' * 1024)
            f.flush()
            # give the kernel some time to see the new file
            call_until(lambda: len(p.open_files()) != len(files))
            files = p.open_files()
            filenames = [os.path.normcase(x.path) for x in files]
            assert os.path.normcase(testfn) in filenames
            if LINUX:
                for file in files:
                    if file.path == testfn:
                        assert file.position == 1024
        for file in files:
            assert os.path.isfile(file.path), file

        # another process
        cmdline = (
            "import time; f = open(r'%s', 'r'); [time.sleep(0.1) for x in"
            " range(100)];" % testfn
        )
        p = self.spawn_psproc([PYTHON_EXE, "-c", cmdline])

        for x in range(100):
            filenames = [os.path.normcase(x.path) for x in p.open_files()]
            if testfn in filenames:
                break
            time.sleep(0.01)
        else:
            assert os.path.normcase(testfn) in filenames
        for file in filenames:
            assert os.path.isfile(file), file

    # TODO: #595
    @pytest.mark.skipif(BSD, reason="broken on BSD")
    # can't find any process file on Appveyor
    @pytest.mark.skipif(APPVEYOR, reason="unreliable on APPVEYOR")
    def test_open_files_2(self):
        # test fd and path fields
        p = psutil.Process()
        normcase = os.path.normcase
        testfn = self.get_testfn()
        with open(testfn, 'w') as fileobj:
            for file in p.open_files():
                if (
                    normcase(file.path) == normcase(fileobj.name)
                    or file.fd == fileobj.fileno()
                ):
                    break
            else:
                raise self.fail(
                    "no file found; files=%s" % (repr(p.open_files()))
                )
            assert normcase(file.path) == normcase(fileobj.name)
            if WINDOWS:
                assert file.fd == -1
            else:
                assert file.fd == fileobj.fileno()
            # test positions
            ntuple = p.open_files()[0]
            assert ntuple[0] == ntuple.path
            assert ntuple[1] == ntuple.fd
            # test file is gone
            assert fileobj.name not in p.open_files()

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_num_fds(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        start = p.num_fds()
        file = open(testfn, 'w')
        self.addCleanup(file.close)
        assert p.num_fds() == start + 1
        sock = socket.socket()
        self.addCleanup(sock.close)
        assert p.num_fds() == start + 2
        file.close()
        sock.close()
        assert p.num_fds() == start

    @skip_on_not_implemented(only_if=LINUX)
    @pytest.mark.skipif(
        OPENBSD or NETBSD, reason="not reliable on OPENBSD & NETBSD"
    )
    def test_num_ctx_switches(self):
        p = psutil.Process()
        before = sum(p.num_ctx_switches())
        for _ in range(2):
            time.sleep(0.05)  # this shall ensure a context switch happens
            after = sum(p.num_ctx_switches())
            if after > before:
                return
        raise self.fail("num ctx switches still the same after 2 iterations")

    def test_ppid(self):
        p = psutil.Process()
        if hasattr(os, 'getppid'):
            assert p.ppid() == os.getppid()
        p = self.spawn_psproc()
        assert p.ppid() == os.getpid()

    def test_parent(self):
        p = self.spawn_psproc()
        assert p.parent().pid == os.getpid()

        lowest_pid = psutil.pids()[0]
        assert psutil.Process(lowest_pid).parent() is None

    def test_parent_multi(self):
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        assert grandchild.parent() == child
        assert child.parent() == parent

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    @retry_on_failure()
    def test_parents(self):
        parent = psutil.Process()
        assert parent.parents()
        child, grandchild = self.spawn_children_pair()
        assert child.parents()[0] == parent
        assert grandchild.parents()[0] == child
        assert grandchild.parents()[1] == parent

    def test_children(self):
        parent = psutil.Process()
        assert parent.children() == []
        assert parent.children(recursive=True) == []
        # On Windows we set the flag to 0 in order to cancel out the
        # CREATE_NO_WINDOW flag (enabled by default) which creates
        # an extra "conhost.exe" child.
        child = self.spawn_psproc(creationflags=0)
        children1 = parent.children()
        children2 = parent.children(recursive=True)
        for children in (children1, children2):
            assert len(children) == 1
            assert children[0].pid == child.pid
            assert children[0].ppid() == parent.pid

    def test_children_recursive(self):
        # Test children() against two sub processes, p1 and p2, where
        # p1 (our child) spawned p2 (our grandchild).
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        assert parent.children() == [child]
        assert parent.children(recursive=True) == [child, grandchild]
        # If the intermediate process is gone there's no way for
        # children() to recursively find it.
        child.terminate()
        child.wait()
        assert parent.children(recursive=True) == []

    def test_children_duplicates(self):
        # find the process which has the highest number of children
        table = collections.defaultdict(int)
        for p in psutil.process_iter():
            try:
                table[p.ppid()] += 1
            except psutil.Error:
                pass
        # this is the one, now let's make sure there are no duplicates
        pid = sorted(table.items(), key=lambda x: x[1])[-1][0]
        if LINUX and pid == 0:
            raise pytest.skip("PID 0")
        p = psutil.Process(pid)
        try:
            c = p.children(recursive=True)
        except psutil.AccessDenied:  # windows
            pass
        else:
            assert len(c) == len(set(c))

    def test_parents_and_children(self):
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        # forward
        children = parent.children(recursive=True)
        assert len(children) == 2
        assert children[0] == child
        assert children[1] == grandchild
        # backward
        parents = grandchild.parents()
        assert parents[0] == child
        assert parents[1] == parent

    def test_suspend_resume(self):
        p = self.spawn_psproc()
        p.suspend()
        for _ in range(100):
            if p.status() == psutil.STATUS_STOPPED:
                break
            time.sleep(0.01)
        p.resume()
        assert p.status() != psutil.STATUS_STOPPED

    def test_invalid_pid(self):
        with pytest.raises(TypeError):
            psutil.Process("1")
        with pytest.raises(ValueError):
            psutil.Process(-1)

    def test_as_dict(self):
        p = psutil.Process()
        d = p.as_dict(attrs=['exe', 'name'])
        assert sorted(d.keys()) == ['exe', 'name']

        p = psutil.Process(min(psutil.pids()))
        d = p.as_dict(attrs=['net_connections'], ad_value='foo')
        if not isinstance(d['net_connections'], list):
            assert d['net_connections'] == 'foo'

        # Test ad_value is set on AccessDenied.
        with mock.patch(
            'psutil.Process.nice', create=True, side_effect=psutil.AccessDenied
        ):
            assert p.as_dict(attrs=["nice"], ad_value=1) == {"nice": 1}

        # Test that NoSuchProcess bubbles up.
        with mock.patch(
            'psutil.Process.nice',
            create=True,
            side_effect=psutil.NoSuchProcess(p.pid, "name"),
        ):
            with pytest.raises(psutil.NoSuchProcess):
                p.as_dict(attrs=["nice"])

        # Test that ZombieProcess is swallowed.
        with mock.patch(
            'psutil.Process.nice',
            create=True,
            side_effect=psutil.ZombieProcess(p.pid, "name"),
        ):
            assert p.as_dict(attrs=["nice"], ad_value="foo") == {"nice": "foo"}

        # By default APIs raising NotImplementedError are
        # supposed to be skipped.
        with mock.patch(
            'psutil.Process.nice', create=True, side_effect=NotImplementedError
        ):
            d = p.as_dict()
            assert 'nice' not in list(d.keys())
            # ...unless the user explicitly asked for some attr.
            with pytest.raises(NotImplementedError):
                p.as_dict(attrs=["nice"])

        # errors
        with pytest.raises(TypeError):
            p.as_dict('name')
        with pytest.raises(ValueError):
            p.as_dict(['foo'])
        with pytest.raises(ValueError):
            p.as_dict(['foo', 'bar'])

    def test_oneshot(self):
        p = psutil.Process()
        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            with p.oneshot():
                p.cpu_times()
                p.cpu_times()
            assert m.call_count == 1

        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            p.cpu_times()
            p.cpu_times()
        assert m.call_count == 2

    def test_oneshot_twice(self):
        # Test the case where the ctx manager is __enter__ed twice.
        # The second __enter__ is supposed to resut in a NOOP.
        p = psutil.Process()
        with mock.patch("psutil._psplatform.Process.cpu_times") as m1:
            with mock.patch("psutil._psplatform.Process.oneshot_enter") as m2:
                with p.oneshot():
                    p.cpu_times()
                    p.cpu_times()
                    with p.oneshot():
                        p.cpu_times()
                        p.cpu_times()
                assert m1.call_count == 1
                assert m2.call_count == 1

        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            p.cpu_times()
            p.cpu_times()
        assert m.call_count == 2

    def test_oneshot_cache(self):
        # Make sure oneshot() cache is nonglobal. Instead it's
        # supposed to be bound to the Process instance, see:
        # https://github.com/giampaolo/psutil/issues/1373
        p1, p2 = self.spawn_children_pair()
        p1_ppid = p1.ppid()
        p2_ppid = p2.ppid()
        assert p1_ppid != p2_ppid
        with p1.oneshot():
            assert p1.ppid() == p1_ppid
            assert p2.ppid() == p2_ppid
        with p2.oneshot():
            assert p1.ppid() == p1_ppid
            assert p2.ppid() == p2_ppid

    def test_halfway_terminated_process(self):
        # Test that NoSuchProcess exception gets raised in case the
        # process dies after we create the Process object.
        # Example:
        # >>> proc = Process(1234)
        # >>> time.sleep(2)  # time-consuming task, process dies in meantime
        # >>> proc.name()
        # Refers to Issue #15
        def assert_raises_nsp(fun, fun_name):
            try:
                ret = fun()
            except psutil.ZombieProcess:  # differentiate from NSP
                raise
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if OPENBSD and fun_name in ('threads', 'num_threads'):
                    return
                raise
            else:
                # NtQuerySystemInformation succeeds even if process is gone.
                if WINDOWS and fun_name in ('exe', 'name'):
                    return
                raise self.fail(
                    "%r didn't raise NSP and returned %r instead" % (fun, ret)
                )

        p = self.spawn_psproc()
        p.terminate()
        p.wait()
        if WINDOWS:  # XXX
            call_until(lambda: p.pid not in psutil.pids())
        self.assertProcessGone(p)

        ns = process_namespace(p)
        for fun, name in ns.iter(ns.all):
            assert_raises_nsp(fun, name)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_zombie_process(self):
        _parent, zombie = self.spawn_zombie()
        self.assertProcessZombie(zombie)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_zombie_process_is_running_w_exc(self):
        # Emulate a case where internally is_running() raises
        # ZombieProcess.
        p = psutil.Process()
        with mock.patch(
            "psutil.Process", side_effect=psutil.ZombieProcess(0)
        ) as m:
            assert p.is_running()
            assert m.called

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_zombie_process_status_w_exc(self):
        # Emulate a case where internally status() raises
        # ZombieProcess.
        p = psutil.Process()
        with mock.patch(
            "psutil._psplatform.Process.status",
            side_effect=psutil.ZombieProcess(0),
        ) as m:
            assert p.status() == psutil.STATUS_ZOMBIE
            assert m.called

    def test_reused_pid(self):
        # Emulate a case where PID has been reused by another process.
        if PY3:
            from io import StringIO
        else:
            from StringIO import StringIO

        subp = self.spawn_testproc()
        p = psutil.Process(subp.pid)
        p._ident = (p.pid, p.create_time() + 100)

        list(psutil.process_iter())
        assert p.pid in psutil._pmap
        assert not p.is_running()

        # make sure is_running() removed PID from process_iter()
        # internal cache
        with mock.patch.object(psutil._common, "PSUTIL_DEBUG", True):
            with redirect_stderr(StringIO()) as f:
                list(psutil.process_iter())
        assert (
            "refreshing Process instance for reused PID %s" % p.pid
            in f.getvalue()
        )
        assert p.pid not in psutil._pmap

        assert p != psutil.Process(subp.pid)
        msg = "process no longer exists and its PID has been reused"
        ns = process_namespace(p)
        for fun, name in ns.iter(ns.setters + ns.killers, clear_cache=False):
            with self.subTest(name=name):
                with pytest.raises(psutil.NoSuchProcess, match=msg):
                    fun()

        assert "terminated + PID reused" in str(p)
        assert "terminated + PID reused" in repr(p)

        with pytest.raises(psutil.NoSuchProcess, match=msg):
            p.ppid()
        with pytest.raises(psutil.NoSuchProcess, match=msg):
            p.parent()
        with pytest.raises(psutil.NoSuchProcess, match=msg):
            p.parents()
        with pytest.raises(psutil.NoSuchProcess, match=msg):
            p.children()

    def test_pid_0(self):
        # Process(0) is supposed to work on all platforms except Linux
        if 0 not in psutil.pids():
            with pytest.raises(psutil.NoSuchProcess):
                psutil.Process(0)
            # These 2 are a contradiction, but "ps" says PID 1's parent
            # is PID 0.
            assert not psutil.pid_exists(0)
            assert psutil.Process(1).ppid() == 0
            return

        p = psutil.Process(0)
        exc = psutil.AccessDenied if WINDOWS else ValueError
        with pytest.raises(exc):
            p.wait()
        with pytest.raises(exc):
            p.terminate()
        with pytest.raises(exc):
            p.suspend()
        with pytest.raises(exc):
            p.resume()
        with pytest.raises(exc):
            p.kill()
        with pytest.raises(exc):
            p.send_signal(signal.SIGTERM)

        # test all methods
        ns = process_namespace(p)
        for fun, name in ns.iter(ns.getters + ns.setters):
            try:
                ret = fun()
            except psutil.AccessDenied:
                pass
            else:
                if name in ("uids", "gids"):
                    assert ret.real == 0
                elif name == "username":
                    user = 'NT AUTHORITY\\SYSTEM' if WINDOWS else 'root'
                    assert p.username() == user
                elif name == "name":
                    assert name, name

        if not OPENBSD:
            assert 0 in psutil.pids()
            assert psutil.pid_exists(0)

    @pytest.mark.skipif(not HAS_ENVIRON, reason="not supported")
    def test_environ(self):
        def clean_dict(d):
            exclude = ["PLAT", "HOME", "PYTEST_CURRENT_TEST", "PYTEST_VERSION"]
            if MACOS:
                exclude.extend([
                    "__CF_USER_TEXT_ENCODING",
                    "VERSIONER_PYTHON_PREFER_32_BIT",
                    "VERSIONER_PYTHON_VERSION",
                    "VERSIONER_PYTHON_VERSION",
                ])
            for name in exclude:
                d.pop(name, None)
            return dict([
                (
                    k.replace("\r", "").replace("\n", ""),
                    v.replace("\r", "").replace("\n", ""),
                )
                for k, v in d.items()
            ])

        self.maxDiff = None
        p = psutil.Process()
        d1 = clean_dict(p.environ())
        d2 = clean_dict(os.environ.copy())
        if not OSX and GITHUB_ACTIONS:
            assert d1 == d2

    @pytest.mark.skipif(not HAS_ENVIRON, reason="not supported")
    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @pytest.mark.skipif(
        MACOS_11PLUS,
        reason="macOS 11+ can't get another process environment, issue #2084",
    )
    @pytest.mark.skipif(
        NETBSD, reason="sometimes fails on `assert is_running()`"
    )
    def test_weird_environ(self):
        # environment variables can contain values without an equals sign
        code = textwrap.dedent("""
            #include <unistd.h>
            #include <fcntl.h>

            char * const argv[] = {"cat", 0};
            char * const envp[] = {"A=1", "X", "C=3", 0};

            int main(void) {
                // Close stderr on exec so parent can wait for the
                // execve to finish.
                if (fcntl(2, F_SETFD, FD_CLOEXEC) != 0)
                    return 0;
                return execve("/bin/cat", argv, envp);
            }
            """)
        cexe = create_c_exe(self.get_testfn(), c_code=code)
        sproc = self.spawn_testproc(
            [cexe], stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        p = psutil.Process(sproc.pid)
        wait_for_pid(p.pid)
        assert p.is_running()
        # Wait for process to exec or exit.
        assert sproc.stderr.read() == b""
        if MACOS and CI_TESTING:
            try:
                env = p.environ()
            except psutil.AccessDenied:
                # XXX: fails sometimes with:
                # PermissionError from 'sysctl(KERN_PROCARGS2) -> EIO'
                return
        else:
            env = p.environ()
        assert env == {"A": "1", "C": "3"}
        sproc.communicate()
        assert sproc.returncode == 0


# ===================================================================
# --- Limited user tests
# ===================================================================


if POSIX and os.getuid() == 0:

    class LimitedUserTestCase(TestProcess):
        """Repeat the previous tests by using a limited user.
        Executed only on UNIX and only if the user who run the test script
        is root.
        """

        # the uid/gid the test suite runs under
        if hasattr(os, 'getuid'):
            PROCESS_UID = os.getuid()
            PROCESS_GID = os.getgid()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # re-define all existent test methods in order to
            # ignore AccessDenied exceptions
            for attr in [x for x in dir(self) if x.startswith('test')]:
                meth = getattr(self, attr)

                def test_(self):
                    try:
                        meth()  # noqa
                    except psutil.AccessDenied:
                        pass

                setattr(self, attr, types.MethodType(test_, self))

        def setUp(self):
            super().setUp()
            os.setegid(1000)
            os.seteuid(1000)

        def tearDown(self):
            os.setegid(self.PROCESS_UID)
            os.seteuid(self.PROCESS_GID)
            super().tearDown()

        def test_nice(self):
            try:
                psutil.Process().nice(-1)
            except psutil.AccessDenied:
                pass
            else:
                raise self.fail("exception not raised")

        @pytest.mark.skipif(True, reason="causes problem as root")
        def test_zombie_process(self):
            pass


# ===================================================================
# --- psutil.Popen tests
# ===================================================================


class TestPopen(PsutilTestCase):
    """Tests for psutil.Popen class."""

    @classmethod
    def tearDownClass(cls):
        reap_children()

    def test_misc(self):
        # XXX this test causes a ResourceWarning on Python 3 because
        # psutil.__subproc instance doesn't get properly freed.
        # Not sure what to do though.
        cmd = [
            PYTHON_EXE,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)];",
        ]
        with psutil.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=PYTHON_EXE_ENV,
        ) as proc:
            proc.name()
            proc.cpu_times()
            proc.stdin  # noqa
            assert dir(proc)
            with pytest.raises(AttributeError):
                proc.foo  # noqa
            proc.terminate()
        if POSIX:
            assert proc.wait(5) == -signal.SIGTERM
        else:
            assert proc.wait(5) == signal.SIGTERM

    def test_ctx_manager(self):
        with psutil.Popen(
            [PYTHON_EXE, "-V"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=PYTHON_EXE_ENV,
        ) as proc:
            proc.communicate()
        assert proc.stdout.closed
        assert proc.stderr.closed
        assert proc.stdin.closed
        assert proc.returncode == 0

    def test_kill_terminate(self):
        # subprocess.Popen()'s terminate(), kill() and send_signal() do
        # not raise exception after the process is gone. psutil.Popen
        # diverges from that.
        cmd = [
            PYTHON_EXE,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)];",
        ]
        with psutil.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=PYTHON_EXE_ENV,
        ) as proc:
            proc.terminate()
            proc.wait()
            with pytest.raises(psutil.NoSuchProcess):
                proc.terminate()
            with pytest.raises(psutil.NoSuchProcess):
                proc.kill()
            with pytest.raises(psutil.NoSuchProcess):
                proc.send_signal(signal.SIGTERM)
            if WINDOWS:
                with pytest.raises(psutil.NoSuchProcess):
                    proc.send_signal(signal.CTRL_C_EVENT)
                with pytest.raises(psutil.NoSuchProcess):
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
