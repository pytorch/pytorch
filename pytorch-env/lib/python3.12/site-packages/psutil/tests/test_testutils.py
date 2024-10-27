#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for testing utils (psutil.tests namespace)."""

import collections
import contextlib
import errno
import os
import socket
import stat
import subprocess
import textwrap
import unittest
import warnings

import psutil
import psutil.tests
from psutil import FREEBSD
from psutil import NETBSD
from psutil import POSIX
from psutil._common import open_binary
from psutil._common import open_text
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil.tests import CI_TESTING
from psutil.tests import COVERAGE
from psutil.tests import HAS_NET_CONNECTIONS_UNIX
from psutil.tests import HERE
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import TestMemoryLeak
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import call_until
from psutil.tests import chdir
from psutil.tests import create_sockets
from psutil.tests import fake_pytest
from psutil.tests import filter_proc_net_connections
from psutil.tests import get_free_port
from psutil.tests import is_namedtuple
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import pytest
from psutil.tests import reap_children
from psutil.tests import retry
from psutil.tests import retry_on_failure
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import system_namespace
from psutil.tests import tcp_socketpair
from psutil.tests import terminate
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
from psutil.tests import wait_for_pid


# ===================================================================
# --- Unit tests for test utilities.
# ===================================================================


class TestRetryDecorator(PsutilTestCase):
    @mock.patch('time.sleep')
    def test_retry_success(self, sleep):
        # Fail 3 times out of 5; make sure the decorated fun returns.

        @retry(retries=5, interval=1, logfun=None)
        def foo():
            while queue:
                queue.pop()
                1 / 0  # noqa
            return 1

        queue = list(range(3))
        assert foo() == 1
        assert sleep.call_count == 3

    @mock.patch('time.sleep')
    def test_retry_failure(self, sleep):
        # Fail 6 times out of 5; th function is supposed to raise exc.
        @retry(retries=5, interval=1, logfun=None)
        def foo():
            while queue:
                queue.pop()
                1 / 0  # noqa
            return 1

        queue = list(range(6))
        with pytest.raises(ZeroDivisionError):
            foo()
        assert sleep.call_count == 5

    @mock.patch('time.sleep')
    def test_exception_arg(self, sleep):
        @retry(exception=ValueError, interval=1)
        def foo():
            raise TypeError

        with pytest.raises(TypeError):
            foo()
        assert sleep.call_count == 0

    @mock.patch('time.sleep')
    def test_no_interval_arg(self, sleep):
        # if interval is not specified sleep is not supposed to be called

        @retry(retries=5, interval=None, logfun=None)
        def foo():
            1 / 0  # noqa

        with pytest.raises(ZeroDivisionError):
            foo()
        assert sleep.call_count == 0

    @mock.patch('time.sleep')
    def test_retries_arg(self, sleep):
        @retry(retries=5, interval=1, logfun=None)
        def foo():
            1 / 0  # noqa

        with pytest.raises(ZeroDivisionError):
            foo()
        assert sleep.call_count == 5

    @mock.patch('time.sleep')
    def test_retries_and_timeout_args(self, sleep):
        with pytest.raises(ValueError):
            retry(retries=5, timeout=1)


class TestSyncTestUtils(PsutilTestCase):
    def test_wait_for_pid(self):
        wait_for_pid(os.getpid())
        nopid = max(psutil.pids()) + 99999
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            with pytest.raises(psutil.NoSuchProcess):
                wait_for_pid(nopid)

    def test_wait_for_file(self):
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn)
        assert not os.path.exists(testfn)

    def test_wait_for_file_empty(self):
        testfn = self.get_testfn()
        with open(testfn, 'w'):
            pass
        wait_for_file(testfn, empty=True)
        assert not os.path.exists(testfn)

    def test_wait_for_file_no_file(self):
        testfn = self.get_testfn()
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            with pytest.raises(IOError):
                wait_for_file(testfn)

    def test_wait_for_file_no_delete(self):
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn, delete=False)
        assert os.path.exists(testfn)

    def test_call_until(self):
        call_until(lambda: 1)
        # TODO: test for timeout


class TestFSTestUtils(PsutilTestCase):
    def test_open_text(self):
        with open_text(__file__) as f:
            assert f.mode == 'r'

    def test_open_binary(self):
        with open_binary(__file__) as f:
            assert f.mode == 'rb'

    def test_safe_mkdir(self):
        testfn = self.get_testfn()
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)

    def test_safe_rmpath(self):
        # test file is removed
        testfn = self.get_testfn()
        open(testfn, 'w').close()
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        # test no exception if path does not exist
        safe_rmpath(testfn)
        # test dir is removed
        os.mkdir(testfn)
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        # test other exceptions are raised
        with mock.patch(
            'psutil.tests.os.stat', side_effect=OSError(errno.EINVAL, "")
        ) as m:
            with pytest.raises(OSError):
                safe_rmpath(testfn)
            assert m.called

    def test_chdir(self):
        testfn = self.get_testfn()
        base = os.getcwd()
        os.mkdir(testfn)
        with chdir(testfn):
            assert os.getcwd() == os.path.join(base, testfn)
        assert os.getcwd() == base


class TestProcessUtils(PsutilTestCase):
    def test_reap_children(self):
        subp = self.spawn_testproc()
        p = psutil.Process(subp.pid)
        assert p.is_running()
        reap_children()
        assert not p.is_running()
        assert not psutil.tests._pids_started
        assert not psutil.tests._subprocesses_started

    def test_spawn_children_pair(self):
        child, grandchild = self.spawn_children_pair()
        assert child.pid != grandchild.pid
        assert child.is_running()
        assert grandchild.is_running()
        children = psutil.Process().children()
        assert children == [child]
        children = psutil.Process().children(recursive=True)
        assert len(children) == 2
        assert child in children
        assert grandchild in children
        assert child.ppid() == os.getpid()
        assert grandchild.ppid() == child.pid

        terminate(child)
        assert not child.is_running()
        assert grandchild.is_running()

        terminate(grandchild)
        assert not grandchild.is_running()

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_spawn_zombie(self):
        _parent, zombie = self.spawn_zombie()
        assert zombie.status() == psutil.STATUS_ZOMBIE

    def test_terminate(self):
        # by subprocess.Popen
        p = self.spawn_testproc()
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        # by psutil.Process
        p = psutil.Process(self.spawn_testproc().pid)
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        # by psutil.Popen
        cmd = [
            PYTHON_EXE,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)];",
        ]
        p = psutil.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=PYTHON_EXE_ENV,
        )
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        # by PID
        pid = self.spawn_testproc().pid
        terminate(pid)
        self.assertPidGone(p.pid)
        terminate(pid)
        # zombie
        if POSIX:
            parent, zombie = self.spawn_zombie()
            terminate(parent)
            terminate(zombie)
            self.assertPidGone(parent.pid)
            self.assertPidGone(zombie.pid)


class TestNetUtils(PsutilTestCase):
    def bind_socket(self):
        port = get_free_port()
        with contextlib.closing(bind_socket(addr=('', port))) as s:
            assert s.getsockname()[1] == port

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_bind_unix_socket(self):
        name = self.get_testfn()
        sock = bind_unix_socket(name)
        with contextlib.closing(sock):
            assert sock.family == socket.AF_UNIX
            assert sock.type == socket.SOCK_STREAM
            assert sock.getsockname() == name
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
        # UDP
        name = self.get_testfn()
        sock = bind_unix_socket(name, type=socket.SOCK_DGRAM)
        with contextlib.closing(sock):
            assert sock.type == socket.SOCK_DGRAM

    def tcp_tcp_socketpair(self):
        addr = ("127.0.0.1", get_free_port())
        server, client = tcp_socketpair(socket.AF_INET, addr=addr)
        with contextlib.closing(server):
            with contextlib.closing(client):
                # Ensure they are connected and the positions are
                # correct.
                assert server.getsockname() == addr
                assert client.getpeername() == addr
                assert client.getsockname() != addr

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @pytest.mark.skipif(
        NETBSD or FREEBSD, reason="/var/run/log UNIX socket opened by default"
    )
    def test_unix_socketpair(self):
        p = psutil.Process()
        num_fds = p.num_fds()
        assert (
            filter_proc_net_connections(p.net_connections(kind='unix')) == []
        )
        name = self.get_testfn()
        server, client = unix_socketpair(name)
        try:
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
            assert p.num_fds() - num_fds == 2
            assert (
                len(
                    filter_proc_net_connections(p.net_connections(kind='unix'))
                )
                == 2
            )
            assert server.getsockname() == name
            assert client.getpeername() == name
        finally:
            client.close()
            server.close()

    def test_create_sockets(self):
        with create_sockets() as socks:
            fams = collections.defaultdict(int)
            types = collections.defaultdict(int)
            for s in socks:
                fams[s.family] += 1
                # work around http://bugs.python.org/issue30204
                types[s.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)] += 1
            assert fams[socket.AF_INET] >= 2
            if supports_ipv6():
                assert fams[socket.AF_INET6] >= 2
            if POSIX and HAS_NET_CONNECTIONS_UNIX:
                assert fams[socket.AF_UNIX] >= 2
            assert types[socket.SOCK_STREAM] >= 2
            assert types[socket.SOCK_DGRAM] >= 2


@pytest.mark.xdist_group(name="serial")
class TestMemLeakClass(TestMemoryLeak):
    @retry_on_failure()
    def test_times(self):
        def fun():
            cnt['cnt'] += 1

        cnt = {'cnt': 0}
        self.execute(fun, times=10, warmup_times=15)
        assert cnt['cnt'] == 26

    def test_param_err(self):
        with pytest.raises(ValueError):
            self.execute(lambda: 0, times=0)
        with pytest.raises(ValueError):
            self.execute(lambda: 0, times=-1)
        with pytest.raises(ValueError):
            self.execute(lambda: 0, warmup_times=-1)
        with pytest.raises(ValueError):
            self.execute(lambda: 0, tolerance=-1)
        with pytest.raises(ValueError):
            self.execute(lambda: 0, retries=-1)

    @retry_on_failure()
    @pytest.mark.skipif(CI_TESTING, reason="skipped on CI")
    @pytest.mark.skipif(COVERAGE, reason="skipped during test coverage")
    def test_leak_mem(self):
        ls = []

        def fun(ls=ls):
            ls.append("x" * 248 * 1024)

        try:
            # will consume around 60M in total
            with pytest.raises(AssertionError, match="extra-mem"):
                self.execute(fun, times=100)
        finally:
            del ls

    def test_unclosed_files(self):
        def fun():
            f = open(__file__)
            self.addCleanup(f.close)
            box.append(f)

        box = []
        kind = "fd" if POSIX else "handle"
        with pytest.raises(AssertionError, match="unclosed " + kind):
            self.execute(fun)

    def test_tolerance(self):
        def fun():
            ls.append("x" * 24 * 1024)

        ls = []
        times = 100
        self.execute(
            fun, times=times, warmup_times=0, tolerance=200 * 1024 * 1024
        )
        assert len(ls) == times + 1

    def test_execute_w_exc(self):
        def fun_1():
            1 / 0  # noqa

        self.execute_w_exc(ZeroDivisionError, fun_1)
        with pytest.raises(ZeroDivisionError):
            self.execute_w_exc(OSError, fun_1)

        def fun_2():
            pass

        with pytest.raises(AssertionError):
            self.execute_w_exc(ZeroDivisionError, fun_2)


class TestFakePytest(PsutilTestCase):
    def run_test_class(self, klass):
        suite = unittest.TestSuite()
        suite.addTest(klass)
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        return result

    def test_raises(self):
        with fake_pytest.raises(ZeroDivisionError) as cm:
            1 / 0  # noqa
        assert isinstance(cm.value, ZeroDivisionError)

        with fake_pytest.raises(ValueError, match="foo") as cm:
            raise ValueError("foo")

        try:
            with fake_pytest.raises(ValueError, match="foo") as cm:
                raise ValueError("bar")
        except AssertionError as err:
            assert str(err) == '"foo" does not match "bar"'
        else:
            raise self.fail("exception not raised")

    def test_mark(self):
        @fake_pytest.mark.xdist_group(name="serial")
        def foo():
            return 1

        assert foo() == 1

        @fake_pytest.mark.xdist_group(name="serial")
        class Foo:
            def bar(self):
                return 1

        assert Foo().bar() == 1

    def test_skipif(self):
        class TestCase(unittest.TestCase):
            @fake_pytest.mark.skipif(True, reason="reason")
            def foo(self):
                assert 1 == 1  # noqa

        result = self.run_test_class(TestCase("foo"))
        assert result.wasSuccessful()
        assert len(result.skipped) == 1
        assert result.skipped[0][1] == "reason"

        class TestCase(unittest.TestCase):
            @fake_pytest.mark.skipif(False, reason="reason")
            def foo(self):
                assert 1 == 1  # noqa

        result = self.run_test_class(TestCase("foo"))
        assert result.wasSuccessful()
        assert len(result.skipped) == 0

    @pytest.mark.skipif(not PY3, reason="not PY3")
    def test_skip(self):
        class TestCase(unittest.TestCase):
            def foo(self):
                fake_pytest.skip("reason")
                assert 1 == 0  # noqa

        result = self.run_test_class(TestCase("foo"))
        assert result.wasSuccessful()
        assert len(result.skipped) == 1
        assert result.skipped[0][1] == "reason"

    def test_main(self):
        tmpdir = self.get_testfn(dir=HERE)
        os.mkdir(tmpdir)
        with open(os.path.join(tmpdir, "__init__.py"), "w"):
            pass
        with open(os.path.join(tmpdir, "test_file.py"), "w") as f:
            f.write(textwrap.dedent("""\
                import unittest

                class TestCase(unittest.TestCase):
                    def test_passed(self):
                        pass
                """).lstrip())
        with mock.patch.object(psutil.tests, "HERE", tmpdir):
            with self.assertWarnsRegex(
                UserWarning, "Fake pytest module was used"
            ):
                suite = fake_pytest.main()
                assert suite.countTestCases() == 1

    def test_warns(self):
        # success
        with fake_pytest.warns(UserWarning):
            warnings.warn("foo", UserWarning, stacklevel=1)

        # failure
        try:
            with fake_pytest.warns(UserWarning):
                warnings.warn("foo", DeprecationWarning, stacklevel=1)
        except AssertionError:
            pass
        else:
            raise self.fail("exception not raised")

        # match success
        with fake_pytest.warns(UserWarning, match="foo"):
            warnings.warn("foo", UserWarning, stacklevel=1)

        # match failure
        try:
            with fake_pytest.warns(UserWarning, match="foo"):
                warnings.warn("bar", UserWarning, stacklevel=1)
        except AssertionError:
            pass
        else:
            raise self.fail("exception not raised")


class TestTestingUtils(PsutilTestCase):
    def test_process_namespace(self):
        p = psutil.Process()
        ns = process_namespace(p)
        ns.test()
        fun = [x for x in ns.iter(ns.getters) if x[1] == 'ppid'][0][0]
        assert fun() == p.ppid()

    def test_system_namespace(self):
        ns = system_namespace()
        fun = [x for x in ns.iter(ns.getters) if x[1] == 'net_if_addrs'][0][0]
        assert fun() == psutil.net_if_addrs()


class TestOtherUtils(PsutilTestCase):
    def test_is_namedtuple(self):
        assert is_namedtuple(collections.namedtuple('foo', 'a b c')(1, 2, 3))
        assert not is_namedtuple(tuple())
