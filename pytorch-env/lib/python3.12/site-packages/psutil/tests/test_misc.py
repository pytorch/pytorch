#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Miscellaneous tests."""

import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import sys

import psutil
import psutil.tests
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import QEMU_USER
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import pytest
from psutil.tests import reload_module
from psutil.tests import sh
from psutil.tests import system_namespace


# ===================================================================
# --- Test classes' repr(), str(), ...
# ===================================================================


class TestSpecialMethods(PsutilTestCase):
    def test_check_pid_range(self):
        with pytest.raises(OverflowError):
            psutil._psplatform.cext.check_pid_range(2**128)
        with pytest.raises(psutil.NoSuchProcess):
            psutil.Process(2**128)

    def test_process__repr__(self, func=repr):
        p = psutil.Process(self.spawn_testproc().pid)
        r = func(p)
        assert "psutil.Process" in r
        assert "pid=%s" % p.pid in r
        assert "name='%s'" % str(p.name()) in r.replace("name=u'", "name='")
        assert "status=" in r
        assert "exitcode=" not in r
        p.terminate()
        p.wait()
        r = func(p)
        assert "status='terminated'" in r
        assert "exitcode=" in r

        with mock.patch.object(
            psutil.Process,
            "name",
            side_effect=psutil.ZombieProcess(os.getpid()),
        ):
            p = psutil.Process()
            r = func(p)
            assert "pid=%s" % p.pid in r
            assert "status='zombie'" in r
            assert "name=" not in r
        with mock.patch.object(
            psutil.Process,
            "name",
            side_effect=psutil.NoSuchProcess(os.getpid()),
        ):
            p = psutil.Process()
            r = func(p)
            assert "pid=%s" % p.pid in r
            assert "terminated" in r
            assert "name=" not in r
        with mock.patch.object(
            psutil.Process,
            "name",
            side_effect=psutil.AccessDenied(os.getpid()),
        ):
            p = psutil.Process()
            r = func(p)
            assert "pid=%s" % p.pid in r
            assert "name=" not in r

    def test_process__str__(self):
        self.test_process__repr__(func=str)

    def test_error__repr__(self):
        assert repr(psutil.Error()) == "psutil.Error()"

    def test_error__str__(self):
        assert str(psutil.Error()) == ""  # noqa

    def test_no_such_process__repr__(self):
        assert (
            repr(psutil.NoSuchProcess(321))
            == "psutil.NoSuchProcess(pid=321, msg='process no longer exists')"
        )
        assert (
            repr(psutil.NoSuchProcess(321, name="name", msg="msg"))
            == "psutil.NoSuchProcess(pid=321, name='name', msg='msg')"
        )

    def test_no_such_process__str__(self):
        assert (
            str(psutil.NoSuchProcess(321))
            == "process no longer exists (pid=321)"
        )
        assert (
            str(psutil.NoSuchProcess(321, name="name", msg="msg"))
            == "msg (pid=321, name='name')"
        )

    def test_zombie_process__repr__(self):
        assert (
            repr(psutil.ZombieProcess(321))
            == 'psutil.ZombieProcess(pid=321, msg="PID still '
            'exists but it\'s a zombie")'
        )
        assert (
            repr(psutil.ZombieProcess(321, name="name", ppid=320, msg="foo"))
            == "psutil.ZombieProcess(pid=321, ppid=320, name='name',"
            " msg='foo')"
        )

    def test_zombie_process__str__(self):
        assert (
            str(psutil.ZombieProcess(321))
            == "PID still exists but it's a zombie (pid=321)"
        )
        assert (
            str(psutil.ZombieProcess(321, name="name", ppid=320, msg="foo"))
            == "foo (pid=321, ppid=320, name='name')"
        )

    def test_access_denied__repr__(self):
        assert repr(psutil.AccessDenied(321)) == "psutil.AccessDenied(pid=321)"
        assert (
            repr(psutil.AccessDenied(321, name="name", msg="msg"))
            == "psutil.AccessDenied(pid=321, name='name', msg='msg')"
        )

    def test_access_denied__str__(self):
        assert str(psutil.AccessDenied(321)) == "(pid=321)"
        assert (
            str(psutil.AccessDenied(321, name="name", msg="msg"))
            == "msg (pid=321, name='name')"
        )

    def test_timeout_expired__repr__(self):
        assert (
            repr(psutil.TimeoutExpired(5))
            == "psutil.TimeoutExpired(seconds=5, msg='timeout after 5"
            " seconds')"
        )
        assert (
            repr(psutil.TimeoutExpired(5, pid=321, name="name"))
            == "psutil.TimeoutExpired(pid=321, name='name', seconds=5, "
            "msg='timeout after 5 seconds')"
        )

    def test_timeout_expired__str__(self):
        assert str(psutil.TimeoutExpired(5)) == "timeout after 5 seconds"
        assert (
            str(psutil.TimeoutExpired(5, pid=321, name="name"))
            == "timeout after 5 seconds (pid=321, name='name')"
        )

    def test_process__eq__(self):
        p1 = psutil.Process()
        p2 = psutil.Process()
        assert p1 == p2
        p2._ident = (0, 0)
        assert p1 != p2
        assert p1 != 'foo'

    def test_process__hash__(self):
        s = set([psutil.Process(), psutil.Process()])
        assert len(s) == 1


# ===================================================================
# --- Misc, generic, corner cases
# ===================================================================


class TestMisc(PsutilTestCase):
    def test__all__(self):
        dir_psutil = dir(psutil)
        for name in dir_psutil:
            if name in (
                'debug',
                'long',
                'tests',
                'test',
                'PermissionError',
                'ProcessLookupError',
            ):
                continue
            if not name.startswith('_'):
                try:
                    __import__(name)
                except ImportError:
                    if name not in psutil.__all__:
                        fun = getattr(psutil, name)
                        if fun is None:
                            continue
                        if (
                            fun.__doc__ is not None
                            and 'deprecated' not in fun.__doc__.lower()
                        ):
                            raise self.fail('%r not in psutil.__all__' % name)

        # Import 'star' will break if __all__ is inconsistent, see:
        # https://github.com/giampaolo/psutil/issues/656
        # Can't do `from psutil import *` as it won't work on python 3
        # so we simply iterate over __all__.
        for name in psutil.__all__:
            assert name in dir_psutil

    def test_version(self):
        assert (
            '.'.join([str(x) for x in psutil.version_info])
            == psutil.__version__
        )

    def test_process_as_dict_no_new_names(self):
        # See https://github.com/giampaolo/psutil/issues/813
        p = psutil.Process()
        p.foo = '1'
        assert 'foo' not in p.as_dict()

    def test_serialization(self):
        def check(ret):
            json.loads(json.dumps(ret))

            a = pickle.dumps(ret)
            b = pickle.loads(a)
            assert ret == b

        # --- process APIs

        proc = psutil.Process()
        check(psutil.Process().as_dict())

        ns = process_namespace(proc)
        for fun, name in ns.iter(ns.getters, clear_cache=True):
            with self.subTest(proc=proc, name=name):
                try:
                    ret = fun()
                except psutil.Error:
                    pass
                else:
                    check(ret)

        # --- system APIs

        ns = system_namespace()
        for fun, name in ns.iter(ns.getters):
            if name in {"win_service_iter", "win_service_get"}:
                continue
            if QEMU_USER and name == "net_if_stats":
                # OSError: [Errno 38] ioctl(SIOCETHTOOL) not implemented
                continue
            with self.subTest(name=name):
                try:
                    ret = fun()
                except psutil.AccessDenied:
                    pass
                else:
                    check(ret)

        # --- exception classes

        b = pickle.loads(
            pickle.dumps(
                psutil.NoSuchProcess(pid=4567, name='name', msg='msg')
            )
        )
        assert isinstance(b, psutil.NoSuchProcess)
        assert b.pid == 4567
        assert b.name == 'name'
        assert b.msg == 'msg'

        b = pickle.loads(
            pickle.dumps(
                psutil.ZombieProcess(pid=4567, name='name', ppid=42, msg='msg')
            )
        )
        assert isinstance(b, psutil.ZombieProcess)
        assert b.pid == 4567
        assert b.ppid == 42
        assert b.name == 'name'
        assert b.msg == 'msg'

        b = pickle.loads(
            pickle.dumps(psutil.AccessDenied(pid=123, name='name', msg='msg'))
        )
        assert isinstance(b, psutil.AccessDenied)
        assert b.pid == 123
        assert b.name == 'name'
        assert b.msg == 'msg'

        b = pickle.loads(
            pickle.dumps(
                psutil.TimeoutExpired(seconds=33, pid=4567, name='name')
            )
        )
        assert isinstance(b, psutil.TimeoutExpired)
        assert b.seconds == 33
        assert b.pid == 4567
        assert b.name == 'name'

    # # XXX: https://github.com/pypa/setuptools/pull/2896
    # @pytest.mark.skipif(APPVEYOR,
    #     reason="temporarily disabled due to setuptools bug"
    # )
    # def test_setup_script(self):
    #     setup_py = os.path.join(ROOT_DIR, 'setup.py')
    #     if CI_TESTING and not os.path.exists(setup_py):
    #         raise pytest.skip("can't find setup.py")
    #     module = import_module_by_path(setup_py)
    #     self.assertRaises(SystemExit, module.setup)
    #     self.assertEqual(module.get_version(), psutil.__version__)

    def test_ad_on_process_creation(self):
        # We are supposed to be able to instantiate Process also in case
        # of zombie processes or access denied.
        with mock.patch.object(
            psutil.Process, '_get_ident', side_effect=psutil.AccessDenied
        ) as meth:
            psutil.Process()
            assert meth.called

        with mock.patch.object(
            psutil.Process, '_get_ident', side_effect=psutil.ZombieProcess(1)
        ) as meth:
            psutil.Process()
            assert meth.called

        with mock.patch.object(
            psutil.Process, '_get_ident', side_effect=ValueError
        ) as meth:
            with pytest.raises(ValueError):
                psutil.Process()
            assert meth.called

        with mock.patch.object(
            psutil.Process, '_get_ident', side_effect=psutil.NoSuchProcess(1)
        ) as meth:
            with self.assertRaises(psutil.NoSuchProcess):
                psutil.Process()
            assert meth.called

    def test_sanity_version_check(self):
        # see: https://github.com/giampaolo/psutil/issues/564
        with mock.patch(
            "psutil._psplatform.cext.version", return_value="0.0.0"
        ):
            with pytest.raises(ImportError) as cm:
                reload_module(psutil)
            assert "version conflict" in str(cm.value).lower()


# ===================================================================
# --- psutil/_common.py utils
# ===================================================================


class TestMemoizeDecorator(PsutilTestCase):
    def setUp(self):
        self.calls = []

    tearDown = setUp

    def run_against(self, obj, expected_retval=None):
        # no args
        for _ in range(2):
            ret = obj()
            assert self.calls == [((), {})]
            if expected_retval is not None:
                assert ret == expected_retval
        # with args
        for _ in range(2):
            ret = obj(1)
            assert self.calls == [((), {}), ((1,), {})]
            if expected_retval is not None:
                assert ret == expected_retval
        # with args + kwargs
        for _ in range(2):
            ret = obj(1, bar=2)
            assert self.calls == [((), {}), ((1,), {}), ((1,), {'bar': 2})]
            if expected_retval is not None:
                assert ret == expected_retval
        # clear cache
        assert len(self.calls) == 3
        obj.cache_clear()
        ret = obj()
        if expected_retval is not None:
            assert ret == expected_retval
        assert len(self.calls) == 4
        # docstring
        assert obj.__doc__ == "My docstring."

    def test_function(self):
        @memoize
        def foo(*args, **kwargs):
            """My docstring."""
            baseclass.calls.append((args, kwargs))
            return 22

        baseclass = self
        self.run_against(foo, expected_retval=22)

    def test_class(self):
        @memoize
        class Foo:
            """My docstring."""

            def __init__(self, *args, **kwargs):
                baseclass.calls.append((args, kwargs))

            def bar(self):
                return 22

        baseclass = self
        self.run_against(Foo, expected_retval=None)
        assert Foo().bar() == 22

    def test_class_singleton(self):
        # @memoize can be used against classes to create singletons
        @memoize
        class Bar:
            def __init__(self, *args, **kwargs):
                pass

        assert Bar() is Bar()
        assert id(Bar()) == id(Bar())
        assert id(Bar(1)) == id(Bar(1))
        assert id(Bar(1, foo=3)) == id(Bar(1, foo=3))
        assert id(Bar(1)) != id(Bar(2))

    def test_staticmethod(self):
        class Foo:
            @staticmethod
            @memoize
            def bar(*args, **kwargs):
                """My docstring."""
                baseclass.calls.append((args, kwargs))
                return 22

        baseclass = self
        self.run_against(Foo().bar, expected_retval=22)

    def test_classmethod(self):
        class Foo:
            @classmethod
            @memoize
            def bar(cls, *args, **kwargs):
                """My docstring."""
                baseclass.calls.append((args, kwargs))
                return 22

        baseclass = self
        self.run_against(Foo().bar, expected_retval=22)

    def test_original(self):
        # This was the original test before I made it dynamic to test it
        # against different types. Keeping it anyway.
        @memoize
        def foo(*args, **kwargs):
            """Foo docstring."""
            calls.append(None)
            return (args, kwargs)

        calls = []
        # no args
        for _ in range(2):
            ret = foo()
            expected = ((), {})
            assert ret == expected
            assert len(calls) == 1
        # with args
        for _ in range(2):
            ret = foo(1)
            expected = ((1,), {})
            assert ret == expected
            assert len(calls) == 2
        # with args + kwargs
        for _ in range(2):
            ret = foo(1, bar=2)
            expected = ((1,), {'bar': 2})
            assert ret == expected
            assert len(calls) == 3
        # clear cache
        foo.cache_clear()
        ret = foo()
        expected = ((), {})
        assert ret == expected
        assert len(calls) == 4
        # docstring
        assert foo.__doc__ == "Foo docstring."


class TestCommonModule(PsutilTestCase):
    def test_memoize_when_activated(self):
        class Foo:
            @memoize_when_activated
            def foo(self):
                calls.append(None)

        f = Foo()
        calls = []
        f.foo()
        f.foo()
        assert len(calls) == 2

        # activate
        calls = []
        f.foo.cache_activate(f)
        f.foo()
        f.foo()
        assert len(calls) == 1

        # deactivate
        calls = []
        f.foo.cache_deactivate(f)
        f.foo()
        f.foo()
        assert len(calls) == 2

    def test_parse_environ_block(self):
        def k(s):
            return s.upper() if WINDOWS else s

        assert parse_environ_block("a=1\0") == {k("a"): "1"}
        assert parse_environ_block("a=1\0b=2\0\0") == {
            k("a"): "1",
            k("b"): "2",
        }
        assert parse_environ_block("a=1\0b=\0\0") == {k("a"): "1", k("b"): ""}
        # ignore everything after \0\0
        assert parse_environ_block("a=1\0b=2\0\0c=3\0") == {
            k("a"): "1",
            k("b"): "2",
        }
        # ignore everything that is not an assignment
        assert parse_environ_block("xxx\0a=1\0") == {k("a"): "1"}
        assert parse_environ_block("a=1\0=b=2\0") == {k("a"): "1"}
        # do not fail if the block is incomplete
        assert parse_environ_block("a=1\0b=2") == {k("a"): "1"}

    def test_supports_ipv6(self):
        self.addCleanup(supports_ipv6.cache_clear)
        if supports_ipv6():
            with mock.patch('psutil._common.socket') as s:
                s.has_ipv6 = False
                supports_ipv6.cache_clear()
                assert not supports_ipv6()

            supports_ipv6.cache_clear()
            with mock.patch(
                'psutil._common.socket.socket', side_effect=socket.error
            ) as s:
                assert not supports_ipv6()
                assert s.called

            supports_ipv6.cache_clear()
            with mock.patch(
                'psutil._common.socket.socket', side_effect=socket.gaierror
            ) as s:
                assert not supports_ipv6()
                supports_ipv6.cache_clear()
                assert s.called

            supports_ipv6.cache_clear()
            with mock.patch(
                'psutil._common.socket.socket.bind',
                side_effect=socket.gaierror,
            ) as s:
                assert not supports_ipv6()
                supports_ipv6.cache_clear()
                assert s.called
        else:
            with pytest.raises(socket.error):
                sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                try:
                    sock.bind(("::1", 0))
                finally:
                    sock.close()

    def test_isfile_strict(self):
        this_file = os.path.abspath(__file__)
        assert isfile_strict(this_file)
        assert not isfile_strict(os.path.dirname(this_file))
        with mock.patch(
            'psutil._common.os.stat', side_effect=OSError(errno.EPERM, "foo")
        ):
            with pytest.raises(OSError):
                isfile_strict(this_file)
        with mock.patch(
            'psutil._common.os.stat', side_effect=OSError(errno.EACCES, "foo")
        ):
            with pytest.raises(OSError):
                isfile_strict(this_file)
        with mock.patch(
            'psutil._common.os.stat', side_effect=OSError(errno.ENOENT, "foo")
        ):
            assert not isfile_strict(this_file)
        with mock.patch('psutil._common.stat.S_ISREG', return_value=False):
            assert not isfile_strict(this_file)

    def test_debug(self):
        if PY3:
            from io import StringIO
        else:
            from StringIO import StringIO

        with mock.patch.object(psutil._common, "PSUTIL_DEBUG", True):
            with redirect_stderr(StringIO()) as f:
                debug("hello")
                sys.stderr.flush()
        msg = f.getvalue()
        assert msg.startswith("psutil-debug"), msg
        assert "hello" in msg
        assert __file__.replace('.pyc', '.py') in msg

        # supposed to use repr(exc)
        with mock.patch.object(psutil._common, "PSUTIL_DEBUG", True):
            with redirect_stderr(StringIO()) as f:
                debug(ValueError("this is an error"))
        msg = f.getvalue()
        assert "ignoring ValueError" in msg
        assert "'this is an error'" in msg

        # supposed to use str(exc), because of extra info about file name
        with mock.patch.object(psutil._common, "PSUTIL_DEBUG", True):
            with redirect_stderr(StringIO()) as f:
                exc = OSError(2, "no such file")
                exc.filename = "/foo"
                debug(exc)
        msg = f.getvalue()
        assert "no such file" in msg
        assert "/foo" in msg

    def test_cat_bcat(self):
        testfn = self.get_testfn()
        with open(testfn, "w") as f:
            f.write("foo")
        assert cat(testfn) == "foo"
        assert bcat(testfn) == b"foo"
        with pytest.raises(FileNotFoundError):
            cat(testfn + '-invalid')
        with pytest.raises(FileNotFoundError):
            bcat(testfn + '-invalid')
        assert cat(testfn + '-invalid', fallback="bar") == "bar"
        assert bcat(testfn + '-invalid', fallback="bar") == "bar"


# ===================================================================
# --- Tests for wrap_numbers() function.
# ===================================================================


nt = collections.namedtuple('foo', 'a b c')


class TestWrapNumbers(PsutilTestCase):
    def setUp(self):
        wrap_numbers.cache_clear()

    tearDown = setUp

    def test_first_call(self):
        input = {'disk1': nt(5, 5, 5)}
        assert wrap_numbers(input, 'disk_io') == input

    def test_input_hasnt_changed(self):
        input = {'disk1': nt(5, 5, 5)}
        assert wrap_numbers(input, 'disk_io') == input
        assert wrap_numbers(input, 'disk_io') == input

    def test_increase_but_no_wrap(self):
        input = {'disk1': nt(5, 5, 5)}
        assert wrap_numbers(input, 'disk_io') == input
        input = {'disk1': nt(10, 15, 20)}
        assert wrap_numbers(input, 'disk_io') == input
        input = {'disk1': nt(20, 25, 30)}
        assert wrap_numbers(input, 'disk_io') == input
        input = {'disk1': nt(20, 25, 30)}
        assert wrap_numbers(input, 'disk_io') == input

    def test_wrap(self):
        # let's say 100 is the threshold
        input = {'disk1': nt(100, 100, 100)}
        assert wrap_numbers(input, 'disk_io') == input
        # first wrap restarts from 10
        input = {'disk1': nt(100, 100, 10)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(100, 100, 110)}
        # then it remains the same
        input = {'disk1': nt(100, 100, 10)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(100, 100, 110)}
        # then it goes up
        input = {'disk1': nt(100, 100, 90)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(100, 100, 190)}
        # then it wraps again
        input = {'disk1': nt(100, 100, 20)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(100, 100, 210)}
        # and remains the same
        input = {'disk1': nt(100, 100, 20)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(100, 100, 210)}
        # now wrap another num
        input = {'disk1': nt(50, 100, 20)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(150, 100, 210)}
        # and again
        input = {'disk1': nt(40, 100, 20)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(190, 100, 210)}
        # keep it the same
        input = {'disk1': nt(40, 100, 20)}
        assert wrap_numbers(input, 'disk_io') == {'disk1': nt(190, 100, 210)}

    def test_changing_keys(self):
        # Emulate a case where the second call to disk_io()
        # (or whatever) provides a new disk, then the new disk
        # disappears on the third call.
        input = {'disk1': nt(5, 5, 5)}
        assert wrap_numbers(input, 'disk_io') == input
        input = {'disk1': nt(5, 5, 5), 'disk2': nt(7, 7, 7)}
        assert wrap_numbers(input, 'disk_io') == input
        input = {'disk1': nt(8, 8, 8)}
        assert wrap_numbers(input, 'disk_io') == input

    def test_changing_keys_w_wrap(self):
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        assert wrap_numbers(input, 'disk_io') == input
        # disk 2 wraps
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 10)}
        assert wrap_numbers(input, 'disk_io') == {
            'disk1': nt(50, 50, 50),
            'disk2': nt(100, 100, 110),
        }
        # disk 2 disappears
        input = {'disk1': nt(50, 50, 50)}
        assert wrap_numbers(input, 'disk_io') == input

        # then it appears again; the old wrap is supposed to be
        # gone.
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        assert wrap_numbers(input, 'disk_io') == input
        # remains the same
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        assert wrap_numbers(input, 'disk_io') == input
        # and then wraps again
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 10)}
        assert wrap_numbers(input, 'disk_io') == {
            'disk1': nt(50, 50, 50),
            'disk2': nt(100, 100, 110),
        }

    def test_real_data(self):
        d = {
            'nvme0n1': (300, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048),
            'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8),
            'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28),
            'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348),
        }
        assert wrap_numbers(d, 'disk_io') == d
        assert wrap_numbers(d, 'disk_io') == d
        # decrease this   ↓
        d = {
            'nvme0n1': (100, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048),
            'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8),
            'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28),
            'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348),
        }
        out = wrap_numbers(d, 'disk_io')
        assert out['nvme0n1'][0] == 400

    # --- cache tests

    def test_cache_first_call(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        assert cache[1] == {'disk_io': {}}
        assert cache[2] == {'disk_io': {}}

    def test_cache_call_twice(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        input = {'disk1': nt(10, 10, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        assert cache[1] == {
            'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 0}
        }
        assert cache[2] == {'disk_io': {}}

    def test_cache_wrap(self):
        # let's say 100 is the threshold
        input = {'disk1': nt(100, 100, 100)}
        wrap_numbers(input, 'disk_io')

        # first wrap restarts from 10
        input = {'disk1': nt(100, 100, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        assert cache[1] == {
            'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 100}
        }
        assert cache[2] == {'disk_io': {'disk1': set([('disk1', 2)])}}

        def check_cache_info():
            cache = wrap_numbers.cache_info()
            assert cache[1] == {
                'disk_io': {
                    ('disk1', 0): 0,
                    ('disk1', 1): 0,
                    ('disk1', 2): 100,
                }
            }
            assert cache[2] == {'disk_io': {'disk1': set([('disk1', 2)])}}

        # then it remains the same
        input = {'disk1': nt(100, 100, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        check_cache_info()

        # then it goes up
        input = {'disk1': nt(100, 100, 90)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        check_cache_info()

        # then it wraps again
        input = {'disk1': nt(100, 100, 20)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        assert cache[1] == {
            'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 190}
        }
        assert cache[2] == {'disk_io': {'disk1': set([('disk1', 2)])}}

    def test_cache_changing_keys(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        input = {'disk1': nt(5, 5, 5), 'disk2': nt(7, 7, 7)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        assert cache[0] == {'disk_io': input}
        assert cache[1] == {
            'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 0}
        }
        assert cache[2] == {'disk_io': {}}

    def test_cache_clear(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        wrap_numbers(input, 'disk_io')
        wrap_numbers.cache_clear('disk_io')
        assert wrap_numbers.cache_info() == ({}, {}, {})
        wrap_numbers.cache_clear('disk_io')
        wrap_numbers.cache_clear('?!?')

    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_cache_clear_public_apis(self):
        if not psutil.disk_io_counters() or not psutil.net_io_counters():
            raise pytest.skip("no disks or NICs available")
        psutil.disk_io_counters()
        psutil.net_io_counters()
        caches = wrap_numbers.cache_info()
        for cache in caches:
            assert 'psutil.disk_io_counters' in cache
            assert 'psutil.net_io_counters' in cache

        psutil.disk_io_counters.cache_clear()
        caches = wrap_numbers.cache_info()
        for cache in caches:
            assert 'psutil.net_io_counters' in cache
            assert 'psutil.disk_io_counters' not in cache

        psutil.net_io_counters.cache_clear()
        caches = wrap_numbers.cache_info()
        assert caches == ({}, {}, {})


# ===================================================================
# --- Example script tests
# ===================================================================


@pytest.mark.skipif(
    not os.path.exists(SCRIPTS_DIR), reason="can't locate scripts directory"
)
class TestScripts(PsutilTestCase):
    """Tests for scripts in the "scripts" directory."""

    @staticmethod
    def assert_stdout(exe, *args, **kwargs):
        kwargs.setdefault("env", PYTHON_EXE_ENV)
        exe = '%s' % os.path.join(SCRIPTS_DIR, exe)
        cmd = [PYTHON_EXE, exe]
        for arg in args:
            cmd.append(arg)
        try:
            out = sh(cmd, **kwargs).strip()
        except RuntimeError as err:
            if 'AccessDenied' in str(err):
                return str(err)
            else:
                raise
        assert out, out
        return out

    @staticmethod
    def assert_syntax(exe):
        exe = os.path.join(SCRIPTS_DIR, exe)
        with open(exe, encoding="utf8") if PY3 else open(exe) as f:
            src = f.read()
        ast.parse(src)

    def test_coverage(self):
        # make sure all example scripts have a test method defined
        meths = dir(self)
        for name in os.listdir(SCRIPTS_DIR):
            if name.endswith('.py'):
                if 'test_' + os.path.splitext(name)[0] not in meths:
                    # self.assert_stdout(name)
                    raise self.fail(
                        'no test defined for %r script'
                        % os.path.join(SCRIPTS_DIR, name)
                    )

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_executable(self):
        for root, dirs, files in os.walk(SCRIPTS_DIR):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    if not stat.S_IXUSR & os.stat(path)[stat.ST_MODE]:
                        raise self.fail('%r is not executable' % path)

    def test_disk_usage(self):
        self.assert_stdout('disk_usage.py')

    def test_free(self):
        self.assert_stdout('free.py')

    def test_meminfo(self):
        self.assert_stdout('meminfo.py')

    def test_procinfo(self):
        self.assert_stdout('procinfo.py', str(os.getpid()))

    @pytest.mark.skipif(CI_TESTING and not psutil.users(), reason="no users")
    def test_who(self):
        self.assert_stdout('who.py')

    def test_ps(self):
        self.assert_stdout('ps.py')

    def test_pstree(self):
        self.assert_stdout('pstree.py')

    def test_netstat(self):
        self.assert_stdout('netstat.py')

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_ifconfig(self):
        self.assert_stdout('ifconfig.py')

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    def test_pmap(self):
        self.assert_stdout('pmap.py', str(os.getpid()))

    def test_procsmem(self):
        if 'uss' not in psutil.Process().memory_full_info()._fields:
            raise pytest.skip("not supported")
        self.assert_stdout('procsmem.py')

    def test_killall(self):
        self.assert_syntax('killall.py')

    def test_nettop(self):
        self.assert_syntax('nettop.py')

    def test_top(self):
        self.assert_syntax('top.py')

    def test_iotop(self):
        self.assert_syntax('iotop.py')

    def test_pidof(self):
        output = self.assert_stdout('pidof.py', psutil.Process().name())
        assert str(os.getpid()) in output

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_winservices(self):
        self.assert_stdout('winservices.py')

    def test_cpu_distribution(self):
        self.assert_syntax('cpu_distribution.py')

    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_temperatures(self):
        if not psutil.sensors_temperatures():
            raise pytest.skip("no temperatures")
        self.assert_stdout('temperatures.py')

    @pytest.mark.skipif(not HAS_SENSORS_FANS, reason="not supported")
    def test_fans(self):
        if not psutil.sensors_fans():
            raise pytest.skip("no fans")
        self.assert_stdout('fans.py')

    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    @pytest.mark.skipif(not HAS_BATTERY, reason="no battery")
    def test_battery(self):
        self.assert_stdout('battery.py')

    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    @pytest.mark.skipif(not HAS_BATTERY, reason="no battery")
    def test_sensors(self):
        self.assert_stdout('sensors.py')
