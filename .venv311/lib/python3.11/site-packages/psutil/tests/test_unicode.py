#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Notes about unicode handling in psutil
======================================.

Starting from version 5.3.0 psutil adds unicode support, see:
https://github.com/giampaolo/psutil/issues/1040
The notes below apply to *any* API returning a string such as
process exe(), cwd() or username():

* all strings are encoded by using the OS filesystem encoding
  (sys.getfilesystemencoding()) which varies depending on the platform
  (e.g. "UTF-8" on macOS, "mbcs" on Win)
* no API call is supposed to crash with UnicodeDecodeError
* instead, in case of badly encoded data returned by the OS, the
  following error handlers are used to replace the corrupted characters in
  the string:
    * sys.getfilesystemencodeerrors() or "surrogatescape" on POSIX and
      "replace" on Windows.

For a detailed explanation of how psutil handles unicode see #1040.

Tests
=====

List of APIs returning or dealing with a string:
('not tested' means they are not tested to deal with non-ASCII strings):

* Process.cmdline()
* Process.cwd()
* Process.environ()
* Process.exe()
* Process.memory_maps()
* Process.name()
* Process.net_connections('unix')
* Process.open_files()
* Process.username()             (not tested)

* disk_io_counters()             (not tested)
* disk_partitions()              (not tested)
* disk_usage(str)
* net_connections('unix')
* net_if_addrs()                 (not tested)
* net_if_stats()                 (not tested)
* net_io_counters()              (not tested)
* sensors_fans()                 (not tested)
* sensors_temperatures()         (not tested)
* users()                        (not tested)

* WindowsService.binpath()       (not tested)
* WindowsService.description()   (not tested)
* WindowsService.display_name()  (not tested)
* WindowsService.name()          (not tested)
* WindowsService.status()        (not tested)
* WindowsService.username()      (not tested)

In here we create a unicode path with a funky non-ASCII name and (where
possible) make psutil return it back (e.g. on name(), exe(), open_files(),
etc.) and make sure that:

* psutil never crashes with UnicodeDecodeError
* the returned path matches
"""

import os
import shutil
import warnings
from contextlib import closing

import psutil
from psutil import BSD
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import WINDOWS
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_CONNECTIONS_UNIX
from psutil.tests import INVALID_UNICODE_SUFFIX
from psutil.tests import PYPY
from psutil.tests import TESTFN_PREFIX
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import bind_unix_socket
from psutil.tests import chdir
from psutil.tests import copyload_shared_lib
from psutil.tests import create_py_exe
from psutil.tests import get_testfn
from psutil.tests import pytest
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_subproc
from psutil.tests import terminate


def try_unicode(suffix):
    """Return True if both the fs and the subprocess module can
    deal with a unicode file name.
    """
    sproc = None
    testfn = get_testfn(suffix=suffix)
    try:
        safe_rmpath(testfn)
        create_py_exe(testfn)
        sproc = spawn_subproc(cmd=[testfn])
        shutil.copyfile(testfn, testfn + '-2')
        safe_rmpath(testfn + '-2')
    except (UnicodeEncodeError, OSError):
        return False
    else:
        return True
    finally:
        if sproc is not None:
            terminate(sproc)
        safe_rmpath(testfn)


# ===================================================================
# FS APIs
# ===================================================================


class BaseUnicodeTest(PsutilTestCase):
    funky_suffix = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.skip_tests = False
        cls.funky_name = None
        if cls.funky_suffix is not None:
            if not try_unicode(cls.funky_suffix):
                cls.skip_tests = True
            else:
                cls.funky_name = get_testfn(suffix=cls.funky_suffix)
                create_py_exe(cls.funky_name)

    def setUp(self):
        super().setUp()
        if self.skip_tests:
            return pytest.skip("can't handle unicode str")


@pytest.mark.xdist_group(name="serial")
@pytest.mark.skipif(ASCII_FS, reason="ASCII fs")
class TestFSAPIs(BaseUnicodeTest):
    """Test FS APIs with a funky, valid, UTF8 path name."""

    funky_suffix = UNICODE_SUFFIX

    def expect_exact_path_match(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.funky_name in os.listdir(".")

    # ---

    def test_proc_exe(self):
        cmd = [
            self.funky_name,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        subp = self.spawn_subproc(cmd)
        p = psutil.Process(subp.pid)
        exe = p.exe()
        assert isinstance(exe, str)
        if self.expect_exact_path_match():
            assert os.path.normcase(exe) == os.path.normcase(self.funky_name)

    def test_proc_name(self):
        cmd = [
            self.funky_name,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        subp = self.spawn_subproc(cmd)
        name = psutil.Process(subp.pid).name()
        assert isinstance(name, str)
        if self.expect_exact_path_match():
            assert name == os.path.basename(self.funky_name)

    def test_proc_cmdline(self):
        cmd = [
            self.funky_name,
            "-c",
            "import time; [time.sleep(0.1) for x in range(100)]",
        ]
        subp = self.spawn_subproc(cmd)
        p = psutil.Process(subp.pid)
        cmdline = p.cmdline()
        for part in cmdline:
            assert isinstance(part, str)
        if self.expect_exact_path_match():
            assert cmdline == cmd

    def test_proc_cwd(self):
        dname = self.funky_name + "2"
        self.addCleanup(safe_rmpath, dname)
        safe_mkdir(dname)
        with chdir(dname):
            p = psutil.Process()
            cwd = p.cwd()
        assert isinstance(p.cwd(), str)
        if self.expect_exact_path_match():
            assert cwd == dname

    @pytest.mark.skipif(PYPY and WINDOWS, reason="fails on PYPY + WINDOWS")
    @pytest.mark.skipif(
        NETBSD or OPENBSD, reason="broken on NETBSD or OPENBSD"
    )
    def test_proc_open_files(self):
        p = psutil.Process()
        start = set(p.open_files())
        with open(self.funky_name, 'rb'):
            new = set(p.open_files())
        path = (new - start).pop().path
        assert isinstance(path, str)
        if BSD and not path:
            # XXX - see https://github.com/giampaolo/psutil/issues/595
            return pytest.skip("open_files on BSD is broken")
        if self.expect_exact_path_match():
            assert os.path.normcase(path) == os.path.normcase(self.funky_name)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @pytest.mark.skipif(
        not HAS_NET_CONNECTIONS_UNIX, reason="can't list UNIX sockets"
    )
    def test_proc_net_connections(self):
        name = self.get_testfn(suffix=self.funky_suffix)
        sock = bind_unix_socket(name)
        with closing(sock):
            conn = psutil.Process().net_connections('unix')[0]
            assert isinstance(conn.laddr, str)
            if not conn.laddr and MACOS and CI_TESTING:
                return pytest.skip("unreliable on OSX")
            assert conn.laddr == name

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @pytest.mark.skipif(
        not HAS_NET_CONNECTIONS_UNIX, reason="can't list UNIX sockets"
    )
    @skip_on_access_denied()
    def test_net_connections(self):
        def find_sock(cons):
            for conn in cons:
                if os.path.basename(conn.laddr).startswith(TESTFN_PREFIX):
                    return conn
            raise ValueError("connection not found")

        name = self.get_testfn(suffix=self.funky_suffix)
        sock = bind_unix_socket(name)
        with closing(sock):
            cons = psutil.net_connections(kind='unix')
            conn = find_sock(cons)
            assert isinstance(conn.laddr, str)
            assert conn.laddr == name

    def test_disk_usage(self):
        dname = self.funky_name + "2"
        self.addCleanup(safe_rmpath, dname)
        safe_mkdir(dname)
        psutil.disk_usage(dname)

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    def test_memory_maps(self):
        with copyload_shared_lib(suffix=self.funky_suffix) as funky_path:

            def normpath(p):
                return os.path.realpath(os.path.normcase(p))

            libpaths = [
                normpath(x.path) for x in psutil.Process().memory_maps()
            ]
            # ...just to have a clearer msg in case of failure
            libpaths = [x for x in libpaths if TESTFN_PREFIX in x]
            assert normpath(funky_path) in libpaths
            for path in libpaths:
                assert isinstance(path, str)


@pytest.mark.skipif(CI_TESTING, reason="unreliable on CI")
class TestFSAPIsWithInvalidPath(TestFSAPIs):
    """Test FS APIs with a funky, invalid path name."""

    funky_suffix = INVALID_UNICODE_SUFFIX

    def expect_exact_path_match(self):
        return not MACOS


# ===================================================================
# Non fs APIs
# ===================================================================


class TestNonFSAPIS(BaseUnicodeTest):
    """Unicode tests for non fs-related APIs."""

    funky_suffix = UNICODE_SUFFIX

    @pytest.mark.skipif(not HAS_ENVIRON, reason="not supported")
    @pytest.mark.skipif(PYPY and WINDOWS, reason="segfaults on PYPY + WINDOWS")
    def test_proc_environ(self):
        # Note: differently from others, this test does not deal
        # with fs paths.
        env = os.environ.copy()
        env['FUNNY_ARG'] = self.funky_suffix
        sproc = self.spawn_subproc(env=env)
        p = psutil.Process(sproc.pid)
        env = p.environ()
        for k, v in env.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
        assert env['FUNNY_ARG'] == self.funky_suffix
