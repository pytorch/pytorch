#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Iterate over all process PIDs and for each one of them invoke and
test all psutil.Process() methods.
"""

import enum
import errno
import multiprocessing
import os
import stat
import time
import traceback

import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import unicode
from psutil.tests import CI_TESTING
from psutil.tests import PYTEST_PARALLEL
from psutil.tests import QEMU_USER
from psutil.tests import VALID_PROC_STATUSES
from psutil.tests import PsutilTestCase
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import is_namedtuple
from psutil.tests import is_win_secure_system_proc
from psutil.tests import process_namespace
from psutil.tests import pytest


# Cuts the time in half, but (e.g.) on macOS the process pool stays
# alive after join() (multiprocessing bug?), messing up other tests.
USE_PROC_POOL = LINUX and not CI_TESTING and not PYTEST_PARALLEL


def proc_info(pid):
    tcase = PsutilTestCase()

    def check_exception(exc, proc, name, ppid):
        tcase.assertEqual(exc.pid, pid)
        if exc.name is not None:
            tcase.assertEqual(exc.name, name)
        if isinstance(exc, psutil.ZombieProcess):
            tcase.assertProcessZombie(proc)
            if exc.ppid is not None:
                tcase.assertGreaterEqual(exc.ppid, 0)
                tcase.assertEqual(exc.ppid, ppid)
        elif isinstance(exc, psutil.NoSuchProcess):
            tcase.assertProcessGone(proc)
        str(exc)
        repr(exc)

    def do_wait():
        if pid != 0:
            try:
                proc.wait(0)
            except psutil.Error as exc:
                check_exception(exc, proc, name, ppid)

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        tcase.assertPidGone(pid)
        return {}
    try:
        d = proc.as_dict(['ppid', 'name'])
    except psutil.NoSuchProcess:
        tcase.assertProcessGone(proc)
    else:
        name, ppid = d['name'], d['ppid']
        info = {'pid': proc.pid}
        ns = process_namespace(proc)
        # We don't use oneshot() because in order not to fool
        # check_exception() in case of NSP.
        for fun, fun_name in ns.iter(ns.getters, clear_cache=False):
            try:
                info[fun_name] = fun()
            except psutil.Error as exc:
                check_exception(exc, proc, name, ppid)
                continue
        do_wait()
        return info


class TestFetchAllProcesses(PsutilTestCase):
    """Test which iterates over all running processes and performs
    some sanity checks against Process API's returned values.
    Uses a process pool to get info about all processes.
    """

    def setUp(self):
        psutil._set_debug(False)
        # Using a pool in a CI env may result in deadlock, see:
        # https://github.com/giampaolo/psutil/issues/2104
        if USE_PROC_POOL:
            self.pool = multiprocessing.Pool()

    def tearDown(self):
        psutil._set_debug(True)
        if USE_PROC_POOL:
            self.pool.terminate()
            self.pool.join()

    def iter_proc_info(self):
        # Fixes "can't pickle <function proc_info>: it's not the
        # same object as test_process_all.proc_info".
        from psutil.tests.test_process_all import proc_info

        if USE_PROC_POOL:
            return self.pool.imap_unordered(proc_info, psutil.pids())
        else:
            ls = []
            for pid in psutil.pids():
                ls.append(proc_info(pid))
            return ls

    def test_all(self):
        failures = []
        for info in self.iter_proc_info():
            for name, value in info.items():
                meth = getattr(self, name)
                try:
                    meth(value, info)
                except Exception:  # noqa: BLE001
                    s = '\n' + '=' * 70 + '\n'
                    s += "FAIL: name=test_%s, pid=%s, ret=%s\ninfo=%s\n" % (
                        name,
                        info['pid'],
                        repr(value),
                        info,
                    )
                    s += '-' * 70
                    s += "\n%s" % traceback.format_exc()
                    s = "\n".join((" " * 4) + i for i in s.splitlines()) + "\n"
                    failures.append(s)
                else:
                    if value not in (0, 0.0, [], None, '', {}):
                        assert value, value
        if failures:
            raise self.fail(''.join(failures))

    def cmdline(self, ret, info):
        assert isinstance(ret, list)
        for part in ret:
            assert isinstance(part, str)

    def exe(self, ret, info):
        assert isinstance(ret, (str, unicode))
        assert ret.strip() == ret
        if ret:
            if WINDOWS and not ret.endswith('.exe'):
                return  # May be "Registry", "MemCompression", ...
            assert os.path.isabs(ret), ret
            # Note: os.stat() may return False even if the file is there
            # hence we skip the test, see:
            # http://stackoverflow.com/questions/3112546/os-path-exists-lies
            if POSIX and os.path.isfile(ret):
                if hasattr(os, 'access') and hasattr(os, "X_OK"):
                    # XXX: may fail on MACOS
                    try:
                        assert os.access(ret, os.X_OK)
                    except AssertionError:
                        if os.path.exists(ret) and not CI_TESTING:
                            raise

    def pid(self, ret, info):
        assert isinstance(ret, int)
        assert ret >= 0

    def ppid(self, ret, info):
        assert isinstance(ret, (int, long))
        assert ret >= 0
        proc_info(ret)

    def name(self, ret, info):
        assert isinstance(ret, (str, unicode))
        if WINDOWS and not ret and is_win_secure_system_proc(info['pid']):
            # https://github.com/giampaolo/psutil/issues/2338
            return
        # on AIX, "<exiting>" processes don't have names
        if not AIX:
            assert ret, repr(ret)

    def create_time(self, ret, info):
        assert isinstance(ret, float)
        try:
            assert ret >= 0
        except AssertionError:
            # XXX
            if OPENBSD and info['status'] == psutil.STATUS_ZOMBIE:
                pass
            else:
                raise
        # this can't be taken for granted on all platforms
        # self.assertGreaterEqual(ret, psutil.boot_time())
        # make sure returned value can be pretty printed
        # with strftime
        time.strftime("%Y %m %d %H:%M:%S", time.localtime(ret))

    def uids(self, ret, info):
        assert is_namedtuple(ret)
        for uid in ret:
            assert isinstance(uid, int)
            assert uid >= 0

    def gids(self, ret, info):
        assert is_namedtuple(ret)
        # note: testing all gids as above seems not to be reliable for
        # gid == 30 (nodoby); not sure why.
        for gid in ret:
            assert isinstance(gid, int)
            if not MACOS and not NETBSD:
                assert gid >= 0

    def username(self, ret, info):
        assert isinstance(ret, str)
        assert ret.strip() == ret
        assert ret.strip()

    def status(self, ret, info):
        assert isinstance(ret, str)
        assert ret, ret
        if QEMU_USER:
            # status does not work under qemu user
            return
        assert ret != '?'  # XXX
        assert ret in VALID_PROC_STATUSES

    def io_counters(self, ret, info):
        assert is_namedtuple(ret)
        for field in ret:
            assert isinstance(field, (int, long))
            if field != -1:
                assert field >= 0

    def ionice(self, ret, info):
        if LINUX:
            assert isinstance(ret.ioclass, int)
            assert isinstance(ret.value, int)
            assert ret.ioclass >= 0
            assert ret.value >= 0
        else:  # Windows, Cygwin
            choices = [
                psutil.IOPRIO_VERYLOW,
                psutil.IOPRIO_LOW,
                psutil.IOPRIO_NORMAL,
                psutil.IOPRIO_HIGH,
            ]
            assert isinstance(ret, int)
            assert ret >= 0
            assert ret in choices

    def num_threads(self, ret, info):
        assert isinstance(ret, int)
        if WINDOWS and ret == 0 and is_win_secure_system_proc(info['pid']):
            # https://github.com/giampaolo/psutil/issues/2338
            return
        assert ret >= 1

    def threads(self, ret, info):
        assert isinstance(ret, list)
        for t in ret:
            assert is_namedtuple(t)
            assert t.id >= 0
            assert t.user_time >= 0
            assert t.system_time >= 0
            for field in t:
                assert isinstance(field, (int, float))

    def cpu_times(self, ret, info):
        assert is_namedtuple(ret)
        for n in ret:
            assert isinstance(n, float)
            assert n >= 0
        # TODO: check ntuple fields

    def cpu_percent(self, ret, info):
        assert isinstance(ret, float)
        assert 0.0 <= ret <= 100.0, ret

    def cpu_num(self, ret, info):
        assert isinstance(ret, int)
        if FREEBSD and ret == -1:
            return
        assert ret >= 0
        if psutil.cpu_count() == 1:
            assert ret == 0
        assert ret in list(range(psutil.cpu_count()))

    def memory_info(self, ret, info):
        assert is_namedtuple(ret)
        for value in ret:
            assert isinstance(value, (int, long))
            assert value >= 0
        if WINDOWS:
            assert ret.peak_wset >= ret.wset
            assert ret.peak_paged_pool >= ret.paged_pool
            assert ret.peak_nonpaged_pool >= ret.nonpaged_pool
            assert ret.peak_pagefile >= ret.pagefile

    def memory_full_info(self, ret, info):
        assert is_namedtuple(ret)
        total = psutil.virtual_memory().total
        for name in ret._fields:
            value = getattr(ret, name)
            assert isinstance(value, (int, long))
            assert value >= 0
            if LINUX or (OSX and name in ('vms', 'data')):
                # On Linux there are processes (e.g. 'goa-daemon') whose
                # VMS is incredibly high for some reason.
                continue
            assert value <= total, name

        if LINUX:
            assert ret.pss >= ret.uss

    def open_files(self, ret, info):
        assert isinstance(ret, list)
        for f in ret:
            assert isinstance(f.fd, int)
            assert isinstance(f.path, str)
            assert f.path.strip() == f.path
            if WINDOWS:
                assert f.fd == -1
            elif LINUX:
                assert isinstance(f.position, int)
                assert isinstance(f.mode, str)
                assert isinstance(f.flags, int)
                assert f.position >= 0
                assert f.mode in ('r', 'w', 'a', 'r+', 'a+')
                assert f.flags > 0
            elif BSD and not f.path:
                # XXX see: https://github.com/giampaolo/psutil/issues/595
                continue
            assert os.path.isabs(f.path), f
            try:
                st = os.stat(f.path)
            except FileNotFoundError:
                pass
            else:
                assert stat.S_ISREG(st.st_mode), f

    def num_fds(self, ret, info):
        assert isinstance(ret, int)
        assert ret >= 0

    def net_connections(self, ret, info):
        with create_sockets():
            assert len(ret) == len(set(ret))
            for conn in ret:
                assert is_namedtuple(conn)
                check_connection_ntuple(conn)

    def cwd(self, ret, info):
        assert isinstance(ret, (str, unicode))
        assert ret.strip() == ret
        if ret:
            assert os.path.isabs(ret), ret
            try:
                st = os.stat(ret)
            except OSError as err:
                if WINDOWS and psutil._psplatform.is_permission_err(err):
                    pass
                # directory has been removed in mean time
                elif err.errno != errno.ENOENT:
                    raise
            else:
                assert stat.S_ISDIR(st.st_mode)

    def memory_percent(self, ret, info):
        assert isinstance(ret, float)
        assert 0 <= ret <= 100, ret

    def is_running(self, ret, info):
        assert isinstance(ret, bool)

    def cpu_affinity(self, ret, info):
        assert isinstance(ret, list)
        assert ret != []
        cpus = list(range(psutil.cpu_count()))
        for n in ret:
            assert isinstance(n, int)
            assert n in cpus

    def terminal(self, ret, info):
        assert isinstance(ret, (str, type(None)))
        if ret is not None:
            assert os.path.isabs(ret), ret
            assert os.path.exists(ret), ret

    def memory_maps(self, ret, info):
        for nt in ret:
            assert isinstance(nt.addr, str)
            assert isinstance(nt.perms, str)
            assert isinstance(nt.path, str)
            for fname in nt._fields:
                value = getattr(nt, fname)
                if fname == 'path':
                    if not value.startswith(("[", "anon_inode:")):
                        assert os.path.isabs(nt.path), nt.path
                        # commented as on Linux we might get
                        # '/foo/bar (deleted)'
                        # assert os.path.exists(nt.path), nt.path
                elif fname == 'addr':
                    assert value, repr(value)
                elif fname == 'perms':
                    if not WINDOWS:
                        assert value, repr(value)
                else:
                    assert isinstance(value, (int, long))
                    assert value >= 0

    def num_handles(self, ret, info):
        assert isinstance(ret, int)
        assert ret >= 0

    def nice(self, ret, info):
        assert isinstance(ret, int)
        if POSIX:
            assert -20 <= ret <= 20, ret
        else:
            priorities = [
                getattr(psutil, x)
                for x in dir(psutil)
                if x.endswith('_PRIORITY_CLASS')
            ]
            assert ret in priorities
            if PY3:
                assert isinstance(ret, enum.IntEnum)
            else:
                assert isinstance(ret, int)

    def num_ctx_switches(self, ret, info):
        assert is_namedtuple(ret)
        for value in ret:
            assert isinstance(value, (int, long))
            assert value >= 0

    def rlimit(self, ret, info):
        assert isinstance(ret, tuple)
        assert len(ret) == 2
        assert ret[0] >= -1
        assert ret[1] >= -1

    def environ(self, ret, info):
        assert isinstance(ret, dict)
        for k, v in ret.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


class TestPidsRange(PsutilTestCase):
    """Given pid_exists() return value for a range of PIDs which may or
    may not exist, make sure that psutil.Process() and psutil.pids()
    agree with pid_exists(). This guarantees that the 3 APIs are all
    consistent with each other. See:
    https://github.com/giampaolo/psutil/issues/2359

    XXX - Note about Windows: it turns out there are some "hidden" PIDs
    which are not returned by psutil.pids() and are also not revealed
    by taskmgr.exe and ProcessHacker, still they can be instantiated by
    psutil.Process() and queried. One of such PIDs is "conhost.exe".
    Running as_dict() for it reveals that some Process() APIs
    erroneously raise NoSuchProcess, so we know we have problem there.
    Let's ignore this for now, since it's quite a corner case (who even
    imagined hidden PIDs existed on Windows?).
    """

    def setUp(self):
        psutil._set_debug(False)

    def tearDown(self):
        psutil._set_debug(True)

    def test_it(self):
        def is_linux_tid(pid):
            try:
                f = open("/proc/%s/status" % pid, "rb")
            except FileNotFoundError:
                return False
            else:
                with f:
                    for line in f:
                        if line.startswith(b"Tgid:"):
                            tgid = int(line.split()[1])
                            # If tgid and pid are different then we're
                            # dealing with a process TID.
                            return tgid != pid
                    raise ValueError("'Tgid' line not found")

        def check(pid):
            # In case of failure retry up to 3 times in order to avoid
            # race conditions, especially when running in a CI
            # environment where PIDs may appear and disappear at any
            # time.
            x = 3
            while True:
                exists = psutil.pid_exists(pid)
                try:
                    if exists:
                        psutil.Process(pid)
                        if not WINDOWS:  # see docstring
                            assert pid in psutil.pids()
                    else:
                        # On OpenBSD thread IDs can be instantiated,
                        # and oneshot() succeeds, but other APIs fail
                        # with EINVAL.
                        if not OPENBSD:
                            with pytest.raises(psutil.NoSuchProcess):
                                psutil.Process(pid)
                        if not WINDOWS:  # see docstring
                            assert pid not in psutil.pids()
                except (psutil.Error, AssertionError):
                    x -= 1
                    if x == 0:
                        raise
                else:
                    return

        for pid in range(1, 3000):
            if LINUX and is_linux_tid(pid):
                # On Linux a TID (thread ID) can be passed to the
                # Process class and is querable like a PID (process
                # ID). Skip it.
                continue
            with self.subTest(pid=pid):
                check(pid)
