#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""POSIX specific tests."""

import datetime
import errno
import os
import re
import shutil
import subprocess
import time
from unittest import mock

import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil.tests import AARCH64
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import PYTHON_EXE
from psutil.tests import PsutilTestCase
from psutil.tests import pytest
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_subproc
from psutil.tests import terminate

if POSIX:
    import mmap
    import resource


def ps(fmt, pid=None):
    """Wrapper for calling the ps command with a little bit of cross-platform
    support for a narrow range of features.
    """

    cmd = ['ps']

    if LINUX:
        cmd.append('--no-headers')

    if pid is not None:
        cmd.extend(['-p', str(pid)])
    elif SUNOS or AIX:
        cmd.append('-A')
    else:
        cmd.append('ax')

    if SUNOS:
        fmt = fmt.replace("start", "stime")

    cmd.extend(['-o', fmt])

    output = sh(cmd)

    output = output.splitlines() if LINUX else output.splitlines()[1:]

    all_output = []
    for line in output:
        line = line.strip()

        try:
            line = int(line)
        except ValueError:
            pass

        all_output.append(line)

    if pid is None:
        return all_output
    else:
        return all_output[0]


# ps "-o" field names differ wildly between platforms.
# "comm" means "only executable name" but is not available on BSD platforms.
# "args" means "command with all its arguments", and is also not available
# on BSD platforms.
# "command" is like "args" on most platforms, but like "comm" on AIX,
# and not available on SUNOS.
# so for the executable name we can use "comm" on Solaris and split "command"
# on other platforms.
# to get the cmdline (with args) we have to use "args" on AIX and
# Solaris, and can use "command" on all others.


def ps_name(pid):
    field = "command"
    if SUNOS:
        field = "comm"
    command = ps(field, pid).split()
    return command[0]


def ps_args(pid):
    field = "command"
    if AIX or SUNOS:
        field = "args"
    out = ps(field, pid)
    # observed on BSD + Github CI: '/usr/local/bin/python3 -E -O (python3.9)'
    out = re.sub(r"\(python.*?\)$", "", out)
    return out.strip()


def ps_rss(pid):
    field = "rss"
    if AIX:
        field = "rssize"
    return ps(field, pid)


def ps_vsz(pid):
    field = "vsz"
    if AIX:
        field = "vsize"
    return ps(field, pid)


def df(device):
    try:
        out = sh(f"df -k {device}").strip()
    except RuntimeError as err:
        if "device busy" in str(err).lower():
            return pytest.skip("df returned EBUSY")
        raise
    line = out.split('\n')[1]
    fields = line.split()
    sys_total = int(fields[1]) * 1024
    sys_used = int(fields[2]) * 1024
    sys_free = int(fields[3]) * 1024
    sys_percent = float(fields[4].replace('%', ''))
    return (sys_total, sys_used, sys_free, sys_percent)


@pytest.mark.skipif(not POSIX, reason="POSIX only")
class TestProcess(PsutilTestCase):
    """Compare psutil results against 'ps' command line utility (mainly)."""

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_subproc(
            [PYTHON_EXE, "-E", "-O"], stdin=subprocess.PIPE
        ).pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_ppid(self):
        ppid_ps = ps('ppid', self.pid)
        ppid_psutil = psutil.Process(self.pid).ppid()
        assert ppid_ps == ppid_psutil

    def test_uid(self):
        uid_ps = ps('uid', self.pid)
        uid_psutil = psutil.Process(self.pid).uids().real
        assert uid_ps == uid_psutil

    def test_gid(self):
        gid_ps = ps('rgid', self.pid)
        gid_psutil = psutil.Process(self.pid).gids().real
        assert gid_ps == gid_psutil

    def test_username(self):
        username_ps = ps('user', self.pid)
        username_psutil = psutil.Process(self.pid).username()
        assert username_ps == username_psutil

    def test_username_no_resolution(self):
        # Emulate a case where the system can't resolve the uid to
        # a username in which case psutil is supposed to return
        # the stringified uid.
        p = psutil.Process()
        with mock.patch("psutil.pwd.getpwuid", side_effect=KeyError) as fun:
            assert p.username() == str(p.uids().real)
            assert fun.called

    @skip_on_access_denied()
    @retry_on_failure()
    def test_rss_memory(self):
        # give python interpreter some time to properly initialize
        # so that the results are the same
        time.sleep(0.1)
        rss_ps = ps_rss(self.pid)
        rss_psutil = psutil.Process(self.pid).memory_info()[0] / 1024
        assert rss_ps == rss_psutil

    @skip_on_access_denied()
    @retry_on_failure()
    def test_vsz_memory(self):
        # give python interpreter some time to properly initialize
        # so that the results are the same
        time.sleep(0.1)
        vsz_ps = ps_vsz(self.pid)
        vsz_psutil = psutil.Process(self.pid).memory_info()[1] / 1024
        assert vsz_ps == vsz_psutil

    def test_name(self):
        name_ps = ps_name(self.pid)
        # remove path if there is any, from the command
        name_ps = os.path.basename(name_ps).lower()
        name_psutil = psutil.Process(self.pid).name().lower()
        # ...because of how we calculate PYTHON_EXE; on MACOS this may
        # be "pythonX.Y".
        name_ps = re.sub(r"\d.\d", "", name_ps)
        name_psutil = re.sub(r"\d.\d", "", name_psutil)
        # ...may also be "python.X"
        name_ps = re.sub(r"\d", "", name_ps)
        name_psutil = re.sub(r"\d", "", name_psutil)
        assert name_ps == name_psutil

    def test_name_long(self):
        # On UNIX the kernel truncates the name to the first 15
        # characters. In such a case psutil tries to determine the
        # full name from the cmdline.
        name = "long-program-name"
        cmdline = ["long-program-name-extended", "foo", "bar"]
        with mock.patch("psutil._psplatform.Process.name", return_value=name):
            with mock.patch(
                "psutil._psplatform.Process.cmdline", return_value=cmdline
            ):
                p = psutil.Process()
                assert p.name() == "long-program-name-extended"

    def test_name_long_cmdline_ad_exc(self):
        # Same as above but emulates a case where cmdline() raises
        # AccessDenied in which case psutil is supposed to return
        # the truncated name instead of crashing.
        name = "long-program-name"
        with mock.patch("psutil._psplatform.Process.name", return_value=name):
            with mock.patch(
                "psutil._psplatform.Process.cmdline",
                side_effect=psutil.AccessDenied(0, ""),
            ):
                p = psutil.Process()
                assert p.name() == "long-program-name"

    def test_name_long_cmdline_nsp_exc(self):
        # Same as above but emulates a case where cmdline() raises NSP
        # which is supposed to propagate.
        name = "long-program-name"
        with mock.patch("psutil._psplatform.Process.name", return_value=name):
            with mock.patch(
                "psutil._psplatform.Process.cmdline",
                side_effect=psutil.NoSuchProcess(0, ""),
            ):
                p = psutil.Process()
                with pytest.raises(psutil.NoSuchProcess):
                    p.name()

    @pytest.mark.skipif(MACOS or BSD, reason="ps -o start not available")
    def test_create_time(self):
        time_ps = ps('start', self.pid)
        time_psutil = psutil.Process(self.pid).create_time()
        time_psutil_tstamp = datetime.datetime.fromtimestamp(
            time_psutil
        ).strftime("%H:%M:%S")
        # sometimes ps shows the time rounded up instead of down, so we check
        # for both possible values
        round_time_psutil = round(time_psutil)
        round_time_psutil_tstamp = datetime.datetime.fromtimestamp(
            round_time_psutil
        ).strftime("%H:%M:%S")
        assert time_ps in {time_psutil_tstamp, round_time_psutil_tstamp}

    def test_exe(self):
        ps_pathname = ps_name(self.pid)
        psutil_pathname = psutil.Process(self.pid).exe()
        try:
            assert ps_pathname == psutil_pathname
        except AssertionError:
            # certain platforms such as BSD are more accurate returning:
            # "/usr/local/bin/python3.7"
            # ...instead of:
            # "/usr/local/bin/python"
            # We do not want to consider this difference in accuracy
            # an error.
            adjusted_ps_pathname = ps_pathname[: len(ps_pathname)]
            assert ps_pathname == adjusted_ps_pathname

    # On macOS the official python installer exposes a python wrapper that
    # executes a python executable hidden inside an application bundle inside
    # the Python framework.
    # There's a race condition between the ps call & the psutil call below
    # depending on the completion of the execve call so let's retry on failure
    @retry_on_failure()
    def test_cmdline(self):
        ps_cmdline = ps_args(self.pid)
        psutil_cmdline = " ".join(psutil.Process(self.pid).cmdline())
        if AARCH64 and len(ps_cmdline) < len(psutil_cmdline):
            assert psutil_cmdline.startswith(ps_cmdline)
        else:
            assert ps_cmdline == psutil_cmdline

    # On SUNOS "ps" reads niceness /proc/pid/psinfo which returns an
    # incorrect value (20); the real deal is getpriority(2) which
    # returns 0; psutil relies on it, see:
    # https://github.com/giampaolo/psutil/issues/1082
    # AIX has the same issue
    @pytest.mark.skipif(SUNOS, reason="not reliable on SUNOS")
    @pytest.mark.skipif(AIX, reason="not reliable on AIX")
    def test_nice(self):
        ps_nice = ps('nice', self.pid)
        psutil_nice = psutil.Process().nice()
        assert ps_nice == psutil_nice


@pytest.mark.skipif(not POSIX, reason="POSIX only")
class TestSystemAPIs(PsutilTestCase):
    """Test some system APIs."""

    @retry_on_failure()
    def test_pids(self):
        # Note: this test might fail if the OS is starting/killing
        # other processes in the meantime
        pids_ps = sorted(ps("pid"))
        pids_psutil = psutil.pids()

        # on MACOS and OPENBSD ps doesn't show pid 0
        if MACOS or (OPENBSD and 0 not in pids_ps):
            pids_ps.insert(0, 0)

        # There will often be one more process in pids_ps for ps itself
        if len(pids_ps) - len(pids_psutil) > 1:
            difference = [x for x in pids_psutil if x not in pids_ps] + [
                x for x in pids_ps if x not in pids_psutil
            ]
            return pytest.fail("difference: " + str(difference))

    # for some reason ifconfig -a does not report all interfaces
    # returned by psutil
    @pytest.mark.skipif(SUNOS, reason="unreliable on SUNOS")
    @pytest.mark.skipif(not shutil.which("ifconfig"), reason="no ifconfig cmd")
    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_nic_names(self):
        output = sh("ifconfig -a")
        for nic in psutil.net_io_counters(pernic=True):
            for line in output.split():
                if line.startswith(nic):
                    break
            else:
                return pytest.fail(
                    f"couldn't find {nic} nic in 'ifconfig -a'"
                    f" output\n{output}"
                )

    @retry_on_failure()
    def test_users(self):
        out = sh("who -u")
        if not out.strip():
            return pytest.skip("no users on this system")

        susers = []
        for line in out.splitlines():
            user = line.split()[0]
            terminal = line.split()[1]
            if LINUX or MACOS:
                try:
                    pid = int(line.split()[-2])
                except ValueError:
                    pid = int(line.split()[-1])
                susers.append((user, terminal, pid))
            else:
                susers.append((user, terminal))

        if LINUX or MACOS:
            pusers = [(u.name, u.terminal, u.pid) for u in psutil.users()]
        else:
            pusers = [(u.name, u.terminal) for u in psutil.users()]

        assert len(susers) == len(pusers)
        assert sorted(susers) == sorted(pusers)

        for user in psutil.users():
            if user.pid is not None:
                assert user.pid > 0

    @retry_on_failure()
    def test_users_started(self):
        out = sh("who -u")
        if not out.strip():
            return pytest.skip("no users on this system")
        tstamp = None
        # '2023-04-11 09:31' (Linux)
        started = re.findall(r"\d\d\d\d-\d\d-\d\d \d\d:\d\d", out)
        if started:
            tstamp = "%Y-%m-%d %H:%M"
        else:
            # 'Apr 10 22:27' (macOS)
            started = re.findall(r"[A-Z][a-z][a-z] \d\d \d\d:\d\d", out)
            if started:
                tstamp = "%b %d %H:%M"
            else:
                # 'Apr 10'
                started = re.findall(r"[A-Z][a-z][a-z] \d\d", out)
                if started:
                    tstamp = "%b %d"
                else:
                    # 'apr 10' (sunOS)
                    started = re.findall(r"[a-z][a-z][a-z] \d\d", out)
                    if started:
                        tstamp = "%b %d"
                        started = [x.capitalize() for x in started]

        if not tstamp:
            return pytest.skip(f"cannot interpret tstamp in who output\n{out}")

        with self.subTest(psutil=psutil.users(), who=out):
            for idx, u in enumerate(psutil.users()):
                psutil_value = datetime.datetime.fromtimestamp(
                    u.started
                ).strftime(tstamp)
                assert psutil_value == started[idx]

    def test_pid_exists_let_raise(self):
        # According to "man 2 kill" possible error values for kill
        # are (EINVAL, EPERM, ESRCH). Test that any other errno
        # results in an exception.
        with mock.patch(
            "psutil._psposix.os.kill", side_effect=OSError(errno.EBADF, "")
        ) as m:
            with pytest.raises(OSError):
                psutil._psposix.pid_exists(os.getpid())
            assert m.called

    def test_os_waitpid_let_raise(self):
        # os.waitpid() is supposed to catch EINTR and ECHILD only.
        # Test that any other errno results in an exception.
        with mock.patch(
            "psutil._psposix.os.waitpid", side_effect=OSError(errno.EBADF, "")
        ) as m:
            with pytest.raises(OSError):
                psutil._psposix.wait_pid(os.getpid())
            assert m.called

    def test_os_waitpid_eintr(self):
        # os.waitpid() is supposed to "retry" on EINTR.
        with mock.patch(
            "psutil._psposix.os.waitpid", side_effect=OSError(errno.EINTR, "")
        ) as m:
            with pytest.raises(psutil._psposix.TimeoutExpired):
                psutil._psposix.wait_pid(os.getpid(), timeout=0.01)
            assert m.called

    def test_os_waitpid_bad_ret_status(self):
        # Simulate os.waitpid() returning a bad status.
        with mock.patch(
            "psutil._psposix.os.waitpid", return_value=(1, -1)
        ) as m:
            with pytest.raises(ValueError):
                psutil._psposix.wait_pid(os.getpid())
            assert m.called

    # AIX can return '-' in df output instead of numbers, e.g. for /proc
    @pytest.mark.skipif(AIX, reason="unreliable on AIX")
    @retry_on_failure()
    def test_disk_usage(self):
        tolerance = 4 * 1024 * 1024  # 4MB
        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            try:
                sys_total, sys_used, sys_free, sys_percent = df(part.device)
            except RuntimeError as err:
                # see:
                # https://travis-ci.org/giampaolo/psutil/jobs/138338464
                # https://travis-ci.org/giampaolo/psutil/jobs/138343361
                err = str(err).lower()
                if (
                    "no such file or directory" in err
                    or "raw devices not supported" in err
                    or "permission denied" in err
                ):
                    continue
                raise
            else:
                assert abs(usage.total - sys_total) < tolerance
                assert abs(usage.used - sys_used) < tolerance
                assert abs(usage.free - sys_free) < tolerance
                assert abs(usage.percent - sys_percent) <= 1


@pytest.mark.skipif(not POSIX, reason="POSIX only")
class TestMisc(PsutilTestCase):
    def test_getpagesize(self):
        pagesize = psutil._psplatform.cext.getpagesize()
        assert pagesize > 0
        assert pagesize == resource.getpagesize()
        assert pagesize == mmap.PAGESIZE
