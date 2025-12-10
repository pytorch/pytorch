#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for system APIS."""

import datetime
import enum
import errno
import os
import pprint
import shutil
import signal
import socket
import sys
import time
from unittest import mock

import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import broadcast_addr
from psutil.tests import AARCH64
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import pytest
from psutil.tests import retry_on_failure

# ===================================================================
# --- System-related API tests
# ===================================================================


class TestProcessIter(PsutilTestCase):
    def test_pid_presence(self):
        assert os.getpid() in [x.pid for x in psutil.process_iter()]
        sproc = self.spawn_subproc()
        assert sproc.pid in [x.pid for x in psutil.process_iter()]
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        assert sproc.pid not in [x.pid for x in psutil.process_iter()]

    def test_no_duplicates(self):
        ls = list(psutil.process_iter())
        assert sorted(ls, key=lambda x: x.pid) == sorted(
            set(ls), key=lambda x: x.pid
        )

    def test_emulate_nsp(self):
        list(psutil.process_iter())  # populate cache
        for x in range(2):
            with mock.patch(
                'psutil.Process.as_dict',
                side_effect=psutil.NoSuchProcess(os.getpid()),
            ):
                assert not list(psutil.process_iter(attrs=["cpu_times"]))
            psutil.process_iter.cache_clear()  # repeat test without cache

    def test_emulate_access_denied(self):
        list(psutil.process_iter())  # populate cache
        for x in range(2):
            with mock.patch(
                'psutil.Process.as_dict',
                side_effect=psutil.AccessDenied(os.getpid()),
            ):
                with pytest.raises(psutil.AccessDenied):
                    list(psutil.process_iter(attrs=["cpu_times"]))
            psutil.process_iter.cache_clear()  # repeat test without cache

    def test_attrs(self):
        for p in psutil.process_iter(attrs=['pid']):
            assert list(p.info.keys()) == ['pid']
        # yield again
        for p in psutil.process_iter(attrs=['pid']):
            assert list(p.info.keys()) == ['pid']
        with pytest.raises(ValueError):
            list(psutil.process_iter(attrs=['foo']))
        with mock.patch(
            "psutil._psplatform.Process.cpu_times",
            side_effect=psutil.AccessDenied(0, ""),
        ) as m:
            for p in psutil.process_iter(attrs=["pid", "cpu_times"]):
                assert p.info['cpu_times'] is None
                assert p.info['pid'] >= 0
            assert m.called
        with mock.patch(
            "psutil._psplatform.Process.cpu_times",
            side_effect=psutil.AccessDenied(0, ""),
        ) as m:
            flag = object()
            for p in psutil.process_iter(
                attrs=["pid", "cpu_times"], ad_value=flag
            ):
                assert p.info['cpu_times'] is flag
                assert p.info['pid'] >= 0
            assert m.called

    def test_cache_clear(self):
        list(psutil.process_iter())  # populate cache
        assert psutil._pmap
        psutil.process_iter.cache_clear()
        assert not psutil._pmap


class TestProcessAPIs(PsutilTestCase):
    @pytest.mark.skipif(
        PYPY and WINDOWS,
        reason="spawn_subproc() unreliable on PYPY + WINDOWS",
    )
    def test_wait_procs(self):
        def callback(p):
            pids.append(p.pid)

        pids = []
        sproc1 = self.spawn_subproc()
        sproc2 = self.spawn_subproc()
        sproc3 = self.spawn_subproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        with pytest.raises(ValueError):
            psutil.wait_procs(procs, timeout=-1)
        with pytest.raises(TypeError):
            psutil.wait_procs(procs, callback=1)
        t = time.time()
        gone, alive = psutil.wait_procs(procs, timeout=0.01, callback=callback)

        assert time.time() - t < 0.5
        assert not gone
        assert len(alive) == 3
        assert not pids
        for p in alive:
            assert not hasattr(p, 'returncode')

        @retry_on_failure(30)
        def test_1(procs, callback):
            gone, alive = psutil.wait_procs(
                procs, timeout=0.03, callback=callback
            )
            assert len(gone) == 1
            assert len(alive) == 2
            return gone, alive

        sproc3.terminate()
        gone, alive = test_1(procs, callback)
        assert sproc3.pid in [x.pid for x in gone]
        if POSIX:
            assert gone.pop().returncode == -signal.SIGTERM
        else:
            assert gone.pop().returncode == 1
        assert pids == [sproc3.pid]
        for p in alive:
            assert not hasattr(p, 'returncode')

        @retry_on_failure(30)
        def test_2(procs, callback):
            gone, alive = psutil.wait_procs(
                procs, timeout=0.03, callback=callback
            )
            assert len(gone) == 3
            assert len(alive) == 0
            return gone, alive

        sproc1.terminate()
        sproc2.terminate()
        gone, alive = test_2(procs, callback)
        assert set(pids) == {sproc1.pid, sproc2.pid, sproc3.pid}
        for p in gone:
            assert hasattr(p, 'returncode')

    @pytest.mark.skipif(
        PYPY and WINDOWS,
        reason="spawn_subproc() unreliable on PYPY + WINDOWS",
    )
    def test_wait_procs_no_timeout(self):
        sproc1 = self.spawn_subproc()
        sproc2 = self.spawn_subproc()
        sproc3 = self.spawn_subproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        for p in procs:
            p.terminate()
        psutil.wait_procs(procs)

    def test_pid_exists(self):
        sproc = self.spawn_subproc()
        assert psutil.pid_exists(sproc.pid)
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        assert not psutil.pid_exists(sproc.pid)
        assert not psutil.pid_exists(-1)
        assert psutil.pid_exists(0) == (0 in psutil.pids())

    def test_pid_exists_2(self):
        pids = psutil.pids()
        for pid in pids:
            try:
                assert psutil.pid_exists(pid)
            except AssertionError:
                # in case the process disappeared in meantime fail only
                # if it is no longer in psutil.pids()
                time.sleep(0.1)
                assert pid not in psutil.pids()
        pids = range(max(pids) + 15000, max(pids) + 16000)
        for pid in pids:
            assert not psutil.pid_exists(pid)


class TestMiscAPIs(PsutilTestCase):
    def test_boot_time(self):
        bt = psutil.boot_time()
        assert isinstance(bt, float)
        assert bt > 0
        assert bt < time.time()

    @pytest.mark.skipif(
        CI_TESTING and not psutil.users(), reason="unreliable on CI"
    )
    def test_users(self):
        users = psutil.users()
        assert users
        for user in users:
            with self.subTest(user=user):
                assert user.name
                assert isinstance(user.name, str)
                assert isinstance(user.terminal, (str, type(None)))
                if user.host is not None:
                    assert isinstance(user.host, (str, type(None)))
                user.terminal  # noqa: B018
                user.host  # noqa: B018
                assert user.started > 0.0
                datetime.datetime.fromtimestamp(user.started)
                if WINDOWS or OPENBSD:
                    assert user.pid is None
                else:
                    psutil.Process(user.pid)

    def test_os_constants(self):
        names = [
            "POSIX",
            "WINDOWS",
            "LINUX",
            "MACOS",
            "FREEBSD",
            "OPENBSD",
            "NETBSD",
            "BSD",
            "SUNOS",
        ]
        for name in names:
            assert isinstance(getattr(psutil, name), bool), name

        if os.name == 'posix':
            assert psutil.POSIX
            assert not psutil.WINDOWS
            names.remove("POSIX")
            if "linux" in sys.platform.lower():
                assert psutil.LINUX
                names.remove("LINUX")
            elif "bsd" in sys.platform.lower():
                assert psutil.BSD
                assert [psutil.FREEBSD, psutil.OPENBSD, psutil.NETBSD].count(
                    True
                ) == 1
                names.remove("BSD")
                names.remove("FREEBSD")
                names.remove("OPENBSD")
                names.remove("NETBSD")
            elif (
                "sunos" in sys.platform.lower()
                or "solaris" in sys.platform.lower()
            ):
                assert psutil.SUNOS
                names.remove("SUNOS")
            elif "darwin" in sys.platform.lower():
                assert psutil.MACOS
                names.remove("MACOS")
        else:
            assert psutil.WINDOWS
            assert not psutil.POSIX
            names.remove("WINDOWS")

        # assert all other constants are set to False
        for name in names:
            assert not getattr(psutil, name), name


class TestMemoryAPIs(PsutilTestCase):
    def test_virtual_memory(self):
        mem = psutil.virtual_memory()
        assert mem.total > 0, mem
        assert mem.available > 0, mem
        assert 0 <= mem.percent <= 100, mem
        assert mem.used > 0, mem
        assert mem.free >= 0, mem
        for name in mem._fields:
            value = getattr(mem, name)
            if name != 'percent':
                assert isinstance(value, int)
            if name != 'total':
                if not value >= 0:
                    return pytest.fail(f"{name!r} < 0 ({value})")
                if value > mem.total:
                    return pytest.fail(
                        f"{name!r} > total (total={mem.total}, {name}={value})"
                    )

    def test_swap_memory(self):
        mem = psutil.swap_memory()
        assert mem._fields == (
            'total',
            'used',
            'free',
            'percent',
            'sin',
            'sout',
        )

        assert mem.total >= 0, mem
        assert mem.used >= 0, mem
        if mem.total > 0:
            # likely a system with no swap partition
            assert mem.free > 0, mem
        else:
            assert mem.free == 0, mem
        assert 0 <= mem.percent <= 100, mem
        assert mem.sin >= 0, mem
        assert mem.sout >= 0, mem


class TestCpuAPIs(PsutilTestCase):
    def test_cpu_count_logical(self):
        logical = psutil.cpu_count()
        assert logical is not None
        assert logical == len(psutil.cpu_times(percpu=True))
        assert logical >= 1

        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fd:
                cpuinfo_data = fd.read()
            if "physical id" not in cpuinfo_data:
                return pytest.skip("cpuinfo doesn't include physical id")

    def test_cpu_count_cores(self):
        logical = psutil.cpu_count()
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            return pytest.skip("cpu_count_cores() is None")
        if WINDOWS and sys.getwindowsversion()[:2] <= (6, 1):  # <= Vista
            assert cores is None
        else:
            assert cores >= 1
            assert logical >= cores

    def test_cpu_count_none(self):
        # https://github.com/giampaolo/psutil/issues/1085
        for val in (-1, 0, None):
            with mock.patch(
                'psutil._psplatform.cpu_count_logical', return_value=val
            ) as m:
                assert psutil.cpu_count() is None
                assert m.called
            with mock.patch(
                'psutil._psplatform.cpu_count_cores', return_value=val
            ) as m:
                assert psutil.cpu_count(logical=False) is None
                assert m.called

    def test_cpu_times(self):
        # Check type, value >= 0, str().
        total = 0
        times = psutil.cpu_times()
        sum(times)
        for cp_time in times:
            assert isinstance(cp_time, float)
            assert cp_time >= 0.0
            total += cp_time
        assert round(abs(total - sum(times)), 6) == 0
        str(times)
        # CPU times are always supposed to increase over time
        # or at least remain the same and that's because time
        # cannot go backwards.
        # Surprisingly sometimes this might not be the case (at
        # least on Windows and Linux), see:
        # https://github.com/giampaolo/psutil/issues/392
        # https://github.com/giampaolo/psutil/issues/645
        # if not WINDOWS:
        #     last = psutil.cpu_times()
        #     for x in range(100):
        #         new = psutil.cpu_times()
        #         for field in new._fields:
        #             new_t = getattr(new, field)
        #             last_t = getattr(last, field)
        #             assert new_t >= last_t
        #         last = new

    def test_cpu_times_time_increases(self):
        # Make sure time increases between calls.
        t1 = sum(psutil.cpu_times())
        stop_at = time.time() + GLOBAL_TIMEOUT
        while time.time() < stop_at:
            t2 = sum(psutil.cpu_times())
            if t2 > t1:
                return None
        return pytest.fail("time remained the same")

    def test_per_cpu_times(self):
        # Check type, value >= 0, str().
        for times in psutil.cpu_times(percpu=True):
            total = 0
            sum(times)
            for cp_time in times:
                assert isinstance(cp_time, float)
                assert cp_time >= 0.0
                total += cp_time
            assert round(abs(total - sum(times)), 6) == 0
            str(times)
        assert len(psutil.cpu_times(percpu=True)[0]) == len(
            psutil.cpu_times(percpu=False)
        )

        # Note: in theory CPU times are always supposed to increase over
        # time or remain the same but never go backwards. In practice
        # sometimes this is not the case.
        # This issue seemd to be afflict Windows:
        # https://github.com/giampaolo/psutil/issues/392
        # ...but it turns out also Linux (rarely) behaves the same.
        # last = psutil.cpu_times(percpu=True)
        # for x in range(100):
        #     new = psutil.cpu_times(percpu=True)
        #     for index in range(len(new)):
        #         newcpu = new[index]
        #         lastcpu = last[index]
        #         for field in newcpu._fields:
        #             new_t = getattr(newcpu, field)
        #             last_t = getattr(lastcpu, field)
        #             assert new_t >= last_t
        #     last = new

    def test_per_cpu_times_2(self):
        # Simulate some work load then make sure time have increased
        # between calls.
        tot1 = psutil.cpu_times(percpu=True)
        giveup_at = time.time() + GLOBAL_TIMEOUT
        while True:
            if time.time() >= giveup_at:
                return pytest.fail("timeout")
            tot2 = psutil.cpu_times(percpu=True)
            for t1, t2 in zip(tot1, tot2):
                t1, t2 = psutil._cpu_busy_time(t1), psutil._cpu_busy_time(t2)
                difference = t2 - t1
                if difference >= 0.05:
                    return None

    @pytest.mark.skipif(
        (CI_TESTING and OPENBSD) or MACOS, reason="unreliable on OPENBSD + CI"
    )
    @retry_on_failure(30)
    def test_cpu_times_comparison(self):
        # Make sure the sum of all per cpu times is almost equal to
        # base "one cpu" times. On OpenBSD the sum of per-CPUs is
        # higher for some reason.
        base = psutil.cpu_times()
        per_cpu = psutil.cpu_times(percpu=True)
        summed_values = base._make([sum(num) for num in zip(*per_cpu)])
        for field in base._fields:
            with self.subTest(field=field, base=base, per_cpu=per_cpu):
                assert (
                    abs(getattr(base, field) - getattr(summed_values, field))
                    < 2
                )

    def _test_cpu_percent(self, percent, last_ret, new_ret):
        try:
            assert isinstance(percent, float)
            assert percent >= 0.0
            assert percent <= 100.0 * psutil.cpu_count()
        except AssertionError as err:
            raise AssertionError(
                "\n{}\nlast={}\nnew={}".format(
                    err, pprint.pformat(last_ret), pprint.pformat(new_ret)
                )
            )

    def test_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_percent(interval=None)
            self._test_cpu_percent(new, last, new)
            last = new
        with pytest.raises(ValueError):
            psutil.cpu_percent(interval=-1)

    def test_per_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001, percpu=True)
        assert len(last) == psutil.cpu_count()
        for _ in range(100):
            new = psutil.cpu_percent(interval=None, percpu=True)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            last = new
        with pytest.raises(ValueError):
            psutil.cpu_percent(interval=-1, percpu=True)

    def test_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            self._test_cpu_percent(sum(new), last, new)
            last = new
        with pytest.raises(ValueError):
            psutil.cpu_times_percent(interval=-1)

    def test_per_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001, percpu=True)
        assert len(last) == psutil.cpu_count()
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None, percpu=True)
            for cpu in new:
                for percent in cpu:
                    self._test_cpu_percent(percent, last, new)
                self._test_cpu_percent(sum(cpu), last, new)
            last = new

    def test_per_cpu_times_percent_negative(self):
        # see: https://github.com/giampaolo/psutil/issues/645
        psutil.cpu_times_percent(percpu=True)
        zero_times = [
            x._make([0 for x in range(len(x._fields))])
            for x in psutil.cpu_times(percpu=True)
        ]
        with mock.patch('psutil.cpu_times', return_value=zero_times):
            for cpu in psutil.cpu_times_percent(percpu=True):
                for percent in cpu:
                    self._test_cpu_percent(percent, None, None)

    def test_cpu_stats(self):
        # Tested more extensively in per-platform test modules.
        infos = psutil.cpu_stats()
        assert infos._fields == (
            'ctx_switches',
            'interrupts',
            'soft_interrupts',
            'syscalls',
        )
        for name in infos._fields:
            value = getattr(infos, name)
            assert value >= 0
            # on AIX, ctx_switches is always 0
            if not AIX and name in {'ctx_switches', 'interrupts'}:
                assert value > 0

    # TODO: remove this once 1892 is fixed
    @pytest.mark.skipif(MACOS and AARCH64, reason="skipped due to #1892")
    @pytest.mark.skipif(not HAS_CPU_FREQ, reason="not supported")
    def test_cpu_freq(self):
        def check_ls(ls):
            for nt in ls:
                assert nt._fields == ('current', 'min', 'max')
                for name in nt._fields:
                    value = getattr(nt, name)
                    assert isinstance(value, (int, float))
                    assert value >= 0

        ls = psutil.cpu_freq(percpu=True)
        if (FREEBSD or AARCH64) and not ls:
            return pytest.skip(
                "returns empty list on FreeBSD and Linux aarch64"
            )

        assert ls, ls
        check_ls([psutil.cpu_freq(percpu=False)])

        if LINUX:
            assert len(ls) == psutil.cpu_count()

    @pytest.mark.skipif(not HAS_GETLOADAVG, reason="not supported")
    def test_getloadavg(self):
        loadavg = psutil.getloadavg()
        assert len(loadavg) == 3
        for load in loadavg:
            assert isinstance(load, float)
            assert load >= 0.0


class TestDiskAPIs(PsutilTestCase):
    def test_disk_usage(self):
        usage = psutil.disk_usage(os.getcwd())
        assert usage._fields == ('total', 'used', 'free', 'percent')
        assert usage.total > 0, usage
        assert usage.used > 0, usage
        assert usage.free > 0, usage
        assert usage.total > usage.used, usage
        assert usage.total > usage.free, usage
        assert 0 <= usage.percent <= 100, usage.percent

        shutil_usage = shutil.disk_usage(os.getcwd())
        tolerance = 5 * 1024 * 1024  # 5MB
        assert usage.total == shutil_usage.total
        assert abs(usage.free - shutil_usage.free) < tolerance
        if not MACOS_12PLUS:
            # see https://github.com/giampaolo/psutil/issues/2147
            assert abs(usage.used - shutil_usage.used) < tolerance

        # if path does not exist OSError ENOENT is expected across
        # all platforms
        fname = self.get_testfn()
        with pytest.raises(FileNotFoundError):
            psutil.disk_usage(fname)

    @pytest.mark.skipif(not ASCII_FS, reason="not an ASCII fs")
    def test_disk_usage_unicode(self):
        # See: https://github.com/giampaolo/psutil/issues/416
        with pytest.raises(UnicodeEncodeError):
            psutil.disk_usage(UNICODE_SUFFIX)

    def test_disk_usage_bytes(self):
        psutil.disk_usage(b'.')

    def test_disk_partitions(self):
        def check_ntuple(nt):
            assert isinstance(nt.device, str)
            assert isinstance(nt.mountpoint, str)
            assert isinstance(nt.fstype, str)
            assert isinstance(nt.opts, str)

        # all = False
        ls = psutil.disk_partitions(all=False)
        assert ls
        for disk in ls:
            check_ntuple(disk)
            if WINDOWS and 'cdrom' in disk.opts:
                continue
            if not POSIX:
                assert os.path.exists(disk.device), disk
            else:
                # we cannot make any assumption about this, see:
                # http://goo.gl/p9c43
                disk.device  # noqa: B018
            # on modern systems mount points can also be files
            assert os.path.exists(disk.mountpoint), disk
            assert disk.fstype, disk

        # all = True
        ls = psutil.disk_partitions(all=True)
        assert ls
        for disk in psutil.disk_partitions(all=True):
            check_ntuple(disk)
            if not WINDOWS and disk.mountpoint:
                try:
                    os.stat(disk.mountpoint)
                except OSError as err:
                    if GITHUB_ACTIONS and MACOS and err.errno == errno.EIO:
                        continue
                    # http://mail.python.org/pipermail/python-dev/
                    #     2012-June/120787.html
                    if err.errno not in {errno.EPERM, errno.EACCES}:
                        raise
                else:
                    assert os.path.exists(disk.mountpoint), disk

        # ---

        def find_mount_point(path):
            path = os.path.abspath(path)
            while not os.path.ismount(path):
                path = os.path.dirname(path)
            return path.lower()

        mount = find_mount_point(__file__)
        mounts = [
            x.mountpoint.lower()
            for x in psutil.disk_partitions(all=True)
            if x.mountpoint
        ]
        assert mount in mounts

    @pytest.mark.skipif(
        LINUX and not os.path.exists('/proc/diskstats'),
        reason="/proc/diskstats not available on this linux version",
    )
    @pytest.mark.skipif(
        CI_TESTING and not psutil.disk_io_counters(), reason="unreliable on CI"
    )  # no visible disks
    def test_disk_io_counters(self):
        def check_ntuple(nt):
            assert nt[0] == nt.read_count
            assert nt[1] == nt.write_count
            assert nt[2] == nt.read_bytes
            assert nt[3] == nt.write_bytes
            if not (OPENBSD or NETBSD):
                assert nt[4] == nt.read_time
                assert nt[5] == nt.write_time
                if LINUX:
                    assert nt[6] == nt.read_merged_count
                    assert nt[7] == nt.write_merged_count
                    assert nt[8] == nt.busy_time
                elif FREEBSD:
                    assert nt[6] == nt.busy_time
            for name in nt._fields:
                assert getattr(nt, name) >= 0, nt

        ret = psutil.disk_io_counters(perdisk=False)
        assert ret is not None, "no disks on this system?"
        check_ntuple(ret)
        ret = psutil.disk_io_counters(perdisk=True)
        # make sure there are no duplicates
        assert len(ret) == len(set(ret))
        for key in ret:
            assert key, key
            check_ntuple(ret[key])

    def test_disk_io_counters_no_disks(self):
        # Emulate a case where no disks are installed, see:
        # https://github.com/giampaolo/psutil/issues/1062
        with mock.patch(
            'psutil._psplatform.disk_io_counters', return_value={}
        ) as m:
            assert psutil.disk_io_counters(perdisk=False) is None
            assert psutil.disk_io_counters(perdisk=True) == {}
            assert m.called


class TestNetAPIs(PsutilTestCase):
    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_net_io_counters(self):
        def check_ntuple(nt):
            assert nt[0] == nt.bytes_sent
            assert nt[1] == nt.bytes_recv
            assert nt[2] == nt.packets_sent
            assert nt[3] == nt.packets_recv
            assert nt[4] == nt.errin
            assert nt[5] == nt.errout
            assert nt[6] == nt.dropin
            assert nt[7] == nt.dropout
            assert nt.bytes_sent >= 0, nt
            assert nt.bytes_recv >= 0, nt
            assert nt.packets_sent >= 0, nt
            assert nt.packets_recv >= 0, nt
            assert nt.errin >= 0, nt
            assert nt.errout >= 0, nt
            assert nt.dropin >= 0, nt
            assert nt.dropout >= 0, nt

        ret = psutil.net_io_counters(pernic=False)
        check_ntuple(ret)
        ret = psutil.net_io_counters(pernic=True)
        assert ret != []
        for key in ret:
            assert key
            assert isinstance(key, str)
            check_ntuple(ret[key])

    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_net_io_counters_no_nics(self):
        # Emulate a case where no NICs are installed, see:
        # https://github.com/giampaolo/psutil/issues/1062
        with mock.patch(
            'psutil._psplatform.net_io_counters', return_value={}
        ) as m:
            assert psutil.net_io_counters(pernic=False) is None
            assert psutil.net_io_counters(pernic=True) == {}
            assert m.called

    def test_net_if_addrs(self):
        nics = psutil.net_if_addrs()
        assert nics, nics

        nic_stats = psutil.net_if_stats()

        # Not reliable on all platforms (net_if_addrs() reports more
        # interfaces).
        # assert sorted(nics.keys()) == sorted(
        #     psutil.net_io_counters(pernic=True).keys()
        # )

        families = {socket.AF_INET, socket.AF_INET6, psutil.AF_LINK}
        for nic, addrs in nics.items():
            assert isinstance(nic, str)
            assert len(set(addrs)) == len(addrs)
            for addr in addrs:
                assert isinstance(addr.family, int)
                assert isinstance(addr.address, str)
                assert isinstance(addr.netmask, (str, type(None)))
                assert isinstance(addr.broadcast, (str, type(None)))
                assert addr.family in families
                assert isinstance(addr.family, enum.IntEnum)
                if nic_stats[nic].isup:
                    # Do not test binding to addresses of interfaces
                    # that are down
                    if addr.family == socket.AF_INET:
                        with socket.socket(addr.family) as s:
                            s.bind((addr.address, 0))
                    elif addr.family == socket.AF_INET6:
                        info = socket.getaddrinfo(
                            addr.address,
                            0,
                            socket.AF_INET6,
                            socket.SOCK_STREAM,
                            0,
                            socket.AI_PASSIVE,
                        )[0]
                        af, socktype, proto, _canonname, sa = info
                        with socket.socket(af, socktype, proto) as s:
                            s.bind(sa)
                for ip in (
                    addr.address,
                    addr.netmask,
                    addr.broadcast,
                    addr.ptp,
                ):
                    if ip is not None:
                        # TODO: skip AF_INET6 for now because I get:
                        # AddressValueError: Only hex digits permitted in
                        # u'c6f3%lxcbr0' in u'fe80::c8e0:fff:fe54:c6f3%lxcbr0'
                        if addr.family != socket.AF_INET6:
                            check_net_address(ip, addr.family)
                # broadcast and ptp addresses are mutually exclusive
                if addr.broadcast:
                    assert addr.ptp is None
                elif addr.ptp:
                    assert addr.broadcast is None

                # check broadcast address
                if (
                    addr.broadcast
                    and addr.netmask
                    and addr.family in {socket.AF_INET, socket.AF_INET6}
                ):
                    assert addr.broadcast == broadcast_addr(addr)

        if BSD or MACOS or SUNOS:
            if hasattr(socket, "AF_LINK"):
                assert psutil.AF_LINK == socket.AF_LINK
        elif LINUX:
            assert psutil.AF_LINK == socket.AF_PACKET
        elif WINDOWS:
            assert psutil.AF_LINK == -1

    def test_net_if_addrs_mac_null_bytes(self):
        # Simulate that the underlying C function returns an incomplete
        # MAC address. psutil is supposed to fill it with null bytes.
        # https://github.com/giampaolo/psutil/issues/786
        if POSIX:
            ret = [('em1', psutil.AF_LINK, '06:3d:29', None, None, None)]
        else:
            ret = [('em1', -1, '06-3d-29', None, None, None)]
        with mock.patch(
            'psutil._psplatform.net_if_addrs', return_value=ret
        ) as m:
            addr = psutil.net_if_addrs()['em1'][0]
            assert m.called
            if POSIX:
                assert addr.address == '06:3d:29:00:00:00'
            else:
                assert addr.address == '06-3d-29-00-00-00'

    def test_net_if_stats(self):
        nics = psutil.net_if_stats()
        assert nics, nics
        all_duplexes = (
            psutil.NIC_DUPLEX_FULL,
            psutil.NIC_DUPLEX_HALF,
            psutil.NIC_DUPLEX_UNKNOWN,
        )
        for name, stats in nics.items():
            assert isinstance(name, str)
            isup, duplex, speed, mtu, flags = stats
            assert isinstance(isup, bool)
            assert duplex in all_duplexes
            assert duplex in all_duplexes
            assert speed >= 0
            assert mtu >= 0
            assert isinstance(flags, str)

    @pytest.mark.skipif(
        not (LINUX or BSD or MACOS), reason="LINUX or BSD or MACOS specific"
    )
    def test_net_if_stats_enodev(self):
        # See: https://github.com/giampaolo/psutil/issues/1279
        with mock.patch(
            'psutil._psplatform.cext.net_if_mtu',
            side_effect=OSError(errno.ENODEV, ""),
        ) as m:
            ret = psutil.net_if_stats()
            assert ret == {}
            assert m.called


class TestSensorsAPIs(PsutilTestCase):
    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_sensors_temperatures(self):
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            assert isinstance(name, str)
            for entry in entries:
                assert isinstance(entry.label, str)
                if entry.current is not None:
                    assert entry.current >= 0
                if entry.high is not None:
                    assert entry.high >= 0
                if entry.critical is not None:
                    assert entry.critical >= 0

    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_sensors_temperatures_fahreneit(self):
        d = {'coretemp': [('label', 50.0, 60.0, 70.0)]}
        with mock.patch(
            "psutil._psplatform.sensors_temperatures", return_value=d
        ) as m:
            temps = psutil.sensors_temperatures(fahrenheit=True)['coretemp'][0]
            assert m.called
            assert temps.current == 122.0
            assert temps.high == 140.0
            assert temps.critical == 158.0

    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    @pytest.mark.skipif(not HAS_BATTERY, reason="no battery")
    def test_sensors_battery(self):
        ret = psutil.sensors_battery()
        assert ret.percent >= 0
        assert ret.percent <= 100
        if ret.secsleft not in {
            psutil.POWER_TIME_UNKNOWN,
            psutil.POWER_TIME_UNLIMITED,
        }:
            assert ret.secsleft >= 0
        elif ret.secsleft == psutil.POWER_TIME_UNLIMITED:
            assert ret.power_plugged
        assert isinstance(ret.power_plugged, bool)

    @pytest.mark.skipif(not HAS_SENSORS_FANS, reason="not supported")
    def test_sensors_fans(self):
        fans = psutil.sensors_fans()
        for name, entries in fans.items():
            assert isinstance(name, str)
            for entry in entries:
                assert isinstance(entry.label, str)
                assert isinstance(entry.current, int)
                assert entry.current >= 0
