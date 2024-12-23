#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for detecting function memory leaks (typically the ones
implemented in C). It does so by calling a function many times and
checking whether process memory usage keeps increasing between
calls or over time.
Note that this may produce false positives (especially on Windows
for some reason).
PyPy appears to be completely unstable for this framework, probably
because of how its JIT handles memory, so tests are skipped.
"""

from __future__ import print_function

import functools
import os
import platform

import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import QEMU_USER
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import pytest
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate


cext = psutil._psplatform.cext
thisproc = psutil.Process()
FEW_TIMES = 5


def fewtimes_if_linux():
    """Decorator for those Linux functions which are implemented in pure
    Python, and which we want to run faster.
    """

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            if LINUX:
                before = self.__class__.times
                try:
                    self.__class__.times = FEW_TIMES
                    return fun(self, *args, **kwargs)
                finally:
                    self.__class__.times = before
            else:
                return fun(self, *args, **kwargs)

        return wrapper

    return decorator


# ===================================================================
# Process class
# ===================================================================


class TestProcessObjectLeaks(TestMemoryLeak):
    """Test leaks of Process class methods."""

    proc = thisproc

    def test_coverage(self):
        ns = process_namespace(None)
        ns.test_class_coverage(self, ns.getters + ns.setters)

    @fewtimes_if_linux()
    def test_name(self):
        self.execute(self.proc.name)

    @fewtimes_if_linux()
    def test_cmdline(self):
        self.execute(self.proc.cmdline)

    @fewtimes_if_linux()
    def test_exe(self):
        self.execute(self.proc.exe)

    @fewtimes_if_linux()
    def test_ppid(self):
        self.execute(self.proc.ppid)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_uids(self):
        self.execute(self.proc.uids)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_gids(self):
        self.execute(self.proc.gids)

    @fewtimes_if_linux()
    def test_status(self):
        self.execute(self.proc.status)

    def test_nice(self):
        self.execute(self.proc.nice)

    def test_nice_set(self):
        niceness = thisproc.nice()
        self.execute(lambda: self.proc.nice(niceness))

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    def test_ionice(self):
        self.execute(self.proc.ionice)

    @pytest.mark.skipif(not HAS_IONICE, reason="not supported")
    def test_ionice_set(self):
        if WINDOWS:
            value = thisproc.ionice()
            self.execute(lambda: self.proc.ionice(value))
        else:
            self.execute(lambda: self.proc.ionice(psutil.IOPRIO_CLASS_NONE))
            fun = functools.partial(cext.proc_ioprio_set, os.getpid(), -1, 0)
            self.execute_w_exc(OSError, fun)

    @pytest.mark.skipif(not HAS_PROC_IO_COUNTERS, reason="not supported")
    @fewtimes_if_linux()
    def test_io_counters(self):
        self.execute(self.proc.io_counters)

    @pytest.mark.skipif(POSIX, reason="worthless on POSIX")
    def test_username(self):
        # always open 1 handle on Windows (only once)
        psutil.Process().username()
        self.execute(self.proc.username)

    @fewtimes_if_linux()
    def test_create_time(self):
        self.execute(self.proc.create_time)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_num_threads(self):
        self.execute(self.proc.num_threads)

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_num_handles(self):
        self.execute(self.proc.num_handles)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_num_fds(self):
        self.execute(self.proc.num_fds)

    @fewtimes_if_linux()
    def test_num_ctx_switches(self):
        self.execute(self.proc.num_ctx_switches)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_threads(self):
        self.execute(self.proc.threads)

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(self.proc.cpu_times)

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_PROC_CPU_NUM, reason="not supported")
    def test_cpu_num(self):
        self.execute(self.proc.cpu_num)

    @fewtimes_if_linux()
    def test_memory_info(self):
        self.execute(self.proc.memory_info)

    @fewtimes_if_linux()
    def test_memory_full_info(self):
        self.execute(self.proc.memory_full_info)

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    @fewtimes_if_linux()
    def test_terminal(self):
        self.execute(self.proc.terminal)

    def test_resume(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(self.proc.resume, times=times)

    @fewtimes_if_linux()
    def test_cwd(self):
        self.execute(self.proc.cwd)

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity(self):
        self.execute(self.proc.cpu_affinity)

    @pytest.mark.skipif(not HAS_CPU_AFFINITY, reason="not supported")
    def test_cpu_affinity_set(self):
        affinity = thisproc.cpu_affinity()
        self.execute(lambda: self.proc.cpu_affinity(affinity))
        self.execute_w_exc(ValueError, lambda: self.proc.cpu_affinity([-1]))

    @fewtimes_if_linux()
    def test_open_files(self):
        with open(get_testfn(), 'w'):
            self.execute(self.proc.open_files)

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    @fewtimes_if_linux()
    def test_memory_maps(self):
        self.execute(self.proc.memory_maps)

    @pytest.mark.skipif(not LINUX, reason="LINUX only")
    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit(self):
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE))

    @pytest.mark.skipif(not LINUX, reason="LINUX only")
    @pytest.mark.skipif(not HAS_RLIMIT, reason="not supported")
    def test_rlimit_set(self):
        limit = thisproc.rlimit(psutil.RLIMIT_NOFILE)
        self.execute(lambda: self.proc.rlimit(psutil.RLIMIT_NOFILE, limit))
        self.execute_w_exc((OSError, ValueError), lambda: self.proc.rlimit(-1))

    @fewtimes_if_linux()
    # Windows implementation is based on a single system-wide
    # function (tested later).
    @pytest.mark.skipif(WINDOWS, reason="worthless on WINDOWS")
    def test_net_connections(self):
        # TODO: UNIX sockets are temporarily implemented by parsing
        # 'pfiles' cmd  output; we don't want that part of the code to
        # be executed.
        with create_sockets():
            kind = 'inet' if SUNOS else 'all'
            self.execute(lambda: self.proc.net_connections(kind))

    @pytest.mark.skipif(not HAS_ENVIRON, reason="not supported")
    def test_environ(self):
        self.execute(self.proc.environ)

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_proc_info(self):
        self.execute(lambda: cext.proc_info(os.getpid()))


class TestTerminatedProcessLeaks(TestProcessObjectLeaks):
    """Repeat the tests above looking for leaks occurring when dealing
    with terminated processes raising NoSuchProcess exception.
    The C functions are still invoked but will follow different code
    paths. We'll check those code paths.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.subp = spawn_testproc()
        cls.proc = psutil.Process(cls.subp.pid)
        cls.proc.kill()
        cls.proc.wait()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        terminate(cls.subp)

    def call(self, fun):
        try:
            fun()
        except psutil.NoSuchProcess:
            pass

    if WINDOWS:

        def test_kill(self):
            self.execute(self.proc.kill)

        def test_terminate(self):
            self.execute(self.proc.terminate)

        def test_suspend(self):
            self.execute(self.proc.suspend)

        def test_resume(self):
            self.execute(self.proc.resume)

        def test_wait(self):
            self.execute(self.proc.wait)

        def test_proc_info(self):
            # test dual implementation
            def call():
                try:
                    return cext.proc_info(self.proc.pid)
                except ProcessLookupError:
                    pass

            self.execute(call)


@pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
class TestProcessDualImplementation(TestMemoryLeak):
    def test_cmdline_peb_true(self):
        self.execute(lambda: cext.proc_cmdline(os.getpid(), use_peb=True))

    def test_cmdline_peb_false(self):
        self.execute(lambda: cext.proc_cmdline(os.getpid(), use_peb=False))


# ===================================================================
# system APIs
# ===================================================================


class TestModuleFunctionsLeaks(TestMemoryLeak):
    """Test leaks of psutil module functions."""

    def test_coverage(self):
        ns = system_namespace()
        ns.test_class_coverage(self, ns.all)

    # --- cpu

    @fewtimes_if_linux()
    def test_cpu_count(self):  # logical
        self.execute(lambda: psutil.cpu_count(logical=True))

    @fewtimes_if_linux()
    def test_cpu_count_cores(self):
        self.execute(lambda: psutil.cpu_count(logical=False))

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(psutil.cpu_times)

    @fewtimes_if_linux()
    def test_per_cpu_times(self):
        self.execute(lambda: psutil.cpu_times(percpu=True))

    @fewtimes_if_linux()
    def test_cpu_stats(self):
        self.execute(psutil.cpu_stats)

    @fewtimes_if_linux()
    # TODO: remove this once 1892 is fixed
    @pytest.mark.skipif(
        MACOS and platform.machine() == 'arm64', reason="skipped due to #1892"
    )
    @pytest.mark.skipif(not HAS_CPU_FREQ, reason="not supported")
    def test_cpu_freq(self):
        self.execute(psutil.cpu_freq)

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_getloadavg(self):
        psutil.getloadavg()
        self.execute(psutil.getloadavg)

    # --- mem

    def test_virtual_memory(self):
        self.execute(psutil.virtual_memory)

    # TODO: remove this skip when this gets fixed
    @pytest.mark.skipif(SUNOS, reason="worthless on SUNOS (uses a subprocess)")
    def test_swap_memory(self):
        self.execute(psutil.swap_memory)

    def test_pid_exists(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.pid_exists(os.getpid()), times=times)

    # --- disk

    def test_disk_usage(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.disk_usage('.'), times=times)

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_disk_partitions(self):
        self.execute(psutil.disk_partitions)

    @pytest.mark.skipif(
        LINUX and not os.path.exists('/proc/diskstats'),
        reason="/proc/diskstats not available on this Linux version",
    )
    @fewtimes_if_linux()
    def test_disk_io_counters(self):
        self.execute(lambda: psutil.disk_io_counters(nowrap=False))

    # --- proc

    @fewtimes_if_linux()
    def test_pids(self):
        self.execute(psutil.pids)

    # --- net

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_NET_IO_COUNTERS, reason="not supported")
    def test_net_io_counters(self):
        self.execute(lambda: psutil.net_io_counters(nowrap=False))

    @fewtimes_if_linux()
    @pytest.mark.skipif(MACOS and os.getuid() != 0, reason="need root access")
    def test_net_connections(self):
        # always opens and handle on Windows() (once)
        psutil.net_connections(kind='all')
        with create_sockets():
            self.execute(lambda: psutil.net_connections(kind='all'))

    def test_net_if_addrs(self):
        # Note: verified that on Windows this was a false positive.
        tolerance = 80 * 1024 if WINDOWS else self.tolerance
        self.execute(psutil.net_if_addrs, tolerance=tolerance)

    @pytest.mark.skipif(QEMU_USER, reason="QEMU user not supported")
    def test_net_if_stats(self):
        self.execute(psutil.net_if_stats)

    # --- sensors

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    def test_sensors_battery(self):
        self.execute(psutil.sensors_battery)

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_sensors_temperatures(self):
        self.execute(psutil.sensors_temperatures)

    @fewtimes_if_linux()
    @pytest.mark.skipif(not HAS_SENSORS_FANS, reason="not supported")
    def test_sensors_fans(self):
        self.execute(psutil.sensors_fans)

    # --- others

    @fewtimes_if_linux()
    def test_boot_time(self):
        self.execute(psutil.boot_time)

    def test_users(self):
        self.execute(psutil.users)

    def test_set_debug(self):
        self.execute(lambda: psutil._set_debug(False))

    if WINDOWS:

        # --- win services

        def test_win_service_iter(self):
            self.execute(cext.winservice_enumerate)

        def test_win_service_get(self):
            pass

        def test_win_service_get_config(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_config(name))

        def test_win_service_get_status(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_status(name))

        def test_win_service_get_description(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_descr(name))
