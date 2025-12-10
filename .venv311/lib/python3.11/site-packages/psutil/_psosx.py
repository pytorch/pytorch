# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""macOS platform implementation."""

import errno
import functools
import os
from collections import namedtuple

from . import _common
from . import _psposix
from . import _psutil_osx as cext
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent

__extra__all__ = []


# =====================================================================
# --- globals
# =====================================================================


PAGESIZE = cext.getpagesize()
AF_LINK = cext.AF_LINK

TCP_STATUSES = {
    cext.TCPS_ESTABLISHED: _common.CONN_ESTABLISHED,
    cext.TCPS_SYN_SENT: _common.CONN_SYN_SENT,
    cext.TCPS_SYN_RECEIVED: _common.CONN_SYN_RECV,
    cext.TCPS_FIN_WAIT_1: _common.CONN_FIN_WAIT1,
    cext.TCPS_FIN_WAIT_2: _common.CONN_FIN_WAIT2,
    cext.TCPS_TIME_WAIT: _common.CONN_TIME_WAIT,
    cext.TCPS_CLOSED: _common.CONN_CLOSE,
    cext.TCPS_CLOSE_WAIT: _common.CONN_CLOSE_WAIT,
    cext.TCPS_LAST_ACK: _common.CONN_LAST_ACK,
    cext.TCPS_LISTEN: _common.CONN_LISTEN,
    cext.TCPS_CLOSING: _common.CONN_CLOSING,
    cext.PSUTIL_CONN_NONE: _common.CONN_NONE,
}

PROC_STATUSES = {
    cext.SIDL: _common.STATUS_IDLE,
    cext.SRUN: _common.STATUS_RUNNING,
    cext.SSLEEP: _common.STATUS_SLEEPING,
    cext.SSTOP: _common.STATUS_STOPPED,
    cext.SZOMB: _common.STATUS_ZOMBIE,
}

kinfo_proc_map = dict(
    ppid=0,
    ruid=1,
    euid=2,
    suid=3,
    rgid=4,
    egid=5,
    sgid=6,
    ttynr=7,
    ctime=8,
    status=9,
    name=10,
)

pidtaskinfo_map = dict(
    cpuutime=0,
    cpustime=1,
    rss=2,
    vms=3,
    pfaults=4,
    pageins=5,
    numthreads=6,
    volctxsw=7,
)


# =====================================================================
# --- named tuples
# =====================================================================


# fmt: off
# psutil.cpu_times()
scputimes = namedtuple('scputimes', ['user', 'nice', 'system', 'idle'])
# psutil.virtual_memory()
svmem = namedtuple(
    'svmem', ['total', 'available', 'percent', 'used', 'free',
              'active', 'inactive', 'wired'])
# psutil.Process.memory_info()
pmem = namedtuple('pmem', ['rss', 'vms', 'pfaults', 'pageins'])
# psutil.Process.memory_full_info()
pfullmem = namedtuple('pfullmem', pmem._fields + ('uss', ))
# fmt: on


# =====================================================================
# --- memory
# =====================================================================


def virtual_memory():
    """System virtual memory as a namedtuple."""
    total, active, inactive, wired, free, speculative = cext.virtual_mem()
    # This is how Zabbix calculate avail and used mem:
    # https://github.com/zabbix/zabbix/blob/master/src/libs/zbxsysinfo/osx/memory.c
    # Also see: https://github.com/giampaolo/psutil/issues/1277
    avail = inactive + free
    used = active + wired
    # This is NOT how Zabbix calculates free mem but it matches "free"
    # cmdline utility.
    free -= speculative
    percent = usage_percent((total - avail), total, round_=1)
    return svmem(total, avail, percent, used, free, active, inactive, wired)


def swap_memory():
    """Swap system memory as a (total, used, free, sin, sout) tuple."""
    total, used, free, sin, sout = cext.swap_mem()
    percent = usage_percent(used, total, round_=1)
    return _common.sswap(total, used, free, percent, sin, sout)


# =====================================================================
# --- CPU
# =====================================================================


def cpu_times():
    """Return system CPU times as a namedtuple."""
    user, nice, system, idle = cext.cpu_times()
    return scputimes(user, nice, system, idle)


def per_cpu_times():
    """Return system CPU times as a named tuple."""
    ret = []
    for cpu_t in cext.per_cpu_times():
        user, nice, system, idle = cpu_t
        item = scputimes(user, nice, system, idle)
        ret.append(item)
    return ret


def cpu_count_logical():
    """Return the number of logical CPUs in the system."""
    return cext.cpu_count_logical()


def cpu_count_cores():
    """Return the number of CPU cores in the system."""
    return cext.cpu_count_cores()


def cpu_stats():
    ctx_switches, interrupts, soft_interrupts, syscalls, _traps = (
        cext.cpu_stats()
    )
    return _common.scpustats(
        ctx_switches, interrupts, soft_interrupts, syscalls
    )


if cext.has_cpu_freq():  # not always available on ARM64

    def cpu_freq():
        """Return CPU frequency.
        On macOS per-cpu frequency is not supported.
        Also, the returned frequency never changes, see:
        https://arstechnica.com/civis/viewtopic.php?f=19&t=465002.
        """
        curr, min_, max_ = cext.cpu_freq()
        return [_common.scpufreq(curr, min_, max_)]


# =====================================================================
# --- disks
# =====================================================================


disk_usage = _psposix.disk_usage
disk_io_counters = cext.disk_io_counters


def disk_partitions(all=False):
    """Return mounted disk partitions as a list of namedtuples."""
    retlist = []
    partitions = cext.disk_partitions()
    for partition in partitions:
        device, mountpoint, fstype, opts = partition
        if device == 'none':
            device = ''
        if not all:
            if not os.path.isabs(device) or not os.path.exists(device):
                continue
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts)
        retlist.append(ntuple)
    return retlist


# =====================================================================
# --- sensors
# =====================================================================


def sensors_battery():
    """Return battery information."""
    try:
        percent, minsleft, power_plugged = cext.sensors_battery()
    except NotImplementedError:
        # no power source - return None according to interface
        return None
    power_plugged = power_plugged == 1
    if power_plugged:
        secsleft = _common.POWER_TIME_UNLIMITED
    elif minsleft == -1:
        secsleft = _common.POWER_TIME_UNKNOWN
    else:
        secsleft = minsleft * 60
    return _common.sbattery(percent, secsleft, power_plugged)


# =====================================================================
# --- network
# =====================================================================


net_io_counters = cext.net_io_counters
net_if_addrs = cext.net_if_addrs


def net_connections(kind='inet'):
    """System-wide network connections."""
    # Note: on macOS this will fail with AccessDenied unless
    # the process is owned by root.
    ret = []
    for pid in pids():
        try:
            cons = Process(pid).net_connections(kind)
        except NoSuchProcess:
            continue
        else:
            if cons:
                for c in cons:
                    c = list(c) + [pid]
                    ret.append(_common.sconn(*c))
    return ret


def net_if_stats():
    """Get NIC stats (isup, duplex, speed, mtu)."""
    names = net_io_counters().keys()
    ret = {}
    for name in names:
        try:
            mtu = cext.net_if_mtu(name)
            flags = cext.net_if_flags(name)
            duplex, speed = cext.net_if_duplex_speed(name)
        except OSError as err:
            # https://github.com/giampaolo/psutil/issues/1279
            if err.errno != errno.ENODEV:
                raise
        else:
            if hasattr(_common, 'NicDuplex'):
                duplex = _common.NicDuplex(duplex)
            output_flags = ','.join(flags)
            isup = 'running' in flags
            ret[name] = _common.snicstats(
                isup, duplex, speed, mtu, output_flags
            )
    return ret


# =====================================================================
# --- other system functions
# =====================================================================


def boot_time():
    """The system boot time expressed in seconds since the epoch."""
    return cext.boot_time()


try:
    INIT_BOOT_TIME = boot_time()
except Exception as err:  # noqa: BLE001
    # Don't want to crash at import time.
    debug(f"ignoring exception on import: {err!r}")
    INIT_BOOT_TIME = 0


def adjust_proc_create_time(ctime):
    """Account for system clock updates."""
    if INIT_BOOT_TIME == 0:
        return ctime

    diff = INIT_BOOT_TIME - boot_time()
    if diff == 0 or abs(diff) < 1:
        return ctime

    debug("system clock was updated; adjusting process create_time()")
    if diff < 0:
        return ctime - diff
    return ctime + diff


def users():
    """Return currently connected users as a list of namedtuples."""
    retlist = []
    rawlist = cext.users()
    for item in rawlist:
        user, tty, hostname, tstamp, pid = item
        if tty == '~':
            continue  # reboot or shutdown
        if not tstamp:
            continue
        nt = _common.suser(user, tty or None, hostname or None, tstamp, pid)
        retlist.append(nt)
    return retlist


# =====================================================================
# --- processes
# =====================================================================


def pids():
    ls = cext.pids()
    if 0 not in ls:
        # On certain macOS versions pids() C doesn't return PID 0 but
        # "ps" does and the process is querable via sysctl():
        # https://travis-ci.org/giampaolo/psutil/jobs/309619941
        try:
            Process(0).create_time()
            ls.insert(0, 0)
        except NoSuchProcess:
            pass
        except AccessDenied:
            ls.insert(0, 0)
    return ls


pid_exists = _psposix.pid_exists


def wrap_exceptions(fun):
    """Decorator which translates bare OSError exceptions into
    NoSuchProcess and AccessDenied.
    """

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        pid, ppid, name = self.pid, self._ppid, self._name
        try:
            return fun(self, *args, **kwargs)
        except ProcessLookupError as err:
            if cext.proc_is_zombie(pid):
                raise ZombieProcess(pid, name, ppid) from err
            raise NoSuchProcess(pid, name) from err
        except PermissionError as err:
            raise AccessDenied(pid, name) from err
        except cext.ZombieProcessError as err:
            raise ZombieProcess(pid, name, ppid) from err

    return wrapper


class Process:
    """Wrapper class around underlying C implementation."""

    __slots__ = ["_cache", "_name", "_ppid", "pid"]

    def __init__(self, pid):
        self.pid = pid
        self._name = None
        self._ppid = None

    @wrap_exceptions
    @memoize_when_activated
    def _get_kinfo_proc(self):
        # Note: should work with all PIDs without permission issues.
        ret = cext.proc_kinfo_oneshot(self.pid)
        assert len(ret) == len(kinfo_proc_map)
        return ret

    @wrap_exceptions
    @memoize_when_activated
    def _get_pidtaskinfo(self):
        # Note: should work for PIDs owned by user only.
        ret = cext.proc_pidtaskinfo_oneshot(self.pid)
        assert len(ret) == len(pidtaskinfo_map)
        return ret

    def oneshot_enter(self):
        self._get_kinfo_proc.cache_activate(self)
        self._get_pidtaskinfo.cache_activate(self)

    def oneshot_exit(self):
        self._get_kinfo_proc.cache_deactivate(self)
        self._get_pidtaskinfo.cache_deactivate(self)

    @wrap_exceptions
    def name(self):
        name = self._get_kinfo_proc()[kinfo_proc_map['name']]
        return name if name is not None else cext.proc_name(self.pid)

    @wrap_exceptions
    def exe(self):
        return cext.proc_exe(self.pid)

    @wrap_exceptions
    def cmdline(self):
        return cext.proc_cmdline(self.pid)

    @wrap_exceptions
    def environ(self):
        return parse_environ_block(cext.proc_environ(self.pid))

    @wrap_exceptions
    def ppid(self):
        self._ppid = self._get_kinfo_proc()[kinfo_proc_map['ppid']]
        return self._ppid

    @wrap_exceptions
    def cwd(self):
        return cext.proc_cwd(self.pid)

    @wrap_exceptions
    def uids(self):
        rawtuple = self._get_kinfo_proc()
        return _common.puids(
            rawtuple[kinfo_proc_map['ruid']],
            rawtuple[kinfo_proc_map['euid']],
            rawtuple[kinfo_proc_map['suid']],
        )

    @wrap_exceptions
    def gids(self):
        rawtuple = self._get_kinfo_proc()
        return _common.puids(
            rawtuple[kinfo_proc_map['rgid']],
            rawtuple[kinfo_proc_map['egid']],
            rawtuple[kinfo_proc_map['sgid']],
        )

    @wrap_exceptions
    def terminal(self):
        tty_nr = self._get_kinfo_proc()[kinfo_proc_map['ttynr']]
        tmap = _psposix.get_terminal_map()
        try:
            return tmap[tty_nr]
        except KeyError:
            return None

    @wrap_exceptions
    def memory_info(self):
        rawtuple = self._get_pidtaskinfo()
        return pmem(
            rawtuple[pidtaskinfo_map['rss']],
            rawtuple[pidtaskinfo_map['vms']],
            rawtuple[pidtaskinfo_map['pfaults']],
            rawtuple[pidtaskinfo_map['pageins']],
        )

    @wrap_exceptions
    def memory_full_info(self):
        basic_mem = self.memory_info()
        uss = cext.proc_memory_uss(self.pid)
        return pfullmem(*basic_mem + (uss,))

    @wrap_exceptions
    def cpu_times(self):
        rawtuple = self._get_pidtaskinfo()
        return _common.pcputimes(
            rawtuple[pidtaskinfo_map['cpuutime']],
            rawtuple[pidtaskinfo_map['cpustime']],
            # children user / system times are not retrievable (set to 0)
            0.0,
            0.0,
        )

    @wrap_exceptions
    def create_time(self, monotonic=False):
        ctime = self._get_kinfo_proc()[kinfo_proc_map['ctime']]
        if not monotonic:
            ctime = adjust_proc_create_time(ctime)
        return ctime

    @wrap_exceptions
    def num_ctx_switches(self):
        # Unvoluntary value seems not to be available;
        # getrusage() numbers seems to confirm this theory.
        # We set it to 0.
        vol = self._get_pidtaskinfo()[pidtaskinfo_map['volctxsw']]
        return _common.pctxsw(vol, 0)

    @wrap_exceptions
    def num_threads(self):
        return self._get_pidtaskinfo()[pidtaskinfo_map['numthreads']]

    @wrap_exceptions
    def open_files(self):
        if self.pid == 0:
            return []
        files = []
        rawlist = cext.proc_open_files(self.pid)
        for path, fd in rawlist:
            if isfile_strict(path):
                ntuple = _common.popenfile(path, fd)
                files.append(ntuple)
        return files

    @wrap_exceptions
    def net_connections(self, kind='inet'):
        families, types = conn_tmap[kind]
        rawlist = cext.proc_net_connections(self.pid, families, types)
        ret = []
        for item in rawlist:
            fd, fam, type, laddr, raddr, status = item
            nt = conn_to_ntuple(
                fd, fam, type, laddr, raddr, status, TCP_STATUSES
            )
            ret.append(nt)
        return ret

    @wrap_exceptions
    def num_fds(self):
        if self.pid == 0:
            return 0
        return cext.proc_num_fds(self.pid)

    @wrap_exceptions
    def wait(self, timeout=None):
        return _psposix.wait_pid(self.pid, timeout, self._name)

    @wrap_exceptions
    def nice_get(self):
        return cext.proc_priority_get(self.pid)

    @wrap_exceptions
    def nice_set(self, value):
        return cext.proc_priority_set(self.pid, value)

    @wrap_exceptions
    def status(self):
        code = self._get_kinfo_proc()[kinfo_proc_map['status']]
        # XXX is '?' legit? (we're not supposed to return it anyway)
        return PROC_STATUSES.get(code, '?')

    @wrap_exceptions
    def threads(self):
        rawlist = cext.proc_threads(self.pid)
        retlist = []
        for thread_id, utime, stime in rawlist:
            ntuple = _common.pthread(thread_id, utime, stime)
            retlist.append(ntuple)
        return retlist
