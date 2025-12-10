# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Sun OS Solaris platform implementation."""

import errno
import functools
import os
import socket
import subprocess
import sys
from collections import namedtuple
from socket import AF_INET

from . import _common
from . import _psposix
from . import _psutil_sunos as cext
from ._common import AF_INET6
from ._common import ENCODING
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import debug
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import sockfam_to_enum
from ._common import socktype_to_enum
from ._common import usage_percent

__extra__all__ = ["CONN_IDLE", "CONN_BOUND", "PROCFS_PATH"]


# =====================================================================
# --- globals
# =====================================================================


PAGE_SIZE = cext.getpagesize()
AF_LINK = cext.AF_LINK
IS_64_BIT = sys.maxsize > 2**32

CONN_IDLE = "IDLE"
CONN_BOUND = "BOUND"

PROC_STATUSES = {
    cext.SSLEEP: _common.STATUS_SLEEPING,
    cext.SRUN: _common.STATUS_RUNNING,
    cext.SZOMB: _common.STATUS_ZOMBIE,
    cext.SSTOP: _common.STATUS_STOPPED,
    cext.SIDL: _common.STATUS_IDLE,
    cext.SONPROC: _common.STATUS_RUNNING,  # same as run
    cext.SWAIT: _common.STATUS_WAITING,
}

TCP_STATUSES = {
    cext.TCPS_ESTABLISHED: _common.CONN_ESTABLISHED,
    cext.TCPS_SYN_SENT: _common.CONN_SYN_SENT,
    cext.TCPS_SYN_RCVD: _common.CONN_SYN_RECV,
    cext.TCPS_FIN_WAIT_1: _common.CONN_FIN_WAIT1,
    cext.TCPS_FIN_WAIT_2: _common.CONN_FIN_WAIT2,
    cext.TCPS_TIME_WAIT: _common.CONN_TIME_WAIT,
    cext.TCPS_CLOSED: _common.CONN_CLOSE,
    cext.TCPS_CLOSE_WAIT: _common.CONN_CLOSE_WAIT,
    cext.TCPS_LAST_ACK: _common.CONN_LAST_ACK,
    cext.TCPS_LISTEN: _common.CONN_LISTEN,
    cext.TCPS_CLOSING: _common.CONN_CLOSING,
    cext.PSUTIL_CONN_NONE: _common.CONN_NONE,
    cext.TCPS_IDLE: CONN_IDLE,  # sunos specific
    cext.TCPS_BOUND: CONN_BOUND,  # sunos specific
}

proc_info_map = dict(
    ppid=0,
    rss=1,
    vms=2,
    create_time=3,
    nice=4,
    num_threads=5,
    status=6,
    ttynr=7,
    uid=8,
    euid=9,
    gid=10,
    egid=11,
)


# =====================================================================
# --- named tuples
# =====================================================================


# psutil.cpu_times()
scputimes = namedtuple('scputimes', ['user', 'system', 'idle', 'iowait'])
# psutil.cpu_times(percpu=True)
pcputimes = namedtuple(
    'pcputimes', ['user', 'system', 'children_user', 'children_system']
)
# psutil.virtual_memory()
svmem = namedtuple('svmem', ['total', 'available', 'percent', 'used', 'free'])
# psutil.Process.memory_info()
pmem = namedtuple('pmem', ['rss', 'vms'])
pfullmem = pmem
# psutil.Process.memory_maps(grouped=True)
pmmap_grouped = namedtuple(
    'pmmap_grouped', ['path', 'rss', 'anonymous', 'locked']
)
# psutil.Process.memory_maps(grouped=False)
pmmap_ext = namedtuple(
    'pmmap_ext', 'addr perms ' + ' '.join(pmmap_grouped._fields)
)


# =====================================================================
# --- memory
# =====================================================================


def virtual_memory():
    """Report virtual memory metrics."""
    # we could have done this with kstat, but IMHO this is good enough
    total = os.sysconf('SC_PHYS_PAGES') * PAGE_SIZE
    # note: there's no difference on Solaris
    free = avail = os.sysconf('SC_AVPHYS_PAGES') * PAGE_SIZE
    used = total - free
    percent = usage_percent(used, total, round_=1)
    return svmem(total, avail, percent, used, free)


def swap_memory():
    """Report swap memory metrics."""
    sin, sout = cext.swap_mem()
    # XXX
    # we are supposed to get total/free by doing so:
    # http://cvs.opensolaris.org/source/xref/onnv/onnv-gate/
    #     usr/src/cmd/swap/swap.c
    # ...nevertheless I can't manage to obtain the same numbers as 'swap'
    # cmdline utility, so let's parse its output (sigh!)
    p = subprocess.Popen(
        [
            '/usr/bin/env',
            f"PATH=/usr/sbin:/sbin:{os.environ['PATH']}",
            'swap',
            '-l',
        ],
        stdout=subprocess.PIPE,
    )
    stdout, _ = p.communicate()
    stdout = stdout.decode(sys.stdout.encoding)
    if p.returncode != 0:
        msg = f"'swap -l' failed (retcode={p.returncode})"
        raise RuntimeError(msg)

    lines = stdout.strip().split('\n')[1:]
    if not lines:
        msg = 'no swap device(s) configured'
        raise RuntimeError(msg)
    total = free = 0
    for line in lines:
        line = line.split()
        t, f = line[3:5]
        total += int(int(t) * 512)
        free += int(int(f) * 512)
    used = total - free
    percent = usage_percent(used, total, round_=1)
    return _common.sswap(
        total, used, free, percent, sin * PAGE_SIZE, sout * PAGE_SIZE
    )


# =====================================================================
# --- CPU
# =====================================================================


def cpu_times():
    """Return system-wide CPU times as a named tuple."""
    ret = cext.per_cpu_times()
    return scputimes(*[sum(x) for x in zip(*ret)])


def per_cpu_times():
    """Return system per-CPU times as a list of named tuples."""
    ret = cext.per_cpu_times()
    return [scputimes(*x) for x in ret]


def cpu_count_logical():
    """Return the number of logical CPUs in the system."""
    try:
        return os.sysconf("SC_NPROCESSORS_ONLN")
    except ValueError:
        # mimic os.cpu_count() behavior
        return None


def cpu_count_cores():
    """Return the number of CPU cores in the system."""
    return cext.cpu_count_cores()


def cpu_stats():
    """Return various CPU stats as a named tuple."""
    ctx_switches, interrupts, syscalls, _traps = cext.cpu_stats()
    soft_interrupts = 0
    return _common.scpustats(
        ctx_switches, interrupts, soft_interrupts, syscalls
    )


# =====================================================================
# --- disks
# =====================================================================


disk_io_counters = cext.disk_io_counters
disk_usage = _psposix.disk_usage


def disk_partitions(all=False):
    """Return system disk partitions."""
    # TODO - the filtering logic should be better checked so that
    # it tries to reflect 'df' as much as possible
    retlist = []
    partitions = cext.disk_partitions()
    for partition in partitions:
        device, mountpoint, fstype, opts = partition
        if device == 'none':
            device = ''
        if not all:
            # Differently from, say, Linux, we don't have a list of
            # common fs types so the best we can do, AFAIK, is to
            # filter by filesystem having a total size > 0.
            try:
                if not disk_usage(mountpoint).total:
                    continue
            except OSError as err:
                # https://github.com/giampaolo/psutil/issues/1674
                debug(f"skipping {mountpoint!r}: {err}")
                continue
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts)
        retlist.append(ntuple)
    return retlist


# =====================================================================
# --- network
# =====================================================================


net_io_counters = cext.net_io_counters
net_if_addrs = cext.net_if_addrs


def net_connections(kind, _pid=-1):
    """Return socket connections.  If pid == -1 return system-wide
    connections (as opposed to connections opened by one process only).
    Only INET sockets are returned (UNIX are not).
    """
    families, types = _common.conn_tmap[kind]
    rawlist = cext.net_connections(_pid)
    ret = set()
    for item in rawlist:
        fd, fam, type_, laddr, raddr, status, pid = item
        if fam not in families:
            continue
        if type_ not in types:
            continue
        # TODO: refactor and use _common.conn_to_ntuple.
        if fam in {AF_INET, AF_INET6}:
            if laddr:
                laddr = _common.addr(*laddr)
            if raddr:
                raddr = _common.addr(*raddr)
        status = TCP_STATUSES[status]
        fam = sockfam_to_enum(fam)
        type_ = socktype_to_enum(type_)
        if _pid == -1:
            nt = _common.sconn(fd, fam, type_, laddr, raddr, status, pid)
        else:
            nt = _common.pconn(fd, fam, type_, laddr, raddr, status)
        ret.add(nt)
    return list(ret)


def net_if_stats():
    """Get NIC stats (isup, duplex, speed, mtu)."""
    ret = cext.net_if_stats()
    for name, items in ret.items():
        isup, duplex, speed, mtu = items
        if hasattr(_common, 'NicDuplex'):
            duplex = _common.NicDuplex(duplex)
        ret[name] = _common.snicstats(isup, duplex, speed, mtu, '')
    return ret


# =====================================================================
# --- other system functions
# =====================================================================


def boot_time():
    """The system boot time expressed in seconds since the epoch."""
    return cext.boot_time()


def users():
    """Return currently connected users as a list of namedtuples."""
    retlist = []
    rawlist = cext.users()
    localhost = (':0.0', ':0')
    for item in rawlist:
        user, tty, hostname, tstamp, user_process, pid = item
        # note: the underlying C function includes entries about
        # system boot, run level and others.  We might want
        # to use them in the future.
        if not user_process:
            continue
        if hostname in localhost:
            hostname = 'localhost'
        nt = _common.suser(user, tty, hostname, tstamp, pid)
        retlist.append(nt)
    return retlist


# =====================================================================
# --- processes
# =====================================================================


def pids():
    """Returns a list of PIDs currently running on the system."""
    path = get_procfs_path().encode(ENCODING)
    return [int(x) for x in os.listdir(path) if x.isdigit()]


def pid_exists(pid):
    """Check for the existence of a unix pid."""
    return _psposix.pid_exists(pid)


def wrap_exceptions(fun):
    """Call callable into a try/except clause and translate ENOENT,
    EACCES and EPERM in NoSuchProcess or AccessDenied exceptions.
    """

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        pid, ppid, name = self.pid, self._ppid, self._name
        try:
            return fun(self, *args, **kwargs)
        except (FileNotFoundError, ProcessLookupError) as err:
            # ENOENT (no such file or directory) gets raised on open().
            # ESRCH (no such process) can get raised on read() if
            # process is gone in meantime.
            if not pid_exists(pid):
                raise NoSuchProcess(pid, name) from err
            raise ZombieProcess(pid, name, ppid) from err
        except PermissionError as err:
            raise AccessDenied(pid, name) from err
        except OSError as err:
            if pid == 0:
                if 0 in pids():
                    raise AccessDenied(pid, name) from err
                raise
            raise

    return wrapper


class Process:
    """Wrapper class around underlying C implementation."""

    __slots__ = ["_cache", "_name", "_ppid", "_procfs_path", "pid"]

    def __init__(self, pid):
        self.pid = pid
        self._name = None
        self._ppid = None
        self._procfs_path = get_procfs_path()

    def _assert_alive(self):
        """Raise NSP if the process disappeared on us."""
        # For those C function who do not raise NSP, possibly returning
        # incorrect or incomplete result.
        os.stat(f"{self._procfs_path}/{self.pid}")

    def oneshot_enter(self):
        self._proc_name_and_args.cache_activate(self)
        self._proc_basic_info.cache_activate(self)
        self._proc_cred.cache_activate(self)

    def oneshot_exit(self):
        self._proc_name_and_args.cache_deactivate(self)
        self._proc_basic_info.cache_deactivate(self)
        self._proc_cred.cache_deactivate(self)

    @wrap_exceptions
    @memoize_when_activated
    def _proc_name_and_args(self):
        return cext.proc_name_and_args(self.pid, self._procfs_path)

    @wrap_exceptions
    @memoize_when_activated
    def _proc_basic_info(self):
        if self.pid == 0 and not os.path.exists(
            f"{self._procfs_path}/{self.pid}/psinfo"
        ):
            raise AccessDenied(self.pid)
        ret = cext.proc_basic_info(self.pid, self._procfs_path)
        assert len(ret) == len(proc_info_map)
        return ret

    @wrap_exceptions
    @memoize_when_activated
    def _proc_cred(self):
        return cext.proc_cred(self.pid, self._procfs_path)

    @wrap_exceptions
    def name(self):
        # note: max len == 15
        return self._proc_name_and_args()[0]

    @wrap_exceptions
    def exe(self):
        try:
            return os.readlink(f"{self._procfs_path}/{self.pid}/path/a.out")
        except OSError:
            pass  # continue and guess the exe name from the cmdline
        # Will be guessed later from cmdline but we want to explicitly
        # invoke cmdline here in order to get an AccessDenied
        # exception if the user has not enough privileges.
        self.cmdline()
        return ""

    @wrap_exceptions
    def cmdline(self):
        return self._proc_name_and_args()[1]

    @wrap_exceptions
    def environ(self):
        return cext.proc_environ(self.pid, self._procfs_path)

    @wrap_exceptions
    def create_time(self):
        return self._proc_basic_info()[proc_info_map['create_time']]

    @wrap_exceptions
    def num_threads(self):
        return self._proc_basic_info()[proc_info_map['num_threads']]

    @wrap_exceptions
    def nice_get(self):
        # Note #1: getpriority(3) doesn't work for realtime processes.
        # Psinfo is what ps uses, see:
        # https://github.com/giampaolo/psutil/issues/1194
        return self._proc_basic_info()[proc_info_map['nice']]

    @wrap_exceptions
    def nice_set(self, value):
        if self.pid in {2, 3}:
            # Special case PIDs: internally setpriority(3) return ESRCH
            # (no such process), no matter what.
            # The process actually exists though, as it has a name,
            # creation time, etc.
            raise AccessDenied(self.pid, self._name)
        return cext.proc_priority_set(self.pid, value)

    @wrap_exceptions
    def ppid(self):
        self._ppid = self._proc_basic_info()[proc_info_map['ppid']]
        return self._ppid

    @wrap_exceptions
    def uids(self):
        try:
            real, effective, saved, _, _, _ = self._proc_cred()
        except AccessDenied:
            real = self._proc_basic_info()[proc_info_map['uid']]
            effective = self._proc_basic_info()[proc_info_map['euid']]
            saved = None
        return _common.puids(real, effective, saved)

    @wrap_exceptions
    def gids(self):
        try:
            _, _, _, real, effective, saved = self._proc_cred()
        except AccessDenied:
            real = self._proc_basic_info()[proc_info_map['gid']]
            effective = self._proc_basic_info()[proc_info_map['egid']]
            saved = None
        return _common.puids(real, effective, saved)

    @wrap_exceptions
    def cpu_times(self):
        try:
            times = cext.proc_cpu_times(self.pid, self._procfs_path)
        except OSError as err:
            if err.errno == errno.EOVERFLOW and not IS_64_BIT:
                # We may get here if we attempt to query a 64bit process
                # with a 32bit python.
                # Error originates from read() and also tools like "cat"
                # fail in the same way (!).
                # Since there simply is no way to determine CPU times we
                # return 0.0 as a fallback. See:
                # https://github.com/giampaolo/psutil/issues/857
                times = (0.0, 0.0, 0.0, 0.0)
            else:
                raise
        return _common.pcputimes(*times)

    @wrap_exceptions
    def cpu_num(self):
        return cext.proc_cpu_num(self.pid, self._procfs_path)

    @wrap_exceptions
    def terminal(self):
        procfs_path = self._procfs_path
        hit_enoent = False
        tty = wrap_exceptions(self._proc_basic_info()[proc_info_map['ttynr']])
        if tty != cext.PRNODEV:
            for x in (0, 1, 2, 255):
                try:
                    return os.readlink(f"{procfs_path}/{self.pid}/path/{x}")
                except FileNotFoundError:
                    hit_enoent = True
                    continue
        if hit_enoent:
            self._assert_alive()

    @wrap_exceptions
    def cwd(self):
        # /proc/PID/path/cwd may not be resolved by readlink() even if
        # it exists (ls shows it). If that's the case and the process
        # is still alive return None (we can return None also on BSD).
        # Reference: https://groups.google.com/g/comp.unix.solaris/c/tcqvhTNFCAs
        procfs_path = self._procfs_path
        try:
            return os.readlink(f"{procfs_path}/{self.pid}/path/cwd")
        except FileNotFoundError:
            os.stat(f"{procfs_path}/{self.pid}")  # raise NSP or AD
            return ""

    @wrap_exceptions
    def memory_info(self):
        ret = self._proc_basic_info()
        rss = ret[proc_info_map['rss']] * 1024
        vms = ret[proc_info_map['vms']] * 1024
        return pmem(rss, vms)

    memory_full_info = memory_info

    @wrap_exceptions
    def status(self):
        code = self._proc_basic_info()[proc_info_map['status']]
        # XXX is '?' legit? (we're not supposed to return it anyway)
        return PROC_STATUSES.get(code, '?')

    @wrap_exceptions
    def threads(self):
        procfs_path = self._procfs_path
        ret = []
        tids = os.listdir(f"{procfs_path}/{self.pid}/lwp")
        hit_enoent = False
        for tid in tids:
            tid = int(tid)
            try:
                utime, stime = cext.query_process_thread(
                    self.pid, tid, procfs_path
                )
            except OSError as err:
                if err.errno == errno.EOVERFLOW and not IS_64_BIT:
                    # We may get here if we attempt to query a 64bit process
                    # with a 32bit python.
                    # Error originates from read() and also tools like "cat"
                    # fail in the same way (!).
                    # Since there simply is no way to determine CPU times we
                    # return 0.0 as a fallback. See:
                    # https://github.com/giampaolo/psutil/issues/857
                    continue
                # ENOENT == thread gone in meantime
                if err.errno == errno.ENOENT:
                    hit_enoent = True
                    continue
                raise
            else:
                nt = _common.pthread(tid, utime, stime)
                ret.append(nt)
        if hit_enoent:
            self._assert_alive()
        return ret

    @wrap_exceptions
    def open_files(self):
        retlist = []
        hit_enoent = False
        procfs_path = self._procfs_path
        pathdir = f"{procfs_path}/{self.pid}/path"
        for fd in os.listdir(f"{procfs_path}/{self.pid}/fd"):
            path = os.path.join(pathdir, fd)
            if os.path.islink(path):
                try:
                    file = os.readlink(path)
                except FileNotFoundError:
                    hit_enoent = True
                    continue
                else:
                    if isfile_strict(file):
                        retlist.append(_common.popenfile(file, int(fd)))
        if hit_enoent:
            self._assert_alive()
        return retlist

    def _get_unix_sockets(self, pid):
        """Get UNIX sockets used by process by parsing 'pfiles' output."""
        # TODO: rewrite this in C (...but the damn netstat source code
        # does not include this part! Argh!!)
        cmd = ["pfiles", str(pid)]
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        stdout, stderr = (
            x.decode(sys.stdout.encoding) for x in (stdout, stderr)
        )
        if p.returncode != 0:
            if 'permission denied' in stderr.lower():
                raise AccessDenied(self.pid, self._name)
            if 'no such process' in stderr.lower():
                raise NoSuchProcess(self.pid, self._name)
            msg = f"{cmd!r} command error\n{stderr}"
            raise RuntimeError(msg)

        lines = stdout.split('\n')[2:]
        for i, line in enumerate(lines):
            line = line.lstrip()
            if line.startswith('sockname: AF_UNIX'):
                path = line.split(' ', 2)[2]
                type = lines[i - 2].strip()
                if type == 'SOCK_STREAM':
                    type = socket.SOCK_STREAM
                elif type == 'SOCK_DGRAM':
                    type = socket.SOCK_DGRAM
                else:
                    type = -1
                yield (-1, socket.AF_UNIX, type, path, "", _common.CONN_NONE)

    @wrap_exceptions
    def net_connections(self, kind='inet'):
        ret = net_connections(kind, _pid=self.pid)
        # The underlying C implementation retrieves all OS connections
        # and filters them by PID.  At this point we can't tell whether
        # an empty list means there were no connections for process or
        # process is no longer active so we force NSP in case the PID
        # is no longer there.
        if not ret:
            # will raise NSP if process is gone
            os.stat(f"{self._procfs_path}/{self.pid}")

        # UNIX sockets
        if kind in {'all', 'unix'}:
            ret.extend([
                _common.pconn(*conn)
                for conn in self._get_unix_sockets(self.pid)
            ])
        return ret

    nt_mmap_grouped = namedtuple('mmap', 'path rss anon locked')
    nt_mmap_ext = namedtuple('mmap', 'addr perms path rss anon locked')

    @wrap_exceptions
    def memory_maps(self):
        def toaddr(start, end):
            return "{}-{}".format(
                hex(start)[2:].strip('L'), hex(end)[2:].strip('L')
            )

        procfs_path = self._procfs_path
        retlist = []
        try:
            rawlist = cext.proc_memory_maps(self.pid, procfs_path)
        except OSError as err:
            if err.errno == errno.EOVERFLOW and not IS_64_BIT:
                # We may get here if we attempt to query a 64bit process
                # with a 32bit python.
                # Error originates from read() and also tools like "cat"
                # fail in the same way (!).
                # Since there simply is no way to determine CPU times we
                # return 0.0 as a fallback. See:
                # https://github.com/giampaolo/psutil/issues/857
                return []
            else:
                raise
        hit_enoent = False
        for item in rawlist:
            addr, addrsize, perm, name, rss, anon, locked = item
            addr = toaddr(addr, addrsize)
            if not name.startswith('['):
                try:
                    name = os.readlink(f"{procfs_path}/{self.pid}/path/{name}")
                except OSError as err:
                    if err.errno == errno.ENOENT:
                        # sometimes the link may not be resolved by
                        # readlink() even if it exists (ls shows it).
                        # If that's the case we just return the
                        # unresolved link path.
                        # This seems an inconsistency with /proc similar
                        # to: http://goo.gl/55XgO
                        name = f"{procfs_path}/{self.pid}/path/{name}"
                        hit_enoent = True
                    else:
                        raise
            retlist.append((addr, perm, name, rss, anon, locked))
        if hit_enoent:
            self._assert_alive()
        return retlist

    @wrap_exceptions
    def num_fds(self):
        return len(os.listdir(f"{self._procfs_path}/{self.pid}/fd"))

    @wrap_exceptions
    def num_ctx_switches(self):
        return _common.pctxsw(
            *cext.proc_num_ctx_switches(self.pid, self._procfs_path)
        )

    @wrap_exceptions
    def wait(self, timeout=None):
        return _psposix.wait_pid(self.pid, timeout, self._name)
