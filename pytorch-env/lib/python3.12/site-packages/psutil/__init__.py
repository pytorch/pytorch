# -*- coding: utf-8 -*-

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""psutil is a cross-platform library for retrieving information on
running processes and system utilization (CPU, memory, disks, network,
sensors) in Python. Supported platforms:

 - Linux
 - Windows
 - macOS
 - FreeBSD
 - OpenBSD
 - NetBSD
 - Sun Solaris
 - AIX

Works with Python versions 2.7 and 3.6+.
"""

from __future__ import division

import collections
import contextlib
import datetime
import functools
import os
import signal
import subprocess
import sys
import threading
import time


try:
    import pwd
except ImportError:
    pwd = None

from . import _common
from ._common import AIX
from ._common import BSD
from ._common import CONN_CLOSE
from ._common import CONN_CLOSE_WAIT
from ._common import CONN_CLOSING
from ._common import CONN_ESTABLISHED
from ._common import CONN_FIN_WAIT1
from ._common import CONN_FIN_WAIT2
from ._common import CONN_LAST_ACK
from ._common import CONN_LISTEN
from ._common import CONN_NONE
from ._common import CONN_SYN_RECV
from ._common import CONN_SYN_SENT
from ._common import CONN_TIME_WAIT
from ._common import FREEBSD  # NOQA
from ._common import LINUX
from ._common import MACOS
from ._common import NETBSD  # NOQA
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import OPENBSD  # NOQA
from ._common import OSX  # deprecated alias
from ._common import POSIX  # NOQA
from ._common import POWER_TIME_UNKNOWN
from ._common import POWER_TIME_UNLIMITED
from ._common import STATUS_DEAD
from ._common import STATUS_DISK_SLEEP
from ._common import STATUS_IDLE
from ._common import STATUS_LOCKED
from ._common import STATUS_PARKED
from ._common import STATUS_RUNNING
from ._common import STATUS_SLEEPING
from ._common import STATUS_STOPPED
from ._common import STATUS_TRACING_STOP
from ._common import STATUS_WAITING
from ._common import STATUS_WAKING
from ._common import STATUS_ZOMBIE
from ._common import SUNOS
from ._common import WINDOWS
from ._common import AccessDenied
from ._common import Error
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import ZombieProcess
from ._common import debug
from ._common import memoize_when_activated
from ._common import wrap_numbers as _wrap_numbers
from ._compat import PY3 as _PY3
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import SubprocessTimeoutExpired as _SubprocessTimeoutExpired
from ._compat import long


if LINUX:
    # This is public API and it will be retrieved from _pslinux.py
    # via sys.modules.
    PROCFS_PATH = "/proc"

    from . import _pslinux as _psplatform
    from ._pslinux import IOPRIO_CLASS_BE  # NOQA
    from ._pslinux import IOPRIO_CLASS_IDLE  # NOQA
    from ._pslinux import IOPRIO_CLASS_NONE  # NOQA
    from ._pslinux import IOPRIO_CLASS_RT  # NOQA

elif WINDOWS:
    from . import _pswindows as _psplatform
    from ._psutil_windows import ABOVE_NORMAL_PRIORITY_CLASS  # NOQA
    from ._psutil_windows import BELOW_NORMAL_PRIORITY_CLASS  # NOQA
    from ._psutil_windows import HIGH_PRIORITY_CLASS  # NOQA
    from ._psutil_windows import IDLE_PRIORITY_CLASS  # NOQA
    from ._psutil_windows import NORMAL_PRIORITY_CLASS  # NOQA
    from ._psutil_windows import REALTIME_PRIORITY_CLASS  # NOQA
    from ._pswindows import CONN_DELETE_TCB  # NOQA
    from ._pswindows import IOPRIO_HIGH  # NOQA
    from ._pswindows import IOPRIO_LOW  # NOQA
    from ._pswindows import IOPRIO_NORMAL  # NOQA
    from ._pswindows import IOPRIO_VERYLOW  # NOQA

elif MACOS:
    from . import _psosx as _psplatform

elif BSD:
    from . import _psbsd as _psplatform

elif SUNOS:
    from . import _pssunos as _psplatform
    from ._pssunos import CONN_BOUND  # NOQA
    from ._pssunos import CONN_IDLE  # NOQA

    # This is public writable API which is read from _pslinux.py and
    # _pssunos.py via sys.modules.
    PROCFS_PATH = "/proc"

elif AIX:
    from . import _psaix as _psplatform

    # This is public API and it will be retrieved from _pslinux.py
    # via sys.modules.
    PROCFS_PATH = "/proc"

else:  # pragma: no cover
    raise NotImplementedError('platform %s is not supported' % sys.platform)


# fmt: off
__all__ = [
    # exceptions
    "Error", "NoSuchProcess", "ZombieProcess", "AccessDenied",
    "TimeoutExpired",

    # constants
    "version_info", "__version__",

    "STATUS_RUNNING", "STATUS_IDLE", "STATUS_SLEEPING", "STATUS_DISK_SLEEP",
    "STATUS_STOPPED", "STATUS_TRACING_STOP", "STATUS_ZOMBIE", "STATUS_DEAD",
    "STATUS_WAKING", "STATUS_LOCKED", "STATUS_WAITING", "STATUS_LOCKED",
    "STATUS_PARKED",

    "CONN_ESTABLISHED", "CONN_SYN_SENT", "CONN_SYN_RECV", "CONN_FIN_WAIT1",
    "CONN_FIN_WAIT2", "CONN_TIME_WAIT", "CONN_CLOSE", "CONN_CLOSE_WAIT",
    "CONN_LAST_ACK", "CONN_LISTEN", "CONN_CLOSING", "CONN_NONE",
    # "CONN_IDLE", "CONN_BOUND",

    "AF_LINK",

    "NIC_DUPLEX_FULL", "NIC_DUPLEX_HALF", "NIC_DUPLEX_UNKNOWN",

    "POWER_TIME_UNKNOWN", "POWER_TIME_UNLIMITED",

    "BSD", "FREEBSD", "LINUX", "NETBSD", "OPENBSD", "MACOS", "OSX", "POSIX",
    "SUNOS", "WINDOWS", "AIX",

    # "RLIM_INFINITY", "RLIMIT_AS", "RLIMIT_CORE", "RLIMIT_CPU", "RLIMIT_DATA",
    # "RLIMIT_FSIZE", "RLIMIT_LOCKS", "RLIMIT_MEMLOCK", "RLIMIT_NOFILE",
    # "RLIMIT_NPROC", "RLIMIT_RSS", "RLIMIT_STACK", "RLIMIT_MSGQUEUE",
    # "RLIMIT_NICE", "RLIMIT_RTPRIO", "RLIMIT_RTTIME", "RLIMIT_SIGPENDING",

    # classes
    "Process", "Popen",

    # functions
    "pid_exists", "pids", "process_iter", "wait_procs",             # proc
    "virtual_memory", "swap_memory",                                # memory
    "cpu_times", "cpu_percent", "cpu_times_percent", "cpu_count",   # cpu
    "cpu_stats",  # "cpu_freq", "getloadavg"
    "net_io_counters", "net_connections", "net_if_addrs",           # network
    "net_if_stats",
    "disk_io_counters", "disk_partitions", "disk_usage",            # disk
    # "sensors_temperatures", "sensors_battery", "sensors_fans"     # sensors
    "users", "boot_time",                                           # others
]
# fmt: on


__all__.extend(_psplatform.__extra__all__)

# Linux, FreeBSD
if hasattr(_psplatform.Process, "rlimit"):
    # Populate global namespace with RLIM* constants.
    from . import _psutil_posix

    _globals = globals()
    _name = None
    for _name in dir(_psutil_posix):
        if _name.startswith('RLIM') and _name.isupper():
            _globals[_name] = getattr(_psutil_posix, _name)
            __all__.append(_name)
    del _globals, _name

AF_LINK = _psplatform.AF_LINK

__author__ = "Giampaolo Rodola'"
__version__ = "6.1.0"
version_info = tuple([int(num) for num in __version__.split('.')])

_timer = getattr(time, 'monotonic', time.time)
_TOTAL_PHYMEM = None
_LOWEST_PID = None
_SENTINEL = object()

# Sanity check in case the user messed up with psutil installation
# or did something weird with sys.path. In this case we might end
# up importing a python module using a C extension module which
# was compiled for a different version of psutil.
# We want to prevent that by failing sooner rather than later.
# See: https://github.com/giampaolo/psutil/issues/564
if int(__version__.replace('.', '')) != getattr(
    _psplatform.cext, 'version', None
):
    msg = "version conflict: %r C extension " % _psplatform.cext.__file__
    msg += "module was built for another version of psutil"
    if hasattr(_psplatform.cext, 'version'):
        msg += " (%s instead of %s)" % (
            '.'.join([x for x in str(_psplatform.cext.version)]),
            __version__,
        )
    else:
        msg += " (different than %s)" % __version__
    msg += "; you may try to 'pip uninstall psutil', manually remove %s" % (
        getattr(
            _psplatform.cext,
            "__file__",
            "the existing psutil install directory",
        )
    )
    msg += " or clean the virtual env somehow, then reinstall"
    raise ImportError(msg)


# =====================================================================
# --- Utils
# =====================================================================


if hasattr(_psplatform, 'ppid_map'):
    # Faster version (Windows and Linux).
    _ppid_map = _psplatform.ppid_map
else:  # pragma: no cover

    def _ppid_map():
        """Return a {pid: ppid, ...} dict for all running processes in
        one shot. Used to speed up Process.children().
        """
        ret = {}
        for pid in pids():
            try:
                ret[pid] = _psplatform.Process(pid).ppid()
            except (NoSuchProcess, ZombieProcess):
                pass
        return ret


def _pprint_secs(secs):
    """Format seconds in a human readable form."""
    now = time.time()
    secs_ago = int(now - secs)
    fmt = "%H:%M:%S" if secs_ago < 60 * 60 * 24 else "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.fromtimestamp(secs).strftime(fmt)


# =====================================================================
# --- Process class
# =====================================================================


class Process(object):  # noqa: UP004
    """Represents an OS process with the given PID.
    If PID is omitted current process PID (os.getpid()) is used.
    Raise NoSuchProcess if PID does not exist.

    Note that most of the methods of this class do not make sure that
    the PID of the process being queried has been reused. That means
    that you may end up retrieving information for another process.

    The only exceptions for which process identity is pre-emptively
    checked and guaranteed are:

     - parent()
     - children()
     - nice() (set)
     - ionice() (set)
     - rlimit() (set)
     - cpu_affinity (set)
     - suspend()
     - resume()
     - send_signal()
     - terminate()
     - kill()

    To prevent this problem for all other methods you can use
    is_running() before querying the process.
    """

    def __init__(self, pid=None):
        self._init(pid)

    def _init(self, pid, _ignore_nsp=False):
        if pid is None:
            pid = os.getpid()
        else:
            if not _PY3 and not isinstance(pid, (int, long)):
                msg = "pid must be an integer (got %r)" % pid
                raise TypeError(msg)
            if pid < 0:
                msg = "pid must be a positive integer (got %s)" % pid
                raise ValueError(msg)
            try:
                _psplatform.cext.check_pid_range(pid)
            except OverflowError:
                msg = "process PID out of range (got %s)" % pid
                raise NoSuchProcess(pid, msg=msg)

        self._pid = pid
        self._name = None
        self._exe = None
        self._create_time = None
        self._gone = False
        self._pid_reused = False
        self._hash = None
        self._lock = threading.RLock()
        # used for caching on Windows only (on POSIX ppid may change)
        self._ppid = None
        # platform-specific modules define an _psplatform.Process
        # implementation class
        self._proc = _psplatform.Process(pid)
        self._last_sys_cpu_times = None
        self._last_proc_cpu_times = None
        self._exitcode = _SENTINEL
        self._ident = (self.pid, None)
        try:
            self._ident = self._get_ident()
        except AccessDenied:
            # This should happen on Windows only, since we use the fast
            # create time method. AFAIK, on all other platforms we are
            # able to get create time for all PIDs.
            pass
        except ZombieProcess:
            # Zombies can still be queried by this class (although
            # not always) and pids() return them so just go on.
            pass
        except NoSuchProcess:
            if not _ignore_nsp:
                msg = "process PID not found"
                raise NoSuchProcess(pid, msg=msg)
            else:
                self._gone = True

    def _get_ident(self):
        """Return a (pid, uid) tuple which is supposed to identify a
        Process instance univocally over time. The PID alone is not
        enough, as it can be assigned to a new process after this one
        terminates, so we add process creation time to the mix. We need
        this in order to prevent killing the wrong process later on.
        This is also known as PID reuse or PID recycling problem.

        The reliability of this strategy mostly depends on
        create_time() precision, which is 0.01 secs on Linux. The
        assumption is that, after a process terminates, the kernel
        won't reuse the same PID after such a short period of time
        (0.01 secs). Technically this is inherently racy, but
        practically it should be good enough.
        """
        if WINDOWS:
            # Use create_time() fast method in order to speedup
            # `process_iter()`. This means we'll get AccessDenied for
            # most ADMIN processes, but that's fine since it means
            # we'll also get AccessDenied on kill().
            # https://github.com/giampaolo/psutil/issues/2366#issuecomment-2381646555
            self._create_time = self._proc.create_time(fast_only=True)
            return (self.pid, self._create_time)
        else:
            return (self.pid, self.create_time())

    def __str__(self):
        info = collections.OrderedDict()
        info["pid"] = self.pid
        if self._name:
            info['name'] = self._name
        with self.oneshot():
            if self._pid_reused:
                info["status"] = "terminated + PID reused"
            else:
                try:
                    info["name"] = self.name()
                    info["status"] = self.status()
                except ZombieProcess:
                    info["status"] = "zombie"
                except NoSuchProcess:
                    info["status"] = "terminated"
                except AccessDenied:
                    pass

            if self._exitcode not in (_SENTINEL, None):
                info["exitcode"] = self._exitcode
            if self._create_time is not None:
                info['started'] = _pprint_secs(self._create_time)

            return "%s.%s(%s)" % (
                self.__class__.__module__,
                self.__class__.__name__,
                ", ".join(["%s=%r" % (k, v) for k, v in info.items()]),
            )

    __repr__ = __str__

    def __eq__(self, other):
        # Test for equality with another Process object based
        # on PID and creation time.
        if not isinstance(other, Process):
            return NotImplemented
        if OPENBSD or NETBSD:  # pragma: no cover
            # Zombie processes on Open/NetBSD have a creation time of
            # 0.0. This covers the case when a process started normally
            # (so it has a ctime), then it turned into a zombie. It's
            # important to do this because is_running() depends on
            # __eq__.
            pid1, ident1 = self._ident
            pid2, ident2 = other._ident
            if pid1 == pid2:
                if ident1 and not ident2:
                    try:
                        return self.status() == STATUS_ZOMBIE
                    except Error:
                        pass
        return self._ident == other._ident

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._ident)
        return self._hash

    def _raise_if_pid_reused(self):
        """Raises NoSuchProcess in case process PID has been reused."""
        if self._pid_reused or (not self.is_running() and self._pid_reused):
            # We may directly raise NSP in here already if PID is just
            # not running, but I prefer NSP to be raised naturally by
            # the actual Process API call. This way unit tests will tell
            # us if the API is broken (aka don't raise NSP when it
            # should). We also remain consistent with all other "get"
            # APIs which don't use _raise_if_pid_reused().
            msg = "process no longer exists and its PID has been reused"
            raise NoSuchProcess(self.pid, self._name, msg=msg)

    @property
    def pid(self):
        """The process PID."""
        return self._pid

    # --- utility methods

    @contextlib.contextmanager
    def oneshot(self):
        """Utility context manager which considerably speeds up the
        retrieval of multiple process information at the same time.

        Internally different process info (e.g. name, ppid, uids,
        gids, ...) may be fetched by using the same routine, but
        only one information is returned and the others are discarded.
        When using this context manager the internal routine is
        executed once (in the example below on name()) and the
        other info are cached.

        The cache is cleared when exiting the context manager block.
        The advice is to use this every time you retrieve more than
        one information about the process. If you're lucky, you'll
        get a hell of a speedup.

        >>> import psutil
        >>> p = psutil.Process()
        >>> with p.oneshot():
        ...     p.name()  # collect multiple info
        ...     p.cpu_times()  # return cached value
        ...     p.cpu_percent()  # return cached value
        ...     p.create_time()  # return cached value
        ...
        >>>
        """
        with self._lock:
            if hasattr(self, "_cache"):
                # NOOP: this covers the use case where the user enters the
                # context twice:
                #
                # >>> with p.oneshot():
                # ...    with p.oneshot():
                # ...
                #
                # Also, since as_dict() internally uses oneshot()
                # I expect that the code below will be a pretty common
                # "mistake" that the user will make, so let's guard
                # against that:
                #
                # >>> with p.oneshot():
                # ...    p.as_dict()
                # ...
                yield
            else:
                try:
                    # cached in case cpu_percent() is used
                    self.cpu_times.cache_activate(self)
                    # cached in case memory_percent() is used
                    self.memory_info.cache_activate(self)
                    # cached in case parent() is used
                    self.ppid.cache_activate(self)
                    # cached in case username() is used
                    if POSIX:
                        self.uids.cache_activate(self)
                    # specific implementation cache
                    self._proc.oneshot_enter()
                    yield
                finally:
                    self.cpu_times.cache_deactivate(self)
                    self.memory_info.cache_deactivate(self)
                    self.ppid.cache_deactivate(self)
                    if POSIX:
                        self.uids.cache_deactivate(self)
                    self._proc.oneshot_exit()

    def as_dict(self, attrs=None, ad_value=None):
        """Utility method returning process information as a
        hashable dictionary.
        If *attrs* is specified it must be a list of strings
        reflecting available Process class' attribute names
        (e.g. ['cpu_times', 'name']) else all public (read
        only) attributes are assumed.
        *ad_value* is the value which gets assigned in case
        AccessDenied or ZombieProcess exception is raised when
        retrieving that particular process information.
        """
        valid_names = _as_dict_attrnames
        if attrs is not None:
            if not isinstance(attrs, (list, tuple, set, frozenset)):
                msg = "invalid attrs type %s" % type(attrs)
                raise TypeError(msg)
            attrs = set(attrs)
            invalid_names = attrs - valid_names
            if invalid_names:
                msg = "invalid attr name%s %s" % (
                    "s" if len(invalid_names) > 1 else "",
                    ", ".join(map(repr, invalid_names)),
                )
                raise ValueError(msg)

        retdict = {}
        ls = attrs or valid_names
        with self.oneshot():
            for name in ls:
                try:
                    if name == 'pid':
                        ret = self.pid
                    else:
                        meth = getattr(self, name)
                        ret = meth()
                except (AccessDenied, ZombieProcess):
                    ret = ad_value
                except NotImplementedError:
                    # in case of not implemented functionality (may happen
                    # on old or exotic systems) we want to crash only if
                    # the user explicitly asked for that particular attr
                    if attrs:
                        raise
                    continue
                retdict[name] = ret
        return retdict

    def parent(self):
        """Return the parent process as a Process object pre-emptively
        checking whether PID has been reused.
        If no parent is known return None.
        """
        lowest_pid = _LOWEST_PID if _LOWEST_PID is not None else pids()[0]
        if self.pid == lowest_pid:
            return None
        ppid = self.ppid()
        if ppid is not None:
            ctime = self.create_time()
            try:
                parent = Process(ppid)
                if parent.create_time() <= ctime:
                    return parent
                # ...else ppid has been reused by another process
            except NoSuchProcess:
                pass

    def parents(self):
        """Return the parents of this process as a list of Process
        instances. If no parents are known return an empty list.
        """
        parents = []
        proc = self.parent()
        while proc is not None:
            parents.append(proc)
            proc = proc.parent()
        return parents

    def is_running(self):
        """Return whether this process is running.

        It also checks if PID has been reused by another process, in
        which case it will remove the process from `process_iter()`
        internal cache and return False.
        """
        if self._gone or self._pid_reused:
            return False
        try:
            # Checking if PID is alive is not enough as the PID might
            # have been reused by another process. Process identity /
            # uniqueness over time is guaranteed by (PID + creation
            # time) and that is verified in __eq__.
            self._pid_reused = self != Process(self.pid)
            if self._pid_reused:
                _pids_reused.add(self.pid)
                raise NoSuchProcess(self.pid)
            return True
        except ZombieProcess:
            # We should never get here as it's already handled in
            # Process.__init__; here just for extra safety.
            return True
        except NoSuchProcess:
            self._gone = True
            return False

    # --- actual API

    @memoize_when_activated
    def ppid(self):
        """The process parent PID.
        On Windows the return value is cached after first call.
        """
        # On POSIX we don't want to cache the ppid as it may unexpectedly
        # change to 1 (init) in case this process turns into a zombie:
        # https://github.com/giampaolo/psutil/issues/321
        # http://stackoverflow.com/questions/356722/

        # XXX should we check creation time here rather than in
        # Process.parent()?
        self._raise_if_pid_reused()
        if POSIX:
            return self._proc.ppid()
        else:  # pragma: no cover
            self._ppid = self._ppid or self._proc.ppid()
            return self._ppid

    def name(self):
        """The process name. The return value is cached after first call."""
        # Process name is only cached on Windows as on POSIX it may
        # change, see:
        # https://github.com/giampaolo/psutil/issues/692
        if WINDOWS and self._name is not None:
            return self._name
        name = self._proc.name()
        if POSIX and len(name) >= 15:
            # On UNIX the name gets truncated to the first 15 characters.
            # If it matches the first part of the cmdline we return that
            # one instead because it's usually more explicative.
            # Examples are "gnome-keyring-d" vs. "gnome-keyring-daemon".
            try:
                cmdline = self.cmdline()
            except (AccessDenied, ZombieProcess):
                # Just pass and return the truncated name: it's better
                # than nothing. Note: there are actual cases where a
                # zombie process can return a name() but not a
                # cmdline(), see:
                # https://github.com/giampaolo/psutil/issues/2239
                pass
            else:
                if cmdline:
                    extended_name = os.path.basename(cmdline[0])
                    if extended_name.startswith(name):
                        name = extended_name
        self._name = name
        self._proc._name = name
        return name

    def exe(self):
        """The process executable as an absolute path.
        May also be an empty string.
        The return value is cached after first call.
        """

        def guess_it(fallback):
            # try to guess exe from cmdline[0] in absence of a native
            # exe representation
            cmdline = self.cmdline()
            if cmdline and hasattr(os, 'access') and hasattr(os, 'X_OK'):
                exe = cmdline[0]  # the possible exe
                # Attempt to guess only in case of an absolute path.
                # It is not safe otherwise as the process might have
                # changed cwd.
                if (
                    os.path.isabs(exe)
                    and os.path.isfile(exe)
                    and os.access(exe, os.X_OK)
                ):
                    return exe
            if isinstance(fallback, AccessDenied):
                raise fallback
            return fallback

        if self._exe is None:
            try:
                exe = self._proc.exe()
            except AccessDenied as err:
                return guess_it(fallback=err)
            else:
                if not exe:
                    # underlying implementation can legitimately return an
                    # empty string; if that's the case we don't want to
                    # raise AD while guessing from the cmdline
                    try:
                        exe = guess_it(fallback=exe)
                    except AccessDenied:
                        pass
                self._exe = exe
        return self._exe

    def cmdline(self):
        """The command line this process has been called with."""
        return self._proc.cmdline()

    def status(self):
        """The process current status as a STATUS_* constant."""
        try:
            return self._proc.status()
        except ZombieProcess:
            return STATUS_ZOMBIE

    def username(self):
        """The name of the user that owns the process.
        On UNIX this is calculated by using *real* process uid.
        """
        if POSIX:
            if pwd is None:
                # might happen if python was installed from sources
                msg = "requires pwd module shipped with standard python"
                raise ImportError(msg)
            real_uid = self.uids().real
            try:
                return pwd.getpwuid(real_uid).pw_name
            except KeyError:
                # the uid can't be resolved by the system
                return str(real_uid)
        else:
            return self._proc.username()

    def create_time(self):
        """The process creation time as a floating point number
        expressed in seconds since the epoch.
        The return value is cached after first call.
        """
        if self._create_time is None:
            self._create_time = self._proc.create_time()
        return self._create_time

    def cwd(self):
        """Process current working directory as an absolute path."""
        return self._proc.cwd()

    def nice(self, value=None):
        """Get or set process niceness (priority)."""
        if value is None:
            return self._proc.nice_get()
        else:
            self._raise_if_pid_reused()
            self._proc.nice_set(value)

    if POSIX:

        @memoize_when_activated
        def uids(self):
            """Return process UIDs as a (real, effective, saved)
            namedtuple.
            """
            return self._proc.uids()

        def gids(self):
            """Return process GIDs as a (real, effective, saved)
            namedtuple.
            """
            return self._proc.gids()

        def terminal(self):
            """The terminal associated with this process, if any,
            else None.
            """
            return self._proc.terminal()

        def num_fds(self):
            """Return the number of file descriptors opened by this
            process (POSIX only).
            """
            return self._proc.num_fds()

    # Linux, BSD, AIX and Windows only
    if hasattr(_psplatform.Process, "io_counters"):

        def io_counters(self):
            """Return process I/O statistics as a
            (read_count, write_count, read_bytes, write_bytes)
            namedtuple.
            Those are the number of read/write calls performed and the
            amount of bytes read and written by the process.
            """
            return self._proc.io_counters()

    # Linux and Windows
    if hasattr(_psplatform.Process, "ionice_get"):

        def ionice(self, ioclass=None, value=None):
            """Get or set process I/O niceness (priority).

            On Linux *ioclass* is one of the IOPRIO_CLASS_* constants.
            *value* is a number which goes from 0 to 7. The higher the
            value, the lower the I/O priority of the process.

            On Windows only *ioclass* is used and it can be set to 2
            (normal), 1 (low) or 0 (very low).

            Available on Linux and Windows > Vista only.
            """
            if ioclass is None:
                if value is not None:
                    msg = "'ioclass' argument must be specified"
                    raise ValueError(msg)
                return self._proc.ionice_get()
            else:
                self._raise_if_pid_reused()
                return self._proc.ionice_set(ioclass, value)

    # Linux / FreeBSD only
    if hasattr(_psplatform.Process, "rlimit"):

        def rlimit(self, resource, limits=None):
            """Get or set process resource limits as a (soft, hard)
            tuple.

            *resource* is one of the RLIMIT_* constants.
            *limits* is supposed to be a (soft, hard) tuple.

            See "man prlimit" for further info.
            Available on Linux and FreeBSD only.
            """
            if limits is not None:
                self._raise_if_pid_reused()
            return self._proc.rlimit(resource, limits)

    # Windows, Linux and FreeBSD only
    if hasattr(_psplatform.Process, "cpu_affinity_get"):

        def cpu_affinity(self, cpus=None):
            """Get or set process CPU affinity.
            If specified, *cpus* must be a list of CPUs for which you
            want to set the affinity (e.g. [0, 1]).
            If an empty list is passed, all egible CPUs are assumed
            (and set).
            (Windows, Linux and BSD only).
            """
            if cpus is None:
                return sorted(set(self._proc.cpu_affinity_get()))
            else:
                self._raise_if_pid_reused()
                if not cpus:
                    if hasattr(self._proc, "_get_eligible_cpus"):
                        cpus = self._proc._get_eligible_cpus()
                    else:
                        cpus = tuple(range(len(cpu_times(percpu=True))))
                self._proc.cpu_affinity_set(list(set(cpus)))

    # Linux, FreeBSD, SunOS
    if hasattr(_psplatform.Process, "cpu_num"):

        def cpu_num(self):
            """Return what CPU this process is currently running on.
            The returned number should be <= psutil.cpu_count()
            and <= len(psutil.cpu_percent(percpu=True)).
            It may be used in conjunction with
            psutil.cpu_percent(percpu=True) to observe the system
            workload distributed across CPUs.
            """
            return self._proc.cpu_num()

    # All platforms has it, but maybe not in the future.
    if hasattr(_psplatform.Process, "environ"):

        def environ(self):
            """The environment variables of the process as a dict.  Note: this
            might not reflect changes made after the process started.
            """
            return self._proc.environ()

    if WINDOWS:

        def num_handles(self):
            """Return the number of handles opened by this process
            (Windows only).
            """
            return self._proc.num_handles()

    def num_ctx_switches(self):
        """Return the number of voluntary and involuntary context
        switches performed by this process.
        """
        return self._proc.num_ctx_switches()

    def num_threads(self):
        """Return the number of threads used by this process."""
        return self._proc.num_threads()

    if hasattr(_psplatform.Process, "threads"):

        def threads(self):
            """Return threads opened by process as a list of
            (id, user_time, system_time) namedtuples representing
            thread id and thread CPU times (user/system).
            On OpenBSD this method requires root access.
            """
            return self._proc.threads()

    def children(self, recursive=False):
        """Return the children of this process as a list of Process
        instances, pre-emptively checking whether PID has been reused.
        If *recursive* is True return all the parent descendants.

        Example (A == this process):

         A ─┐
            │
            ├─ B (child) ─┐
            │             └─ X (grandchild) ─┐
            │                                └─ Y (great grandchild)
            ├─ C (child)
            └─ D (child)

        >>> import psutil
        >>> p = psutil.Process()
        >>> p.children()
        B, C, D
        >>> p.children(recursive=True)
        B, X, Y, C, D

        Note that in the example above if process X disappears
        process Y won't be listed as the reference to process A
        is lost.
        """
        self._raise_if_pid_reused()
        ppid_map = _ppid_map()
        ret = []
        if not recursive:
            for pid, ppid in ppid_map.items():
                if ppid == self.pid:
                    try:
                        child = Process(pid)
                        # if child happens to be older than its parent
                        # (self) it means child's PID has been reused
                        if self.create_time() <= child.create_time():
                            ret.append(child)
                    except (NoSuchProcess, ZombieProcess):
                        pass
        else:
            # Construct a {pid: [child pids]} dict
            reverse_ppid_map = collections.defaultdict(list)
            for pid, ppid in ppid_map.items():
                reverse_ppid_map[ppid].append(pid)
            # Recursively traverse that dict, starting from self.pid,
            # such that we only call Process() on actual children
            seen = set()
            stack = [self.pid]
            while stack:
                pid = stack.pop()
                if pid in seen:
                    # Since pids can be reused while the ppid_map is
                    # constructed, there may be rare instances where
                    # there's a cycle in the recorded process "tree".
                    continue
                seen.add(pid)
                for child_pid in reverse_ppid_map[pid]:
                    try:
                        child = Process(child_pid)
                        # if child happens to be older than its parent
                        # (self) it means child's PID has been reused
                        intime = self.create_time() <= child.create_time()
                        if intime:
                            ret.append(child)
                            stack.append(child_pid)
                    except (NoSuchProcess, ZombieProcess):
                        pass
        return ret

    def cpu_percent(self, interval=None):
        """Return a float representing the current process CPU
        utilization as a percentage.

        When *interval* is 0.0 or None (default) compares process times
        to system CPU times elapsed since last call, returning
        immediately (non-blocking). That means that the first time
        this is called it will return a meaningful 0.0 value.

        When *interval* is > 0.0 compares process times to system CPU
        times elapsed before and after the interval (blocking).

        In this case is recommended for accuracy that this function
        be called with at least 0.1 seconds between calls.

        A value > 100.0 can be returned in case of processes running
        multiple threads on different CPU cores.

        The returned value is explicitly NOT split evenly between
        all available logical CPUs. This means that a busy loop process
        running on a system with 2 logical CPUs will be reported as
        having 100% CPU utilization instead of 50%.

        Examples:

          >>> import psutil
          >>> p = psutil.Process(os.getpid())
          >>> # blocking
          >>> p.cpu_percent(interval=1)
          2.0
          >>> # non-blocking (percentage since last call)
          >>> p.cpu_percent(interval=None)
          2.9
          >>>
        """
        blocking = interval is not None and interval > 0.0
        if interval is not None and interval < 0:
            msg = "interval is not positive (got %r)" % interval
            raise ValueError(msg)
        num_cpus = cpu_count() or 1

        def timer():
            return _timer() * num_cpus

        if blocking:
            st1 = timer()
            pt1 = self._proc.cpu_times()
            time.sleep(interval)
            st2 = timer()
            pt2 = self._proc.cpu_times()
        else:
            st1 = self._last_sys_cpu_times
            pt1 = self._last_proc_cpu_times
            st2 = timer()
            pt2 = self._proc.cpu_times()
            if st1 is None or pt1 is None:
                self._last_sys_cpu_times = st2
                self._last_proc_cpu_times = pt2
                return 0.0

        delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
        delta_time = st2 - st1
        # reset values for next call in case of interval == None
        self._last_sys_cpu_times = st2
        self._last_proc_cpu_times = pt2

        try:
            # This is the utilization split evenly between all CPUs.
            # E.g. a busy loop process on a 2-CPU-cores system at this
            # point is reported as 50% instead of 100%.
            overall_cpus_percent = (delta_proc / delta_time) * 100
        except ZeroDivisionError:
            # interval was too low
            return 0.0
        else:
            # Note 1:
            # in order to emulate "top" we multiply the value for the num
            # of CPU cores. This way the busy process will be reported as
            # having 100% (or more) usage.
            #
            # Note 2:
            # taskmgr.exe on Windows differs in that it will show 50%
            # instead.
            #
            # Note 3:
            # a percentage > 100 is legitimate as it can result from a
            # process with multiple threads running on different CPU
            # cores (top does the same), see:
            # http://stackoverflow.com/questions/1032357
            # https://github.com/giampaolo/psutil/issues/474
            single_cpu_percent = overall_cpus_percent * num_cpus
            return round(single_cpu_percent, 1)

    @memoize_when_activated
    def cpu_times(self):
        """Return a (user, system, children_user, children_system)
        namedtuple representing the accumulated process time, in
        seconds.
        This is similar to os.times() but per-process.
        On macOS and Windows children_user and children_system are
        always set to 0.
        """
        return self._proc.cpu_times()

    @memoize_when_activated
    def memory_info(self):
        """Return a namedtuple with variable fields depending on the
        platform, representing memory information about the process.

        The "portable" fields available on all platforms are `rss` and `vms`.

        All numbers are expressed in bytes.
        """
        return self._proc.memory_info()

    @_common.deprecated_method(replacement="memory_info")
    def memory_info_ex(self):
        return self.memory_info()

    def memory_full_info(self):
        """This method returns the same information as memory_info(),
        plus, on some platform (Linux, macOS, Windows), also provides
        additional metrics (USS, PSS and swap).
        The additional metrics provide a better representation of actual
        process memory usage.

        Namely USS is the memory which is unique to a process and which
        would be freed if the process was terminated right now.

        It does so by passing through the whole process address.
        As such it usually requires higher user privileges than
        memory_info() and is considerably slower.
        """
        return self._proc.memory_full_info()

    def memory_percent(self, memtype="rss"):
        """Compare process memory to total physical system memory and
        calculate process memory utilization as a percentage.
        *memtype* argument is a string that dictates what type of
        process memory you want to compare against (defaults to "rss").
        The list of available strings can be obtained like this:

        >>> psutil.Process().memory_info()._fields
        ('rss', 'vms', 'shared', 'text', 'lib', 'data', 'dirty', 'uss', 'pss')
        """
        valid_types = list(_psplatform.pfullmem._fields)
        if memtype not in valid_types:
            msg = "invalid memtype %r; valid types are %r" % (
                memtype,
                tuple(valid_types),
            )
            raise ValueError(msg)
        fun = (
            self.memory_info
            if memtype in _psplatform.pmem._fields
            else self.memory_full_info
        )
        metrics = fun()
        value = getattr(metrics, memtype)

        # use cached value if available
        total_phymem = _TOTAL_PHYMEM or virtual_memory().total
        if not total_phymem > 0:
            # we should never get here
            msg = (
                "can't calculate process memory percent because total physical"
                " system memory is not positive (%r)" % (total_phymem)
            )
            raise ValueError(msg)
        return (value / float(total_phymem)) * 100

    if hasattr(_psplatform.Process, "memory_maps"):

        def memory_maps(self, grouped=True):
            """Return process' mapped memory regions as a list of namedtuples
            whose fields are variable depending on the platform.

            If *grouped* is True the mapped regions with the same 'path'
            are grouped together and the different memory fields are summed.

            If *grouped* is False every mapped region is shown as a single
            entity and the namedtuple will also include the mapped region's
            address space ('addr') and permission set ('perms').
            """
            it = self._proc.memory_maps()
            if grouped:
                d = {}
                for tupl in it:
                    path = tupl[2]
                    nums = tupl[3:]
                    try:
                        d[path] = map(lambda x, y: x + y, d[path], nums)
                    except KeyError:
                        d[path] = nums
                nt = _psplatform.pmmap_grouped
                return [nt(path, *d[path]) for path in d]  # NOQA
            else:
                nt = _psplatform.pmmap_ext
                return [nt(*x) for x in it]

    def open_files(self):
        """Return files opened by process as a list of
        (path, fd) namedtuples including the absolute file name
        and file descriptor number.
        """
        return self._proc.open_files()

    def net_connections(self, kind='inet'):
        """Return socket connections opened by process as a list of
        (fd, family, type, laddr, raddr, status) namedtuples.
        The *kind* parameter filters for connections that match the
        following criteria:

        +------------+----------------------------------------------------+
        | Kind Value | Connections using                                  |
        +------------+----------------------------------------------------+
        | inet       | IPv4 and IPv6                                      |
        | inet4      | IPv4                                               |
        | inet6      | IPv6                                               |
        | tcp        | TCP                                                |
        | tcp4       | TCP over IPv4                                      |
        | tcp6       | TCP over IPv6                                      |
        | udp        | UDP                                                |
        | udp4       | UDP over IPv4                                      |
        | udp6       | UDP over IPv6                                      |
        | unix       | UNIX socket (both UDP and TCP protocols)           |
        | all        | the sum of all the possible families and protocols |
        +------------+----------------------------------------------------+
        """
        return self._proc.net_connections(kind)

    @_common.deprecated_method(replacement="net_connections")
    def connections(self, kind="inet"):
        return self.net_connections(kind=kind)

    # --- signals

    if POSIX:

        def _send_signal(self, sig):
            assert not self.pid < 0, self.pid
            self._raise_if_pid_reused()
            if self.pid == 0:
                # see "man 2 kill"
                msg = (
                    "preventing sending signal to process with PID 0 as it "
                    "would affect every process in the process group of the "
                    "calling process (os.getpid()) instead of PID 0"
                )
                raise ValueError(msg)
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                if OPENBSD and pid_exists(self.pid):
                    # We do this because os.kill() lies in case of
                    # zombie processes.
                    raise ZombieProcess(self.pid, self._name, self._ppid)
                else:
                    self._gone = True
                    raise NoSuchProcess(self.pid, self._name)
            except PermissionError:
                raise AccessDenied(self.pid, self._name)

    def send_signal(self, sig):
        """Send a signal *sig* to process pre-emptively checking
        whether PID has been reused (see signal module constants) .
        On Windows only SIGTERM is valid and is treated as an alias
        for kill().
        """
        if POSIX:
            self._send_signal(sig)
        else:  # pragma: no cover
            self._raise_if_pid_reused()
            if sig != signal.SIGTERM and not self.is_running():
                msg = "process no longer exists"
                raise NoSuchProcess(self.pid, self._name, msg=msg)
            self._proc.send_signal(sig)

    def suspend(self):
        """Suspend process execution with SIGSTOP pre-emptively checking
        whether PID has been reused.
        On Windows this has the effect of suspending all process threads.
        """
        if POSIX:
            self._send_signal(signal.SIGSTOP)
        else:  # pragma: no cover
            self._raise_if_pid_reused()
            self._proc.suspend()

    def resume(self):
        """Resume process execution with SIGCONT pre-emptively checking
        whether PID has been reused.
        On Windows this has the effect of resuming all process threads.
        """
        if POSIX:
            self._send_signal(signal.SIGCONT)
        else:  # pragma: no cover
            self._raise_if_pid_reused()
            self._proc.resume()

    def terminate(self):
        """Terminate the process with SIGTERM pre-emptively checking
        whether PID has been reused.
        On Windows this is an alias for kill().
        """
        if POSIX:
            self._send_signal(signal.SIGTERM)
        else:  # pragma: no cover
            self._raise_if_pid_reused()
            self._proc.kill()

    def kill(self):
        """Kill the current process with SIGKILL pre-emptively checking
        whether PID has been reused.
        """
        if POSIX:
            self._send_signal(signal.SIGKILL)
        else:  # pragma: no cover
            self._raise_if_pid_reused()
            self._proc.kill()

    def wait(self, timeout=None):
        """Wait for process to terminate and, if process is a children
        of os.getpid(), also return its exit code, else None.
        On Windows there's no such limitation (exit code is always
        returned).

        If the process is already terminated immediately return None
        instead of raising NoSuchProcess.

        If *timeout* (in seconds) is specified and process is still
        alive raise TimeoutExpired.

        To wait for multiple Process(es) use psutil.wait_procs().
        """
        if timeout is not None and not timeout >= 0:
            msg = "timeout must be a positive integer"
            raise ValueError(msg)
        if self._exitcode is not _SENTINEL:
            return self._exitcode
        self._exitcode = self._proc.wait(timeout)
        return self._exitcode


# The valid attr names which can be processed by Process.as_dict().
# fmt: off
_as_dict_attrnames = set(
    [x for x in dir(Process) if not x.startswith('_') and x not in
     {'send_signal', 'suspend', 'resume', 'terminate', 'kill', 'wait',
      'is_running', 'as_dict', 'parent', 'parents', 'children', 'rlimit',
      'memory_info_ex', 'connections', 'oneshot'}])
# fmt: on


# =====================================================================
# --- Popen class
# =====================================================================


class Popen(Process):
    """Same as subprocess.Popen, but in addition it provides all
    psutil.Process methods in a single class.
    For the following methods which are common to both classes, psutil
    implementation takes precedence:

    * send_signal()
    * terminate()
    * kill()

    This is done in order to avoid killing another process in case its
    PID has been reused, fixing BPO-6973.

      >>> import psutil
      >>> from subprocess import PIPE
      >>> p = psutil.Popen(["python", "-c", "print 'hi'"], stdout=PIPE)
      >>> p.name()
      'python'
      >>> p.uids()
      user(real=1000, effective=1000, saved=1000)
      >>> p.username()
      'giampaolo'
      >>> p.communicate()
      ('hi', None)
      >>> p.terminate()
      >>> p.wait(timeout=2)
      0
      >>>
    """

    def __init__(self, *args, **kwargs):
        # Explicitly avoid to raise NoSuchProcess in case the process
        # spawned by subprocess.Popen terminates too quickly, see:
        # https://github.com/giampaolo/psutil/issues/193
        self.__subproc = subprocess.Popen(*args, **kwargs)
        self._init(self.__subproc.pid, _ignore_nsp=True)

    def __dir__(self):
        return sorted(set(dir(Popen) + dir(subprocess.Popen)))

    def __enter__(self):
        if hasattr(self.__subproc, '__enter__'):
            self.__subproc.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        if hasattr(self.__subproc, '__exit__'):
            return self.__subproc.__exit__(*args, **kwargs)
        else:
            if self.stdout:
                self.stdout.close()
            if self.stderr:
                self.stderr.close()
            try:
                # Flushing a BufferedWriter may raise an error.
                if self.stdin:
                    self.stdin.close()
            finally:
                # Wait for the process to terminate, to avoid zombies.
                self.wait()

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                return object.__getattribute__(self.__subproc, name)
            except AttributeError:
                msg = "%s instance has no attribute '%s'" % (
                    self.__class__.__name__,
                    name,
                )
                raise AttributeError(msg)

    def wait(self, timeout=None):
        if self.__subproc.returncode is not None:
            return self.__subproc.returncode
        ret = super(Popen, self).wait(timeout)  # noqa
        self.__subproc.returncode = ret
        return ret


# =====================================================================
# --- system processes related functions
# =====================================================================


def pids():
    """Return a list of current running PIDs."""
    global _LOWEST_PID
    ret = sorted(_psplatform.pids())
    _LOWEST_PID = ret[0]
    return ret


def pid_exists(pid):
    """Return True if given PID exists in the current process list.
    This is faster than doing "pid in psutil.pids()" and
    should be preferred.
    """
    if pid < 0:
        return False
    elif pid == 0 and POSIX:
        # On POSIX we use os.kill() to determine PID existence.
        # According to "man 2 kill" PID 0 has a special meaning
        # though: it refers to <<every process in the process
        # group of the calling process>> and that is not we want
        # to do here.
        return pid in pids()
    else:
        return _psplatform.pid_exists(pid)


_pmap = {}
_pids_reused = set()


def process_iter(attrs=None, ad_value=None):
    """Return a generator yielding a Process instance for all
    running processes.

    Every new Process instance is only created once and then cached
    into an internal table which is updated every time this is used.
    Cache can optionally be cleared via `process_iter.clear_cache()`.

    The sorting order in which processes are yielded is based on
    their PIDs.

    *attrs* and *ad_value* have the same meaning as in
    Process.as_dict(). If *attrs* is specified as_dict() is called
    and the resulting dict is stored as a 'info' attribute attached
    to returned Process instance.
    If *attrs* is an empty list it will retrieve all process info
    (slow).
    """
    global _pmap

    def add(pid):
        proc = Process(pid)
        pmap[proc.pid] = proc
        return proc

    def remove(pid):
        pmap.pop(pid, None)

    pmap = _pmap.copy()
    a = set(pids())
    b = set(pmap.keys())
    new_pids = a - b
    gone_pids = b - a
    for pid in gone_pids:
        remove(pid)
    while _pids_reused:
        pid = _pids_reused.pop()
        debug("refreshing Process instance for reused PID %s" % pid)
        remove(pid)
    try:
        ls = sorted(list(pmap.items()) + list(dict.fromkeys(new_pids).items()))
        for pid, proc in ls:
            try:
                if proc is None:  # new process
                    proc = add(pid)
                if attrs is not None:
                    proc.info = proc.as_dict(attrs=attrs, ad_value=ad_value)
                yield proc
            except NoSuchProcess:
                remove(pid)
    finally:
        _pmap = pmap


process_iter.cache_clear = lambda: _pmap.clear()  # noqa
process_iter.cache_clear.__doc__ = "Clear process_iter() internal cache."


def wait_procs(procs, timeout=None, callback=None):
    """Convenience function which waits for a list of processes to
    terminate.

    Return a (gone, alive) tuple indicating which processes
    are gone and which ones are still alive.

    The gone ones will have a new *returncode* attribute indicating
    process exit status (may be None).

    *callback* is a function which gets called every time a process
    terminates (a Process instance is passed as callback argument).

    Function will return as soon as all processes terminate or when
    *timeout* occurs.
    Differently from Process.wait() it will not raise TimeoutExpired if
    *timeout* occurs.

    Typical use case is:

     - send SIGTERM to a list of processes
     - give them some time to terminate
     - send SIGKILL to those ones which are still alive

    Example:

    >>> def on_terminate(proc):
    ...     print("process {} terminated".format(proc))
    ...
    >>> for p in procs:
    ...    p.terminate()
    ...
    >>> gone, alive = wait_procs(procs, timeout=3, callback=on_terminate)
    >>> for p in alive:
    ...     p.kill()
    """

    def check_gone(proc, timeout):
        try:
            returncode = proc.wait(timeout=timeout)
        except TimeoutExpired:
            pass
        except _SubprocessTimeoutExpired:
            pass
        else:
            if returncode is not None or not proc.is_running():
                # Set new Process instance attribute.
                proc.returncode = returncode
                gone.add(proc)
                if callback is not None:
                    callback(proc)

    if timeout is not None and not timeout >= 0:
        msg = "timeout must be a positive integer, got %s" % timeout
        raise ValueError(msg)
    gone = set()
    alive = set(procs)
    if callback is not None and not callable(callback):
        msg = "callback %r is not a callable" % callback
        raise TypeError(msg)
    if timeout is not None:
        deadline = _timer() + timeout

    while alive:
        if timeout is not None and timeout <= 0:
            break
        for proc in alive:
            # Make sure that every complete iteration (all processes)
            # will last max 1 sec.
            # We do this because we don't want to wait too long on a
            # single process: in case it terminates too late other
            # processes may disappear in the meantime and their PID
            # reused.
            max_timeout = 1.0 / len(alive)
            if timeout is not None:
                timeout = min((deadline - _timer()), max_timeout)
                if timeout <= 0:
                    break
                check_gone(proc, timeout)
            else:
                check_gone(proc, max_timeout)
        alive = alive - gone  # noqa PLR6104

    if alive:
        # Last attempt over processes survived so far.
        # timeout == 0 won't make this function wait any further.
        for proc in alive:
            check_gone(proc, 0)
        alive = alive - gone  # noqa: PLR6104

    return (list(gone), list(alive))


# =====================================================================
# --- CPU related functions
# =====================================================================


def cpu_count(logical=True):
    """Return the number of logical CPUs in the system (same as
    os.cpu_count() in Python 3.4).

    If *logical* is False return the number of physical cores only
    (e.g. hyper thread CPUs are excluded).

    Return None if undetermined.

    The return value is cached after first call.
    If desired cache can be cleared like this:

    >>> psutil.cpu_count.cache_clear()
    """
    if logical:
        ret = _psplatform.cpu_count_logical()
    else:
        ret = _psplatform.cpu_count_cores()
    if ret is not None and ret < 1:
        ret = None
    return ret


def cpu_times(percpu=False):
    """Return system-wide CPU times as a namedtuple.
    Every CPU time represents the seconds the CPU has spent in the
    given mode. The namedtuple's fields availability varies depending on the
    platform:

     - user
     - system
     - idle
     - nice (UNIX)
     - iowait (Linux)
     - irq (Linux, FreeBSD)
     - softirq (Linux)
     - steal (Linux >= 2.6.11)
     - guest (Linux >= 2.6.24)
     - guest_nice (Linux >= 3.2.0)

    When *percpu* is True return a list of namedtuples for each CPU.
    First element of the list refers to first CPU, second element
    to second CPU and so on.
    The order of the list is consistent across calls.
    """
    if not percpu:
        return _psplatform.cpu_times()
    else:
        return _psplatform.per_cpu_times()


try:
    _last_cpu_times = {threading.current_thread().ident: cpu_times()}
except Exception:  # noqa: BLE001
    # Don't want to crash at import time.
    _last_cpu_times = {}

try:
    _last_per_cpu_times = {
        threading.current_thread().ident: cpu_times(percpu=True)
    }
except Exception:  # noqa: BLE001
    # Don't want to crash at import time.
    _last_per_cpu_times = {}


def _cpu_tot_time(times):
    """Given a cpu_time() ntuple calculates the total CPU time
    (including idle time).
    """
    tot = sum(times)
    if LINUX:
        # On Linux guest times are already accounted in "user" or
        # "nice" times, so we subtract them from total.
        # Htop does the same. References:
        # https://github.com/giampaolo/psutil/pull/940
        # http://unix.stackexchange.com/questions/178045
        # https://github.com/torvalds/linux/blob/
        #     447976ef4fd09b1be88b316d1a81553f1aa7cd07/kernel/sched/
        #     cputime.c#L158
        tot -= getattr(times, "guest", 0)  # Linux 2.6.24+
        tot -= getattr(times, "guest_nice", 0)  # Linux 3.2.0+
    return tot


def _cpu_busy_time(times):
    """Given a cpu_time() ntuple calculates the busy CPU time.
    We do so by subtracting all idle CPU times.
    """
    busy = _cpu_tot_time(times)
    busy -= times.idle
    # Linux: "iowait" is time during which the CPU does not do anything
    # (waits for IO to complete). On Linux IO wait is *not* accounted
    # in "idle" time so we subtract it. Htop does the same.
    # References:
    # https://github.com/torvalds/linux/blob/
    #     447976ef4fd09b1be88b316d1a81553f1aa7cd07/kernel/sched/cputime.c#L244
    busy -= getattr(times, "iowait", 0)
    return busy


def _cpu_times_deltas(t1, t2):
    assert t1._fields == t2._fields, (t1, t2)
    field_deltas = []
    for field in _psplatform.scputimes._fields:
        field_delta = getattr(t2, field) - getattr(t1, field)
        # CPU times are always supposed to increase over time
        # or at least remain the same and that's because time
        # cannot go backwards.
        # Surprisingly sometimes this might not be the case (at
        # least on Windows and Linux), see:
        # https://github.com/giampaolo/psutil/issues/392
        # https://github.com/giampaolo/psutil/issues/645
        # https://github.com/giampaolo/psutil/issues/1210
        # Trim negative deltas to zero to ignore decreasing fields.
        # top does the same. Reference:
        # https://gitlab.com/procps-ng/procps/blob/v3.3.12/top/top.c#L5063
        field_delta = max(0, field_delta)
        field_deltas.append(field_delta)
    return _psplatform.scputimes(*field_deltas)


def cpu_percent(interval=None, percpu=False):
    """Return a float representing the current system-wide CPU
    utilization as a percentage.

    When *interval* is > 0.0 compares system CPU times elapsed before
    and after the interval (blocking).

    When *interval* is 0.0 or None compares system CPU times elapsed
    since last call or module import, returning immediately (non
    blocking). That means the first time this is called it will
    return a meaningless 0.0 value which you should ignore.
    In this case is recommended for accuracy that this function be
    called with at least 0.1 seconds between calls.

    When *percpu* is True returns a list of floats representing the
    utilization as a percentage for each CPU.
    First element of the list refers to first CPU, second element
    to second CPU and so on.
    The order of the list is consistent across calls.

    Examples:

      >>> # blocking, system-wide
      >>> psutil.cpu_percent(interval=1)
      2.0
      >>>
      >>> # blocking, per-cpu
      >>> psutil.cpu_percent(interval=1, percpu=True)
      [2.0, 1.0]
      >>>
      >>> # non-blocking (percentage since last call)
      >>> psutil.cpu_percent(interval=None)
      2.9
      >>>
    """
    tid = threading.current_thread().ident
    blocking = interval is not None and interval > 0.0
    if interval is not None and interval < 0:
        msg = "interval is not positive (got %r)" % interval
        raise ValueError(msg)

    def calculate(t1, t2):
        times_delta = _cpu_times_deltas(t1, t2)
        all_delta = _cpu_tot_time(times_delta)
        busy_delta = _cpu_busy_time(times_delta)

        try:
            busy_perc = (busy_delta / all_delta) * 100
        except ZeroDivisionError:
            return 0.0
        else:
            return round(busy_perc, 1)

    # system-wide usage
    if not percpu:
        if blocking:
            t1 = cpu_times()
            time.sleep(interval)
        else:
            t1 = _last_cpu_times.get(tid) or cpu_times()
        _last_cpu_times[tid] = cpu_times()
        return calculate(t1, _last_cpu_times[tid])
    # per-cpu usage
    else:
        ret = []
        if blocking:
            tot1 = cpu_times(percpu=True)
            time.sleep(interval)
        else:
            tot1 = _last_per_cpu_times.get(tid) or cpu_times(percpu=True)
        _last_per_cpu_times[tid] = cpu_times(percpu=True)
        for t1, t2 in zip(tot1, _last_per_cpu_times[tid]):
            ret.append(calculate(t1, t2))
        return ret


# Use a separate dict for cpu_times_percent(), so it's independent from
# cpu_percent() and they can both be used within the same program.
_last_cpu_times_2 = _last_cpu_times.copy()
_last_per_cpu_times_2 = _last_per_cpu_times.copy()


def cpu_times_percent(interval=None, percpu=False):
    """Same as cpu_percent() but provides utilization percentages
    for each specific CPU time as is returned by cpu_times().
    For instance, on Linux we'll get:

      >>> cpu_times_percent()
      cpupercent(user=4.8, nice=0.0, system=4.8, idle=90.5, iowait=0.0,
                 irq=0.0, softirq=0.0, steal=0.0, guest=0.0, guest_nice=0.0)
      >>>

    *interval* and *percpu* arguments have the same meaning as in
    cpu_percent().
    """
    tid = threading.current_thread().ident
    blocking = interval is not None and interval > 0.0
    if interval is not None and interval < 0:
        msg = "interval is not positive (got %r)" % interval
        raise ValueError(msg)

    def calculate(t1, t2):
        nums = []
        times_delta = _cpu_times_deltas(t1, t2)
        all_delta = _cpu_tot_time(times_delta)
        # "scale" is the value to multiply each delta with to get percentages.
        # We use "max" to avoid division by zero (if all_delta is 0, then all
        # fields are 0 so percentages will be 0 too. all_delta cannot be a
        # fraction because cpu times are integers)
        scale = 100.0 / max(1, all_delta)
        for field_delta in times_delta:
            field_perc = field_delta * scale
            field_perc = round(field_perc, 1)
            # make sure we don't return negative values or values over 100%
            field_perc = min(max(0.0, field_perc), 100.0)
            nums.append(field_perc)
        return _psplatform.scputimes(*nums)

    # system-wide usage
    if not percpu:
        if blocking:
            t1 = cpu_times()
            time.sleep(interval)
        else:
            t1 = _last_cpu_times_2.get(tid) or cpu_times()
        _last_cpu_times_2[tid] = cpu_times()
        return calculate(t1, _last_cpu_times_2[tid])
    # per-cpu usage
    else:
        ret = []
        if blocking:
            tot1 = cpu_times(percpu=True)
            time.sleep(interval)
        else:
            tot1 = _last_per_cpu_times_2.get(tid) or cpu_times(percpu=True)
        _last_per_cpu_times_2[tid] = cpu_times(percpu=True)
        for t1, t2 in zip(tot1, _last_per_cpu_times_2[tid]):
            ret.append(calculate(t1, t2))
        return ret


def cpu_stats():
    """Return CPU statistics."""
    return _psplatform.cpu_stats()


if hasattr(_psplatform, "cpu_freq"):

    def cpu_freq(percpu=False):
        """Return CPU frequency as a namedtuple including current,
        min and max frequency expressed in Mhz.

        If *percpu* is True and the system supports per-cpu frequency
        retrieval (Linux only) a list of frequencies is returned for
        each CPU. If not a list with one element is returned.
        """
        ret = _psplatform.cpu_freq()
        if percpu:
            return ret
        else:
            num_cpus = float(len(ret))
            if num_cpus == 0:
                return None
            elif num_cpus == 1:
                return ret[0]
            else:
                currs, mins, maxs = 0.0, 0.0, 0.0
                set_none = False
                for cpu in ret:
                    currs += cpu.current
                    # On Linux if /proc/cpuinfo is used min/max are set
                    # to None.
                    if LINUX and cpu.min is None:
                        set_none = True
                        continue
                    mins += cpu.min
                    maxs += cpu.max

                current = currs / num_cpus

                if set_none:
                    min_ = max_ = None
                else:
                    min_ = mins / num_cpus
                    max_ = maxs / num_cpus

                return _common.scpufreq(current, min_, max_)

    __all__.append("cpu_freq")


if hasattr(os, "getloadavg") or hasattr(_psplatform, "getloadavg"):
    # Perform this hasattr check once on import time to either use the
    # platform based code or proxy straight from the os module.
    if hasattr(os, "getloadavg"):
        getloadavg = os.getloadavg
    else:
        getloadavg = _psplatform.getloadavg

    __all__.append("getloadavg")


# =====================================================================
# --- system memory related functions
# =====================================================================


def virtual_memory():
    """Return statistics about system memory usage as a namedtuple
    including the following fields, expressed in bytes:

     - total:
       total physical memory available.

     - available:
       the memory that can be given instantly to processes without the
       system going into swap.
       This is calculated by summing different memory values depending
       on the platform and it is supposed to be used to monitor actual
       memory usage in a cross platform fashion.

     - percent:
       the percentage usage calculated as (total - available) / total * 100

     - used:
        memory used, calculated differently depending on the platform and
        designed for informational purposes only:
        macOS: active + wired
        BSD: active + wired + cached
        Linux: total - free

     - free:
       memory not being used at all (zeroed) that is readily available;
       note that this doesn't reflect the actual memory available
       (use 'available' instead)

    Platform-specific fields:

     - active (UNIX):
       memory currently in use or very recently used, and so it is in RAM.

     - inactive (UNIX):
       memory that is marked as not used.

     - buffers (BSD, Linux):
       cache for things like file system metadata.

     - cached (BSD, macOS):
       cache for various things.

     - wired (macOS, BSD):
       memory that is marked to always stay in RAM. It is never moved to disk.

     - shared (BSD):
       memory that may be simultaneously accessed by multiple processes.

    The sum of 'used' and 'available' does not necessarily equal total.
    On Windows 'available' and 'free' are the same.
    """
    global _TOTAL_PHYMEM
    ret = _psplatform.virtual_memory()
    # cached for later use in Process.memory_percent()
    _TOTAL_PHYMEM = ret.total
    return ret


def swap_memory():
    """Return system swap memory statistics as a namedtuple including
    the following fields:

     - total:   total swap memory in bytes
     - used:    used swap memory in bytes
     - free:    free swap memory in bytes
     - percent: the percentage usage
     - sin:     no. of bytes the system has swapped in from disk (cumulative)
     - sout:    no. of bytes the system has swapped out from disk (cumulative)

    'sin' and 'sout' on Windows are meaningless and always set to 0.
    """
    return _psplatform.swap_memory()


# =====================================================================
# --- disks/partitions related functions
# =====================================================================


def disk_usage(path):
    """Return disk usage statistics about the given *path* as a
    namedtuple including total, used and free space expressed in bytes
    plus the percentage usage.
    """
    return _psplatform.disk_usage(path)


def disk_partitions(all=False):
    """Return mounted partitions as a list of
    (device, mountpoint, fstype, opts) namedtuple.
    'opts' field is a raw string separated by commas indicating mount
    options which may vary depending on the platform.

    If *all* parameter is False return physical devices only and ignore
    all others.
    """
    return _psplatform.disk_partitions(all)


def disk_io_counters(perdisk=False, nowrap=True):
    """Return system disk I/O statistics as a namedtuple including
    the following fields:

     - read_count:  number of reads
     - write_count: number of writes
     - read_bytes:  number of bytes read
     - write_bytes: number of bytes written
     - read_time:   time spent reading from disk (in ms)
     - write_time:  time spent writing to disk (in ms)

    Platform specific:

     - busy_time: (Linux, FreeBSD) time spent doing actual I/Os (in ms)
     - read_merged_count (Linux): number of merged reads
     - write_merged_count (Linux): number of merged writes

    If *perdisk* is True return the same information for every
    physical disk installed on the system as a dictionary
    with partition names as the keys and the namedtuple
    described above as the values.

    If *nowrap* is True it detects and adjust the numbers which overflow
    and wrap (restart from 0) and add "old value" to "new value" so that
    the returned numbers will always be increasing or remain the same,
    but never decrease.
    "disk_io_counters.cache_clear()" can be used to invalidate the
    cache.

    On recent Windows versions 'diskperf -y' command may need to be
    executed first otherwise this function won't find any disk.
    """
    kwargs = dict(perdisk=perdisk) if LINUX else {}
    rawdict = _psplatform.disk_io_counters(**kwargs)
    if not rawdict:
        return {} if perdisk else None
    if nowrap:
        rawdict = _wrap_numbers(rawdict, 'psutil.disk_io_counters')
    nt = getattr(_psplatform, "sdiskio", _common.sdiskio)
    if perdisk:
        for disk, fields in rawdict.items():
            rawdict[disk] = nt(*fields)
        return rawdict
    else:
        return nt(*(sum(x) for x in zip(*rawdict.values())))


disk_io_counters.cache_clear = functools.partial(
    _wrap_numbers.cache_clear, 'psutil.disk_io_counters'
)
disk_io_counters.cache_clear.__doc__ = "Clears nowrap argument cache"


# =====================================================================
# --- network related functions
# =====================================================================


def net_io_counters(pernic=False, nowrap=True):
    """Return network I/O statistics as a namedtuple including
    the following fields:

     - bytes_sent:   number of bytes sent
     - bytes_recv:   number of bytes received
     - packets_sent: number of packets sent
     - packets_recv: number of packets received
     - errin:        total number of errors while receiving
     - errout:       total number of errors while sending
     - dropin:       total number of incoming packets which were dropped
     - dropout:      total number of outgoing packets which were dropped
                     (always 0 on macOS and BSD)

    If *pernic* is True return the same information for every
    network interface installed on the system as a dictionary
    with network interface names as the keys and the namedtuple
    described above as the values.

    If *nowrap* is True it detects and adjust the numbers which overflow
    and wrap (restart from 0) and add "old value" to "new value" so that
    the returned numbers will always be increasing or remain the same,
    but never decrease.
    "net_io_counters.cache_clear()" can be used to invalidate the
    cache.
    """
    rawdict = _psplatform.net_io_counters()
    if not rawdict:
        return {} if pernic else None
    if nowrap:
        rawdict = _wrap_numbers(rawdict, 'psutil.net_io_counters')
    if pernic:
        for nic, fields in rawdict.items():
            rawdict[nic] = _common.snetio(*fields)
        return rawdict
    else:
        return _common.snetio(*[sum(x) for x in zip(*rawdict.values())])


net_io_counters.cache_clear = functools.partial(
    _wrap_numbers.cache_clear, 'psutil.net_io_counters'
)
net_io_counters.cache_clear.__doc__ = "Clears nowrap argument cache"


def net_connections(kind='inet'):
    """Return system-wide socket connections as a list of
    (fd, family, type, laddr, raddr, status, pid) namedtuples.
    In case of limited privileges 'fd' and 'pid' may be set to -1
    and None respectively.
    The *kind* parameter filters for connections that fit the
    following criteria:

    +------------+----------------------------------------------------+
    | Kind Value | Connections using                                  |
    +------------+----------------------------------------------------+
    | inet       | IPv4 and IPv6                                      |
    | inet4      | IPv4                                               |
    | inet6      | IPv6                                               |
    | tcp        | TCP                                                |
    | tcp4       | TCP over IPv4                                      |
    | tcp6       | TCP over IPv6                                      |
    | udp        | UDP                                                |
    | udp4       | UDP over IPv4                                      |
    | udp6       | UDP over IPv6                                      |
    | unix       | UNIX socket (both UDP and TCP protocols)           |
    | all        | the sum of all the possible families and protocols |
    +------------+----------------------------------------------------+

    On macOS this function requires root privileges.
    """
    return _psplatform.net_connections(kind)


def net_if_addrs():
    """Return the addresses associated to each NIC (network interface
    card) installed on the system as a dictionary whose keys are the
    NIC names and value is a list of namedtuples for each address
    assigned to the NIC. Each namedtuple includes 5 fields:

     - family: can be either socket.AF_INET, socket.AF_INET6 or
               psutil.AF_LINK, which refers to a MAC address.
     - address: is the primary address and it is always set.
     - netmask: and 'broadcast' and 'ptp' may be None.
     - ptp: stands for "point to point" and references the
            destination address on a point to point interface
            (typically a VPN).
     - broadcast: and *ptp* are mutually exclusive.

    Note: you can have more than one address of the same family
    associated with each interface.
    """
    has_enums = _PY3
    if has_enums:
        import socket
    rawlist = _psplatform.net_if_addrs()
    rawlist.sort(key=lambda x: x[1])  # sort by family
    ret = collections.defaultdict(list)
    for name, fam, addr, mask, broadcast, ptp in rawlist:
        if has_enums:
            try:
                fam = socket.AddressFamily(fam)
            except ValueError:
                if WINDOWS and fam == -1:
                    fam = _psplatform.AF_LINK
                elif (
                    hasattr(_psplatform, "AF_LINK")
                    and fam == _psplatform.AF_LINK
                ):
                    # Linux defines AF_LINK as an alias for AF_PACKET.
                    # We re-set the family here so that repr(family)
                    # will show AF_LINK rather than AF_PACKET
                    fam = _psplatform.AF_LINK
        if fam == _psplatform.AF_LINK:
            # The underlying C function may return an incomplete MAC
            # address in which case we fill it with null bytes, see:
            # https://github.com/giampaolo/psutil/issues/786
            separator = ":" if POSIX else "-"
            while addr.count(separator) < 5:
                addr += "%s00" % separator
        ret[name].append(_common.snicaddr(fam, addr, mask, broadcast, ptp))
    return dict(ret)


def net_if_stats():
    """Return information about each NIC (network interface card)
    installed on the system as a dictionary whose keys are the
    NIC names and value is a namedtuple with the following fields:

     - isup: whether the interface is up (bool)
     - duplex: can be either NIC_DUPLEX_FULL, NIC_DUPLEX_HALF or
               NIC_DUPLEX_UNKNOWN
     - speed: the NIC speed expressed in mega bits (MB); if it can't
              be determined (e.g. 'localhost') it will be set to 0.
     - mtu: the maximum transmission unit expressed in bytes.
    """
    return _psplatform.net_if_stats()


# =====================================================================
# --- sensors
# =====================================================================


# Linux, macOS
if hasattr(_psplatform, "sensors_temperatures"):

    def sensors_temperatures(fahrenheit=False):
        """Return hardware temperatures. Each entry is a namedtuple
        representing a certain hardware sensor (it may be a CPU, an
        hard disk or something else, depending on the OS and its
        configuration).
        All temperatures are expressed in celsius unless *fahrenheit*
        is set to True.
        """

        def convert(n):
            if n is not None:
                return (float(n) * 9 / 5) + 32 if fahrenheit else n

        ret = collections.defaultdict(list)
        rawdict = _psplatform.sensors_temperatures()

        for name, values in rawdict.items():
            while values:
                label, current, high, critical = values.pop(0)
                current = convert(current)
                high = convert(high)
                critical = convert(critical)

                if high and not critical:
                    critical = high
                elif critical and not high:
                    high = critical

                ret[name].append(
                    _common.shwtemp(label, current, high, critical)
                )

        return dict(ret)

    __all__.append("sensors_temperatures")


# Linux
if hasattr(_psplatform, "sensors_fans"):

    def sensors_fans():
        """Return fans speed. Each entry is a namedtuple
        representing a certain hardware sensor.
        All speed are expressed in RPM (rounds per minute).
        """
        return _psplatform.sensors_fans()

    __all__.append("sensors_fans")


# Linux, Windows, FreeBSD, macOS
if hasattr(_psplatform, "sensors_battery"):

    def sensors_battery():
        """Return battery information. If no battery is installed
        returns None.

         - percent: battery power left as a percentage.
         - secsleft: a rough approximation of how many seconds are left
                     before the battery runs out of power. May be
                     POWER_TIME_UNLIMITED or POWER_TIME_UNLIMITED.
         - power_plugged: True if the AC power cable is connected.
        """
        return _psplatform.sensors_battery()

    __all__.append("sensors_battery")


# =====================================================================
# --- other system related functions
# =====================================================================


def boot_time():
    """Return the system boot time expressed in seconds since the epoch."""
    # Note: we are not caching this because it is subject to
    # system clock updates.
    return _psplatform.boot_time()


def users():
    """Return users currently connected on the system as a list of
    namedtuples including the following fields.

     - user: the name of the user
     - terminal: the tty or pseudo-tty associated with the user, if any.
     - host: the host name associated with the entry, if any.
     - started: the creation time as a floating point number expressed in
       seconds since the epoch.
    """
    return _psplatform.users()


# =====================================================================
# --- Windows services
# =====================================================================


if WINDOWS:

    def win_service_iter():
        """Return a generator yielding a WindowsService instance for all
        Windows services installed.
        """
        return _psplatform.win_service_iter()

    def win_service_get(name):
        """Get a Windows service by *name*.
        Raise NoSuchProcess if no service with such name exists.
        """
        return _psplatform.win_service_get(name)


# =====================================================================


def _set_debug(value):
    """Enable or disable PSUTIL_DEBUG option, which prints debugging
    messages to stderr.
    """
    import psutil._common

    psutil._common.PSUTIL_DEBUG = bool(value)
    _psplatform.cext.set_debug(bool(value))


def test():  # pragma: no cover
    from ._common import bytes2human
    from ._compat import get_terminal_size

    today_day = datetime.date.today()
    # fmt: off
    templ = "%-10s %5s %5s %7s %7s %5s %6s %6s %6s  %s"
    attrs = ['pid', 'memory_percent', 'name', 'cmdline', 'cpu_times',
             'create_time', 'memory_info', 'status', 'nice', 'username']
    print(templ % ("USER", "PID", "%MEM", "VSZ", "RSS", "NICE",  # NOQA
                   "STATUS", "START", "TIME", "CMDLINE"))
    # fmt: on
    for p in process_iter(attrs, ad_value=None):
        if p.info['create_time']:
            ctime = datetime.datetime.fromtimestamp(p.info['create_time'])
            if ctime.date() == today_day:
                ctime = ctime.strftime("%H:%M")
            else:
                ctime = ctime.strftime("%b%d")
        else:
            ctime = ''
        if p.info['cpu_times']:
            cputime = time.strftime(
                "%M:%S", time.localtime(sum(p.info['cpu_times']))
            )
        else:
            cputime = ''

        user = p.info['username'] or ''
        if not user and POSIX:
            try:
                user = p.uids()[0]
            except Error:
                pass
        if user and WINDOWS and '\\' in user:
            user = user.split('\\')[1]
        user = user[:9]
        vms = (
            bytes2human(p.info['memory_info'].vms)
            if p.info['memory_info'] is not None
            else ''
        )
        rss = (
            bytes2human(p.info['memory_info'].rss)
            if p.info['memory_info'] is not None
            else ''
        )
        memp = (
            round(p.info['memory_percent'], 1)
            if p.info['memory_percent'] is not None
            else ''
        )
        nice = int(p.info['nice']) if p.info['nice'] else ''
        if p.info['cmdline']:
            cmdline = ' '.join(p.info['cmdline'])
        else:
            cmdline = p.info['name']
        status = p.info['status'][:5] if p.info['status'] else ''

        line = templ % (
            user[:10],
            p.info['pid'],
            memp,
            vms,
            rss,
            nice,
            status,
            ctime,
            cputime,
            cmdline,
        )
        print(line[: get_terminal_size()[0]])  # NOQA


del memoize_when_activated, division
if sys.version_info[0] < 3:
    del num, x  # noqa

if __name__ == "__main__":
    test()
