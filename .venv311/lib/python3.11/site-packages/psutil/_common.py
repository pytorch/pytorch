# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Common objects shared by __init__.py and _ps*.py modules.

Note: this module is imported by setup.py, so it should not import
psutil or third-party modules.
"""

import collections
import enum
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM

try:
    from socket import AF_INET6
except ImportError:
    AF_INET6 = None
try:
    from socket import AF_UNIX
except ImportError:
    AF_UNIX = None


PSUTIL_DEBUG = bool(os.getenv('PSUTIL_DEBUG'))
_DEFAULT = object()

# fmt: off
__all__ = [
    # OS constants
    'FREEBSD', 'BSD', 'LINUX', 'NETBSD', 'OPENBSD', 'MACOS', 'OSX', 'POSIX',
    'SUNOS', 'WINDOWS',
    # connection constants
    'CONN_CLOSE', 'CONN_CLOSE_WAIT', 'CONN_CLOSING', 'CONN_ESTABLISHED',
    'CONN_FIN_WAIT1', 'CONN_FIN_WAIT2', 'CONN_LAST_ACK', 'CONN_LISTEN',
    'CONN_NONE', 'CONN_SYN_RECV', 'CONN_SYN_SENT', 'CONN_TIME_WAIT',
    # net constants
    'NIC_DUPLEX_FULL', 'NIC_DUPLEX_HALF', 'NIC_DUPLEX_UNKNOWN',  # noqa: F822
    # process status constants
    'STATUS_DEAD', 'STATUS_DISK_SLEEP', 'STATUS_IDLE', 'STATUS_LOCKED',
    'STATUS_RUNNING', 'STATUS_SLEEPING', 'STATUS_STOPPED', 'STATUS_SUSPENDED',
    'STATUS_TRACING_STOP', 'STATUS_WAITING', 'STATUS_WAKE_KILL',
    'STATUS_WAKING', 'STATUS_ZOMBIE', 'STATUS_PARKED',
    # other constants
    'ENCODING', 'ENCODING_ERRS', 'AF_INET6',
    # named tuples
    'pconn', 'pcputimes', 'pctxsw', 'pgids', 'pio', 'pionice', 'popenfile',
    'pthread', 'puids', 'sconn', 'scpustats', 'sdiskio', 'sdiskpart',
    'sdiskusage', 'snetio', 'snicaddr', 'snicstats', 'sswap', 'suser',
    # utility functions
    'conn_tmap', 'deprecated_method', 'isfile_strict', 'memoize',
    'parse_environ_block', 'path_exists_strict', 'usage_percent',
    'supports_ipv6', 'sockfam_to_enum', 'socktype_to_enum', "wrap_numbers",
    'open_text', 'open_binary', 'cat', 'bcat',
    'bytes2human', 'conn_to_ntuple', 'debug',
    # shell utils
    'hilite', 'term_supports_colors', 'print_color',
]
# fmt: on


# ===================================================================
# --- OS constants
# ===================================================================


POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
OSX = MACOS  # deprecated alias
FREEBSD = sys.platform.startswith(("freebsd", "midnightbsd"))
OPENBSD = sys.platform.startswith("openbsd")
NETBSD = sys.platform.startswith("netbsd")
BSD = FREEBSD or OPENBSD or NETBSD
SUNOS = sys.platform.startswith(("sunos", "solaris"))
AIX = sys.platform.startswith("aix")


# ===================================================================
# --- API constants
# ===================================================================


# Process.status()
STATUS_RUNNING = "running"
STATUS_SLEEPING = "sleeping"
STATUS_DISK_SLEEP = "disk-sleep"
STATUS_STOPPED = "stopped"
STATUS_TRACING_STOP = "tracing-stop"
STATUS_ZOMBIE = "zombie"
STATUS_DEAD = "dead"
STATUS_WAKE_KILL = "wake-kill"
STATUS_WAKING = "waking"
STATUS_IDLE = "idle"  # Linux, macOS, FreeBSD
STATUS_LOCKED = "locked"  # FreeBSD
STATUS_WAITING = "waiting"  # FreeBSD
STATUS_SUSPENDED = "suspended"  # NetBSD
STATUS_PARKED = "parked"  # Linux

# Process.net_connections() and psutil.net_connections()
CONN_ESTABLISHED = "ESTABLISHED"
CONN_SYN_SENT = "SYN_SENT"
CONN_SYN_RECV = "SYN_RECV"
CONN_FIN_WAIT1 = "FIN_WAIT1"
CONN_FIN_WAIT2 = "FIN_WAIT2"
CONN_TIME_WAIT = "TIME_WAIT"
CONN_CLOSE = "CLOSE"
CONN_CLOSE_WAIT = "CLOSE_WAIT"
CONN_LAST_ACK = "LAST_ACK"
CONN_LISTEN = "LISTEN"
CONN_CLOSING = "CLOSING"
CONN_NONE = "NONE"


# net_if_stats()
class NicDuplex(enum.IntEnum):
    NIC_DUPLEX_FULL = 2
    NIC_DUPLEX_HALF = 1
    NIC_DUPLEX_UNKNOWN = 0


globals().update(NicDuplex.__members__)


# sensors_battery()
class BatteryTime(enum.IntEnum):
    POWER_TIME_UNKNOWN = -1
    POWER_TIME_UNLIMITED = -2


globals().update(BatteryTime.__members__)

# --- others

ENCODING = sys.getfilesystemencoding()
ENCODING_ERRS = sys.getfilesystemencodeerrors()


# ===================================================================
# --- namedtuples
# ===================================================================

# --- for system functions

# fmt: off
# psutil.swap_memory()
sswap = namedtuple('sswap', ['total', 'used', 'free', 'percent', 'sin',
                             'sout'])
# psutil.disk_usage()
sdiskusage = namedtuple('sdiskusage', ['total', 'used', 'free', 'percent'])
# psutil.disk_io_counters()
sdiskio = namedtuple('sdiskio', ['read_count', 'write_count',
                                 'read_bytes', 'write_bytes',
                                 'read_time', 'write_time'])
# psutil.disk_partitions()
sdiskpart = namedtuple('sdiskpart', ['device', 'mountpoint', 'fstype', 'opts'])
# psutil.net_io_counters()
snetio = namedtuple('snetio', ['bytes_sent', 'bytes_recv',
                               'packets_sent', 'packets_recv',
                               'errin', 'errout',
                               'dropin', 'dropout'])
# psutil.users()
suser = namedtuple('suser', ['name', 'terminal', 'host', 'started', 'pid'])
# psutil.net_connections()
sconn = namedtuple('sconn', ['fd', 'family', 'type', 'laddr', 'raddr',
                             'status', 'pid'])
# psutil.net_if_addrs()
snicaddr = namedtuple('snicaddr',
                      ['family', 'address', 'netmask', 'broadcast', 'ptp'])
# psutil.net_if_stats()
snicstats = namedtuple('snicstats',
                       ['isup', 'duplex', 'speed', 'mtu', 'flags'])
# psutil.cpu_stats()
scpustats = namedtuple(
    'scpustats', ['ctx_switches', 'interrupts', 'soft_interrupts', 'syscalls'])
# psutil.cpu_freq()
scpufreq = namedtuple('scpufreq', ['current', 'min', 'max'])
# psutil.sensors_temperatures()
shwtemp = namedtuple(
    'shwtemp', ['label', 'current', 'high', 'critical'])
# psutil.sensors_battery()
sbattery = namedtuple('sbattery', ['percent', 'secsleft', 'power_plugged'])
# psutil.sensors_fans()
sfan = namedtuple('sfan', ['label', 'current'])
# fmt: on

# --- for Process methods

# psutil.Process.cpu_times()
pcputimes = namedtuple(
    'pcputimes', ['user', 'system', 'children_user', 'children_system']
)
# psutil.Process.open_files()
popenfile = namedtuple('popenfile', ['path', 'fd'])
# psutil.Process.threads()
pthread = namedtuple('pthread', ['id', 'user_time', 'system_time'])
# psutil.Process.uids()
puids = namedtuple('puids', ['real', 'effective', 'saved'])
# psutil.Process.gids()
pgids = namedtuple('pgids', ['real', 'effective', 'saved'])
# psutil.Process.io_counters()
pio = namedtuple(
    'pio', ['read_count', 'write_count', 'read_bytes', 'write_bytes']
)
# psutil.Process.ionice()
pionice = namedtuple('pionice', ['ioclass', 'value'])
# psutil.Process.ctx_switches()
pctxsw = namedtuple('pctxsw', ['voluntary', 'involuntary'])
# psutil.Process.net_connections()
pconn = namedtuple(
    'pconn', ['fd', 'family', 'type', 'laddr', 'raddr', 'status']
)

# psutil.net_connections() and psutil.Process.net_connections()
addr = namedtuple('addr', ['ip', 'port'])


# ===================================================================
# --- Process.net_connections() 'kind' parameter mapping
# ===================================================================


conn_tmap = {
    "all": ([AF_INET, AF_INET6, AF_UNIX], [SOCK_STREAM, SOCK_DGRAM]),
    "tcp": ([AF_INET, AF_INET6], [SOCK_STREAM]),
    "tcp4": ([AF_INET], [SOCK_STREAM]),
    "udp": ([AF_INET, AF_INET6], [SOCK_DGRAM]),
    "udp4": ([AF_INET], [SOCK_DGRAM]),
    "inet": ([AF_INET, AF_INET6], [SOCK_STREAM, SOCK_DGRAM]),
    "inet4": ([AF_INET], [SOCK_STREAM, SOCK_DGRAM]),
    "inet6": ([AF_INET6], [SOCK_STREAM, SOCK_DGRAM]),
}

if AF_INET6 is not None:
    conn_tmap.update({
        "tcp6": ([AF_INET6], [SOCK_STREAM]),
        "udp6": ([AF_INET6], [SOCK_DGRAM]),
    })

if AF_UNIX is not None and not SUNOS:
    conn_tmap.update({"unix": ([AF_UNIX], [SOCK_STREAM, SOCK_DGRAM])})


# =====================================================================
# --- Exceptions
# =====================================================================


class Error(Exception):
    """Base exception class. All other psutil exceptions inherit
    from this one.
    """

    __module__ = 'psutil'

    def _infodict(self, attrs):
        info = collections.OrderedDict()
        for name in attrs:
            value = getattr(self, name, None)
            if value or (name == "pid" and value == 0):
                info[name] = value
        return info

    def __str__(self):
        # invoked on `raise Error`
        info = self._infodict(("pid", "ppid", "name"))
        if info:
            details = "({})".format(
                ", ".join([f"{k}={v!r}" for k, v in info.items()])
            )
        else:
            details = None
        return " ".join([x for x in (getattr(self, "msg", ""), details) if x])

    def __repr__(self):
        # invoked on `repr(Error)`
        info = self._infodict(("pid", "ppid", "name", "seconds", "msg"))
        details = ", ".join([f"{k}={v!r}" for k, v in info.items()])
        return f"psutil.{self.__class__.__name__}({details})"


class NoSuchProcess(Error):
    """Exception raised when a process with a certain PID doesn't
    or no longer exists.
    """

    __module__ = 'psutil'

    def __init__(self, pid, name=None, msg=None):
        Error.__init__(self)
        self.pid = pid
        self.name = name
        self.msg = msg or "process no longer exists"

    def __reduce__(self):
        return (self.__class__, (self.pid, self.name, self.msg))


class ZombieProcess(NoSuchProcess):
    """Exception raised when querying a zombie process. This is
    raised on macOS, BSD and Solaris only, and not always: depending
    on the query the OS may be able to succeed anyway.
    On Linux all zombie processes are querable (hence this is never
    raised). Windows doesn't have zombie processes.
    """

    __module__ = 'psutil'

    def __init__(self, pid, name=None, ppid=None, msg=None):
        NoSuchProcess.__init__(self, pid, name, msg)
        self.ppid = ppid
        self.msg = msg or "PID still exists but it's a zombie"

    def __reduce__(self):
        return (self.__class__, (self.pid, self.name, self.ppid, self.msg))


class AccessDenied(Error):
    """Exception raised when permission to perform an action is denied."""

    __module__ = 'psutil'

    def __init__(self, pid=None, name=None, msg=None):
        Error.__init__(self)
        self.pid = pid
        self.name = name
        self.msg = msg or ""

    def __reduce__(self):
        return (self.__class__, (self.pid, self.name, self.msg))


class TimeoutExpired(Error):
    """Raised on Process.wait(timeout) if timeout expires and process
    is still alive.
    """

    __module__ = 'psutil'

    def __init__(self, seconds, pid=None, name=None):
        Error.__init__(self)
        self.seconds = seconds
        self.pid = pid
        self.name = name
        self.msg = f"timeout after {seconds} seconds"

    def __reduce__(self):
        return (self.__class__, (self.seconds, self.pid, self.name))


# ===================================================================
# --- utils
# ===================================================================


def usage_percent(used, total, round_=None):
    """Calculate percentage usage of 'used' against 'total'."""
    try:
        ret = (float(used) / total) * 100
    except ZeroDivisionError:
        return 0.0
    else:
        if round_ is not None:
            ret = round(ret, round_)
        return ret


def memoize(fun):
    """A simple memoize decorator for functions supporting (hashable)
    positional arguments.
    It also provides a cache_clear() function for clearing the cache:

    >>> @memoize
    ... def foo()
    ...     return 1
        ...
    >>> foo()
    1
    >>> foo.cache_clear()
    >>>

    It supports:
     - functions
     - classes (acts as a @singleton)
     - staticmethods
     - classmethods

    It does NOT support:
     - methods
    """

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(sorted(kwargs.items())))
        try:
            return cache[key]
        except KeyError:
            try:
                ret = cache[key] = fun(*args, **kwargs)
            except Exception as err:
                raise err from None
            return ret

    def cache_clear():
        """Clear cache."""
        cache.clear()

    cache = {}
    wrapper.cache_clear = cache_clear
    return wrapper


def memoize_when_activated(fun):
    """A memoize decorator which is disabled by default. It can be
    activated and deactivated on request.
    For efficiency reasons it can be used only against class methods
    accepting no arguments.

    >>> class Foo:
    ...     @memoize
    ...     def foo()
    ...         print(1)
    ...
    >>> f = Foo()
    >>> # deactivated (default)
    >>> foo()
    1
    >>> foo()
    1
    >>>
    >>> # activated
    >>> foo.cache_activate(self)
    >>> foo()
    1
    >>> foo()
    >>> foo()
    >>>
    """

    @functools.wraps(fun)
    def wrapper(self):
        try:
            # case 1: we previously entered oneshot() ctx
            ret = self._cache[fun]
        except AttributeError:
            # case 2: we never entered oneshot() ctx
            try:
                return fun(self)
            except Exception as err:
                raise err from None
        except KeyError:
            # case 3: we entered oneshot() ctx but there's no cache
            # for this entry yet
            try:
                ret = fun(self)
            except Exception as err:
                raise err from None
            try:
                self._cache[fun] = ret
            except AttributeError:
                # multi-threading race condition, see:
                # https://github.com/giampaolo/psutil/issues/1948
                pass
        return ret

    def cache_activate(proc):
        """Activate cache. Expects a Process instance. Cache will be
        stored as a "_cache" instance attribute.
        """
        proc._cache = {}

    def cache_deactivate(proc):
        """Deactivate and clear cache."""
        try:
            del proc._cache
        except AttributeError:
            pass

    wrapper.cache_activate = cache_activate
    wrapper.cache_deactivate = cache_deactivate
    return wrapper


def isfile_strict(path):
    """Same as os.path.isfile() but does not swallow EACCES / EPERM
    exceptions, see:
    http://mail.python.org/pipermail/python-dev/2012-June/120787.html.
    """
    try:
        st = os.stat(path)
    except PermissionError:
        raise
    except OSError:
        return False
    else:
        return stat.S_ISREG(st.st_mode)


def path_exists_strict(path):
    """Same as os.path.exists() but does not swallow EACCES / EPERM
    exceptions. See:
    http://mail.python.org/pipermail/python-dev/2012-June/120787.html.
    """
    try:
        os.stat(path)
    except PermissionError:
        raise
    except OSError:
        return False
    else:
        return True


def supports_ipv6():
    """Return True if IPv6 is supported on this platform."""
    if not socket.has_ipv6 or AF_INET6 is None:
        return False
    try:
        with socket.socket(AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
        return True
    except OSError:
        return False


def parse_environ_block(data):
    """Parse a C environ block of environment variables into a dictionary."""
    # The block is usually raw data from the target process.  It might contain
    # trailing garbage and lines that do not look like assignments.
    ret = {}
    pos = 0

    # localize global variable to speed up access.
    WINDOWS_ = WINDOWS
    while True:
        next_pos = data.find("\0", pos)
        # nul byte at the beginning or double nul byte means finish
        if next_pos <= pos:
            break
        # there might not be an equals sign
        equal_pos = data.find("=", pos, next_pos)
        if equal_pos > pos:
            key = data[pos:equal_pos]
            value = data[equal_pos + 1 : next_pos]
            # Windows expects environment variables to be uppercase only
            if WINDOWS_:
                key = key.upper()
            ret[key] = value
        pos = next_pos + 1

    return ret


def sockfam_to_enum(num):
    """Convert a numeric socket family value to an IntEnum member.
    If it's not a known member, return the numeric value itself.
    """
    try:
        return socket.AddressFamily(num)
    except ValueError:
        return num


def socktype_to_enum(num):
    """Convert a numeric socket type value to an IntEnum member.
    If it's not a known member, return the numeric value itself.
    """
    try:
        return socket.SocketKind(num)
    except ValueError:
        return num


def conn_to_ntuple(fd, fam, type_, laddr, raddr, status, status_map, pid=None):
    """Convert a raw connection tuple to a proper ntuple."""
    if fam in {socket.AF_INET, AF_INET6}:
        if laddr:
            laddr = addr(*laddr)
        if raddr:
            raddr = addr(*raddr)
    if type_ == socket.SOCK_STREAM and fam in {AF_INET, AF_INET6}:
        status = status_map.get(status, CONN_NONE)
    else:
        status = CONN_NONE  # ignore whatever C returned to us
    fam = sockfam_to_enum(fam)
    type_ = socktype_to_enum(type_)
    if pid is None:
        return pconn(fd, fam, type_, laddr, raddr, status)
    else:
        return sconn(fd, fam, type_, laddr, raddr, status, pid)


def broadcast_addr(addr):
    """Given the address ntuple returned by ``net_if_addrs()``
    calculates the broadcast address.
    """
    import ipaddress

    if not addr.address or not addr.netmask:
        return None
    if addr.family == socket.AF_INET:
        return str(
            ipaddress.IPv4Network(
                f"{addr.address}/{addr.netmask}", strict=False
            ).broadcast_address
        )
    if addr.family == socket.AF_INET6:
        return str(
            ipaddress.IPv6Network(
                f"{addr.address}/{addr.netmask}", strict=False
            ).broadcast_address
        )


def deprecated_method(replacement):
    """A decorator which can be used to mark a method as deprecated
    'replcement' is the method name which will be called instead.
    """

    def outer(fun):
        msg = (
            f"{fun.__name__}() is deprecated and will be removed; use"
            f" {replacement}() instead"
        )
        if fun.__doc__ is None:
            fun.__doc__ = msg

        @functools.wraps(fun)
        def inner(self, *args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return getattr(self, replacement)(*args, **kwargs)

        return inner

    return outer


class _WrapNumbers:
    """Watches numbers so that they don't overflow and wrap
    (reset to zero).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}
        self.reminders = {}
        self.reminder_keys = {}

    def _add_dict(self, input_dict, name):
        assert name not in self.cache
        assert name not in self.reminders
        assert name not in self.reminder_keys
        self.cache[name] = input_dict
        self.reminders[name] = collections.defaultdict(int)
        self.reminder_keys[name] = collections.defaultdict(set)

    def _remove_dead_reminders(self, input_dict, name):
        """In case the number of keys changed between calls (e.g. a
        disk disappears) this removes the entry from self.reminders.
        """
        old_dict = self.cache[name]
        gone_keys = set(old_dict.keys()) - set(input_dict.keys())
        for gone_key in gone_keys:
            for remkey in self.reminder_keys[name][gone_key]:
                del self.reminders[name][remkey]
            del self.reminder_keys[name][gone_key]

    def run(self, input_dict, name):
        """Cache dict and sum numbers which overflow and wrap.
        Return an updated copy of `input_dict`.
        """
        if name not in self.cache:
            # This was the first call.
            self._add_dict(input_dict, name)
            return input_dict

        self._remove_dead_reminders(input_dict, name)

        old_dict = self.cache[name]
        new_dict = {}
        for key in input_dict:
            input_tuple = input_dict[key]
            try:
                old_tuple = old_dict[key]
            except KeyError:
                # The input dict has a new key (e.g. a new disk or NIC)
                # which didn't exist in the previous call.
                new_dict[key] = input_tuple
                continue

            bits = []
            for i in range(len(input_tuple)):
                input_value = input_tuple[i]
                old_value = old_tuple[i]
                remkey = (key, i)
                if input_value < old_value:
                    # it wrapped!
                    self.reminders[name][remkey] += old_value
                    self.reminder_keys[name][key].add(remkey)
                bits.append(input_value + self.reminders[name][remkey])

            new_dict[key] = tuple(bits)

        self.cache[name] = input_dict
        return new_dict

    def cache_clear(self, name=None):
        """Clear the internal cache, optionally only for function 'name'."""
        with self.lock:
            if name is None:
                self.cache.clear()
                self.reminders.clear()
                self.reminder_keys.clear()
            else:
                self.cache.pop(name, None)
                self.reminders.pop(name, None)
                self.reminder_keys.pop(name, None)

    def cache_info(self):
        """Return internal cache dicts as a tuple of 3 elements."""
        with self.lock:
            return (self.cache, self.reminders, self.reminder_keys)


def wrap_numbers(input_dict, name):
    """Given an `input_dict` and a function `name`, adjust the numbers
    which "wrap" (restart from zero) across different calls by adding
    "old value" to "new value" and return an updated dict.
    """
    with _wn.lock:
        return _wn.run(input_dict, name)


_wn = _WrapNumbers()
wrap_numbers.cache_clear = _wn.cache_clear
wrap_numbers.cache_info = _wn.cache_info


# The read buffer size for open() builtin. This (also) dictates how
# much data we read(2) when iterating over file lines as in:
#   >>> with open(file) as f:
#   ...    for line in f:
#   ...        ...
# Default per-line buffer size for binary files is 1K. For text files
# is 8K. We use a bigger buffer (32K) in order to have more consistent
# results when reading /proc pseudo files on Linux, see:
# https://github.com/giampaolo/psutil/issues/2050
# https://github.com/giampaolo/psutil/issues/708
FILE_READ_BUFFER_SIZE = 32 * 1024


def open_binary(fname):
    return open(fname, "rb", buffering=FILE_READ_BUFFER_SIZE)


def open_text(fname):
    """Open a file in text mode by using the proper FS encoding and
    en/decoding error handlers.
    """
    # See:
    # https://github.com/giampaolo/psutil/issues/675
    # https://github.com/giampaolo/psutil/pull/733
    fobj = open(  # noqa: SIM115
        fname,
        buffering=FILE_READ_BUFFER_SIZE,
        encoding=ENCODING,
        errors=ENCODING_ERRS,
    )
    try:
        # Dictates per-line read(2) buffer size. Defaults is 8k. See:
        # https://github.com/giampaolo/psutil/issues/2050#issuecomment-1013387546
        fobj._CHUNK_SIZE = FILE_READ_BUFFER_SIZE
    except AttributeError:
        pass
    except Exception:
        fobj.close()
        raise

    return fobj


def cat(fname, fallback=_DEFAULT, _open=open_text):
    """Read entire file content and return it as a string. File is
    opened in text mode. If specified, `fallback` is the value
    returned in case of error, either if the file does not exist or
    it can't be read().
    """
    if fallback is _DEFAULT:
        with _open(fname) as f:
            return f.read()
    else:
        try:
            with _open(fname) as f:
                return f.read()
        except OSError:
            return fallback


def bcat(fname, fallback=_DEFAULT):
    """Same as above but opens file in binary mode."""
    return cat(fname, fallback=fallback, _open=open_binary)


def bytes2human(n, format="%(value).1f%(symbol)s"):
    """Used by various scripts. See: https://code.activestate.com/recipes/578019-bytes-to-human-human-to-bytes-converter/?in=user-4178764.

    >>> bytes2human(10000)
    '9.8K'
    >>> bytes2human(100001221)
    '95.4M'
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if abs(n) >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    return sys.modules['psutil'].PROCFS_PATH


def decode(s):
    return s.decode(encoding=ENCODING, errors=ENCODING_ERRS)


# =====================================================================
# --- shell utils
# =====================================================================


@memoize
def term_supports_colors(file=sys.stdout):  # pragma: no cover
    if os.name == 'nt':
        return True
    try:
        import curses

        assert file.isatty()
        curses.setupterm()
        assert curses.tigetnum("colors") > 0
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


def hilite(s, color=None, bold=False):  # pragma: no cover
    """Return an highlighted version of 'string'."""
    if not term_supports_colors():
        return s
    attr = []
    colors = dict(
        blue='34',
        brown='33',
        darkgrey='30',
        green='32',
        grey='37',
        lightblue='36',
        red='91',
        violet='35',
        yellow='93',
    )
    colors[None] = '29'
    try:
        color = colors[color]
    except KeyError:
        msg = f"invalid color {color!r}; choose amongst {list(colors.keys())}"
        raise ValueError(msg) from None
    attr.append(color)
    if bold:
        attr.append('1')
    return f"\x1b[{';'.join(attr)}m{s}\x1b[0m"


def print_color(
    s, color=None, bold=False, file=sys.stdout
):  # pragma: no cover
    """Print a colorized version of string."""
    if not term_supports_colors():
        print(s, file=file)
    elif POSIX:
        print(hilite(s, color, bold), file=file)
    else:
        import ctypes

        DEFAULT_COLOR = 7
        GetStdHandle = ctypes.windll.Kernel32.GetStdHandle
        SetConsoleTextAttribute = (
            ctypes.windll.Kernel32.SetConsoleTextAttribute
        )

        colors = dict(green=2, red=4, brown=6, yellow=6)
        colors[None] = DEFAULT_COLOR
        try:
            color = colors[color]
        except KeyError:
            msg = (
                f"invalid color {color!r}; choose between"
                f" {list(colors.keys())!r}"
            )
            raise ValueError(msg) from None
        if bold and color <= 7:
            color += 8

        handle_id = -12 if file is sys.stderr else -11
        GetStdHandle.restype = ctypes.c_ulong
        handle = GetStdHandle(handle_id)
        SetConsoleTextAttribute(handle, color)
        try:
            print(s, file=file)
        finally:
            SetConsoleTextAttribute(handle, DEFAULT_COLOR)


def debug(msg):
    """If PSUTIL_DEBUG env var is set, print a debug message to stderr."""
    if PSUTIL_DEBUG:
        import inspect

        fname, lineno, _, _lines, _index = inspect.getframeinfo(
            inspect.currentframe().f_back
        )
        if isinstance(msg, Exception):
            if isinstance(msg, OSError):
                # ...because str(exc) may contain info about the file name
                msg = f"ignoring {msg}"
            else:
                msg = f"ignoring {msg!r}"
        print(  # noqa: T201
            f"psutil-debug [{fname}:{lineno}]> {msg}", file=sys.stderr
        )
