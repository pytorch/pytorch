# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Routines common to all posix systems."""

import glob
import os
import signal
import sys
import time

from ._common import MACOS
from ._common import TimeoutExpired
from ._common import memoize
from ._common import sdiskusage
from ._common import usage_percent
from ._compat import PY3
from ._compat import ChildProcessError
from ._compat import FileNotFoundError
from ._compat import InterruptedError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import unicode


if MACOS:
    from . import _psutil_osx


if PY3:
    import enum
else:
    enum = None


__all__ = ['pid_exists', 'wait_pid', 'disk_usage', 'get_terminal_map']


def pid_exists(pid):
    """Check whether pid exists in the current process table."""
    if pid == 0:
        # According to "man 2 kill" PID 0 has a special meaning:
        # it refers to <<every process in the process group of the
        # calling process>> so we don't want to go any further.
        # If we get here it means this UNIX platform *does* have
        # a process with id 0.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM clearly means there's a process to deny access to
        return True
    # According to "man 2 kill" possible error values are
    # (EINVAL, EPERM, ESRCH)
    else:
        return True


# Python 3.5 signals enum (contributed by me ^^):
# https://bugs.python.org/issue21076
if enum is not None and hasattr(signal, "Signals"):
    Negsignal = enum.IntEnum(
        'Negsignal', dict([(x.name, -x.value) for x in signal.Signals])
    )

    def negsig_to_enum(num):
        """Convert a negative signal value to an enum."""
        try:
            return Negsignal(num)
        except ValueError:
            return num

else:  # pragma: no cover

    def negsig_to_enum(num):
        return num


def wait_pid(
    pid,
    timeout=None,
    proc_name=None,
    _waitpid=os.waitpid,
    _timer=getattr(time, 'monotonic', time.time),  # noqa: B008
    _min=min,
    _sleep=time.sleep,
    _pid_exists=pid_exists,
):
    """Wait for a process PID to terminate.

    If the process terminated normally by calling exit(3) or _exit(2),
    or by returning from main(), the return value is the positive integer
    passed to *exit().

    If it was terminated by a signal it returns the negated value of the
    signal which caused the termination (e.g. -SIGTERM).

    If PID is not a children of os.getpid() (current process) just
    wait until the process disappears and return None.

    If PID does not exist at all return None immediately.

    If *timeout* != None and process is still alive raise TimeoutExpired.
    timeout=0 is also possible (either return immediately or raise).
    """
    if pid <= 0:
        # see "man waitpid"
        msg = "can't wait for PID 0"
        raise ValueError(msg)
    interval = 0.0001
    flags = 0
    if timeout is not None:
        flags |= os.WNOHANG
        stop_at = _timer() + timeout

    def sleep(interval):
        # Sleep for some time and return a new increased interval.
        if timeout is not None:
            if _timer() >= stop_at:
                raise TimeoutExpired(timeout, pid=pid, name=proc_name)
        _sleep(interval)
        return _min(interval * 2, 0.04)

    # See: https://linux.die.net/man/2/waitpid
    while True:
        try:
            retpid, status = os.waitpid(pid, flags)
        except InterruptedError:
            interval = sleep(interval)
        except ChildProcessError:
            # This has two meanings:
            # - PID is not a child of os.getpid() in which case
            #   we keep polling until it's gone
            # - PID never existed in the first place
            # In both cases we'll eventually return None as we
            # can't determine its exit status code.
            while _pid_exists(pid):
                interval = sleep(interval)
            return
        else:
            if retpid == 0:
                # WNOHANG flag was used and PID is still running.
                interval = sleep(interval)
                continue

            if os.WIFEXITED(status):
                # Process terminated normally by calling exit(3) or _exit(2),
                # or by returning from main(). The return value is the
                # positive integer passed to *exit().
                return os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                # Process exited due to a signal. Return the negative value
                # of that signal.
                return negsig_to_enum(-os.WTERMSIG(status))
            # elif os.WIFSTOPPED(status):
            #     # Process was stopped via SIGSTOP or is being traced, and
            #     # waitpid() was called with WUNTRACED flag. PID is still
            #     # alive. From now on waitpid() will keep returning (0, 0)
            #     # until the process state doesn't change.
            #     # It may make sense to catch/enable this since stopped PIDs
            #     # ignore SIGTERM.
            #     interval = sleep(interval)
            #     continue
            # elif os.WIFCONTINUED(status):
            #     # Process was resumed via SIGCONT and waitpid() was called
            #     # with WCONTINUED flag.
            #     interval = sleep(interval)
            #     continue
            else:
                # Should never happen.
                raise ValueError("unknown process exit status %r" % status)


def disk_usage(path):
    """Return disk usage associated with path.
    Note: UNIX usually reserves 5% disk space which is not accessible
    by user. In this function "total" and "used" values reflect the
    total and used disk space whereas "free" and "percent" represent
    the "free" and "used percent" user disk space.
    """
    if PY3:
        st = os.statvfs(path)
    else:  # pragma: no cover
        # os.statvfs() does not support unicode on Python 2:
        # - https://github.com/giampaolo/psutil/issues/416
        # - http://bugs.python.org/issue18695
        try:
            st = os.statvfs(path)
        except UnicodeEncodeError:
            if isinstance(path, unicode):
                try:
                    path = path.encode(sys.getfilesystemencoding())
                except UnicodeEncodeError:
                    pass
                st = os.statvfs(path)
            else:
                raise

    # Total space which is only available to root (unless changed
    # at system level).
    total = st.f_blocks * st.f_frsize
    # Remaining free space usable by root.
    avail_to_root = st.f_bfree * st.f_frsize
    # Remaining free space usable by user.
    avail_to_user = st.f_bavail * st.f_frsize
    # Total space being used in general.
    used = total - avail_to_root
    if MACOS:
        # see: https://github.com/giampaolo/psutil/pull/2152
        used = _psutil_osx.disk_usage_used(path, used)
    # Total space which is available to user (same as 'total' but
    # for the user).
    total_user = used + avail_to_user
    # User usage percent compared to the total amount of space
    # the user can use. This number would be higher if compared
    # to root's because the user has less space (usually -5%).
    usage_percent_user = usage_percent(used, total_user, round_=1)

    # NB: the percentage is -5% than what shown by df due to
    # reserved blocks that we are currently not considering:
    # https://github.com/giampaolo/psutil/issues/829#issuecomment-223750462
    return sdiskusage(
        total=total, used=used, free=avail_to_user, percent=usage_percent_user
    )


@memoize
def get_terminal_map():
    """Get a map of device-id -> path as a dict.
    Used by Process.terminal().
    """
    ret = {}
    ls = glob.glob('/dev/tty*') + glob.glob('/dev/pts/*')
    for name in ls:
        assert name not in ret, name
        try:
            ret[os.stat(name).st_rdev] = name
        except FileNotFoundError:
            pass
    return ret
