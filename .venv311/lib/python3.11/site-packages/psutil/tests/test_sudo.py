#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests which are meant to be run as root.

NOTE: keep this module compatible with unittest: we want to run this
file with the unittest runner, since pytest may not be installed for
the root user.
"""

import datetime
import time
import unittest

import psutil
from psutil import FREEBSD
from psutil import LINUX
from psutil import OPENBSD
from psutil import WINDOWS
from psutil.tests import CI_TESTING
from psutil.tests import PsutilTestCase


def get_systime():
    if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_REALTIME"):
        return time.clock_gettime(time.CLOCK_REALTIME)
    return time.time()


def set_systime(secs):  # secs since the epoch
    if hasattr(time, "clock_settime") and hasattr(time, "CLOCK_REALTIME"):
        try:
            time.clock_settime(time.CLOCK_REALTIME, secs)
        except PermissionError:
            raise unittest.SkipTest("needs root")
    elif WINDOWS:
        import pywintypes
        import win32api

        dt = datetime.datetime.fromtimestamp(secs, datetime.timezone.utc)
        try:
            win32api.SetSystemTime(
                dt.year,
                dt.month,
                dt.isoweekday() % 7,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                int(dt.microsecond / 1000),
            )
        except pywintypes.error as err:
            if err.winerror == 1314:
                raise unittest.SkipTest("needs Administrator user")
            raise
    else:
        raise unittest.SkipTest("setting systime not supported")


class TestUpdatedSystemTime(PsutilTestCase):
    """Tests which update the system clock."""

    def setUp(self):
        self.time_updated = False
        self.orig_time = get_systime()
        self.time_started = time.monotonic()

    def tearDown(self):
        if self.time_updated:
            extra_t = time.monotonic() - self.time_started
            set_systime(self.orig_time + extra_t)

    def update_systime(self):
        # set system time 1 hour later
        set_systime(self.orig_time + 3600)
        self.time_updated = True

    def test_boot_time(self):
        # Test that boot_time() reflects system clock updates.
        t1 = psutil.boot_time()
        self.update_systime()
        t2 = psutil.boot_time()
        self.assertGreater(t2, t1)
        diff = int(t2 - t1)
        self.assertAlmostEqual(diff, 3600, delta=1)

    @unittest.skipIf(WINDOWS, "broken on WINDOWS")  # TODO: fix it
    def test_proc_create_time(self):
        # Test that Process.create_time() reflects system clock
        # updates. On systems such as Linux this is added on top of the
        # process monotonic time returned by the kernel.
        t1 = psutil.Process().create_time()
        self.update_systime()
        t2 = psutil.Process().create_time()
        diff = int(t2 - t1)
        self.assertAlmostEqual(diff, 3600, delta=1)

    @unittest.skipIf(CI_TESTING, "skipped on CI for now")  # TODO: fix it
    @unittest.skipIf(OPENBSD, "broken on OPENBSD")  # TODO: fix it
    @unittest.skipIf(FREEBSD, "broken on FREEBSD")  # TODO: fix it
    def test_proc_ident(self):
        p1 = psutil.Process()
        self.update_systime()
        p2 = psutil.Process()
        self.assertEqual(p1._get_ident(), p2._get_ident())
        self.assertEqual(p1, p2)

    @unittest.skipIf(not LINUX, "LINUX only")
    def test_linux_monotonic_proc_time(self):
        t1 = psutil.Process()._proc.create_time(monotonic=True)
        self.update_systime()
        time.sleep(0.05)
        t2 = psutil.Process()._proc.create_time(monotonic=True)
        self.assertEqual(t1, t2)
