#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'
# Copyright (c) 2017, Arnon Yaari
# All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""AIX specific tests."""

import re

import psutil
from psutil import AIX
from psutil.tests import PsutilTestCase
from psutil.tests import pytest
from psutil.tests import sh


@pytest.mark.skipif(not AIX, reason="AIX only")
class AIXSpecificTestCase(PsutilTestCase):
    def test_virtual_memory(self):
        out = sh('/usr/bin/svmon -O unit=KB')
        re_pattern = r"memory\s*"
        for field in [
            "size",
            "inuse",
            "free",
            "pin",
            "virtual",
            "available",
            "mmode",
        ]:
            re_pattern += rf"(?P<{field}>\S+)\s+"
        matchobj = re.search(re_pattern, out)

        assert matchobj is not None

        KB = 1024
        total = int(matchobj.group("size")) * KB
        available = int(matchobj.group("available")) * KB
        used = int(matchobj.group("inuse")) * KB
        free = int(matchobj.group("free")) * KB

        psutil_result = psutil.virtual_memory()

        # TOLERANCE_SYS_MEM from psutil.tests is not enough. For some reason
        # we're seeing differences of ~1.2 MB. 2 MB is still a good tolerance
        # when compared to GBs.
        TOLERANCE_SYS_MEM = 2 * KB * KB  # 2 MB
        assert psutil_result.total == total
        assert abs(psutil_result.used - used) < TOLERANCE_SYS_MEM
        assert abs(psutil_result.available - available) < TOLERANCE_SYS_MEM
        assert abs(psutil_result.free - free) < TOLERANCE_SYS_MEM

    def test_swap_memory(self):
        out = sh('/usr/sbin/lsps -a')
        # From the man page, "The size is given in megabytes" so we assume
        # we'll always have 'MB' in the result
        # TODO maybe try to use "swap -l" to check "used" too, but its units
        # are not guaranteed to be "MB" so parsing may not be consistent
        matchobj = re.search(
            r"(?P<space>\S+)\s+"
            r"(?P<vol>\S+)\s+"
            r"(?P<vg>\S+)\s+"
            r"(?P<size>\d+)MB",
            out,
        )

        assert matchobj is not None

        total_mb = int(matchobj.group("size"))
        MB = 1024**2
        psutil_result = psutil.swap_memory()
        # we divide our result by MB instead of multiplying the lsps value by
        # MB because lsps may round down, so we round down too
        assert int(psutil_result.total / MB) == total_mb

    def test_cpu_stats(self):
        out = sh('/usr/bin/mpstat -a')

        re_pattern = r"ALL\s*"
        for field in [
            "min",
            "maj",
            "mpcs",
            "mpcr",
            "dev",
            "soft",
            "dec",
            "ph",
            "cs",
            "ics",
            "bound",
            "rq",
            "push",
            "S3pull",
            "S3grd",
            "S0rd",
            "S1rd",
            "S2rd",
            "S3rd",
            "S4rd",
            "S5rd",
            "sysc",
        ]:
            re_pattern += rf"(?P<{field}>\S+)\s+"
        matchobj = re.search(re_pattern, out)

        assert matchobj is not None

        # numbers are usually in the millions so 1000 is ok for tolerance
        CPU_STATS_TOLERANCE = 1000
        psutil_result = psutil.cpu_stats()
        assert (
            abs(psutil_result.ctx_switches - int(matchobj.group("cs")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.syscalls - int(matchobj.group("sysc")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.interrupts - int(matchobj.group("dev")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.soft_interrupts - int(matchobj.group("soft")))
            < CPU_STATS_TOLERANCE
        )

    def test_cpu_count_logical(self):
        out = sh('/usr/bin/mpstat -a')
        mpstat_lcpu = int(re.search(r"lcpu=(\d+)", out).group(1))
        psutil_lcpu = psutil.cpu_count(logical=True)
        assert mpstat_lcpu == psutil_lcpu

    def test_net_if_addrs_names(self):
        out = sh('/etc/ifconfig -l')
        ifconfig_names = set(out.split())
        psutil_names = set(psutil.net_if_addrs().keys())
        assert ifconfig_names == psutil_names
