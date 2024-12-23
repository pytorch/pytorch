#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Sun OS specific tests."""

import os

import psutil
from psutil import SUNOS
from psutil.tests import PsutilTestCase
from psutil.tests import pytest
from psutil.tests import sh


@pytest.mark.skipif(not SUNOS, reason="SUNOS only")
class SunOSSpecificTestCase(PsutilTestCase):
    def test_swap_memory(self):
        out = sh('env PATH=/usr/sbin:/sbin:%s swap -l' % os.environ['PATH'])
        lines = out.strip().split('\n')[1:]
        if not lines:
            raise ValueError('no swap device(s) configured')
        total = free = 0
        for line in lines:
            fields = line.split()
            total = int(fields[3]) * 512
            free = int(fields[4]) * 512
        used = total - free

        psutil_swap = psutil.swap_memory()
        assert psutil_swap.total == total
        assert psutil_swap.used == used
        assert psutil_swap.free == free

    def test_cpu_count(self):
        out = sh("/usr/sbin/psrinfo")
        assert psutil.cpu_count() == len(out.split('\n'))
