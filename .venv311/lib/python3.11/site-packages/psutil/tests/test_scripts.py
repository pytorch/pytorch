#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Test various scripts."""

import ast
import os
import shutil
import stat
import subprocess

import pytest

from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import ROOT_DIR
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import import_module_by_path
from psutil.tests import psutil
from psutil.tests import sh

INTERNAL_SCRIPTS_DIR = os.path.join(SCRIPTS_DIR, "internal")
SETUP_PY = os.path.join(ROOT_DIR, 'setup.py')


# ===================================================================
# --- Tests scripts in scripts/ directory
# ===================================================================


@pytest.mark.skipif(
    CI_TESTING and not os.path.exists(SCRIPTS_DIR),
    reason="can't find scripts/ directory",
)
class TestExampleScripts(PsutilTestCase):
    @staticmethod
    def assert_stdout(exe, *args):
        env = PYTHON_EXE_ENV.copy()
        env.pop("PSUTIL_DEBUG")  # avoid spamming to stderr
        exe = os.path.join(SCRIPTS_DIR, exe)
        cmd = [PYTHON_EXE, exe, *args]
        try:
            out = sh(cmd, env=env).strip()
        except RuntimeError as err:
            if 'AccessDenied' in str(err):
                return str(err)
            else:
                raise
        assert out, out
        return out

    @staticmethod
    def assert_syntax(exe):
        exe = os.path.join(SCRIPTS_DIR, exe)
        with open(exe, encoding="utf8") as f:
            src = f.read()
        ast.parse(src)

    def test_coverage(self):
        # make sure all example scripts have a test method defined
        meths = dir(self)
        for name in os.listdir(SCRIPTS_DIR):
            if name.endswith('.py'):
                if 'test_' + os.path.splitext(name)[0] not in meths:
                    # self.assert_stdout(name)
                    return pytest.fail(
                        "no test defined for"
                        f" {os.path.join(SCRIPTS_DIR, name)!r} script"
                    )

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_executable(self):
        for root, dirs, files in os.walk(SCRIPTS_DIR):
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    if not stat.S_IXUSR & os.stat(path)[stat.ST_MODE]:
                        return pytest.fail(f"{path!r} is not executable")

    def test_disk_usage(self):
        self.assert_stdout('disk_usage.py')

    def test_free(self):
        self.assert_stdout('free.py')

    def test_meminfo(self):
        self.assert_stdout('meminfo.py')

    def test_procinfo(self):
        self.assert_stdout('procinfo.py', str(os.getpid()))

    @pytest.mark.skipif(CI_TESTING and not psutil.users(), reason="no users")
    def test_who(self):
        self.assert_stdout('who.py')

    def test_ps(self):
        self.assert_stdout('ps.py')

    def test_pstree(self):
        self.assert_stdout('pstree.py')

    def test_netstat(self):
        self.assert_stdout('netstat.py')

    def test_ifconfig(self):
        self.assert_stdout('ifconfig.py')

    @pytest.mark.skipif(not HAS_MEMORY_MAPS, reason="not supported")
    def test_pmap(self):
        self.assert_stdout('pmap.py', str(os.getpid()))

    def test_procsmem(self):
        if 'uss' not in psutil.Process().memory_full_info()._fields:
            return pytest.skip("not supported")
        self.assert_stdout('procsmem.py')

    def test_killall(self):
        self.assert_syntax('killall.py')

    def test_nettop(self):
        self.assert_syntax('nettop.py')

    def test_top(self):
        self.assert_syntax('top.py')

    def test_iotop(self):
        self.assert_syntax('iotop.py')

    def test_pidof(self):
        output = self.assert_stdout('pidof.py', psutil.Process().name())
        assert str(os.getpid()) in output

    @pytest.mark.skipif(not WINDOWS, reason="WINDOWS only")
    def test_winservices(self):
        self.assert_stdout('winservices.py')

    def test_cpu_distribution(self):
        self.assert_syntax('cpu_distribution.py')

    @pytest.mark.skipif(not HAS_SENSORS_TEMPERATURES, reason="not supported")
    def test_temperatures(self):
        if not psutil.sensors_temperatures():
            return pytest.skip("no temperatures")
        self.assert_stdout('temperatures.py')

    @pytest.mark.skipif(not HAS_SENSORS_FANS, reason="not supported")
    def test_fans(self):
        if not psutil.sensors_fans():
            return pytest.skip("no fans")
        self.assert_stdout('fans.py')

    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    @pytest.mark.skipif(not HAS_BATTERY, reason="no battery")
    def test_battery(self):
        self.assert_stdout('battery.py')

    @pytest.mark.skipif(not HAS_SENSORS_BATTERY, reason="not supported")
    @pytest.mark.skipif(not HAS_BATTERY, reason="no battery")
    def test_sensors(self):
        self.assert_stdout('sensors.py')


# ===================================================================
# --- Tests scripts in scripts/internal/ directory
# ===================================================================


@pytest.mark.skipif(
    CI_TESTING and not os.path.exists(INTERNAL_SCRIPTS_DIR),
    reason="can't find scripts/internal/ directory",
)
class TestInternalScripts(PsutilTestCase):
    @staticmethod
    def ls():
        for name in os.listdir(INTERNAL_SCRIPTS_DIR):
            if name.endswith(".py"):
                yield os.path.join(INTERNAL_SCRIPTS_DIR, name)

    def test_syntax_all(self):
        for path in self.ls():
            with open(path, encoding="utf8") as f:
                data = f.read()
            ast.parse(data)

    # don't care about other platforms, this is really just for myself
    @pytest.mark.skipif(not LINUX, reason="not on LINUX")
    @pytest.mark.skipif(CI_TESTING, reason="not on CI")
    def test_import_all(self):
        for path in self.ls():
            try:
                import_module_by_path(path)
            except SystemExit:
                pass


# ===================================================================
# --- Tests for setup.py script
# ===================================================================


@pytest.mark.skipif(
    CI_TESTING and not os.path.exists(SETUP_PY), reason="can't find setup.py"
)
class TestSetupScript(PsutilTestCase):
    def test_invocation(self):
        module = import_module_by_path(SETUP_PY)
        with pytest.raises(SystemExit):
            module.setup()
        assert module.get_version() == psutil.__version__

    @pytest.mark.skipif(
        not shutil.which("python2.7"), reason="python2.7 not installed"
    )
    def test_python2(self):
        # There's a duplicate of this test in scripts/internal
        # directory, which is only executed by CI. We replicate it here
        # to run it when developing locally.
        p = subprocess.Popen(
            [shutil.which("python2.7"), SETUP_PY],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = p.communicate()
        assert p.wait() == 1
        assert not stdout
        assert "psutil no longer supports Python 2.7" in stderr
        assert "Latest version supporting Python 2.7 is" in stderr
