"""Tests for distutils._msvccompiler."""

import os
import sys
import threading
import unittest.mock as mock
from distutils import _msvccompiler
from distutils.errors import DistutilsPlatformError
from distutils.tests import support

import pytest

needs_winreg = pytest.mark.skipif('not hasattr(_msvccompiler, "winreg")')


class Testmsvccompiler(support.TempdirManager):
    def test_no_compiler(self):
        # makes sure query_vcvarsall raises
        # a DistutilsPlatformError if the compiler
        # is not found
        def _find_vcvarsall(plat_spec):
            return None, None

        old_find_vcvarsall = _msvccompiler._find_vcvarsall
        _msvccompiler._find_vcvarsall = _find_vcvarsall
        try:
            with pytest.raises(DistutilsPlatformError):
                _msvccompiler._get_vc_env(
                    'wont find this version',
                )
        finally:
            _msvccompiler._find_vcvarsall = old_find_vcvarsall

    @needs_winreg
    def test_get_vc_env_unicode(self):
        test_var = 'ṰḖṤṪ┅ṼẨṜ'
        test_value = '₃⁴₅'

        # Ensure we don't early exit from _get_vc_env
        old_distutils_use_sdk = os.environ.pop('DISTUTILS_USE_SDK', None)
        os.environ[test_var] = test_value
        try:
            env = _msvccompiler._get_vc_env('x86')
            assert test_var.lower() in env
            assert test_value == env[test_var.lower()]
        finally:
            os.environ.pop(test_var)
            if old_distutils_use_sdk:
                os.environ['DISTUTILS_USE_SDK'] = old_distutils_use_sdk

    @needs_winreg
    @pytest.mark.parametrize('ver', (2015, 2017))
    def test_get_vc(self, ver):
        # This function cannot be mocked, so pass if VC is found
        # and skip otherwise.
        lookup = getattr(_msvccompiler, f'_find_vc{ver}')
        expected_version = {2015: 14, 2017: 15}[ver]
        version, path = lookup()
        if not version:
            pytest.skip(f"VS {ver} is not installed")
        assert version >= expected_version
        assert os.path.isdir(path)


class CheckThread(threading.Thread):
    exc_info = None

    def run(self):
        try:
            super().run()
        except Exception:
            self.exc_info = sys.exc_info()

    def __bool__(self):
        return not self.exc_info


class TestSpawn:
    def test_concurrent_safe(self):
        """
        Concurrent calls to spawn should have consistent results.
        """
        compiler = _msvccompiler.MSVCCompiler()
        compiler._paths = "expected"
        inner_cmd = 'import os; assert os.environ["PATH"] == "expected"'
        command = [sys.executable, '-c', inner_cmd]

        threads = [
            CheckThread(target=compiler.spawn, args=[command]) for n in range(100)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(threads)

    def test_concurrent_safe_fallback(self):
        """
        If CCompiler.spawn has been monkey-patched without support
        for an env, it should still execute.
        """
        from distutils import ccompiler

        compiler = _msvccompiler.MSVCCompiler()
        compiler._paths = "expected"

        def CCompiler_spawn(self, cmd):
            "A spawn without an env argument."
            assert os.environ["PATH"] == "expected"

        with mock.patch.object(ccompiler.CCompiler, 'spawn', CCompiler_spawn):
            compiler.spawn(["n/a"])

        assert os.environ.get("PATH") != "expected"
