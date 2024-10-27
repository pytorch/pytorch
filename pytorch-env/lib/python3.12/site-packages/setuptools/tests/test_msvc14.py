"""
Tests for msvc support module (msvc14 unit tests).
"""

import os
from distutils.errors import DistutilsPlatformError
import pytest
import sys


@pytest.mark.skipif(sys.platform != "win32", reason="These tests are only for win32")
class TestMSVC14:
    """Python 3.8 "distutils/tests/test_msvccompiler.py" backport"""

    def test_no_compiler(self):
        import setuptools.msvc as _msvccompiler

        # makes sure query_vcvarsall raises
        # a DistutilsPlatformError if the compiler
        # is not found

        def _find_vcvarsall(plat_spec):
            return None, None

        old_find_vcvarsall = _msvccompiler._msvc14_find_vcvarsall
        _msvccompiler._msvc14_find_vcvarsall = _find_vcvarsall
        try:
            pytest.raises(
                DistutilsPlatformError,
                _msvccompiler._msvc14_get_vc_env,
                'wont find this version',
            )
        finally:
            _msvccompiler._msvc14_find_vcvarsall = old_find_vcvarsall

    def test_get_vc_env_unicode(self):
        import setuptools.msvc as _msvccompiler

        test_var = 'ṰḖṤṪ┅ṼẨṜ'
        test_value = '₃⁴₅'

        # Ensure we don't early exit from _get_vc_env
        old_distutils_use_sdk = os.environ.pop('DISTUTILS_USE_SDK', None)
        os.environ[test_var] = test_value
        try:
            env = _msvccompiler._msvc14_get_vc_env('x86')
            assert test_var.lower() in env
            assert test_value == env[test_var.lower()]
        finally:
            os.environ.pop(test_var)
            if old_distutils_use_sdk:
                os.environ['DISTUTILS_USE_SDK'] = old_distutils_use_sdk

    def test_get_vc2017(self):
        import setuptools.msvc as _msvccompiler

        # This function cannot be mocked, so pass it if we find VS 2017
        # and mark it skipped if we do not.
        version, path = _msvccompiler._msvc14_find_vc2017()
        if os.environ.get('APPVEYOR_BUILD_WORKER_IMAGE', '') == 'Visual Studio 2017':
            assert version
        if version:
            assert version >= 15
            assert os.path.isdir(path)
        else:
            pytest.skip("VS 2017 is not installed")

    def test_get_vc2015(self):
        import setuptools.msvc as _msvccompiler

        # This function cannot be mocked, so pass it if we find VS 2015
        # and mark it skipped if we do not.
        version, path = _msvccompiler._msvc14_find_vc2015()
        if os.environ.get('APPVEYOR_BUILD_WORKER_IMAGE', '') in [
            'Visual Studio 2015',
            'Visual Studio 2017',
        ]:
            assert version
        if version:
            assert version >= 14
            assert os.path.isdir(path)
        else:
            pytest.skip("VS 2015 is not installed")
