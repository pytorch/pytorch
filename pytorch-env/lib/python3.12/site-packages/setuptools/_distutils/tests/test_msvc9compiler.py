"""Tests for distutils.msvc9compiler."""

import os
import sys
from distutils.errors import DistutilsPlatformError
from distutils.tests import support

import pytest

# A manifest with the only assembly reference being the msvcrt assembly, so
# should have the assembly completely stripped.  Note that although the
# assembly has a <security> reference the assembly is removed - that is
# currently a "feature", not a bug :)
_MANIFEST_WITH_ONLY_MSVC_REFERENCE = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1"
          manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false">
        </requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"
         version="9.0.21022.8" processorArchitecture="x86"
         publicKeyToken="XXXX">
      </assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>
"""

# A manifest with references to assemblies other than msvcrt.  When processed,
# this assembly should be returned with just the msvcrt part removed.
_MANIFEST_WITH_MULTIPLE_REFERENCES = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1"
          manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false">
        </requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"
         version="9.0.21022.8" processorArchitecture="x86"
         publicKeyToken="XXXX">
      </assemblyIdentity>
    </dependentAssembly>
  </dependency>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"
        version="9.0.21022.8" processorArchitecture="x86"
        publicKeyToken="XXXX"></assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>
"""

_CLEANED_MANIFEST = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1"
          manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false">
        </requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>

  </dependency>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"
        version="9.0.21022.8" processorArchitecture="x86"
        publicKeyToken="XXXX"></assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>"""

if sys.platform == "win32":
    from distutils.msvccompiler import get_build_version

    if get_build_version() >= 8.0:
        SKIP_MESSAGE = None
    else:
        SKIP_MESSAGE = "These tests are only for MSVC8.0 or above"
else:
    SKIP_MESSAGE = "These tests are only for win32"


@pytest.mark.skipif('SKIP_MESSAGE', reason=SKIP_MESSAGE)
class Testmsvc9compiler(support.TempdirManager):
    def test_no_compiler(self):
        # makes sure query_vcvarsall raises
        # a DistutilsPlatformError if the compiler
        # is not found
        from distutils.msvc9compiler import query_vcvarsall

        def _find_vcvarsall(version):
            return None

        from distutils import msvc9compiler

        old_find_vcvarsall = msvc9compiler.find_vcvarsall
        msvc9compiler.find_vcvarsall = _find_vcvarsall
        try:
            with pytest.raises(DistutilsPlatformError):
                query_vcvarsall('wont find this version')
        finally:
            msvc9compiler.find_vcvarsall = old_find_vcvarsall

    def test_reg_class(self):
        from distutils.msvc9compiler import Reg

        with pytest.raises(KeyError):
            Reg.get_value('xxx', 'xxx')

        # looking for values that should exist on all
        # windows registry versions.
        path = r'Control Panel\Desktop'
        v = Reg.get_value(path, 'dragfullwindows')
        assert v in ('0', '1', '2')

        import winreg

        HKCU = winreg.HKEY_CURRENT_USER
        keys = Reg.read_keys(HKCU, 'xxxx')
        assert keys is None

        keys = Reg.read_keys(HKCU, r'Control Panel')
        assert 'Desktop' in keys

    def test_remove_visual_c_ref(self):
        from distutils.msvc9compiler import MSVCCompiler

        tempdir = self.mkdtemp()
        manifest = os.path.join(tempdir, 'manifest')
        f = open(manifest, 'w')
        try:
            f.write(_MANIFEST_WITH_MULTIPLE_REFERENCES)
        finally:
            f.close()

        compiler = MSVCCompiler()
        compiler._remove_visual_c_ref(manifest)

        # see what we got
        f = open(manifest)
        try:
            # removing trailing spaces
            content = '\n'.join([line.rstrip() for line in f])
        finally:
            f.close()

        # makes sure the manifest was properly cleaned
        assert content == _CLEANED_MANIFEST

    def test_remove_entire_manifest(self):
        from distutils.msvc9compiler import MSVCCompiler

        tempdir = self.mkdtemp()
        manifest = os.path.join(tempdir, 'manifest')
        f = open(manifest, 'w')
        try:
            f.write(_MANIFEST_WITH_ONLY_MSVC_REFERENCE)
        finally:
            f.close()

        compiler = MSVCCompiler()
        got = compiler._remove_visual_c_ref(manifest)
        assert got is None
