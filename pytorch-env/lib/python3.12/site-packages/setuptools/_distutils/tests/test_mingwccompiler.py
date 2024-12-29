import pytest

from distutils.util import split_quoted, is_mingw
from distutils.errors import DistutilsPlatformError, CCompilerError
from distutils import sysconfig


class TestMingw32CCompiler:
    @pytest.mark.skipif(not is_mingw(), reason='not on mingw')
    def test_compiler_type(self):
        from distutils.cygwinccompiler import Mingw32CCompiler

        compiler = Mingw32CCompiler()
        assert compiler.compiler_type == 'mingw32'

    @pytest.mark.skipif(not is_mingw(), reason='not on mingw')
    def test_set_executables(self, monkeypatch):
        from distutils.cygwinccompiler import Mingw32CCompiler

        monkeypatch.setenv('CC', 'cc')
        monkeypatch.setenv('CXX', 'c++')

        compiler = Mingw32CCompiler()

        assert compiler.compiler == split_quoted('cc -O -Wall')
        assert compiler.compiler_so == split_quoted('cc -shared -O -Wall')
        assert compiler.compiler_cxx == split_quoted('c++ -O -Wall')
        assert compiler.linker_exe == split_quoted('cc')
        assert compiler.linker_so == split_quoted('cc -shared')

    @pytest.mark.skipif(not is_mingw(), reason='not on mingw')
    def test_runtime_library_dir_option(self):
        from distutils.cygwinccompiler import Mingw32CCompiler

        compiler = Mingw32CCompiler()
        with pytest.raises(DistutilsPlatformError):
            compiler.runtime_library_dir_option('/usr/lib')

    @pytest.mark.skipif(not is_mingw(), reason='not on mingw')
    def test_cygwincc_error(self, monkeypatch):
        import distutils.cygwinccompiler

        monkeypatch.setattr(distutils.cygwinccompiler, 'is_cygwincc', lambda _: True)

        with pytest.raises(CCompilerError):
            distutils.cygwinccompiler.Mingw32CCompiler()

    def test_customize_compiler_with_msvc_python(self):
        from distutils.cygwinccompiler import Mingw32CCompiler

        # In case we have an MSVC Python build, but still want to use
        # Mingw32CCompiler, then customize_compiler() shouldn't fail at least.
        # https://github.com/pypa/setuptools/issues/4456
        compiler = Mingw32CCompiler()
        sysconfig.customize_compiler(compiler)
