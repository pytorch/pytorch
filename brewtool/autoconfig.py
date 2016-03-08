"""autoconfig tries to figure out a bunch of platform-specific stuff.
"""
import atexit
import os
import re
import shlex
import shutil
import subprocess
import sysconfig
import tempfile
import uuid

from brewtool.logging import *

_SCRATCH_DIRECTORY = tempfile.mkdtemp()


def CleanScratch():
    shutil.rmtree(_SCRATCH_DIRECTORY)

atexit.register(CleanScratch)

def _TestFilename(name):
    """Get the internal test file name as its absolute path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _TempFilename(ext=''):
    return os.path.join(_SCRATCH_DIRECTORY, str(uuid.uuid4()) + ext)

def GetSubprocessOutput(command, env):
    """Runs a subprocess command, and tries to get the result. Return None if the
    command fails.

    Args:
        command: a list containing the command to run, such as
            ['rm', '-rf', '/'].

    Returns:
        return_code: the return code of the process. if an exception happens,
            the return code is -1.
        output: the command output. None if the command cannot be run.
    """
    if type(command) is str:
        command = shlex.split(command)
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env)
        out, err = proc.communicate()
        out = out.decode('ascii')
    except OSError as e:
        BuildDebug('Exception found in command {0}. Exception is: {1}',
                   repr(command), str(e))
        return -1, None
    return proc.returncode, out.strip()


def GetGitBuildString(env):
    ret, out = GetSubprocessOutput(['git', 'describe', '--always'], env)
    if ret:
        # Cannot really run git stuff. Will simply say unknown.
        return 'unknown_git_version'
    ret, head_status = GetSubprocessOutput(['git', 'status', '--porcelain'], env)
    out = str(out)
    if ret:
        # For some reason we cannot get git status porcelain.
        return out + ',unknown_head_status'
    else:
        if len(head_status):
            return out + ',with_local_changes'
        else:
            return out + ',clean_head'

def GetCpp11Flag(cc, env):
    """Gets the flag for c++11 for the current compiler."""
    BuildDebug("Trying to figure out the C++11 flag for the compiler.")
    command = [cc, "-std=c++11", _TestFilename("test.cc"),
               "-o", _TempFilename(".out")]
    ret, _ = GetSubprocessOutput(command, env)
    if ret == 0:
        return "-std=c++11"
    else:
        BuildDebug("Compiler does not support -std=c++11. Testing -std=c++0x.")
        command[1] = "-std=c++0x"
        ret, _ = GetSubprocessOutput(command, env)
        if ret == 0:
            return "-std=c++0x"
        else:
            BuildFatal("Cannot figure out the C++11 flag for compiler {0}."
                       "Does it support C++11?", cc)


def NeedLibrtOrNot(cc, env):
    command = [cc, _TestFilename("test.cc"), "-o", _TempFilename(".out"),
               "-lrt"]
    ret, _ = GetSubprocessOutput(command, env)
    if ret == 0:
        BuildDebug("The linker seem to have librt present.")
        return True
    else:
        BuildDebug("The linker cannot find librt. This is expected on e.g. "
                   "Android platforms. But double check if you think librt "
                   "should be present.")
        return False


def GetCompilerType(cc, env):
    _, out = GetSubprocessOutput([cc, '--version'], env)
    if 'clang' in out:
        # This is clang.
        return 'clang'
    elif ('g++' in out or 'Free Software Foundation' in out):
        # This is g++.
        return 'g++'
    else:
        # Return unknown and let upstream caller figure out.
        return 'unknown'


def GetWholeArchiveTemplate(cc, env):
    compiler_type = GetCompilerType(cc, env)
    if compiler_type == 'clang':
        return '-Wl,-force_load,{src}'
    elif compiler_type == 'g++':
        return '-Wl,--whole-archive {src} -Wl,--no-whole-archive'
    else:
        BuildFatal("I don't know the compiler type (got {0} for {1}). "
                   "Cannot set whole archive template.", compiler_type, cc)


def GetRpathTemplate(cc, env):
    compiler_type = GetCompilerType(cc, env)
    if compiler_type == 'clang':
        return '-rpath {path}'
    elif compiler_type == 'g++':
        return '-Wl,-rpath={path}'
    else:
        BuildFatal("I don't know the compiler type (got {0} for {1}). "
                   "Cannot determine rpath template.", compiler_type, cc)


class Env(object):
    def __init__(self, Config):
        """Figures out the invoking details for a bunch of stuff."""
        if not Config.GENDIR.startswith('gen'):
            BuildFatal('The generated directory does not start with gen: '
                       'Currently the build system hard-codes this prefix. '
                       'Please consider renaming the GENDIR to something like '
                       '{0}', 'gen-' + Config.GENDIR)
        self.GENDIR = os.path.abspath(Config.GENDIR)
        self.Config = Config

        self.ENV = dict(os.environ)
        for key in Config.ENVIRONMENTAL_VARS:
            self.ENV[key] = Config.ENVIRONMENTAL_VARS[key]
        if 'PYTHONPATH' not in self.ENV:
            self.ENV['PYTHONPATH'] = ''
        self.ENV['PYTHONPATH'] = '{0}:{1}:{2}'.format(
            os.path.join(self.GENDIR, "third_party"), self.GENDIR,
            self.ENV['PYTHONPATH'])
        self.ENV['CAFFE2_GENDIR'] = os.path.abspath(self.GENDIR)
        self.ENV['CAFFE2_SRCDIR'] = os.path.abspath('.')

        SetVerboseLogging(Config.VERBOSE_BUILD)

        # Default values, with things to be added incrementally.
        self.DEFINES = Config.DEFINES + [
            "-DGTEST_USE_OWN_TR1_TUPLE=1",  # Needed by gtest
            "-DEIGEN_NO_DEBUG",  # So that Eigen is in optimized mode.
            "-DPIC",  # Position independent code
        ]

        # Add a macro for the caffe2 build message.
        self.DEFINES.append(
            r'-DCAFFE2_BUILD_STRING=\"' + GetGitBuildString(self.ENV) + r'\"')
        self.CFLAGS = Config.CFLAGS + [
            '-fPIC',
            '-ffast-math',
            '-pthread',
            '-Wall',
            '-Wextra',
            '-Wno-unused-parameter',  # needed by some third_party code
            '-Wno-sign-compare',  # needed by some third_party code
            '-ffunction-sections',
            '-fdata-sections',
        ] + Config.OPTIMIZATION_FLAGS
        self.INCLUDES = Config.INCLUDES + [
            self.GENDIR,
            os.path.join(self.GENDIR, 'third_party'),
            os.path.join(self.GENDIR, 'third_party', 'include'),
        ]
        self.LIBDIRS = Config.LIBDIRS + []
        self.LINKFLAGS = Config.LINKFLAGS + [
            '-pthread',
        ]
        if GetCompilerType(Config.CC, self.ENV) == 'g++':
            self.LINKFLAGS.append('-Wl,--gc-sections')
        self.LIBS = Config.ADDITIONAL_LIBS
        self.SHARED_LIB_EXT = (
            Config.SHARED_LIB_EXT if len(Config.SHARED_LIB_EXT) else ".so")

        self.NVCC_CFLAGS = Config.CUDA_CFLAGS + ['-std=c++11'] + [
            '-gencode ' + s for s in Config.CUDA_GENCODE
        ] + Config.OPTIMIZATION_FLAGS

        # Check BLAS
        if Config.BLAS_BACKEND == 'eigen':
          self.DEFINES.append('-DCAFFE2_USE_EIGEN_FOR_BLAS')
        elif Config.BLAS_BACKEND == "mkl":
            self.DEFINES.append('-DCAFFE2_USE_MKL')

        # Set C++11 flag. The reason we do not simply add it to the CFLAGS list
        # above is that NVCC cannot pass -std=c++11 via Xcompiler, otherwise the
        # host compiler usually produces an error (clang) or warning (g++)
        # complaining that the code does not work well
        self.CPP11_FLAG = GetCpp11Flag(Config.CC, self.ENV)
        # Determine if we need librt: for example, Android does not need it.
        self.NEED_LIBRT = NeedLibrtOrNot(Config.CC, self.ENV)

        # Third party flags.
        if not Config.USE_SYSTEM_PROTOBUF:
            self.DEFINES.append("-DHAVE_PTHREAD")
        if Config.USE_GLOG:
            # If we are building with GLOG, enable the glog macro.
            self.DEFINES.append("-DCAFFE2_USE_GOOGLE_GLOG")

        # Proto-lite or Proto.
        if Config.USE_LITE_PROTO:
            self.DEFINES.append("-DCAFFE2_USE_LITE_PROTO")

        # RTTI
        if not Config.USE_RTTI:
            # If we are building protobuf, disable RTTI.
            self.DEFINES.append("-DGOOGLE_PROTOBUF_NO_RTTI")
            self.CFLAGS.append("-fno-rtti")

        # OpenMP
        if Config.USE_OPENMP:
            self.CFLAGS.append("-fopenmp")
            self.LIBS.append("gomp")

        # MPI
        self.MPIRUN = Config.MPIRUN
        ret, out = GetSubprocessOutput(
            [Config.MPICC, '--showme:incdirs'], self.ENV)
        if ret == 0:
            BuildDebug("Adding MPI-specific includes and library directories.")
            self.INCLUDES += out.split(' ')
            _, out = GetSubprocessOutput(
                [Config.MPICC, '--showme:libdirs'], self.ENV)
            mpi_libdirs = out.split(' ')
            if Config.MPI_ADD_TO_RPATH:
                rpath_template = GetRpathTemplate(Config.CC, self.ENV)
                self.LINKFLAGS += [
                    rpath_template.format(path=p) for p in mpi_libdirs]
            self.LIBDIRS += mpi_libdirs
            _, out = GetSubprocessOutput(
                [Config.MPICC, '--showme:libs'], self.ENV)
            self.MPI_LIBS = out.split(' ')
            if Config.FORCE_FALLBACK_CUDA_MPI:
                self.DEFINES.append('-DCAFFE2_FORCE_FALLBACK_CUDA_MPI')
            else:
                # Try to figure out if mpi has cuda support
                ret, out = GetSubprocessOutput(
                    [Config.OMPI_INFO, '--parsable', '--all'], self.ENV)
                if ret == 0:
                    OMPI_CUDA_SUPPORT = [
                        r for r in out.split('\n')
                        if 'mpi_built_with_cuda_support:value' in r]
                    if (len(OMPI_CUDA_SUPPORT) == 0 or
                            (len(OMPI_CUDA_SUPPORT) == 1 and
                             OMPI_CUDA_SUPPORT[0][-5:] == 'false')):
                        BuildWarning(
                            "Your openmpi binary does not seem to have CUDA "
                            "support enabled. Thus, we will disable MPI calls "
                            "with CUDA DMA.")
                        self.DEFINES.append('-DCAFFE2_FORCE_FALLBACK_CUDA_MPI')
                else:
                    # Cannot figure out mpi configuration via ompi_info. In this
                    # case, we will simply assume that MPI cuda support is
                    # present.
                    BuildWarning("I cannot determine if the current MPI "
                                 "supports CUDA. Assuming yes.")
        else:
            BuildWarning(
                'MPI not found, so some libraries and binaries that use MPI '
                'will not compile correctly. If you would like to use those, '
                'you need to install MPI on your machine. The easiest way to '
                'install on ubuntu is via apt-get, and on mac via homebrew.')
            self.MPI_LIBS = []

        # CUDA
        # Check NVCC version
        NVCC = os.path.join(Config.CUDA_DIR, "bin", "nvcc")
        ret, out = GetSubprocessOutput([NVCC, "--version"], self.ENV)
        if ret != 0:
            BuildWarning("NVCC not found. I will not build CUDA.")
            # When nvcc is not present, do not build pycaffe2 with cuda.
            self.DEFINES.append('-DPYCAFFE2_CPU_ONLY')
        else:
            # Figure out all CUDA details.
            version = re.search('release \d', out)
            if version is None or int(version.group()[-1]) < 7:
                BuildWarning("Caffe2 is only tested against CUDA 7 and above. "
                             "For any low versions, it may or may not compile.")
            # Add the include and lib directories
            self.INCLUDES.append(os.path.join(Config.CUDA_DIR, "include"))
            if len(Config.MANUAL_CUDA_LIB_DIRS):
                cuda_dirs = Config.MANUAL_CUDA_LIB_DIRS
            else:
                cuda_dirs = [os.path.join(Config.CUDA_DIR, "lib"),
                             os.path.join(Config.CUDA_DIR, "lib64")]
            self.LIBDIRS += cuda_dirs
            if Config.CUDA_ADD_TO_RPATH:
                rpath_template = GetRpathTemplate(Config.CC, self.ENV)
                self.LINKFLAGS += [
                    rpath_template.format(path=p) for p in cuda_dirs]

        # Python
        # Caffe2 requires numpy. We will try to find the numpy header files.
        try:
            import numpy.distutils
            numpy_includes = numpy.distutils.misc_util.get_numpy_include_dirs()
        except Exception as e:
            BuildWarning("Cannot find numpy. Pycaffe2 based code will not "
                         "compile correctly. Error is: {0}.", str(e))
            numpy_includes = []
        self.PYTHON_CFLAGS = (
            [s for s in sysconfig.get_config_var('CFLAGS').split(' ') if s] +
            ['-I' + s for s in sysconfig.get_config_var('INCLUDEPY').split(' ')] +
            ['-I' + s for s in numpy_includes])
        self.PYTHON_LDFLAGS = [
            s for s in sysconfig.get_config_var('LDFLAGS').split(' ') if s]
        # Add the actual target
        python_target = sysconfig.get_config_var('LDLIBRARY')
        if python_target[:3] != 'lib':
            BuildWarning('Cannot understand python library. Pycaffe2 may not '
                         'compile correctly.')
        python_target = python_target[3:]
        python_target = python_target[:python_target.rfind('.')]
        self.PYTHON_LIBS = [python_target]


        # Now, after all the above commands, we will assemble the templates used
        # for all the commands.
        self.TEMPLATE_PROTOC = ' '.join(
            [Config.PROTOC_BINARY, '-I' + self.GENDIR, '--cpp_out', self.GENDIR,
             '--python_out', self.GENDIR, '{src}'])
        self.TEMPLATE_CC = ' '.join(
            [Config.CC] + self.DEFINES + self.CFLAGS + [self.CPP11_FLAG] +
            ['-I' + s for s in self.INCLUDES] +
            ['-c', '{src}', '-o', '{dst}'])
        self.TEMPLATE_LINK_STATIC = ' '.join(
            [Config.AR, 'cr', '{dst}', '{src}'])
        self.TEMPLATE_LINK_SHARED = ' '.join(
            [Config.CC, '-shared', '-o', '{dst}'] + self.LINKFLAGS +
            ['-L' + s for s in self.LIBDIRS] + ['{src}'] +
            ['-l' + s for s in self.LIBS])
        self.TEMPLATE_LINK_BINARY = ' '.join(
            [Config.CC, '-o', '{dst}'] + self.LINKFLAGS +
            ['-L' + s for s in self.LIBDIRS] + ['{src}'] +
            ['-l' + s for s in self.LIBS])
        self.TEMPLATE_CC_TEST = ' '.join(
            ['{src}', '--caffe_test_root=' + os.path.abspath(self.GENDIR),
             '--gtest_filter=-*.LARGE_*'])
        self.TEMPLATE_NVCC = ' '.join(
            [NVCC, '-ccbin', Config.CC] + self.NVCC_CFLAGS +
            ['-Xcompiler', '"' + ' '.join(self.CFLAGS) + '"'] +
            ['-I' + s for s in self.INCLUDES] +
            self.DEFINES + ['-c', '{src}', '-o', '{dst}'])
        self.TEMPLATE_WHOLE_ARCHIVE = GetWholeArchiveTemplate(
            Config.CC, self.ENV)
        self.TEMPLATE_PYEXT_CC = ' '.join(
            [Config.CC] + self.DEFINES + self.CFLAGS + [self.CPP11_FLAG] +
            self.PYTHON_CFLAGS + ['-I' + s for s in self.INCLUDES] +
            ['-c', '{src}', '-o', '{dst}'])
        self.TEMPLATE_PYEXT_LINK = ' '.join(
            [Config.CC, '-shared', '-o', '{dst}'] + self.LINKFLAGS +
            self.PYTHON_LDFLAGS +
            ['-L' + s for s in self.LIBDIRS] + ['{src}'] +
            ['-l' + s for s in self.LIBS] +
            ['-l' + s for s in self.PYTHON_LIBS])
        # After all are done, we print out the Env setting.
        BuildDebug("Finished autoconfiguring the build environment.")

    def _format(self, template, src, dst=''):
        if type(src) is list:
            src = ' '.join(src)
        if type(dst) is list:
            dst = ' '.join(dst)
        return template.format(src=src, dst=dst)

    def protoc(self, src):
        return self._format(self.TEMPLATE_PROTOC, src)

    def cc(self, src, dst):
        return self._format(self.TEMPLATE_CC, src, dst)

    def link_static(self, src, dst):
        return self._format(self.TEMPLATE_LINK_STATIC, src, dst)

    def link_shared(self, src, dst):
        return self._format(self.TEMPLATE_LINK_SHARED, src, dst)

    def link_binary(self, src, dst):
        return self._format(self.TEMPLATE_LINK_BINARY, src, dst)

    def cc_test(self, src):
        return self._format(self.TEMPLATE_CC_TEST, src)

    def nvcc(self, src, dst):
        return self._format(self.TEMPLATE_NVCC, src, dst)

    def whole_archive(self, src):
        return self._format(self.TEMPLATE_WHOLE_ARCHIVE, src)

    def pyext_cc(self, src, dst):
        return self._format(self.TEMPLATE_PYEXT_CC, src, dst)

    def pyext_link(self, src, dst):
        return self._format(self.TEMPLATE_PYEXT_LINK, src, dst)
