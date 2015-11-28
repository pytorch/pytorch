"""Configuration for the Caffe2 installation targeted for Android.
"""


STANDALONE_TCHAIN_ROOT = '/home/jiayq/NVPACK/android-ndk-r10e/toolchains/arm-linux-androideabi-4.6/gen_standalone/linux-x86_64/'


class Config(object):
    VERBOSE_BUILD = True
    # Specify your compiler.
    CC = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-g++'
    # Specify your archiver.
    AR = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-ar'
    # Specify your output folder.
    GENDIR = "gen-android"

    USE_SYSTEM_PROTOBUF = False
    PROTOC_BINARY = 'gen/third_party/google/protoc'
    # Note for the line above: if you are doing things like cross-compilation,
    # the built protoc compiler will not work on the host, in which case you
    # will need to provide a protoc binary that can run on the host environment.

    # Eigen: Eigen is a third party library that Caffe2 uses for some numerical
    # operations. If you have eigen installed in your system, you can simply use
    # USE_SYSTEM_EIGEN = True. Otherwise (for example when you are cross
    # compiling) you may want to set USE_SYSTEM_EIGEN to False.
    USE_SYSTEM_EIGEN = False

    # google-glog: Caffe can choose to use google glog, which will allow a more
    # sophisticated logging scheme. It also comes with a minimal logging tool
    # that does not depend on glog. If you wish to use glog, set USE_GLOG to
    # True.
    USE_GLOG = False

    # Manually specified defines.
    DEFINES = [
        "-DANDROID",
    ]

    # Manually specified include paths. These include paths are searched before
    # any auto-generated include paths.
    INCLUDES = []

    # Manually specified lib directories. These are searched before any
    # auto-generated library directories.
    LIBDIRS = []

    # Additional cflags you would like to add to the compilation.
    CFLAGS = []

    # Additional link flags you would like to add to the compilation.
    LINKFLAGS = []

    ###########################################################################
    # (optional) CUDA. If you do not specify this, the GPU part of Caffe2 will
    # not be available.
    ############################################################################
    # Specify the cuda directory.
    CUDA_DIR = "/home/jiayq/NVPACK/cuda-7.0"
    # If you are cross compiling, you may need to add paths where the cuda
    # libraries for the target platform can be found. Otherwise, leave it empty.
    MANUAL_CUDA_LIB_DIRS = []
    CUDA_GENCODE = [
        'arch=compute_32,code=sm_32',
    ]
    # additional CUDA cflags to pass to nvcc.
    CUDA_CFLAGS = ["-m32"],
    # You can choose to add the path of the cuda libraries to the rpath, so that
    # during runtime you do not need to hard-code the library paths. You can,
    # of course, set this to False.
    CUDA_ADD_TO_RPATH = False
    # Specify if you want to link cuda as static libraries.
    LINK_CUDA_STATIC = True

    ############################################################################
    # (optional) MPI setting.
    ############################################################################
    # Specify the MPI c++ compiler. You usually don't need to change this.
    MPICC = "non-existing"
    MPIRUN = "non-existing"
    # Specify ompi_info if you are using openmpi.
    OMPI_INFO = "non-existing"
    # Now, the cuda MPI suport is available after certain versions (such as
    # OpenMPI 1.7), but it is possible that the MPI is built without cuda
    # support. We will try to figure out if cuda support is available, but
    # sometimes you may need to manually request MPI operators to go to
    # "fallback" mode: in which case the MPI operations are carried out by CUDA
    # memory copy followed by MPI in the CPU space.
    FORCE_FALLBACK_CUDA_MPI = False
    # Whether to add the MPI library to rpath.
    MPI_ADD_TO_RPATH = False

    ################################################################################
    # (optional) Python.
    ################################################################################
    # Specify the python config command.
    PYTHON_CONFIG = "non-existing"

    ################################################################################
    # Very rarely used configurations.
    ################################################################################
    # If the platform uses a non-conventional shared library extension, manually
    # specify it here.
    SHARED_LIB_EXT = ''
    # If you would like to pass in any specific environmental variables to the
    # build command, do it here.
    ENVIRONMENTAL_VARS = {}
    # Optimization flags: -O2 in default.
    OPTIMIZATION_FLAGS = ["-O2"]


# brew.py
if __name__ == '__main__':
    from brewtool.brewery import Brewery
    import sys
    Brewery.Run(Config, sys.argv)
