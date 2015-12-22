"""Configuration for the Caffe2 installation targeted for Android.
"""


STANDALONE_TCHAIN_ROOT = (
    '/home/jiayq/NVPACK/android-ndk-r10e/toolchains/'
    'arm-linux-androideabi-4.6/gen_standalone/linux-x86_64/')


class Config(object):
    VERBOSE_BUILD = True
    # Specify your compiler.
    CC = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-g++'
    # Specify your archiver.
    AR = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-ar'
    # Specify your output folder.
    GENDIR = "gen-android"

    USE_SYSTEM_PROTOBUF = False
    PROTOC_BINARY = 'brewtool/prebuilt/protoc-Linux-x86_64'
    USE_LITE_PROTO = False

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

    # Whether to use RTTI or not. Note that this might not always work; to
    # disable RTTI means that all your dependencies, most notably protobuf, have
    # to be built without RTTI. If you don't know, leave USE_RTTI True.
    USE_RTTI = False

    # Whether to use openmp or not. Note that currently, a lot of Caffe2's code
    # is not using openmp, but the underlying Eigen library can take advantage
    # of that.
    USE_OPENMP = False

    # Manually specified defines.
    DEFINES = []

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

    # Additional libraries to link against. This will be appended to each link
    # link command.
    ADDITIONAL_LIBS = []

    ###########################################################################
    # (optional) CUDA. If you do not specify this, the GPU part of Caffe2 will
    # not be available.
    ############################################################################
    # Specify the cuda directory.
    CUDA_DIR = "/home/jiayq/NVPACK/cuda-6.5"
    # If you are cross compiling, you may need to add paths where the cuda
    # libraries for the target platform can be found. Otherwise, leave it empty.
    MANUAL_CUDA_LIB_DIRS = [
        "/home/jiayq/NVPACK/cuda-6.5/targets/armv7-linux-androideabi/lib"
    ]
    CUDA_GENCODE = [
        'arch=compute_32,code=sm_32',
    ]
    # additional CUDA cflags to pass to nvcc.
    CUDA_CFLAGS = [
        "-m32",
        "-target-cpu-arch=ARM",
        "-target-os-variant=Android",
        '-Xptxas -dlcm=ca',
    ]
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
    OPTIMIZATION_FLAGS = ["-Os"]


# brew.py
if __name__ == '__main__':
    from brewtool.brewery import Brewery
    import sys
    Brewery.Run(Config, sys.argv)
