"""Configuration for the Caffe2 installation.
"""


class Config(object):
    ############################################################################
    # Common settings that are necessary to build Caffe2's core functionality.
    ############################################################################
    # If you want to show a lot of the build details, set VERBOSE_BUILD to True.
    # This will show the detailed commands being run during the build process.
    VERBOSE_BUILD = True
    # Specify your compiler.
    CC = "c++"
    # Specify your archiver.
    AR = "ar"
    # Specify your output folder.
    GENDIR = "gen"

    # Specify if you want to use the system protocol buffer or not.
    # If you have protobuf installed, use the following two lines usually
    # suffice:
    USE_SYSTEM_PROTOBUF = True
    PROTOC_BINARY = "protoc"
    # Otherwise, use the following line: we will build protobuf using the
    # included source file.
    #USE_SYSTEM_PROTOBUF = False
    #PROTOC_BINARY = 'gen/third_party/protoc'
    # Note for the line above: if you are doing things like cross-compilation,
    # the built protoc compiler will not work on the host, in which case you
    # will need to provide a protoc binary that can run on the host environment.

    # Choose if Caffe2 uses only lite proto or not. Note that this will turn off
    # quite a few capabilities in Caffe2's generated protobuf, notably the text
    # format support. But, if you are using lite proto already, you don't want
    # text format anyway, do you?
    USE_LITE_PROTO = False

    # Eigen: Eigen is a third party library that Caffe2 uses for some numerical
    # operations. If you have eigen installed in your system, you can simply use
    # USE_SYSTEM_EIGEN = True. Otherwise (for example when you are cross
    # compiling) you may want to set USE_SYSTEM_EIGEN to False.
    USE_SYSTEM_EIGEN = False

    # BLAS backend: which backend to use for blas functions.
    # Note that, if the BLAS backend is MKL, we will also assume that the
    # MKL VSL library is present, and we will use the VSL function calls as
    # well.
    # Also note that, if the BLAS backend is eigen, there is actually *no*
    # actual blas function calls. We only routed the caffe-specific functions
    # to use Eigen.
    BLAS_BACKEND = "eigen"

    # google-glog: Caffe can choose to use google glog, which will allow a more
    # sophisticated logging scheme. It also comes with a minimal logging tool
    # that does not depend on glog. If you wish to use glog, set USE_GLOG to
    # True.
    USE_GLOG = True

    # gflags: Caffe can choose to use google glog, which will allow a more
    # feature complete flags registration mechanism. If you wish to use gflags,
    # set USE_GFLAGS to True.
    USE_GFLAGS = True

    # Whether to use RTTI or not. Note that this might not always work; to
    # disable RTTI means that all your dependencies, most notably protobuf, have
    # to be built without RTTI. If you don't know, leave USE_RTTI True.
    USE_RTTI = True

    # Whether to use openmp or not. Note that currently, a lot of Caffe2's code
    # is not using openmp, but the underlying Eigen library can take advantage
    # of that.
    USE_OPENMP = True

    # Manually specified defines.
    DEFINES = ["-DNDEBUG"]

    # Manually specified include paths. These include paths are searched before
    # any auto-generated include paths.
    INCLUDES = []

    # Manually specified lib directories. These are searched before any
    # auto-generated library directories.
    LIBDIRS = []

    # Additional cflags you would like to add to the compilation.
    CFLAGS = []
    # If you have a nice CPU, you can enable several intrinsics. Make sure you know
    # that these are available on your CPU though, otherwise you will get illegal
    # instruction errors.
    #CFLAGS = ["-mavx", "-mavx2", "-mfma"]

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
    CUDA_DIR = "/usr/local/cuda"
    # If you are cross compiling, you may need to add paths where the cuda
    # libraries for the target platform can be found. Otherwise, leave it empty.
    MANUAL_CUDA_LIB_DIRS = []
    CUDA_GENCODE = [
        'arch=compute_30,code=sm_30',
        'arch=compute_35,code=sm_35',
        'arch=compute_50,code=sm_50',
        'arch=compute_61,code=sm_61',
    ]
    # additional CUDA cflags to pass to nvcc.
    CUDA_CFLAGS = []

    # You can choose to add the path of the cuda libraries to the rpath, so that
    # during runtime you do not need to hard-code the library paths. You can,
    # of course, set this to False.
    CUDA_ADD_TO_RPATH = True
    # Specify if you want to link cuda as static libraries.
    LINK_CUDA_STATIC = True

    ############################################################################
    # (optional) MPI setting.
    ############################################################################
    # Specify the MPI c++ compiler. You usually don't need to change this.
    MPICC = "mpic++"
    MPIRUN = "mpirun"
    # Specify ompi_info if you are using openmpi.
    OMPI_INFO = 'ompi_info'
    # Now, the cuda MPI suport is available after certain versions (such as
    # OpenMPI 1.7), but it is possible that the MPI is built without cuda
    # support. We will try to figure out if cuda support is available, but
    # sometimes you may need to manually request MPI operators to go to
    # "fallback" mode: in which case the MPI operations are carried out by CUDA
    # memory copy followed by MPI in the CPU space.
    FORCE_FALLBACK_CUDA_MPI = False
    # Whether to add the MPI library to rpath.
    MPI_ADD_TO_RPATH = True

    ################################################################################
    # Very rarely used configurations.
    ################################################################################
    # If the platform uses a non-conventional shared library extension, manually
    # specify it here.
    SHARED_LIB_EXT = ''
    # If you would like to pass in any specific environmental variables to the
    # build command, do it here.
    ENVIRONMENTAL_VARS = {}
    # Optimization flags: -O2 in default. The reason we do not include it
    # directly in the CFLAGS option is because it will be inserted to both
    # c++ and nvcc: some cflags may not be compatible with nvcc so we do not
    # want to put all cflags into nvcc.
    OPTIMIZATION_FLAGS = ["-O2"]


# brew.py
if __name__ == '__main__':
    from brewtool.brewery import Brewery
    import sys
    Brewery.Run(Config, sys.argv)
