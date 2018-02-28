# ---[ Declare variables that we are going to use across the Caffe2 build.
# This file defines common, Caffe2-wide variables that we use to collect
# source files and other things. Each variable is annotated with their
# intended uses.
# Note that adding and / or deleting these variables usually involves
# changing the whole build system, so make sure you send a PR early if you
# want to change them.

# Caffe2_{CPU,GPU}_SRCS is the list that will have all the related source
# files for CPU and GPU respectively. They will be filled with the
# CMakeLists.txt files under each folder respectively.
set(Caffe2_CPU_SRCS)
set(Caffe2_GPU_SRCS)

# Caffe2_{CPU,GPU}_TEST_SRCS is the list that will have all the related source
# files for CPU and GPU tests respectively.
set(Caffe2_CPU_TEST_SRCS)
set(Caffe2_GPU_TEST_SRCS)

# Caffe2_MAIN_LIBS is a list of the libraries that a dependent library should
# depend on when it links against Caffe2.
set(Caffe2_MAIN_LIBS)

# Lists for Caffe2 dependency libraries, for CPU and CUDA respectively.
set(Caffe2_DEPENDENCY_LIBS "")
set(Caffe2_CUDA_DEPENDENCY_LIBS "")

# Lists for Caffe2 public dependency libraries. These libraries will be
# transitive to any libraries that depends on Caffe2.
set(Caffe2_PUBLIC_DEPENDENCY_LIBS "")
set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS "")