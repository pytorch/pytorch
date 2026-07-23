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

# Caffe2_{CPU,GPU}_INCLUDE is the list that will have all the include
# directories for CPU and GPU respectively.
set(Caffe2_CPU_INCLUDE)
set(Caffe2_GPU_INCLUDE)

# Lists for Caffe2 dependency libraries, for CPU and CUDA respectively.
set(Caffe2_DEPENDENCY_LIBS "")
set(Caffe2_CUDA_DEPENDENCY_LIBS "")
# This variable contains dependency libraries of Caffe2 which requires whole
# symbol linkage. One example is the onnx lib where we need all its schema
# symbols. However, if the lib is whole linked in caffe2 lib, we don't want
# it to be linked in binaries that will link caffe2 lib. Because if caffe2 lib
# is built as dynamic library, it will result in two copied of symbols of
# Caffe2_DEPENDENCY_WHOLE_LINK_LIBS existing in caffe2.so and the binary, which
# will cause issues. Therefore Caffe2_DEPENDENCY_WHOLE_LINK_LIBS will only
# be linked by caffe2 lib.
set(Caffe2_DEPENDENCY_WHOLE_LINK_LIBS "")

# Lists for Caffe2 public dependency libraries. These libraries will be
# transitive to any libraries that depends on Caffe2.
set(Caffe2_PUBLIC_DEPENDENCY_LIBS "")
set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS "")

# List of modules that is built as part of the main Caffe2 build. For all
# binary targets, such as Python and native binaries, they will be linked
# automatically with these modules.
set(Caffe2_MODULES "")
