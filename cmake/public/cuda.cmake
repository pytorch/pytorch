# ---[ cuda

# sccache is only supported in CMake master and not in the newest official
# release (3.11.3) yet. Hence we need our own Modules_CUDA_fix to enable sccache.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../Modules_CUDA_fix)

# Find CUDA.
find_package(CUDA 7.0)
if(NOT CUDA_FOUND)
  message(WARNING
    "Caffe2: CUDA cannot be found. Depending on whether you are building "
    "Caffe2 or a Caffe2 dependent library, the next warning / error will "
    "give you more info.")
  set(CAFFE2_USE_CUDA OFF)
  return()
endif()
message(STATUS "Caffe2: CUDA detected: " ${CUDA_VERSION})
message(STATUS "Caffe2: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "Caffe2: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})

if(CUDA_FOUND)
  # Sometimes, we may mismatch nvcc with the CUDA headers we are
  # compiling with, e.g., if a ccache nvcc is fed to us by CUDA_NVCC_EXECUTABLE
  # but the PATH is not consistent with CUDA_HOME.  It's better safe
  # than sorry: make sure everything is consistent.
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_version.cc")
  file(WRITE ${file} ""
    "#include <cuda.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "  printf(\"%d.%d\", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);\n"
    "  return 0;\n"
    "}\n"
    )
  try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
    LINK_LIBRARIES ${CUDA_LIBRARIES}
    RUN_OUTPUT_VARIABLE cuda_version_from_header
    COMPILE_OUTPUT_VARIABLE output_var
    )
  if(NOT compile_result)
    message(FATAL_ERROR "Caffe2: Couldn't determine version from header: " ${output_var})
  endif()
  message(STATUS "Caffe2: Header version is: " ${cuda_version_from_header})
  if(NOT ${cuda_version_from_header} STREQUAL ${CUDA_VERSION})
    # Force CUDA to be processed for again next time
    # TODO: I'm not sure if this counts as an implementation detail of
    # FindCUDA
    set(${cuda_version_from_findcuda} ${CUDA_VERSION})
    unset(CUDA_TOOLKIT_ROOT_DIR_INTERNAL CACHE)
    # Not strictly necessary, but for good luck.
    unset(CUDA_VERSION CACHE)
    # Error out
    message(FATAL_ERROR "FindCUDA says CUDA version is ${cuda_version_from_findcuda} (usually determined by nvcc), "
      "but the CUDA headers say the version is ${cuda_version_from_header}.  This often occurs "
      "when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to "
      "non-standard locations, without also setting PATH to point to the correct nvcc.  "
      "Perhaps, try re-running this command again with PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$PATH.  "
      "See above log messages for more diagnostics, and see https://github.com/pytorch/pytorch/issues/8092 for more details.")
  endif()
endif()

# Find cuDNN.
if(CAFFE2_STATIC_LINK_CUDA)
  SET(CUDNN_LIBNAME "libcudnn_static.a")
else()
  SET(CUDNN_LIBNAME "cudnn")
endif()
include(FindPackageHandleStandardArgs)

if(DEFINED ENV{CUDNN_ROOT_DIR})
  set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder contains NVIDIA cuDNN")
else()
  set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
endif()

if(DEFINED ENV{CUDNN_INCLUDE_DIR})
  set(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR})
else()
  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
endif()

if(DEFINED ENV{CUDNN_LIBRARY})
  set(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY})
else()
  find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
endif()

find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
if(NOT CUDNN_FOUND)
  message(WARNING
    "Caffe2: Cannot find cuDNN library. Turning the option off")
  set(CAFFE2_USE_CUDNN OFF)
else()
  set(CAFFE2_USE_CUDNN ON)
endif()

# Optionally, find TensorRT
if(CAFFE2_USE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
  find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
  find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
  if(NOT TENSORRT_FOUND)
    message(WARNING
      "Caffe2: Cannot find TensorRT library. Turning the option off")
    set(CAFFE2_USE_TENSORRT OFF)
  endif()
endif()

# ---[ Extract versions
if(CAFFE2_USE_CUDNN)
  # Get cuDNN version
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
               CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
               CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
               CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
               CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION
        "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARY})")
endif()

# ---[ CUDA libraries wrapper

# find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed. On windows, we also search under the
# folder lib/x64.
find_library(CUDA_CUDA_LIB cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
find_library(CUDA_NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cuda
add_library(caffe2::cuda UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY IMPORTED_LOCATION
    ${CUDA_CUDA_LIB})
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudart. CUDA_LIBRARIES is actually a list, so we will make an interface
# library.
add_library(caffe2::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a" rt)
else()
    set_property(
        TARGET caffe2::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudnn
# static linking is handled by USE_STATIC_CUDNN environment variable
if(CAFFE2_USE_CUDNN)
  add_library(caffe2::cudnn UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::cudnn PROPERTY IMPORTED_LOCATION
      ${CUDNN_LIBRARY})
  set_property(
      TARGET caffe2::cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${CUDNN_INCLUDE_DIR})
endif()

# curand
add_library(caffe2::curand UNKNOWN IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a")
else()
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        ${CUDA_curand_LIBRARY})
endif()
set_property(
    TARGET caffe2::curand PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cufft. CUDA_CUFFT_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cufft INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static.a")
else()
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUFFT_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# TensorRT
if(CAFFE2_USE_TENSORRT)
  add_library(caffe2::tensorrt UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::tensorrt PROPERTY IMPORTED_LOCATION
      ${TENSORRT_LIBRARY})
  set_property(
      TARGET caffe2::tensorrt PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${TENSORRT_INCLUDE_DIR})
endif()

# cublas. CUDA_CUBLAS_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cublas INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
else()
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUBLAS_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# nvrtc
add_library(caffe2::nvrtc UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::nvrtc PROPERTY IMPORTED_LOCATION
    ${CUDA_NVRTC_LIB})
set_property(
    TARGET caffe2::nvrtc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# Note: in theory, we can add similar dependent library wrappers. For
# now, Caffe2 only uses the above libraries, so we will only wrap
# these.

# Special care for windows platform: we know that 32-bit windows does not
# support cuda.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR
            "CUDA support not available with 32-bit windows. Did you "
            "forget to set Win64 in the generator target?")
    return()
  endif()
endif()

if (${CUDA_VERSION} LESS 8.0) # CUDA 7.x
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
elseif (${CUDA_VERSION} LESS 9.0) # CUDA 8.x
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
  # CUDA 8 may complain that sm_20 is no longer supported. Suppress the
  # warning for now.
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif()

# Add onnx namepsace definition to nvcc
if (ONNX_NAMESPACE)
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# CUDA 9.0 & 9.1 require GCC version <= 5
# Although they support GCC 6, but a bug that wasn't fixed until 9.2 prevents
# them from compiling the std::tuple header of GCC 6.
# See Sec. 2.2.1 of
# https://developer.download.nvidia.com/compute/cuda/9.2/Prod/docs/sidebar/CUDA_Toolkit_Release_Notes.pdf
if ((CUDA_VERSION VERSION_EQUAL   9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
      NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
      CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with std::tuple from GCC version "
      ">= 6. Please upgrade to CUDA 9.2 or use the following option to use "
      "another version (for example): \n"
      "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
  endif()
elseif (CUDA_VERSION VERSION_EQUAL 8.0)
  # CUDA 8.0 requires GCC version <= 5
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
      NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
      CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "CUDA 8.0 is not compatible with GCC version >= 6. "
      "Use the following option to use another version (for example): \n"
      "  -DCUDA_HOST_COMPILER=/usr/bin/gcc-5\n")
  endif()
endif()

# setting nvcc arch flags
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS_BLACKLIST "-Werror")
if (NOT MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-fPIC")
endif()

# Debug and Release symbol support
if (MSVC)
  if ((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo") OR (${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel"))
    if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MT")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MD")
    endif()
  elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MTd")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MDd")
    endif()
  else()
    message(FATAL_ERROR "Unknown cmake build type: " ${CMAKE_BUILD_TYPE})
  endif()
elseif (CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
