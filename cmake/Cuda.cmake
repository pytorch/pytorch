# Known NVIDIA GPU achitectures Caffe2 can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(Caffe2_known_gpu_archs "20 21(20) 30 35 50 52 60 61")
set(Caffe2_known_gpu_archs7 "20 21(20) 30 35 50 52")

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   caffe_detect_installed_gpus(out_variable)
function(caffe2_detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin=${CUDA_HOST_COMPILER}" ${CUDA_NVCC_FLAGS} "--run" "${__cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(__nvcc_res EQUAL 0)
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from caffe_detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${Caffe2_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   caffe_select_nvcc_arch_flags(out_variable)
function(caffe2_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Kepler" "Maxwell" "Pascal" "All" "Manual")
  set(__archs_name_default "All")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND __archs_names "Auto")
    set(__archs_name_default "Auto")
  endif()

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${Caffe2_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${Caffe2_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    caffe2_detect_installed_gpus(__cuda_arch_bin)
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  caffe_list_unique(__cuda_arch_bin __cuda_arch_ptx)

  set(__nvcc_flags "")
  set(__nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for cuda compilation
# Usage:
#   caffe_cuda_compile(<objlist_variable> <cuda_files>)
macro(caffe2_cuda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_cuda_compile_ "${${var}}")

    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

  endforeach()

  if(APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -Wno-unused-function)
  endif()

  cuda_compile(cuda_objcs ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()

  set(${objlist_variable} ${cuda_objcs})
endmacro()

################################################################################################
###  Non macro section
################################################################################################

# Special care for windows platform: we know that 32-bit windows does not support cuda.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR
            "CUDA support not available with 32-bit windows. Did you "
            "forget to set Win64 in the generator target?")
    return()
  endif()
endif()

find_package(CUDA 7.0 QUIET)
find_cuda_helper_libs(curand)  # cmake 2.8.7 compartibility which doesn't search for curand

if(NOT CUDA_FOUND)
  return()
endif()

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${CUDA_VERSION})
if (${CUDA_VERSION} LESS 8.0)
  set(Caffe2_known_gpu_archs ${Caffe2_known_gpu_archs7})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
else()
  # CUDA 8 may complain that sm_20 is no longer supported. Suppress the
  # warning for now.
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif()

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND Caffe2_DEPENDENCY_LIBS ${CUDA_CUDART_LIBRARY}
                              ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

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

# setting nvcc arch flags
caffe2_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")

if(CUDA_CUDA_LIB)
    message(STATUS "Found libcuda: ${CUDA_CUDA_LIB}")
    list(APPEND Caffe2_DEPENDENCY_LIBS ${CUDA_CUDA_LIB})
else()
    message(FATAL_ERROR "Cannot find libcuda.so. Please file an issue on https://github.com/caffe2/caffe2 with your build output.")
endif()
if(CUDA_NVRTC_LIB)
  message(STATUS "Found libnvrtc: ${CUDA_NVRTC_LIB}")
  list(APPEND Caffe2_DEPENDENCY_LIBS ${CUDA_NVRTC_LIB})
else()
    message(FATAL_ERROR "Cannot find libnvrtc.so. Please file an issue on https://github.com/caffe2/caffe2 with your build output.")
endif()

# disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
if (NOT MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
endif()

# Debug and Release symbol support
if (MSVC)
  if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    if (${BUILD_SHARED_LIBS})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -MD")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -MT")
    endif()
  elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    message(FATAL_ERROR
            "Caffe2 currently does not support the combination of MSVC, Cuda "
            "and Debug mode. Either set USE_CUDA=OFF or set the build type "
            "to Release")
    if (${BUILD_SHARED_LIBS})
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -MDd")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -MTd")
    endif()
  else()
    message(FATAL_ERROR "Unknown cmake build type: " ${CMAKE_BUILD_TYPE})
  endif()
endif()


if(OpenMP_FOUND)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS}")
endif()

# Set :expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
