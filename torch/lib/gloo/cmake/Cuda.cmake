# Known NVIDIA GPU achitectures Gloo can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(gloo_known_gpu_archs "30 35 50 52 60 61 70")
set(gloo_known_gpu_archs7 "30 35 50 52")
set(gloo_known_gpu_archs8 "30 35 50 52 60 61")

################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   gloo_select_nvcc_arch_flags(out_variable)
function(gloo_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "All")
  set(__archs_name_default "All")

  # Set CUDA_ARCH_NAME strings (so it will be seen as dropbox in the CMake GUI)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU architecture")
  set_property(CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names})
  mark_as_advanced(CUDA_ARCH_NAME)

  # Verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Invalid CUDA_ARCH_NAME, supported values: ${__archs_names}")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    set(__cuda_arch_bin "70")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${gloo_known_gpu_archs})
  else()
    message(FATAL_ERROR "Invalid CUDA_ARCH_NAME")
  endif()

  # Remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  list(REMOVE_DUPLICATES __cuda_arch_bin)
  list(REMOVE_DUPLICATES __cuda_arch_ptx)

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

################################################################################
# Function to append to list if specified sequence does not yet exist in list.
# Usage:
#   gloo_list_append_if_unique(list_variable arg1 arg2 ...)
function(gloo_list_append_if_unique list)
  list(LENGTH ARGN __match_length)
  set(__match_index 0)
  set(__match OFF)
  foreach(__elem ${${list}})
    list(GET ARGN ${__match_index} __match_elem)
    if("${__elem}" STREQUAL "${__match_elem}")
      MATH(EXPR __match_index "${__match_index}+1")
      if(${__match_index} EQUAL ${__match_length})
        set(__match ON)
        break()
      endif()
    else()
      # Mismatch; start from scratch.
      # This doesn't do backtracking but shouldn't be needed either.
      set(__match_index 0)
    endif()
  endforeach()

  # Only append arguments if we didn't find a match.
  if(NOT __match)
    list(APPEND ${list} ${ARGN})
    set(${list} ${${list}} PARENT_SCOPE)
  endif()
endfunction()

################################################################################
# Short command for cuda compilation
# Usage:
#   gloo_cuda_compile(<objlist_variable> <cuda_files>)
macro(gloo_cuda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_cuda_compile_ "${${var}}")
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

################################################################################
###  Non macro section
################################################################################

find_package(CUDA 7.0)
if(NOT CUDA_FOUND)
  return()
endif()

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${CUDA_VERSION})
if (${CUDA_VERSION} LESS 8.0)
  set(gloo_known_gpu_archs ${gloo_known_gpu_archs7})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
elseif (${CUDA_VERSION} LESS 9.0)
  set(gloo_known_gpu_archs ${gloo_known_gpu_archs8})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
else()
  # CUDA 8 may complain that sm_20 is no longer supported. Suppress the warning for now.
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
endif()

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND gloo_DEPENDENCY_LIBS ${CUDA_CUDART_LIBRARY})

# Find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed.
find_library(CUDA_CUDA_LIB cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs)
find_library(CUDA_NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

# Setting nvcc arch flags (or inherit if already set)
if (NOT ";${CUDA_NVCC_FLAGS};" MATCHES ";-gencode;")
  gloo_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
  list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
  message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")
endif()

if(CUDA_CUDA_LIB)
  message(STATUS "Found libcuda: ${CUDA_CUDA_LIB}")
  list(APPEND gloo_DEPENDENCY_LIBS ${CUDA_CUDA_LIB})
else()
  message(FATAL_ERROR "Cannot find libcuda.so. Please file an issue on https://github.com/facebookincubator/gloo with your build output.")
endif()

if(CUDA_NVRTC_LIB)
  message(STATUS "Found libnvrtc: ${CUDA_NVRTC_LIB}")
  list(APPEND gloo_DEPENDENCY_LIBS ${CUDA_NVRTC_LIB})
else()
  message(FATAL_ERROR "Cannot find libnvrtc.so. Please file an issue on https://github.com/facebookincubator/gloo with your build output.")
endif()

# Disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  gloo_list_append_if_unique(CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
gloo_list_append_if_unique(CUDA_NVCC_FLAGS "-std=c++11")
gloo_list_append_if_unique(CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

# Set :expt-relaxed-constexpr to suppress Eigen warnings
gloo_list_append_if_unique(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
