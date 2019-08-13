if (NOT INTERN_BUILD_ATEN_OPS)
  return()
endif()

# Find modules
if (NOT INTERN_BUILD_MOBILE)
  list(APPEND CMAKE_MODULE_PATH /usr/lib/x86_64-linux-gnu/)
  list(APPEND CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ /usr/lib/aarch64-linux-gnu/)
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules_CUDA_fix)

cmake_policy(SET CMP0012 NEW)

#############################################

set(ATen_CPU_SRCS)
set(ATen_CPU_TEST_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_THIRD_PARTY_INCLUDE)
set(ATen_CUDA_SRCS)
set(ATen_CUDA_TEST_SRCS)
set(ATen_CUDA_INCLUDE)
set(ATen_NVRTC_STUB_SRCS)
set(ATen_HIP_SRCS)
set(ATen_HIP_TEST_SRCS)
set(ATen_HIP_INCLUDE)
set(ATen_CPU_DEPENDENCY_LIBS)
set(ATen_CUDA_DEPENDENCY_LIBS)
set(ATen_HIP_DEPENDENCY_LIBS)
set(ATen_PUBLIC_CUDA_DEPENDENCY_LIBS)
set(ATen_PUBLIC_HIP_DEPENDENCY_LIBS)
SET(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
SET(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
SET(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

if(USE_CUDA)
  list(APPEND ATen_CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
endif()

set(TH_LINK_STYLE STATIC)
add_subdirectory(src/TH)
set(TH_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/aten/src)
list(APPEND ATen_CPU_INCLUDE ${TH_CPU_INCLUDE})

add_subdirectory(src/THNN)

# Find the HIP package, set the HIP paths, load the HIP CMake.
IF(USE_ROCM)
  include(LoadHIP)
  if (NOT PYTORCH_FOUND_HIP)
    MESSAGE(FATAL_ERROR
      "Could not find HIP installation")
  endif()
ENDIF()

IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
ENDIF(MSVC)

if(USE_ROCM)
  # TODO: AT_HIP_ENABLED (change this once we represent HIP as HIP in
  # ATen proper)
  SET(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THH)
  add_subdirectory(src/THHUNN)
  message("ROCm is enabled.")
elseif(USE_CUDA)
  SET(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
else()
  message("disabling CUDA because USE_CUDA is set false")
  SET(AT_CUDA_ENABLED 0)
endif()
if(NOT USE_CUDA)
  # we still parse THCUNN even if cuda is disabled to make sure to
  # install it
  INSTALL(FILES src/THCUNN/generic/THCUNN.h DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCUNN/generic")
endif()

if(NOT USE_NNPACK)
  set(AT_NNPACK_ENABLED 0)
else()
  set(AT_NNPACK_ENABLED 1)
endif()

list(APPEND ATen_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catch/single_include)
add_subdirectory(src/ATen)

# Pass source, includes, and libs to parent
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
