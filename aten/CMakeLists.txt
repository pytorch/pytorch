if(NOT INTERN_BUILD_ATEN_OPS)
  return()
endif()

# Find modules
if(NOT INTERN_BUILD_MOBILE)
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
set(ATen_CUDA_SRCS_W_SORT_BY_KEY)
set(ATen_CUDA_TEST_SRCS)
set(ATen_CUDA_INCLUDE)
set(ATen_NVRTC_STUB_SRCS)
set(ATen_HIP_SRCS)
set(ATen_HIP_SRCS_W_SORT_BY_KEY)
set(ATen_HIP_TEST_SRCS)
set(ATen_HIP_INCLUDE)
set(ATen_VULKAN_TEST_SRCS)
set(ATen_CPU_DEPENDENCY_LIBS)
set(ATen_CUDA_DEPENDENCY_LIBS)
set(ATen_HIP_DEPENDENCY_LIBS)
set(ATen_PUBLIC_CUDA_DEPENDENCY_LIBS)
set(ATen_PUBLIC_HIP_DEPENDENCY_LIBS)
set(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
set(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
set(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

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

if(USE_VULKAN)
  list(APPEND ATen_CPU_INCLUDE ${CMAKE_BINARY_DIR}/vulkan)
endif()

# Find the HIP package, set the HIP paths, load the HIP CMake.
if(USE_ROCM)
  include(LoadHIP)
  if(NOT PYTORCH_FOUND_HIP)
    set(USE_ROCM OFF)
  endif()
endif()

# Both CUDA and ROCM are enabled and found. Report an error.
if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "Both CUDA and ROCm are enabled and found. PyTorch can only be built with either of them. Please turn one off by using either USE_CUDA=OFF or USE_ROCM=OFF.")
endif()

if(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE=1)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "/wd4819" "-Xcompiler" "/wd4503" "-Xcompiler" "/wd4190" "-Xcompiler" "/wd4244" "-Xcompiler" "/wd4251" "-Xcompiler" "/wd4275" "-Xcompiler" "/wd4522")
endif(MSVC)

if(USE_ROCM)
  # TODO: AT_HIP_ENABLED (change this once we represent HIP as HIP in
  # ATen proper)
  set(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THH)
  add_subdirectory(src/THHUNN)
  message("ROCm is enabled.")
elseif(USE_CUDA)
  set(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
else()
  message("disabling CUDA because USE_CUDA is set false")
  set(AT_CUDA_ENABLED 0)
endif()
if(NOT USE_CUDA)
  # we still parse THCUNN even if cuda is disabled to make sure to
  # install it
  install(FILES src/THCUNN/generic/THCUNN.h DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCUNN/generic")
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
set(ATen_CUDA_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS_W_SORT_BY_KEY ${ATen_HIP_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_BENCHMARK_SRCS ${ATen_MOBILE_BENCHMARK_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_TEST_SRCS ${ATen_MOBILE_TEST_SRCS} PARENT_SCOPE)
set(ATen_VEC256_TEST_SRCS ${ATen_VEC256_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
