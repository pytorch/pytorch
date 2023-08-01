cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-ignored-qualifiers")
  string(APPEND CMAKE_C_FLAGS " -Wno-ignored-qualifiers")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-absolute-value")
  string(APPEND CMAKE_C_FLAGS " -Wno-absolute-value")
endif(NOT MSVC)

# Can be compiled standalone
if(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  set(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  set(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  set(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  set(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
endif()

# These flag are used in Config but set externally. We must normalize them to
# 0/1 otherwise `#if ON` will be evaluated to false.
macro(set_bool OUT IN)
  if(${IN})
    set(${OUT} 1)
  else()
    set(${OUT} 0)
  endif()
endmacro()

set_bool(AT_BUILD_WITH_BLAS USE_BLAS)
set_bool(AT_BUILD_WITH_LAPACK USE_LAPACK)
set_bool(AT_BLAS_F2C BLAS_F2C)
set_bool(AT_BLAS_USE_CBLAS_DOT BLAS_USE_CBLAS_DOT)
set_bool(AT_MAGMA_ENABLED USE_MAGMA)
set_bool(CAFFE2_STATIC_LINK_CUDA_INT CAFFE2_STATIC_LINK_CUDA)
set_bool(AT_CUDNN_ENABLED CAFFE2_USE_CUDNN)

configure_file(Config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h")
# TODO: Do not generate CUDAConfig.h for ROCm BUILDS
# At the moment, `jit_macors.h` include CUDAConfig.h for both CUDA and HIP builds
if(USE_CUDA OR USE_ROCM)
  configure_file(cuda/CUDAConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDAConfig.h")
endif()
if(USE_ROCM)
  configure_file(hip/HIPConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/hip/HIPConfig.h")
endif()

# NB: If you edit these globs, you'll have to update setup.py package_data as well
file(GLOB_RECURSE ATen_CORE_HEADERS  "core/*.h")
file(GLOB_RECURSE ATen_CORE_SRCS "core/*.cpp")
if(NOT BUILD_LITE_INTERPRETER)
  file(GLOB_RECURSE ATen_CORE_TEST_SRCS "core/*_test.cpp")
endif()
EXCLUDE(ATen_CORE_SRCS "${ATen_CORE_SRCS}" ${ATen_CORE_TEST_SRCS})
# Exclude TensorImpl_test.cpp if compiling without Caffe2
if(NOT BUILD_CAFFE2 AND NOT BUILD_LITE_INTERPRETER)
  file(GLOB_RECURSE ATen_CORE_EXCLUDED_TEST_SRCS "core/TensorImpl_test.cpp")
  EXCLUDE(ATen_CORE_TEST_SRCS "${ATen_CORE_TEST_SRCS}" ${ATen_CORE_EXCLUDED_TEST_SRCS})
endif()

file(GLOB base_h "*.h" "detail/*.h" "cpu/*.h" "cpu/vec/vec512/*.h" "cpu/vec/vec256/*.h" "cpu/vec/vec256/vsx/*.h" "cpu/vec/*.h" "quantized/*.h" "functorch/*.h")
file(GLOB base_cpp "*.cpp" "detail/*.cpp" "cpu/*.cpp" "functorch/*.cpp")
file(GLOB cuda_h "cuda/*.h" "cuda/detail/*.h" "cuda/*.cuh" "cuda/detail/*.cuh")
file(GLOB cuda_cpp "cuda/*.cpp" "cuda/detail/*.cpp")
file(GLOB cuda_nvrtc_stub_h "cuda/nvrtc_stub/*.h")
file(GLOB cuda_nvrtc_stub_cpp "cuda/nvrtc_stub/*.cpp")
file(GLOB cuda_cu "cuda/*.cu" "cuda/detail/*.cu")
file(GLOB cudnn_h "cudnn/*.h" "cudnn/*.cuh")
file(GLOB cudnn_cpp "cudnn/*.cpp")
file(GLOB ops_h "ops/*.h")

file(GLOB hip_h "hip/*.h" "hip/detail/*.h" "hip/*.cuh" "hip/detail/*.cuh" "hip/impl/*.h")
file(GLOB hip_cpp "hip/*.cpp" "hip/detail/*.cpp" "hip/impl/*.cpp")
list(REMOVE_ITEM hip_cpp "${CMAKE_CURRENT_SOURCE_DIR}/hip/detail/LazyNVRTC.cpp")
file(GLOB hip_hip "hip/*.hip" "hip/detail/*.hip" "hip/impl/*.hip")
file(GLOB hip_nvrtc_stub_h "hip/nvrtc_stub/*.h")
file(GLOB hip_nvrtc_stub_cpp "hip/nvrtc_stub/*.cpp")
file(GLOB miopen_h "miopen/*.h")
file(GLOB miopen_cpp "miopen/*.cpp")

file(GLOB mkl_cpp "mkl/*.cpp")
file(GLOB mkldnn_cpp "mkldnn/*.cpp")

file(GLOB native_cpp "native/*.cpp")
file(GLOB native_mkl_cpp "native/mkl/*.cpp")
file(GLOB native_mkldnn_cpp "native/mkldnn/*.cpp")
file(GLOB vulkan_cpp "vulkan/*.cpp")
file(GLOB native_vulkan_cpp "native/vulkan/*.cpp" "native/vulkan/api/*.cpp" "native/vulkan/impl/*.cpp" "native/vulkan/ops/*.cpp")

# Metal
file(GLOB metal_h "metal/*.h")
file(GLOB metal_cpp "metal/*.cpp")
file(GLOB_RECURSE native_metal_h "native/metal/*.h")
file(GLOB metal_test_srcs "native/metal/mpscnn/tests/*.mm")
file(GLOB_RECURSE native_metal_srcs "native/metal/*.mm" "native/metal/*.cpp")
EXCLUDE(native_metal_srcs "${native_metal_srcs}" ${metal_test_srcs})
file(GLOB metal_prepack_h "native/metal/MetalPrepackOpContext.h")
file(GLOB metal_prepack_cpp "native/metal/MetalPrepackOpRegister.cpp")

file(GLOB native_ao_sparse_cpp
            "native/ao_sparse/*.cpp"
            "native/ao_sparse/cpu/*.cpp"
            "native/ao_sparse/quantized/*.cpp"
            "native/ao_sparse/quantized/cpu/*.cpp")
# MPS
file(GLOB mps_cpp "mps/*.cpp")
file(GLOB mps_mm "mps/*.mm")
file(GLOB mps_h "mps/*.h")
file(GLOB_RECURSE native_mps_cpp "native/mps/*.cpp")
file(GLOB_RECURSE native_mps_mm "native/mps/*.mm")
file(GLOB_RECURSE native_mps_h "native/mps/*.h")

file(GLOB native_sparse_cpp "native/sparse/*.cpp")
file(GLOB native_quantized_cpp
            "native/quantized/*.cpp"
            "native/quantized/cpu/*.cpp")
file(GLOB native_nested_cpp "native/nested/*.cpp")
file(GLOB native_transformers_cpp "native/transformers/*.cpp")

file(GLOB native_h "native/*.h")
file(GLOB native_ao_sparse_h
            "native/ao_sparse/*.h"
            "native/ao_sparse/cpu/*.h"
            "native/ao_sparse/quantized/*.h"
            "native/ao_sparse/quantized/cpu/*.h")
file(GLOB native_quantized_h "native/quantized/*.h" "native/quantized/cpu/*.h" "native/quantized/cudnn/*.h")
file(GLOB native_cpu_h "native/cpu/*.h")

file(GLOB native_cuda_cu "native/cuda/*.cu")
file(GLOB native_cuda_cpp "native/cuda/*.cpp")
file(GLOB native_cuda_h "native/cuda/*.h" "native/cuda/*.cuh")
file(GLOB native_cuda_linalg_cpp "native/cuda/linalg/*.cpp")
file(GLOB native_hip_h "native/hip/*.h" "native/hip/*.cuh")
file(GLOB native_cudnn_cpp "native/cudnn/*.cpp")
file(GLOB native_sparse_cuda_cu "native/sparse/cuda/*.cu")
file(GLOB native_sparse_cuda_cpp "native/sparse/cuda/*.cpp")
file(GLOB native_quantized_cuda_cu "native/quantized/cuda/*.cu")
file(GLOB native_quantized_cuda_cpp "native/quantized/cuda/*.cpp")
file(GLOB native_quantized_cudnn_cpp "native/quantized/cudnn/*.cpp")
file(GLOB native_nested_cuda_cu "native/nested/cuda/*.cu")
file(GLOB native_nested_cuda_cpp "native/nested/cuda/*.cpp")

file(GLOB native_hip_hip "native/hip/*.hip")
file(GLOB native_hip_cpp "native/hip/*.cpp")
file(GLOB native_hip_linalg_cpp "native/hip/linalg/*.cpp")
file(GLOB native_miopen_cpp "native/miopen/*.cpp")
file(GLOB native_cudnn_hip_cpp "native/cudnn/hip/*.cpp")
file(GLOB native_nested_hip_hip "native/nested/hip/*.hip")
file(GLOB native_nested_hip_cpp "native/nested/hip/*.cpp")
file(GLOB native_sparse_hip_hip "native/sparse/hip/*.hip")
file(GLOB native_sparse_hip_cpp "native/sparse/hip/*.cpp")
file(GLOB native_quantized_hip_hip "native/quantized/hip/*.hip")
file(GLOB native_quantized_hip_cpp "native/quantized/hip/*.cpp")
file(GLOB native_transformers_cuda_cu "native/transformers/cuda/*.cu")
file(GLOB native_transformers_cuda_cpp "native/transformers/cuda/*.cpp")
file(GLOB native_transformers_hip_hip "native/transformers/hip/*.hip")
file(GLOB native_transformers_hip_cpp "native/transformers/hip/*.cpp")
file(GLOB native_quantized_cudnn_hip_cpp "native/quantized/cudnn/hip/*.cpp")
file(GLOB native_utils_cpp "native/utils/*.cpp")

# flash_attention sources
file(GLOB flash_attention_cuda_cu "native/transformers/cuda/flash_attn/*.cu")
file(GLOB flash_attention_cuda_cpp "native/transformers/cuda/flash_attn/*.cpp")

#Mem_eff attention sources
file(GLOB mem_eff_attention_cuda_cu "native/transformers/cuda/mem_eff_attention/*.cu")
file(GLOB mem_eff_attention_cuda_kernels_cu "native/transformers/cuda/mem_eff_attention/kernels/*.cu")
file(GLOB mem_eff_attention_cuda_cpp "native/transformers/cuda/mem_eff_attention/*.cpp")

if(USE_FLASH_ATTENTION)
  list(APPEND native_transformers_cuda_cu ${flash_attention_cuda_cu})
  list(APPEND native_transformers_cuda_cpp ${flash_attention_cuda_cpp})

  list(APPEND native_transformers_cuda_cu ${mem_eff_attention_cuda_cu})
  list(APPEND native_transformers_cuda_cu ${mem_eff_attention_cuda_kernels_cu})
  list(APPEND native_transformers_cuda_cpp ${mem_eff_attention_cuda_cpp})
  list(APPEND MEM_EFF_ATTENTION_CUDA_SOURCES ${native_transformers_cuda_cu} ${mem_eff_attention_cuda_cu} ${mem_eff_attention_cuda_kernels_cu})
endif()

# XNNPACK
file(GLOB native_xnnpack "native/xnnpack/*.cpp")

# Add files needed from jit folders
append_filelist("jit_core_headers" ATen_CORE_HEADERS)
append_filelist("jit_core_sources" ATen_CORE_SRCS)

add_subdirectory(quantized)
add_subdirectory(nnapi)

if(BUILD_LITE_INTERPRETER)
  set(all_cpu_cpp ${generated_sources} ${core_generated_sources} ${cpu_kernel_cpp})
  append_filelist("jit_core_sources" all_cpu_cpp)
  append_filelist("aten_cpu_source_non_codegen_list" all_cpu_cpp)
  append_filelist("aten_native_source_non_codegen_list" all_cpu_cpp)
else()
  set(
    all_cpu_cpp ${base_cpp} ${ATen_CORE_SRCS} ${native_cpp}
    ${native_ao_sparse_cpp} ${native_sparse_cpp} ${native_nested_cpp}
    ${native_quantized_cpp} ${native_mkl_cpp} ${native_mkldnn_cpp}
    ${native_transformers_cpp}
    ${native_utils_cpp} ${native_xnnpack} ${generated_sources} ${core_generated_sources}
    ${ATen_CPU_SRCS} ${ATen_QUANTIZED_SRCS} ${ATen_NNAPI_SRCS} ${cpu_kernel_cpp}
  )
endif()

if(USE_LIGHTWEIGHT_DISPATCH)
  set(all_cpu_cpp ${all_cpu_cpp} ${generated_unboxing_sources})
endif()
if(AT_MKL_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkl_cpp})
endif()
if(AT_MKLDNN_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkldnn_cpp})
endif()
if(USE_VULKAN)
  set(all_cpu_cpp ${all_cpu_cpp} ${vulkan_cpp} ${native_vulkan_cpp} ${vulkan_generated_cpp})
else()
  set(all_cpu_cpp ${all_cpu_cpp} ${vulkan_cpp})
endif()

# Metal
if(USE_PYTORCH_METAL_EXPORT)
  # Add files needed from exporting metal models(optimized_for_mobile)
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp} ${metal_prepack_cpp})
elseif(APPLE AND USE_PYTORCH_METAL)
  # Compile Metal kernels
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp} ${native_metal_srcs})
else()
  set(all_cpu_cpp ${all_cpu_cpp} ${metal_cpp})
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "ATen doesn't not currently support simultaneously building with CUDA and ROCM")
endif()

if(USE_CUDA)
  list(APPEND ATen_CUDA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  list(APPEND ATen_CUDA_CU_SRCS
    ${cuda_cu}
    ${native_cuda_cu}
    ${native_nested_cuda_cu}
    ${native_sparse_cuda_cu}
    ${native_quantized_cuda_cu}
    ${native_transformers_cuda_cu}
    ${cuda_generated_sources}
  )
  list(APPEND ATen_CUDA_CPP_SRCS
    ${cuda_cpp}
    ${native_cuda_cpp}
    ${native_cudnn_cpp}
    ${native_miopen_cpp}
    ${native_nested_cuda_cpp}
    ${native_quantized_cuda_cpp}
    ${native_quantized_cudnn_cpp}
    ${native_sparse_cuda_cpp}
    ${native_transformers_cuda_cpp}
  )
  set(ATen_CUDA_LINALG_SRCS ${native_cuda_linalg_cpp})
  if(NOT BUILD_LAZY_CUDA_LINALG)
    list(APPEND ATen_CUDA_CU_SRCS ${native_cuda_linalg_cpp})
  endif()
  if(CAFFE2_USE_CUDNN)
    list(APPEND ATen_CUDA_CPP_SRCS ${cudnn_cpp})
  endif()

  append_filelist("aten_cuda_cu_source_list" ATen_CUDA_CU_SRCS)
  append_filelist("aten_cuda_with_sort_by_key_source_list" ATen_CUDA_SRCS_W_SORT_BY_KEY)
  append_filelist("aten_cuda_cu_with_sort_by_key_source_list" ATen_CUDA_CU_SRCS_W_SORT_BY_KEY)

  exclude(ATen_CUDA_CPP_SRCS "${ATen_CUDA_CPP_SRCS}"
      ${ATen_CUDA_CU_SRCS}
      ${ATen_CUDA_SRCS_W_SORT_BY_KEY} ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
  exclude(ATen_CUDA_CU_SRCS "${ATen_CUDA_CU_SRCS}"
      ${ATen_CUDA_SRCS_W_SORT_BY_KEY} ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
endif()

if(USE_ROCM)
  list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/hip)
  set(ATen_HIP_SRCS ${ATen_HIP_SRCS} ${hip_hip} ${native_hip_hip} ${native_nested_hip_hip} ${native_sparse_hip_hip} ${native_quantized_hip_hip} ${native_transformers_hip_hip})
  # TODO: Codegen separate files for HIP and use those (s/cuda_generated_sources/hip_generated_sources)
  set(all_hip_cpp ${native_nested_hip_cpp} ${native_sparse_hip_cpp} ${native_quantized_hip_cpp} ${native_transformers_hip_cpp} ${native_quantized_cudnn_hip_cpp} ${hip_cpp} ${native_hip_cpp} ${native_hip_linalg_cpp} ${cuda_generated_sources} ${ATen_HIP_SRCS})
  set(all_hip_cpp ${native_miopen_cpp} ${native_cudnn_hip_cpp} ${miopen_cpp} ${all_hip_cpp})
endif()

list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(USE_TBB)
  if(USE_SYSTEM_TBB)
    message("ATen is compiled with system-provided Intel TBB.")
  else()
    message("ATen is compiled with Intel TBB (${TBB_ROOT_DIR}).")
  endif()
  list(APPEND ATen_CPU_INCLUDE ${TBB_INCLUDE_DIR})
  list(APPEND ATen_CPU_DEPENDENCY_LIBS TBB::tbb)
endif()

if(BLAS_FOUND)
  if($ENV{TH_BINARY_BUILD})
    message(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    list(APPEND ATen_CPU_DEPENDENCY_LIBS
      "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
  else($ENV{TH_BINARY_BUILD})
    list(APPEND ATen_CPU_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
  endif($ENV{TH_BINARY_BUILD})
endif(BLAS_FOUND)

if(LAPACK_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  if(USE_CUDA AND MSVC)
    # Although Lapack provides CPU (and thus, one might expect that ATen_cuda
    # would not need this at all), some of our libraries (magma in particular)
    # backend to CPU BLAS/LAPACK implementations, and so it is very important
    # we get the *right* implementation, because even if the symbols are the
    # same, LAPACK implementions may have different calling conventions.
    # This caused https://github.com/pytorch/pytorch/issues/7353
    #
    # We do NOT do this on Linux, since we just rely on torch_cpu to
    # provide all of the symbols we need
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()
endif(LAPACK_FOUND)

if(UNIX AND NOT APPLE)
   include(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   if(NEED_LIBRT)
     list(APPEND ATen_CPU_DEPENDENCY_LIBS rt)
     set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   endif(NEED_LIBRT)
endif(UNIX AND NOT APPLE)

if(UNIX)
  set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  if(HAVE_MMAP)
    add_definitions(-DHAVE_MMAP=1)
  endif(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
  add_definitions(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  if(HAVE_SHM_OPEN)
    add_definitions(-DHAVE_SHM_OPEN=1)
  endif(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  if(HAVE_SHM_UNLINK)
    add_definitions(-DHAVE_SHM_UNLINK=1)
  endif(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  if(HAVE_MALLOC_USABLE_SIZE)
    add_definitions(-DHAVE_MALLOC_USABLE_SIZE=1)
  endif(HAVE_MALLOC_USABLE_SIZE)
endif(UNIX)

ADD_DEFINITIONS(-DUSE_EXTERNAL_MZCRC)

if(NOT MSVC)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS m)
endif()

if(AT_NNPACK_ENABLED)
  include_directories(${NNPACK_INCLUDE_DIRS})
  list(APPEND ATen_CPU_DEPENDENCY_LIBS nnpack) # cpuinfo is added below
endif()

if(MKLDNN_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${MKLDNN_LIBRARIES})
endif(MKLDNN_FOUND)

list(APPEND ATen_CPU_DEPENDENCY_LIBS cpuinfo)

if(NOT MSVC AND NOT EMSCRIPTEN AND NOT INTERN_BUILD_MOBILE)
  # Preserve values for the main build
  set(__aten_sleef_build_shared_libs ${BUILD_SHARED_LIBS})
  set(__aten_sleef_build_tests ${BUILD_TESTS})

  # Unset our restrictive C++ flags here and reset them later.
  # Remove this once we use proper target_compile_options.
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS)

  # Bump up optimization level for sleef to -O1, since at -O0 the compiler
  # excessively spills intermediate vector registers to the stack
  # and makes things run impossibly slowly
  set(OLD_CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  if(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
    string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  else()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
  endif()

  if(NOT USE_SYSTEM_SLEEF)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
    set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
    set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
    set(OLD_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(DISABLE_SVE ON CACHE BOOL "Xcode's clang-12.5 crashes while trying to compile SVE code" FORCE)
      endif()
    endif()
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND
        CMAKE_C_COMPILER_VERSION VERSION_GREATER 6.9 AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
      set(GCC_7 True)
    else()
      set(GCC_7 False)
    endif()
    if(GCC_7)
      set(CMAKE_BUILD_TYPE Release)  # Always build Sleef as a Release build to work around a gcc-7 bug
    endif()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/sleef" ${CMAKE_BINARY_DIR}/sleef)
    if(GCC_7)
      set(CMAKE_BUILD_TYPE ${OLD_CMAKE_BUILD_TYPE})
    endif()
    set_property(TARGET sleef PROPERTY FOLDER "dependencies")
    list(APPEND ATen_THIRD_PARTY_INCLUDE ${CMAKE_BINARY_DIR}/include)
    link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
  else()
    add_library(sleef SHARED IMPORTED)
    find_library(SLEEF_LIBRARY sleef)
    if(NOT SLEEF_LIBRARY)
      message(FATAL_ERROR "Cannot find sleef")
    endif()
    message("Found sleef: ${SLEEF_LIBRARY}")
    set_target_properties(sleef PROPERTIES IMPORTED_LOCATION "${SLEEF_LIBRARY}")
  endif()
  list(APPEND ATen_CPU_DEPENDENCY_LIBS sleef)

  set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

  # Set these back. TODO: Use SLEEF_ to pass these instead
  set(BUILD_SHARED_LIBS ${__aten_sleef_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)
  set(BUILD_TESTS ${__aten_sleef_build_tests} CACHE BOOL "Build tests" FORCE)
endif()

if(USE_CUDA AND NOT USE_ROCM)
  list(APPEND ATen_CUDA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/include)
  if($ENV{ATEN_STATIC_CUDA})
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      CUDA::cusparse_static
      CUDA::curand_static
      CUDA::cufft_static_nocallback
    )
   if(NOT BUILD_LAZY_CUDA_LINALG)
     if(CUDA_VERSION_MAJOR LESS_EQUAL 11)
       list(APPEND ATen_CUDA_DEPENDENCY_LIBS
         CUDA::cusolver_static
         ${CUDAToolkit_LIBRARY_DIR}/liblapack_static.a     # needed for libcusolver_static
       )
     elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
       list(APPEND ATen_CUDA_DEPENDENCY_LIBS
         CUDA::cusolver_static
         ${CUDAToolkit_LIBRARY_DIR}/libcusolver_lapack_static.a     # needed for libcusolver_static
       )
     endif()
   endif()
  else()
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      CUDA::cusparse
      CUDA::curand
      CUDA::cufft
    )
   if(NOT BUILD_LAZY_CUDA_LINALG)
     list(APPEND ATen_CUDA_DEPENDENCY_LIBS
       CUDA::cusolver
     )
   endif()
  endif()

  if(CAFFE2_USE_CUDNN)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
  endif()
  if($ENV{ATEN_STATIC_CUDA})
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      CUDA::culibos
      CUDA::cudart_static
    )
  endif($ENV{ATEN_STATIC_CUDA})
endif()


  if(USE_MAGMA)
    if(USE_CUDA AND NOT BUILD_LAZY_CUDA_LINALG)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS torch::magma)
    endif(USE_CUDA AND NOT BUILD_LAZY_CUDA_LINALG)
    if(USE_ROCM)
      list(APPEND ATen_HIP_DEPENDENCY_LIBS torch::magma)
    endif(USE_ROCM)
    if(MSVC)
      if($ENV{TH_BINARY_BUILD})
        # Do not do this on Linux: see Note [Extra MKL symbols for MAGMA in torch_cpu]
        # in caffe2/CMakeLists.txt
        list(APPEND ATen_CUDA_DEPENDENCY_LIBS
          "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
      endif($ENV{TH_BINARY_BUILD})
    endif(MSVC)
  endif(USE_MAGMA)

# Include CPU paths for CUDA/HIP as well
list(APPEND ATen_CUDA_INCLUDE ${ATen_CPU_INCLUDE})
list(APPEND ATen_HIP_INCLUDE ${ATen_CPU_INCLUDE})
list(APPEND ATen_VULKAN_INCLUDE ${ATen_CPU_INCLUDE})

# We have two libraries: libATen_cpu.so and libATen_cuda.so,
# with libATen_cuda.so depending on libATen_cpu.so.  The CPU library
# contains CPU code only.  libATen_cpu.so is invariant to the setting
# of USE_CUDA (it always builds the same way); libATen_cuda.so is only
# built when USE_CUDA=1 and CUDA is available.  (libATen_hip.so works
# the same way as libATen_cuda.so)
set(ATen_CPU_SRCS ${all_cpu_cpp})
list(APPEND ATen_CPU_DEPENDENCY_LIBS ATEN_CPU_FILES_GEN_LIB)

if(USE_CUDA)
  set(ATen_NVRTC_STUB_SRCS ${cuda_nvrtc_stub_cpp})
  list(APPEND ATen_CUDA_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
endif()

if(USE_MPS)
    set(ATen_MPS_SRCS ${ATen_MPS_SRCS} ${mps_cpp} ${mps_mm} ${mps_h} ${native_mps_cpp} ${native_mps_mm} ${native_mps_h})
endif()

if(USE_ROCM)
  set(ATen_HIP_SRCS ${all_hip_cpp})
  # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
  # See NOTE [ ATen NVRTC Stub and HIP ]
  set(ATen_NVRTC_STUB_SRCS ${hip_nvrtc_stub_cpp})
  # NB: Instead of adding it to this list, we add it by hand
  # to caffe2_hip, because it needs to be a PRIVATE dependency
  # list(APPEND ATen_HIP_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
endif()

set(ATEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_INCLUDE_DIR}")
configure_file(ATenConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake"
  DESTINATION "${AT_INSTALL_SHARE_DIR}/cmake/ATen")

set(INSTALL_HEADERS ${base_h} ${ATen_CORE_HEADERS})
if(NOT INTERN_BUILD_MOBILE)
  list(APPEND INSTALL_HEADERS ${native_h} ${native_cpu_h} ${native_ao_sparse_h} ${native_quantized_h} ${cuda_h} ${native_cuda_h} ${native_hip_h} ${cudnn_h} ${hip_h} ${mps_h} ${native_mps_h} ${miopen_h})
  # Metal
  if(USE_PYTORCH_METAL_EXPORT)
    # Add files needed from exporting metal models(optimized_for_mobile)
    list(APPEND INSTALL_HEADERS ${metal_h} ${metal_prepack_h})
  elseif(APPLE AND USE_PYTORCH_METAL)
    # Needed by Metal kernels
    list(APPEND INSTALL_HEADERS ${metal_h} ${native_metal_h})
  else()
    list(APPEND INSTALL_HEADERS ${metal_h})
  endif()
else()
  if(IOS AND USE_PYTORCH_METAL)
      list(APPEND INSTALL_HEADERS ${metal_h} ${native_metal_h})
  else()
      list(APPEND INSTALL_HEADERS ${metal_h} ${metal_prepack_h})
  endif()
endif()

# https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
foreach(HEADER  ${INSTALL_HEADERS})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "ATen/" HEADER_SUB ${HEADER})
  string(REPLACE "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/" "" HEADER_SUB ${HEADER_SUB})
  get_filename_component(DIR ${HEADER_SUB} DIRECTORY)
  install(FILES ${HEADER} DESTINATION "${AT_INSTALL_INCLUDE_DIR}/${DIR}")
endforeach()

# TODO: Install hip_generated_headers when we have it
foreach(HEADER ${generated_headers} ${cuda_generated_headers})
  # NB: Assumed to be flat
  install(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
endforeach()

message("AT_INSTALL_INCLUDE_DIR ${AT_INSTALL_INCLUDE_DIR}/ATen/core")
foreach(HEADER ${core_generated_headers})
  message("core header install: ${HEADER}")
  install(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/core)
endforeach()

install(FILES ${ops_h} ${ops_generated_headers}
  DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/ops)
install(FILES ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)

if(ATEN_NO_TEST)
  message("disable test because ATEN_NO_TEST is set")
elseif(BUILD_LITE_INTERPRETER)
  message("disable aten test when BUILD_LITE_INTERPRETER is enabled")
else()
  add_subdirectory(test)
endif()

list(APPEND ATen_MOBILE_BENCHMARK_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/tensor_add.cpp)
list(APPEND ATen_MOBILE_BENCHMARK_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/quantize_per_channel.cpp)
list(APPEND ATen_MOBILE_BENCHMARK_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/stateful_conv1d.cpp)

# Pass source, includes, and libs to parent
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS ${ATen_CUDA_CU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_CPP_SRCS ${ATen_CUDA_CPP_SRCS} PARENT_SCOPE)
set(ATen_CUDA_LINALG_SRCS ${ATen_CUDA_LINALG_SRCS} PARENT_SCOPE)
set(ATen_CUDA_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_MPS_SRCS ${ATen_MPS_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_SRCS ${ATen_QUANTIZED_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_BENCHMARK_SRCS ${ATen_MOBILE_BENCHMARK_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_TEST_SRCS ${ATen_MOBILE_TEST_SRCS} ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_VEC_TEST_SRCS  ${ATen_VEC_TEST_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_TEST_SRCS ${ATen_QUANTIZED_TEST_SRCS} PARENT_SCOPE)
set(ATen_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_VULKAN_INCLUDE ${ATen_VULKAN_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
set(MEM_EFF_ATTENTION_CUDA_SOURCES ${MEM_EFF_ATTENTION_CUDA_SOURCES} PARENT_SCOPE)
