cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

IF(NOT MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-ignored-qualifiers")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-absolute-value")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-absolute-value")
ENDIF(NOT MSVC)

# Can be compiled standalone
IF(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  SET(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  SET(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  SET(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  SET(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
ENDIF()

# this flag is used in Config but set externally, we must normalize it
# to 0/1 otherwise `#if ON` will be evaluated to false.
if (CAFFE2_STATIC_LINK_CUDA)
  set(CAFFE2_STATIC_LINK_CUDA_INT 1)
else()
  set(CAFFE2_STATIC_LINK_CUDA_INT 0)
endif()
CONFIGURE_FILE(Config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h")
# TODO: Don't unconditionally generate CUDAConfig.h.in.  Unfortuantely,
# this file generates AT_ROCM_ENABLED() which is required by the miopen
# files, which are compiled even if we are doing a vanilla CUDA build.
# Once we properly split CUDA and HIP in ATen, we can remove this code.
configure_file(cuda/CUDAConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDAConfig.h")
if(USE_ROCM)
  configure_file(hip/HIPConfig.h.in "${CMAKE_CURRENT_SOURCE_DIR}/hip/HIPConfig.h")
endif()

# NB: If you edit these globs, you'll have to update setup.py package_data as well
FILE(GLOB base_h "*.h" "detail/*.h" "cpu/*.h")
FILE(GLOB base_cpp "*.cpp" "detail/*.cpp" "cpu/*.cpp")
add_subdirectory(core)
FILE(GLOB cuda_h "cuda/*.h" "cuda/detail/*.h" "cuda/*.cuh" "cuda/detail/*.cuh")
FILE(GLOB cuda_cpp "cuda/*.cpp" "cuda/detail/*.cpp")
FILE(GLOB cuda_nvrtc_stub_h "cuda/nvrtc_stub/*.h")
FILE(GLOB cuda_nvrtc_stub_cpp "cuda/nvrtc_stub/*.cpp")
FILE(GLOB cuda_cu "cuda/*.cu" "cuda/detail/*.cu")
FILE(GLOB cudnn_h "cudnn/*.h" "cudnn/*.cuh")
FILE(GLOB cudnn_cpp "cudnn/*.cpp")

FILE(GLOB hip_h "hip/*.h" "hip/detail/*.h" "hip/*.cuh" "hip/detail/*.cuh")
FILE(GLOB hip_cpp "hip/*.cpp" "hip/detail/*.cpp" "hip/impl/*.cpp")
FILE(GLOB hip_hip "hip/*.hip" "hip/detail/*.hip" "hip/impl/*.hip")
FILE(GLOB hip_nvrtc_stub_h "hip/nvrtc_stub/*.h")
FILE(GLOB hip_nvrtc_stub_cpp "hip/nvrtc_stub/*.cpp")
FILE(GLOB miopen_h "miopen/*.h")
FILE(GLOB miopen_cpp "miopen/*.cpp")

FILE(GLOB mkl_cpp "mkl/*.cpp")
FILE(GLOB mkldnn_cpp "mkldnn/*.cpp")

FILE(GLOB native_cpp "native/*.cpp")
FILE(GLOB native_mkl_cpp "native/mkl/*.cpp")
FILE(GLOB native_mkldnn_cpp "native/mkldnn/*.cpp")
FILE(GLOB native_sparse_cpp "native/sparse/*.cpp")
FILE(GLOB native_quantized_cpp
            "native/quantized/*.cpp"
            "native/quantized/cpu/*.cpp")

FILE(GLOB native_cuda_cu "native/cuda/*.cu")
FILE(GLOB native_cuda_cpp "native/cuda/*.cpp")
FILE(GLOB native_cudnn_cpp "native/cudnn/*.cpp")
FILE(GLOB native_sparse_cuda_cu "native/sparse/cuda/*.cu")
FILE(GLOB native_sparse_cuda_cpp "native/sparse/cuda/*.cpp")
FILE(GLOB native_quantized_cuda_cu "native/quantized/cuda/*.cu")
FILE(GLOB native_quantized_cuda_cpp "native/quantized/cuda/*.cpp")

FILE(GLOB native_hip_hip "native/hip/*.hip")
FILE(GLOB native_hip_cpp "native/hip/*.cpp")
FILE(GLOB native_miopen_cpp "native/miopen/*.cpp")
FILE(GLOB native_cudnn_hip_cpp "native/cudnn/hip/*.cpp")
FILE(GLOB native_sparse_hip_hip "native/sparse/hip/*.hip")
FILE(GLOB native_sparse_hip_cpp "native/sparse/hip/*.cpp")
FILE(GLOB native_quantized_hip_hip "native/quantized/hip/*.hip")
FILE(GLOB native_quantized_hip_cpp "native/quantized/hip/*.cpp")

add_subdirectory(quantized)
set(all_cpu_cpp ${base_cpp} ${ATen_CORE_SRCS} ${native_cpp} ${native_sparse_cpp} ${native_quantized_cpp} ${native_mkl_cpp} ${native_mkldnn_cpp} ${generated_cpp} ${ATen_CPU_SRCS} ${ATen_QUANTIZED_SRCS} ${cpu_kernel_cpp})
if(AT_MKL_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkl_cpp})
endif()
if(AT_MKLDNN_ENABLED)
  set(all_cpu_cpp ${all_cpu_cpp} ${mkldnn_cpp})
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "ATen doesn't not currently support simultaneously building with CUDA and ROCM")
endif()

IF(USE_CUDA)
  list(APPEND ATen_CUDA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} ${cuda_cu} ${native_cuda_cu} ${native_sparse_cuda_cu} ${native_quantized_cuda_cu})
  set(all_cuda_cpp ${native_sparse_cuda_cpp} ${native_quantized_cuda_cpp} ${cuda_cpp} ${native_cuda_cpp} ${cuda_generated_cpp} ${ATen_CUDA_SRCS})
  SET(all_cuda_cpp ${native_cudnn_cpp} ${native_miopen_cpp} ${all_cuda_cpp})
  IF(CUDNN_FOUND)
    SET(all_cuda_cpp ${all_cuda_cpp} ${cudnn_cpp})
  ENDIF()
endif()

IF(USE_ROCM)
  list(APPEND ATen_HIP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/hip)
  set(ATen_HIP_SRCS ${ATen_HIP_SRCS} ${hip_hip} ${native_hip_hip} ${native_sparse_hip_hip} ${native_quantized_hip_hip})
  # TODO: Codegen separate files for HIP and use those (s/cuda_generated_cpp/hip_generated_cpp)
  set(all_hip_cpp ${native_sparse_hip_cpp} ${native_quantized_hip_cpp} ${hip_cpp} ${native_hip_cpp} ${cuda_generated_cpp} ${ATen_HIP_SRCS})
  set(all_hip_cpp ${native_miopen_cpp} ${native_cudnn_hip_cpp} ${miopen_cpp} ${all_hip_cpp})
endif()

filter_list(generated_h generated_cpp "\\.h$")
filter_list(cuda_generated_h cuda_generated_cpp "\\.h$")
# TODO: When we have hip_generated_cpp
#filter_list(hip_generated_h hip_generated_cpp "\\.h$")

list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..)
# so the build can find the generated header files
list(APPEND ATen_CPU_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
IF(NOT AT_LINK_STYLE)
  SET(AT_LINK_STYLE SHARED)
ENDIF()

IF (USE_TBB)
  message("ATen is compiled with TBB (${TBB_ROOT_DIR})")
  list(APPEND ATen_CPU_INCLUDE ${TBB_ROOT_DIR}/include)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS tbb)
ENDIF()

IF(BLAS_FOUND)
  IF ($ENV{TH_BINARY_BUILD})
    MESSAGE(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    list(APPEND ATen_CPU_DEPENDENCY_LIBS
      "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    if(USE_CUDA)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
        "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    endif()
    if(USE_ROCM)
      list(APPEND ATen_HIP_DEPENDENCY_LIBS
        "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    endif()
  ELSE ($ENV{TH_BINARY_BUILD})
    list(APPEND ATen_CPU_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
    if(USE_CUDA)
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${BLAS_LIBRARIES}")
    endif()
    if(USE_ROCM)
      list(APPEND ATen_HIP_DEPENDENCY_LIBS "${BLAS_LIBRARIES}")
    endif()
  ENDIF ($ENV{TH_BINARY_BUILD})
ENDIF(BLAS_FOUND)

IF(LAPACK_FOUND)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  if(USE_CUDA)
    # Although Lapack provides CPU (and thus, one might expect that ATen_cuda
    # would not need this at all), some of our libraries (magma in particular)
    # backend to CPU BLAS/LAPACK implementations, and so it is very important
    # we get the *right* implementation, because even if the symbols are the
    # same, LAPACK implementions may have different calling conventions.
    # This caused https://github.com/pytorch/pytorch/issues/7353
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()
  if(USE_ROCM)
    # It's not altogether clear that HIP behaves the same way, but it
    # seems safer to assume that it needs it too
    list(APPEND ATen_HIP_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
  endif()
ENDIF(LAPACK_FOUND)

IF (UNIX AND NOT APPLE)
   INCLUDE(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   IF(NEED_LIBRT)
     list(APPEND ATen_CPU_DEPENDENCY_LIBS rt)
     SET(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   ENDIF(NEED_LIBRT)
ENDIF(UNIX AND NOT APPLE)

IF(UNIX)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
  ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  IF(HAVE_SHM_OPEN)
    ADD_DEFINITIONS(-DHAVE_SHM_OPEN=1)
  ENDIF(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  IF(HAVE_SHM_UNLINK)
    ADD_DEFINITIONS(-DHAVE_SHM_UNLINK=1)
  ENDIF(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  IF(HAVE_MALLOC_USABLE_SIZE)
    ADD_DEFINITIONS(-DHAVE_MALLOC_USABLE_SIZE=1)
  ENDIF(HAVE_MALLOC_USABLE_SIZE)
ENDIF(UNIX)

if(NOT MSVC)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS m)
endif()

if(AT_NNPACK_ENABLED)
  include_directories(${NNPACK_INCLUDE_DIRS})
  list(APPEND ATen_CPU_DEPENDENCY_LIBS nnpack pthreadpool) # cpuinfo is added below
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
  IF(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
    string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  ELSE()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
  ENDIF()

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
  set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
  set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/sleef" ${CMAKE_BINARY_DIR}/sleef)
  set_property(TARGET sleef PROPERTY FOLDER "dependencies")
  list(APPEND ATen_THIRD_PARTY_INCLUDE ${CMAKE_BINARY_DIR}/include)
  link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS sleef)

  set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

  # Set these back. TODO: Use SLEEF_ to pass these instead
  set(BUILD_SHARED_LIBS ${__aten_sleef_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)
  set(BUILD_TESTS ${__aten_sleef_build_tests} CACHE BOOL "Build tests" FORCE)
endif()

IF(USE_CUDA AND NOT USE_ROCM)
  IF ($ENV{ATEN_STATIC_CUDA})
    # CuFFT has a complicated static story (especially around CUDA < 9) because it has device callback support
    # we first have to build a fake lib that links with no device callbacks,
    # and then we link against this object file.
    # This was recommended by the CuFFT team at NVIDIA

    # build fake CuFFT lib in build dir
    EXECUTE_PROCESS(COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc)
    if(${CUDA_VERSION_MAJOR} EQUAL "9")
      SET(CUFFT_FAKELINK_OPTIONS
        --generate-code arch=compute_35,code=sm_35
        --generate-code arch=compute_50,code=sm_50
        --generate-code arch=compute_60,code=sm_60
        --generate-code arch=compute_70,code=sm_70)
    elseif(${CUDA_VERSION_MAJOR} EQUAL "10")
      SET(CUFFT_FAKELINK_OPTIONS
        --generate-code arch=compute_35,code=sm_35
        --generate-code arch=compute_50,code=sm_50
        --generate-code arch=compute_60,code=sm_60
        --generate-code arch=compute_70,code=sm_70)
    else()
      MESSAGE(FATAL_ERROR "Unhandled major cuda version ${CUDA_VERSION_MAJOR}")
    endif()
    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a
      COMMAND "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" -o ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a -Xcompiler -fPIC
      ${CUFFT_FAKELINK_OPTIONS}
      --device-link ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc -lcufft_static -lculibos
      )
    ADD_CUSTOM_TARGET(FAKELINKED_CUFFT_TARGET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)
    add_library(FAKELINKED_CUFFT STATIC IMPORTED GLOBAL)
    add_dependencies(FAKELINKED_CUFFT FAKELINKED_CUFFT_TARGET)
    set_target_properties(FAKELINKED_CUFFT PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)

    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a
      FAKELINKED_CUFFT
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static.a
      )
  ELSE()
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS
      ${CUDA_LIBRARIES}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_curand_LIBRARY})
  ENDIF()

  if(CUDNN_FOUND)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${CUDNN_LIBRARIES})
  endif(CUDNN_FOUND)

  IF(USE_MAGMA)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${MAGMA_LIBRARIES})
    IF ($ENV{TH_BINARY_BUILD})
      list(APPEND ATen_CUDA_DEPENDENCY_LIBS
        "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    ENDIF($ENV{TH_BINARY_BUILD})
  ENDIF(USE_MAGMA)
  IF ($ENV{ATEN_STATIC_CUDA})
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a")
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a")
  ENDIF($ENV{ATEN_STATIC_CUDA})
ENDIF()

# NB: We're relying on cmake/Dependencies.cmake to appropriately setup HIP dependencies.
# In principle we could duplicate them, but handling the rocblas
# dependency is nontrivial.  So better not to copy-paste.
# Look for Note [rocblas cmake bug]

# Include CPU paths for CUDA/HIP as well
list(APPEND ATen_CUDA_INCLUDE ${ATen_CPU_INCLUDE})
list(APPEND ATen_HIP_INCLUDE ${ATen_CPU_INCLUDE})

# We have two libraries: libATen_cpu.so and libATen_cuda.so,
# with libATen_cuda.so depending on libATen_cpu.so.  The CPU library
# contains CPU code only.  libATen_cpu.so is invariant to the setting
# of USE_CUDA (it always builds the same way); libATen_cuda.so is only
# built when USE_CUDA=1 and CUDA is available.  (libATen_hip.so works
# the same way as libATen_cuda.so)
set(ATen_CPU_SRCS ${all_cpu_cpp})
if(AT_LINK_STYLE STREQUAL "INTERFACE")
  # Source code can't be added to an interface library, so it is
  # passed back to be compiled into the containing library
  add_library(ATen_cpu INTERFACE)
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ATEN_CPU_FILES_GEN_LIB)
else()
  add_library(ATen_cpu ${AT_LINK_STYLE} ${ATen_CPU_SRCS})
  if (ATen_THIRD_PARTY_INCLUDE)
    target_include_directories(ATen_cpu SYSTEM PRIVATE ${ATen_THIRD_PARTY_INCLUDE})
  endif()
  target_include_directories(ATen_cpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(ATen_cpu PRIVATE ${ATen_CPU_INCLUDE})
  target_link_libraries(ATen_cpu PUBLIC ${ATen_CPU_DEPENDENCY_LIBS})
  target_link_libraries(ATen_cpu PRIVATE ATEN_CPU_FILES_GEN_LIB)
  caffe2_interface_library(ATen_cpu ATen_cpu_library)
  # Set standard properties on the target
  torch_set_target_props(ATen_cpu)

  # Make sure these don't get built by parent
  set(ATen_CPU_SRCS)
endif()

if(USE_CUDA)
  set(ATen_CUDA_SRCS ${all_cuda_cpp})
  set(ATen_NVRTC_STUB_SRCS ${cuda_nvrtc_stub_cpp})
  if(AT_LINK_STYLE STREQUAL "INTERFACE")
    # Source code can't be added to an interface library, so it is
    # passed back to be compiled into the containing library
    add_library(ATen_cuda INTERFACE)
    list(APPEND ATen_CUDA_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
  else()
    message(FATAL_ERROR "Non-INTERFACE AT_LINK_STYLE no longer supported")
  endif()
endif()

if(USE_ROCM)
  set(ATen_HIP_SRCS ${all_hip_cpp})
  # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
  # See NOTE [ ATen NVRTC Stub and HIP ]
  set(ATen_NVRTC_STUB_SRCS ${hip_nvrtc_stub_cpp})
  if(AT_LINK_STYLE STREQUAL "INTERFACE")
    # Source code can't be added to an interface library, so it is
    # passed back to be compiled into the containing library
    add_library(ATen_hip INTERFACE)
    # NB: Instead of adding it to this list, we add it by hand
    # to caffe2_hip, because it needs to be a PRIVATE dependency
    # list(APPEND ATen_HIP_DEPENDENCY_LIBS ATEN_CUDA_FILES_GEN_LIB)
  else()
    message(FATAL_ERROR "Non-INTERFACE AT_LINK_STYLE not (yet) supported for ROCm build")
  endif()
endif()


if(NOT AT_LINK_STYLE STREQUAL "INTERFACE")
  if(USE_CUDA)
    if (NOT $ENV{ATEN_STATIC_CUDA})
      cuda_add_cublas_to_target(ATen_cuda)
      cuda_add_cufft_to_target(ATen_cuda)
    endif()
  endif()

  if(NOT MSVC)
    torch_compile_options(ATen_cpu)
    if(USE_CUDA)
      torch_compile_options(ATen_cuda)
    endif()
    if(USE_ROCM)
      torch_compile_options(ATen_hip)
    endif()
  endif()

  if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
    set_property(TARGET ATen_cpu PROPERTY CXX_STANDARD 11)
    if(USE_CUDA)
      set_property(TARGET ATen_cuda PROPERTY CXX_STANDARD 11)
    endif()
    if(USE_ROCM)
      set_property(TARGET ATen_hip PROPERTY CXX_STANDARD 11)
    endif()
  endif()
endif()

SET(ATEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_INCLUDE_DIR}")
CONFIGURE_FILE(ATenConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake"
  DESTINATION "${AT_INSTALL_SHARE_DIR}/cmake/ATen")

# https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
FOREACH(HEADER ${base_h} ${ATen_CORE_HEADERS} ${cuda_h} ${cudnn_h} ${hip_h} ${miopen_h})
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" HEADER_SUB ${HEADER})
  GET_FILENAME_COMPONENT(DIR ${HEADER_SUB} DIRECTORY)
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/${DIR})
ENDFOREACH()
# TODO: Install hip_generated_h when we have it
FOREACH(HEADER ${generated_h} ${cuda_generated_h})
  # NB: Assumed to be flat
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
INSTALL(FILES ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)

if(ATEN_NO_TEST)
  message("disable test because ATEN_NO_TEST is set")
else()
  add_subdirectory(test)
endif()

# Pass source, includes, and libs to parent
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_SRCS ${ATen_QUANTIZED_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_QUANTIZED_TEST_SRCS ${ATen_QUANTIZED_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
