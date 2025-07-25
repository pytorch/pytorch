# Build file for the C10 CUDA.
#
# C10 CUDA is a minimal library, but it does depend on CUDA.

include(../../cmake/public/utils.cmake)
include(../../cmake/public/cuda.cmake)

# ---[ Configure macro file.
set(C10_CUDA_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
# Probably have to do this :(
configure_file(${CMAKE_CURRENT_LIST_DIR}/impl/cuda_cmake_macros.h.in
               ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h)

if(BUILD_LIBTORCHLESS)
  find_library(
    C10_CUDA_LIB c10_cuda
    PATHS $ENV{LIBTORCH_LIB_PATH}
    NO_DEFAULT_PATH)
else()
  set(C10_CUDA_LIB c10_cuda)
endif()

# Note: if you want to add ANY dependency to the c10 library, make sure you
# check with the core PyTorch developers as the dependency will be transitively
# passed on to all libraries dependent on PyTorch.

# Note: if you add a new source file/header, you will need to update
# torch/utils/hipify/cuda_to_hip_mappings.py for new files and headers you add
set(C10_CUDA_SRCS
    CUDAAllocatorConfig.cpp
    CUDACachingAllocator.cpp
    CUDADeviceAssertionHost.cpp
    CUDAException.cpp
    CUDAFunctions.cpp
    CUDAMallocAsyncAllocator.cpp
    CUDAMiscFunctions.cpp
    CUDAStream.cpp
    impl/CUDAGuardImpl.cpp
    impl/CUDATest.cpp
    driver_api.cpp)
set(C10_CUDA_HEADERS
    CUDAAllocatorConfig.h
    CUDACachingAllocator.h
    CUDADeviceAssertionHost.h
    CUDAException.h
    CUDAFunctions.h
    CUDAGuard.h
    CUDAMacros.h
    CUDAMathCompat.h
    CUDAMiscFunctions.h
    CUDAStream.h
    impl/CUDAGuardImpl.h
    impl/CUDATest.h)
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)

if(NOT BUILD_LIBTORCHLESS)
  torch_cuda_based_add_library(c10_cuda ${C10_CUDA_SRCS} ${C10_CUDA_HEADERS})
  torch_compile_options(c10_cuda)
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  # If building shared library, set dllimport/dllexport proper.
  target_compile_options(c10_cuda PRIVATE "-DC10_CUDA_BUILD_MAIN_LIB")
  # Enable hidden visibility if compiler supports it.
  if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
    target_compile_options(c10_cuda PRIVATE "-fvisibility=hidden")
  endif()

  # ---[ Dependency of c10_cuda
  target_link_libraries(c10_cuda PUBLIC ${C10_LIB} torch::cudart)

  if(NOT WIN32)
    target_link_libraries(c10_cuda PRIVATE dl)
    target_compile_options(c10_cuda
                           PRIVATE "-DPYTORCH_C10_DRIVER_API_SUPPORTED")
  endif()

  target_include_directories(
    c10_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<INSTALL_INTERFACE:include>)

  # ---[ Installation Note: for now, we will put all export path into one single
  # Caffe2Targets group to deal with the cmake deployment need. Inside the
  # Caffe2Targets set, the individual libraries like libc10.so and libcaffe2.so
  # are still self-contained.
  install(
    TARGETS c10_cuda
    EXPORT Caffe2Targets
    DESTINATION lib)

endif()

add_subdirectory(test)

foreach(file ${C10_CUDA_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/cuda/${dir})
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h
        DESTINATION include/c10/cuda/impl)

if(MSVC AND C10_CUDA_BUILD_SHARED_LIBS)
  install(
    FILES $<TARGET_PDB_FILE:c10_cuda>
    DESTINATION lib
    OPTIONAL)
endif()
