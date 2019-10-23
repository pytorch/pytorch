# Build file for the C10 CUDA.

# ---[ Configure macro file.
set(C10_CUDA_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
# Probably have to do this :(
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/cuda_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h)

# Note: if you want to add ANY dependency to the c10 library, make sure you
# check with the core PyTorch developers as the dependendency will be
# transitively passed on to all libraries dependent on PyTorch.

# Note: if you add a new source file/header, you will need to update
# torch/utils/hipify/cuda_to_hip_mappings.py for new files
# and headers you add
set(C10_CUDA_SRCS
    CUDAStream.cpp
    CUDACachingAllocator.cpp
    impl/CUDAGuardImpl.cpp
    impl/CUDATest.cpp
)
set(C10_CUDA_HEADERS
    CUDAException.h
    CUDAGuard.h
    CUDAMacros.h
    CUDAMathCompat.h
    CUDAStream.h
    impl/CUDAGuardImpl.h
    impl/CUDATest.h
)

set(C10_CUDA_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_BINARY_DIR})

add_subdirectory(test)

foreach(file ${C10_CUDA_HEADERS})
  get_filename_component( dir ${file} DIRECTORY )
  install( FILES ${file} DESTINATION include/c10/cuda/${dir} )
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h
  DESTINATION include/c10/cuda/impl)

# Pass source and includes to parents
set(C10_CUDA_SRCS ${C10_CUDA_SRCS} PARENT_SCOPE)
set(C10_CUDA_INCLUDE ${C10_CUDA_INCLUDE} PARENT_SCOPE)
