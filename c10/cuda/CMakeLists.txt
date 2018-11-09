# Build file for the C10 CUDA.
#
# C10 CUDA is a minimal library, but it does depend on CUDA.

include(../../cmake/public/utils.cmake)
include(../../cmake/public/cuda.cmake)

# ---[ Configure macro file.
set(C10_CUDA_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
# Probably have to do this :(
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/cuda_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h)

# Note: if you want to add ANY dependency to the c10 library, make sure you
# check with the core PyTorch developers as the dependendency will be
# transitively passed on to all libraries dependent on PyTorch.
file(GLOB C10_CUDA_SRCS
        *.cpp
        *.cu
        impl/*.cpp
        impl/*.cu
        )
file(GLOB_RECURSE C10_CUDA_ALL_TEST_FILES test/*.cpp)
file(GLOB_RECURSE C10_CUDA_HEADERS *.h)
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
torch_cuda_based_add_library(c10_cuda ${C10_CUDA_SRCS} ${C10_CUDA_HEADERS})
set(CUDA_LINK_LIBRARIES_KEYWORD)
# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10_cuda PRIVATE "-DC10_CUDA_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_cuda PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_cuda
target_link_libraries(c10_cuda PUBLIC c10)

target_link_libraries(c10_cuda INTERFACE caffe2::cudart)

target_include_directories(
    c10_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

add_subdirectory(test)

# ---[ Installation
# Note: for now, we will put all export path into one single Caffe2Targets group
# to deal with the cmake deployment need. Inside the Caffe2Targets set, the
# individual libraries like libc10.so and libcaffe2.so are still self-contained.
install(TARGETS c10_cuda EXPORT Caffe2Targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/c10/cuda/impl/cuda_cmake_macros.h
  DESTINATION include/c10/cuda/impl)
