# GLOO_SRCS is the list of source files that we need to build libgloo.
set(GLOO_SRCS)

# GLOO_HDRS is the list of header files that we need to install.
set(GLOO_HDRS)

# Compiled sources in root directory
list(APPEND GLOO_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
  )

list(APPEND GLOO_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/allgather_ring.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_halving_doubling.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_ring.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_ring_chunked.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_all_to_all.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/barrier_all_to_one.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_one_to_all.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/math.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/pairwise_exchange.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/types.h"
  )

if(USE_CUDA)
  file(GLOB GLOO_CUDA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.cu"
    )

  file(GLOB GLOO_CUDA_HDRS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.h"
    )
endif()

add_subdirectory(common)
add_subdirectory(mpi)
add_subdirectory(nccl)
add_subdirectory(rendezvous)
add_subdirectory(transport)

# Depend on pthreads for transport device threads
list(APPEND gloo_DEPENDENCY_LIBS pthread)

# Enable the following to get a list of source files
if(FALSE)
  message(STATUS "Sources: ")
  foreach(tmp ${GLOO_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# Write configuration header.
# Set variables so macros have GLOO_ prefix.
set(GLOO_USE_CUDA ${USE_CUDA})
set(GLOO_USE_REDIS ${USE_REDIS})
set(GLOO_USE_IBVERBS ${USE_IBVERBS})
configure_file(config.h.in config.h)

# Prepend include path so that generated config.h is picked up.
# Note that it is included as "gloo/config.h" to add parent directory.
get_filename_component(PARENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PATH)
include_directories(BEFORE ${PARENT_BINARY_DIR})

add_library(gloo ${GLOO_STATIC_OR_SHARED} ${GLOO_SRCS})
target_link_libraries(gloo ${gloo_DEPENDENCY_LIBS})
if(USE_CUDA)
  cuda_add_library(gloo_cuda ${GLOO_CUDA_SRCS} ${GLOO_STATIC_OR_SHARED})
  target_link_libraries(gloo_cuda gloo ${gloo_cuda_DEPENDENCY_LIBS})

  # Add explicit dependency if NCCL is built from third-party.
  # Without dependency, make -jN with N>1 can fail if the NCCL build
  # hasn't finished when CUDA targets are linked.
  if(NCCL_EXTERNAL)
    add_dependencies(gloo_cuda nccl_external)
  endif()
endif()

# Add path to gloo/config.h so it is picked up by parent projects.
include_directories(BEFORE ${PARENT_BINARY_DIR})

# Install if necessary.
# If the Gloo build is included from another project's build, it may
# want to statically link with Gloo and not install any artifacts.
if(GLOO_INSTALL)
  install(TARGETS gloo DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  if(USE_CUDA)
    install(TARGETS gloo_cuda DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  endif()
  install(FILES ${CMAKE_BINARY_DIR}/gloo/config.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gloo)
  foreach(HEADER ${GLOO_HDRS})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "gloo" DIR ${DIR})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
  endforeach()
  if(USE_CUDA)
    foreach(HEADER ${GLOO_CUDA_HDRS})
      string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
      string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "gloo" DIR ${DIR})
      install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
    endforeach()
  endif()
endif()

if(BUILD_TEST)
add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
add_subdirectory(benchmark)
endif()
