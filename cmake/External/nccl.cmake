if (NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  # Try finding system-wide NCCL first
  find_package(nccl QUIET)
  if (NCCL_FOUND)
    set(NCCL_EXTERNAL FALSE)
  else()
    # Build directory
    set(nccl_PREFIX ${PROJECT_SOURCE_DIR}/third-party/nccl)

    # Import ExternalProject functions
    include(ExternalProject)

    # Trigger NCCL build
    ExternalProject_Add(nccl_external
      SOURCE_DIR ${nccl_PREFIX}
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make "CXX=${CMAKE_CXX_COMPILER}"
      INSTALL_COMMAND ""
      )

    set(NCCL_FOUND TRUE)
    set(nccl_INCLUDE_DIR ${nccl_PREFIX}/build/include)
    set(nccl_LIBRARIES ${nccl_PREFIX}/build/lib/libnccl_static.a)
    set(nccl_LIBRARY_DIRS ${nccl_PREFIX}/build/lib)
    set(NCCL_EXTERNAL TRUE)
  endif()
endif()
