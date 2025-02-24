# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn
#
# The following variables are used:
#  MKLDNN_USE_NATIVE_ARCH : Whether native CPU instructions should be used in MKLDNN. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF(NOT MKLDNN_FOUND)
  SET(MKLDNN_LIBRARIES)
  SET(MKLDNN_INCLUDE_DIR)

  SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
  SET(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")

  if(USE_XPU) # Build oneDNN GPU library
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # Linux
      # g++ is soft linked to /usr/bin/cxx, oneDNN would not treat it as an absolute path
      set(DNNL_HOST_COMPILER "g++")
      set(SYCL_CXX_DRIVER "icpx")
      set(DNNL_LIB_NAME "libdnnl.a")
    else()
      # Windows
      set(DNNL_HOST_COMPILER "DEFAULT")
      set(SYCL_CXX_DRIVER "icx")
      set(DNNL_LIB_NAME "dnnl.lib")
    endif()

    set(DNNL_MAKE_COMMAND "cmake" "--build" ".")
    include(ProcessorCount)
    ProcessorCount(proc_cnt)
    if((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
      list(APPEND DNNL_MAKE_COMMAND "-j" "$ENV{MAX_JOBS}")
      if(CMAKE_GENERATOR MATCHES "Make|Ninja")
        list(APPEND DNNL_MAKE_COMMAND "--" "-l" "$ENV{MAX_JOBS}")
      endif()
    endif()
    if(XPU_DEVICE_CXX_FLAGS)
      set(DNNL_CXX_FLAGS "-DCMAKE_CXX_FLAGS=${XPU_DEVICE_CXX_FLAGS}")
    else()
      set(DNNL_CXX_FLAGS "")
    endif()
    ExternalProject_Add(xpu_mkldnn_proj
      SOURCE_DIR ${MKLDNN_ROOT}
      PREFIX ${XPU_MKLDNN_DIR_PREFIX}
      BUILD_IN_SOURCE 0
      CMAKE_ARGS  -DCMAKE_C_COMPILER=icx
      -DCMAKE_CXX_COMPILER=${SYCL_CXX_DRIVER}
      ${DNNL_CXX_FLAGS}
      -DDNNL_GPU_RUNTIME=SYCL
      -DDNNL_CPU_RUNTIME=THREADPOOL
      -DDNNL_BUILD_TESTS=OFF
      -DDNNL_BUILD_EXAMPLES=OFF
      -DONEDNN_BUILD_GRAPH=ON
      -DDNNL_LIBRARY_TYPE=STATIC
      -DDNNL_DPCPP_HOST_COMPILER=${DNNL_HOST_COMPILER} # Use global cxx compiler as host compiler
      -G ${CMAKE_GENERATOR} # Align Generator to Torch
      BUILD_COMMAND ${DNNL_MAKE_COMMAND}
      BUILD_BYPRODUCTS "xpu_mkldnn_proj-prefix/src/xpu_mkldnn_proj-build/src/${DNNL_LIB_NAME}"
      INSTALL_COMMAND ""
    )

    ExternalProject_Get_Property(xpu_mkldnn_proj BINARY_DIR)
    set(__XPU_MKLDNN_BUILD_DIR ${BINARY_DIR})
    set(XPU_MKLDNN_LIBRARIES ${__XPU_MKLDNN_BUILD_DIR}/src/${DNNL_LIB_NAME})
    set(XPU_MKLDNN_INCLUDE ${__XPU_MKLDNN_BUILD_DIR}/include)
    # This target would be further linked to libtorch_xpu.so.
    # The libtorch_xpu.so would contain Conv&GEMM operators that depend on
    # oneDNN primitive implementations inside libdnnl.a.
    add_library(xpu_mkldnn INTERFACE)
    add_dependencies(xpu_mkldnn xpu_mkldnn_proj)
    target_link_libraries(xpu_mkldnn INTERFACE ${__XPU_MKLDNN_BUILD_DIR}/src/${DNNL_LIB_NAME})
    target_include_directories(xpu_mkldnn INTERFACE ${XPU_MKLDNN_INCLUDE})
  endif()

  IF(NOT APPLE AND NOT WIN32 AND NOT BUILD_LITE_INTERPRETER)
    MESSAGE("-- Will build oneDNN Graph")
    SET(LLGA_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")
    SET(BUILD_ONEDNN_GRAPH ON)
    SET(ONEDNN_BUILD_GRAPH ON CACHE BOOL "" FORCE)
  ENDIF(NOT APPLE AND NOT WIN32 AND NOT BUILD_LITE_INTERPRETER)

  IF(EXISTS "${MKLDNN_ROOT}/include/oneapi/dnnl/dnnl_ukernel.hpp")
    MESSAGE("-- Will build oneDNN UKERNEL")
    SET(DNNL_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)
  ENDIF(EXISTS "${MKLDNN_ROOT}/include/oneapi/dnnl/dnnl_ukernel.hpp")

  FIND_PACKAGE(BLAS)
  FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
  FIND_PATH(MKLDNN_INCLUDE_DIR dnnl.hpp dnnl.h dnnl_ukernel.hpp dnnl_ukernel.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include/oneapi/dnnl)
  IF(NOT MKLDNN_INCLUDE_DIR)
    MESSAGE("MKLDNN_INCLUDE_DIR not found")
    EXECUTE_PROCESS(COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
    FIND_PATH(MKLDNN_INCLUDE_DIR dnnl.hpp dnnl.h dnnl_ukernel.hpp dnnl_ukernel.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
  ENDIF(NOT MKLDNN_INCLUDE_DIR)
  IF(BUILD_ONEDNN_GRAPH)
    FIND_PATH(LLGA_INCLUDE_DIR dnnl_graph.hpp PATHS ${LLGA_ROOT} PATH_SUFFIXES include/oneapi/dnnl)
  ENDIF(BUILD_ONEDNN_GRAPH)

  IF(NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
    MESSAGE(STATUS "MKLDNN source files not found!")
    RETURN()
  ENDIF(NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
  LIST(APPEND MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})
  IF(BUILD_ONEDNN_GRAPH)
    LIST(APPEND MKLDNN_INCLUDE_DIR ${LLGA_INCLUDE_DIR})
  ENDIF(BUILD_ONEDNN_GRAPH)
  IF(MKL_FOUND)
    ADD_DEFINITIONS(-DIDEEP_USE_MKL)
    # Append to mkldnn dependencies
    LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
    LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  ELSE(MKL_FOUND)
    SET(MKLDNN_USE_MKL "NONE" CACHE STRING "" FORCE)
  ENDIF(MKL_FOUND)

  SET(MKL_cmake_included TRUE)
  IF(NOT MKLDNN_CPU_RUNTIME)
    SET(MKLDNN_CPU_RUNTIME "OMP" CACHE STRING "")
  ELSEIF(MKLDNN_CPU_RUNTIME STREQUAL "TBB")
    IF(TARGET TBB::tbb)
      MESSAGE(STATUS "MKL-DNN is using TBB")

      SET(TBB_cmake_included TRUE)
      SET(Threading_cmake_included TRUE)

      SET(DNNL_CPU_THREADING_RUNTIME ${MKLDNN_CPU_RUNTIME})
      INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIR})
      LIST(APPEND EXTRA_SHARED_LIBS TBB::tbb)
    ELSE()
      MESSAGE(FATAL_ERROR "MKLDNN_CPU_RUNTIME is set to TBB but TBB is not used")
    ENDIF()
  ENDIF()
  MESSAGE(STATUS "MKLDNN_CPU_RUNTIME = ${MKLDNN_CPU_RUNTIME}")

  SET(MKLDNN_CPU_RUNTIME ${MKLDNN_CPU_RUNTIME} CACHE STRING "" FORCE)
  SET(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
  SET(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
  SET(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
  SET(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
  SET(DNNL_GRAPH_CPU_RUNTIME ${MKLDNN_CPU_RUNTIME} CACHE STRING "" FORCE)

  IF(BUILD_ONEDNN_GRAPH)
    SET(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
  ENDIF(BUILD_ONEDNN_GRAPH)
  IF(MKLDNN_USE_NATIVE_ARCH)  # Disable HostOpts in MKLDNN unless MKLDNN_USE_NATIVE_ARCH is set.
    SET(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING "" FORCE)
  ELSE()
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      IF(CPU_INTEL)
        # Do not specify arch in oneDNN build option, for the portability in older systems
        SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
      ELSEIF(CPU_AARCH64)
        SET(DNNL_ARCH_OPT_FLAGS "-mcpu=generic" CACHE STRING "" FORCE)
      ENDIF()
    ELSE()
      SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
    ENDIF()
  ENDIF()

  ADD_SUBDIRECTORY(${MKLDNN_ROOT})

  IF(NOT TARGET dnnl)
    MESSAGE("Failed to include MKL-DNN target")
    RETURN()
  ENDIF(NOT TARGET dnnl)

  IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
    TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-maybe-uninitialized)
    TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-strict-overflow)
    TARGET_COMPILE_OPTIONS(dnnl PRIVATE -Wno-error=strict-overflow)
  ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  LIST(APPEND MKLDNN_LIBRARIES ${MKL_OPENMP_LIBRARY})
  LIST(APPEND MKLDNN_LIBRARIES dnnl)

  SET(MKLDNN_FOUND TRUE)
  MESSAGE(STATUS "Found MKL-DNN: TRUE")

ENDIF(NOT MKLDNN_FOUND)
