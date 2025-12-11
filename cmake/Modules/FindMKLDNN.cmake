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
#  MKLDNN_CPU_GIT_TAG    : Git tag/commit for CPU oneDNN version. Default: v3.9.2
#  MKLDNN_XPU_GIT_TAG    : Git tag/commit for XPU oneDNN version. Default: v3.10.2

IF(NOT MKLDNN_FOUND)
  SET(MKLDNN_LIBRARIES)
  SET(MKLDNN_INCLUDE_DIR)

  SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
  SET(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")
  
  # Use proper binary directory - CMAKE_BINARY_DIR is the top-level build directory
  if(NOT DEFINED MKLDNN_BUILD_DIR)
    if(DEFINED CMAKE_BINARY_DIR)
      set(MKLDNN_BUILD_DIR "${CMAKE_BINARY_DIR}")
    else()
      set(MKLDNN_BUILD_DIR "${PROJECT_BINARY_DIR}")
    endif()
  endif()
  
  # Set default versions
  if(NOT DEFINED MKLDNN_CPU_GIT_TAG)
    set(MKLDNN_CPU_GIT_TAG "v3.9.2" CACHE STRING "Git tag for CPU oneDNN")
  endif()
  if(NOT DEFINED MKLDNN_XPU_GIT_TAG)
    set(MKLDNN_XPU_GIT_TAG "v3.10.2" CACHE STRING "Git tag for XPU oneDNN")
  endif()

  if(USE_XPU) # Build oneDNN GPU library
    if(WIN32)
      # Windows
      set(DNNL_HOST_COMPILER "DEFAULT")
      set(SYCL_CXX_DRIVER "icx")
      set(DNNL_LIB_NAME "dnnl.lib")
    elseif(LINUX)
      # Linux
      # g++ is soft linked to /usr/bin/cxx, oneDNN would not treat it as an absolute path
      set(DNNL_HOST_COMPILER "g++")
      set(SYCL_CXX_DRIVER "icpx")
      set(DNNL_LIB_NAME "libdnnl.a")
    else()
      MESSAGE(FATAL_ERROR "OneDNN for Intel GPU in PyTorch currently supports only Windows and Linux.
                           Detected system '${CMAKE_SYSTEM_NAME}' is not supported.")
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
    ExternalProject_Add(xpu_mkldnn_proj
      GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN
      GIT_TAG ${MKLDNN_XPU_GIT_TAG}
      PREFIX ${MKLDNN_BUILD_DIR}/xpu_mkldnn
      BUILD_IN_SOURCE 0
      CMAKE_ARGS  -DCMAKE_C_COMPILER=icx
      -DCMAKE_CXX_COMPILER=${SYCL_CXX_DRIVER}
      -DDNNL_GPU_RUNTIME=SYCL
      -DDNNL_CPU_RUNTIME=THREADPOOL
      -DDNNL_BUILD_TESTS=OFF
      -DDNNL_BUILD_EXAMPLES=OFF
      -DONEDNN_BUILD_GRAPH=ON
      -DDNNL_LIBRARY_TYPE=STATIC
      -DDNNL_DPCPP_HOST_COMPILER=${DNNL_HOST_COMPILER} # Use global cxx compiler as host compiler
      -G ${CMAKE_GENERATOR} # Align Generator to Torch
      BUILD_COMMAND ${DNNL_MAKE_COMMAND}
      BUILD_BYPRODUCTS "${MKLDNN_BUILD_DIR}/xpu_mkldnn/src/xpu_mkldnn_proj-build/src/${DNNL_LIB_NAME}"
      INSTALL_COMMAND ""
    )

    # Set paths explicitly to avoid path duplication issues
    set(XPU_MKLDNN_SOURCE_DIR "${MKLDNN_BUILD_DIR}/xpu_mkldnn/src/xpu_mkldnn_proj")
    set(XPU_MKLDNN_BINARY_DIR "${MKLDNN_BUILD_DIR}/xpu_mkldnn/src/xpu_mkldnn_proj-build")
    set(XPU_MKLDNN_LIBRARIES ${XPU_MKLDNN_BINARY_DIR}/src/${DNNL_LIB_NAME})
    set(XPU_MKLDNN_INCLUDE ${XPU_MKLDNN_SOURCE_DIR}/include ${XPU_MKLDNN_BINARY_DIR}/include)
    
    # Create placeholder directories to avoid CMake configuration errors
    file(MAKE_DIRECTORY ${XPU_MKLDNN_SOURCE_DIR}/include)
    file(MAKE_DIRECTORY ${XPU_MKLDNN_BINARY_DIR}/include)
    
    # This target would be further linked to libtorch_xpu.so.
    # The libtorch_xpu.so would contain Conv&GEMM operators that depend on
    # oneDNN primitive implementations inside libdnnl.a.
    add_library(xpu_mkldnn INTERFACE)
    add_dependencies(xpu_mkldnn xpu_mkldnn_proj)
    target_link_libraries(xpu_mkldnn INTERFACE ${XPU_MKLDNN_LIBRARIES})
    target_include_directories(xpu_mkldnn INTERFACE
      $<BUILD_INTERFACE:${XPU_MKLDNN_INCLUDE}>
    )
  endif()

  # Build CPU oneDNN using ExternalProject for version control
  IF(NOT BUILD_LITE_INTERPRETER)
    MESSAGE("-- Will build oneDNN for CPU with version ${MKLDNN_CPU_GIT_TAG}")
    # oneDNN Graph is only supported on Linux and Windows
    IF(NOT APPLE AND NOT WIN32)
      SET(BUILD_ONEDNN_GRAPH ON)
      SET(ONEDNN_BUILD_GRAPH ON CACHE BOOL "" FORCE)
    ELSE()
      SET(BUILD_ONEDNN_GRAPH OFF)
      SET(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    ENDIF()
    
    # Prepare build command with parallel jobs
    set(CPU_DNNL_MAKE_COMMAND "cmake" "--build" ".")
    include(ProcessorCount)
    ProcessorCount(cpu_proc_cnt)
    if((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${cpu_proc_cnt}))
      list(APPEND CPU_DNNL_MAKE_COMMAND "-j" "$ENV{MAX_JOBS}")
      if(CMAKE_GENERATOR MATCHES "Make|Ninja")
        list(APPEND CPU_DNNL_MAKE_COMMAND "--" "-l" "$ENV{MAX_JOBS}")
      endif()
    endif()
    
    # Determine library name based on platform
    if(WIN32)
      set(CPU_DNNL_LIB_NAME "dnnl.lib")
    else()
      set(CPU_DNNL_LIB_NAME "libdnnl.a")
    endif()
    
    SET(MKL_cmake_included TRUE)
    # Determine CPU runtime and architecture flags
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
        MESSAGE(FATAL_ERROR "MKLDNN_CPU_RUNTIME is set to TBB but TBB is not found")
      ENDIF()
    ENDIF()
    MESSAGE(STATUS "MKLDNN_CPU_RUNTIME = ${MKLDNN_CPU_RUNTIME}")
    
    SET(MKLDNN_CPU_RUNTIME ${MKLDNN_CPU_RUNTIME} CACHE STRING "" FORCE)
    SET(DNNL_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
    SET(DNNL_BUILD_EXAMPLES FALSE CACHE BOOL "" FORCE)
    SET(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
    SET(DNNL_ENABLE_PRIMITIVE_CACHE TRUE CACHE BOOL "" FORCE)
    SET(DNNL_GRAPH_CPU_RUNTIME ${MKLDNN_CPU_RUNTIME} CACHE STRING "" FORCE)
    SET(DNNL_GRAPH_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
    
    IF(MKLDNN_USE_NATIVE_ARCH)
      SET(CPU_DNNL_ARCH_OPT_FLAGS "HostOpts")
      SET(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING "" FORCE)
    ELSE()
      IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        IF(CPU_INTEL)
          SET(CPU_DNNL_ARCH_OPT_FLAGS "")
          SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
        ELSEIF(CPU_AARCH64)
          SET(CPU_DNNL_ARCH_OPT_FLAGS "-mcpu=generic")
          SET(DNNL_ARCH_OPT_FLAGS "-mcpu=generic" CACHE STRING "" FORCE)
        ELSE()
          SET(CPU_DNNL_ARCH_OPT_FLAGS "")
          SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
        ENDIF()
      ELSE()
        SET(CPU_DNNL_ARCH_OPT_FLAGS "")
        SET(DNNL_ARCH_OPT_FLAGS "" CACHE STRING "" FORCE)
      ENDIF()
    ENDIF()
    
    # Determine UKERNEL setting
    SET(CPU_DNNL_EXPERIMENTAL_UKERNEL OFF)
    IF(NOT CPU_POWER AND NOT CPU_RISCV)
      SET(CPU_DNNL_EXPERIMENTAL_UKERNEL ON)
      SET(DNNL_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)
      MESSAGE("-- Will build oneDNN UKERNEL for CPU")
    ELSE()
      SET(DNNL_EXPERIMENTAL_UKERNEL OFF CACHE BOOL "" FORCE)
    ENDIF()
    
    # Build oneDNN for CPU as external project
    ExternalProject_Add(cpu_mkldnn_proj
      GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN
      GIT_TAG ${MKLDNN_CPU_GIT_TAG}
      PREFIX ${MKLDNN_BUILD_DIR}/cpu_mkldnn
      BUILD_IN_SOURCE 0
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DDNNL_CPU_RUNTIME=${MKLDNN_CPU_RUNTIME}
        -DDNNL_BUILD_TESTS=OFF
        -DDNNL_BUILD_EXAMPLES=OFF
        -DONEDNN_BUILD_GRAPH=${BUILD_ONEDNN_GRAPH}
        -DDNNL_LIBRARY_TYPE=STATIC
        -DDNNL_ENABLE_PRIMITIVE_CACHE=ON
        -DDNNL_ARCH_OPT_FLAGS=${CPU_DNNL_ARCH_OPT_FLAGS}
        -DDNNL_EXPERIMENTAL_UKERNEL=${CPU_DNNL_EXPERIMENTAL_UKERNEL}
        -DDNNL_GRAPH_CPU_RUNTIME=${MKLDNN_CPU_RUNTIME}
        -DDNNL_GRAPH_LIBRARY_TYPE=STATIC
      BUILD_COMMAND ${CPU_DNNL_MAKE_COMMAND}
      BUILD_BYPRODUCTS "${MKLDNN_BUILD_DIR}/cpu_mkldnn/src/cpu_mkldnn_proj-build/src/${CPU_DNNL_LIB_NAME}"
      INSTALL_COMMAND ""
    )
    
    # Set paths explicitly to avoid path duplication issues
    set(CPU_MKLDNN_SOURCE_DIR "${MKLDNN_BUILD_DIR}/cpu_mkldnn/src/cpu_mkldnn_proj")
    set(CPU_MKLDNN_BINARY_DIR "${MKLDNN_BUILD_DIR}/cpu_mkldnn/src/cpu_mkldnn_proj-build")
    set(CPU_MKLDNN_LIBRARIES ${CPU_MKLDNN_BINARY_DIR}/src/${CPU_DNNL_LIB_NAME})
    set(CPU_MKLDNN_INCLUDE ${CPU_MKLDNN_SOURCE_DIR}/include ${CPU_MKLDNN_BINARY_DIR}/include)
    
    # Create placeholder directories to avoid CMake configuration errors
    # ExternalProject will populate these during build
    file(MAKE_DIRECTORY ${CPU_MKLDNN_SOURCE_DIR}/include)
    file(MAKE_DIRECTORY ${CPU_MKLDNN_BINARY_DIR}/include)
    
    # Create interface library for CPU oneDNN
    add_library(cpu_mkldnn INTERFACE)
    add_dependencies(cpu_mkldnn cpu_mkldnn_proj)
    target_link_libraries(cpu_mkldnn INTERFACE ${CPU_MKLDNN_LIBRARIES})
    # Use absolute paths directly, no need for BUILD_INTERFACE generator expression
    # since these are already absolute paths
    foreach(include_dir ${CPU_MKLDNN_INCLUDE})
      target_include_directories(cpu_mkldnn INTERFACE ${include_dir})
    endforeach()
    
    # Also create 'dnnl' target for compatibility
    add_library(dnnl ALIAS cpu_mkldnn)
    
    # Setup MKLDNN variables for rest of the build system
    SET(MKLDNN_INCLUDE_DIR ${CPU_MKLDNN_INCLUDE})
    LIST(APPEND MKLDNN_LIBRARIES cpu_mkldnn)
  ENDIF(NOT BUILD_LITE_INTERPRETER)
  
  # Find and include IDEEP headers (C++ wrapper for oneDNN)
  FIND_PACKAGE(BLAS)
  FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
  IF(NOT IDEEP_INCLUDE_DIR)
    MESSAGE("IDEEP_INCLUDE_DIR not found, attempting to update submodule")
    EXECUTE_PROCESS(COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init ideep WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third_party)
    FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
  ENDIF()
  
  IF(IDEEP_INCLUDE_DIR)
    LIST(APPEND MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})
  ELSE()
    MESSAGE(WARNING "IDEEP headers not found, some CPU features may be unavailable")
  ENDIF()
  
  # MKL integration
  IF(MKL_FOUND)
    ADD_DEFINITIONS(-DIDEEP_USE_MKL)
    LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
    LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  ELSE(MKL_FOUND)
    SET(MKLDNN_USE_MKL "NONE" CACHE STRING "" FORCE)
  ENDIF(MKL_FOUND)
  
  # Append OpenMP library
  LIST(APPEND MKLDNN_LIBRARIES ${MKL_OPENMP_LIBRARY})

  SET(MKLDNN_FOUND TRUE)
  MESSAGE(STATUS "Found MKL-DNN: TRUE")

ENDIF(NOT MKLDNN_FOUND)
