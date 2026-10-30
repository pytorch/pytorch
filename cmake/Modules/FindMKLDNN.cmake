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
#  USE_SYSTEM_ONEDNN      : Use an installed oneDNN (CMake package "dnnl", target DNNL::dnnl)
#                           for the CPU ideep path and, when USE_XPU is on, the XPU path.
#  PYTORCH_ONEDNN_MIN_VERSION : Oldest oneDNN accepted with USE_SYSTEM_ONEDNN. Defaults to the
#                           bundled release. Newer oneDNN 3.x releases are accepted.

# Bundled oneDNN consumed by third_party/ideep/mkl-dnn and the XPU ExternalProject.
set(PYTORCH_BUNDLED_ONEDNN_VERSION "3.13.3")

function(_pytorch_use_system_onednn)
  if(NOT PYTORCH_ONEDNN_MIN_VERSION)
    set(PYTORCH_ONEDNN_MIN_VERSION "${PYTORCH_BUNDLED_ONEDNN_VERSION}" CACHE STRING
      "Minimum oneDNN version when USE_SYSTEM_ONEDNN=ON. Newer oneDNN 3.x releases are accepted.")
    set(PYTORCH_ONEDNN_MIN_VERSION "${PYTORCH_BUNDLED_ONEDNN_VERSION}")
  endif()

  find_package(dnnl CONFIG QUIET)
  if(NOT dnnl_FOUND OR NOT TARGET DNNL::dnnl)
    message(FATAL_ERROR
      "USE_SYSTEM_ONEDNN is ON, but the oneDNN CMake package was not found. "
      "Install oneDNN ${PYTORCH_ONEDNN_MIN_VERSION} or newer and add its prefix to CMAKE_PREFIX_PATH. "
      "The package must provide dnnl-config.cmake and the imported target DNNL::dnnl.")
  endif()
  if(NOT dnnl_VERSION AND DEFINED DNNL_VERSION)
    set(dnnl_VERSION "${DNNL_VERSION}")
  endif()
  if(NOT dnnl_VERSION)
    message(FATAL_ERROR "The oneDNN CMake package did not report a version.")
  endif()
  if(dnnl_VERSION VERSION_LESS "${PYTORCH_ONEDNN_MIN_VERSION}" OR dnnl_VERSION VERSION_GREATER_EQUAL "4.0.0")
    message(FATAL_ERROR
      "Found oneDNN ${dnnl_VERSION}, but PyTorch requires ${PYTORCH_ONEDNN_MIN_VERSION} or newer within the 3.x series. "
      "Install a matching oneDNN, or pass -DPYTORCH_ONEDNN_MIN_VERSION=<version> to accept a different minimum.")
  endif()

  if(NOT MKLDNN_CPU_RUNTIME)
    set(MKLDNN_CPU_RUNTIME "OMP")
  endif()
  if(NOT DNNL_CPU_RUNTIME)
    message(FATAL_ERROR
      "The oneDNN CMake package did not report DNNL_CPU_RUNTIME. "
      "PyTorch's CPU kernels need a oneDNN built with a CPU runtime such as OMP.")
  endif()
  if(NOT DNNL_CPU_RUNTIME STREQUAL "${MKLDNN_CPU_RUNTIME}")
    message(FATAL_ERROR
      "System oneDNN was built with DNNL_CPU_RUNTIME=${DNNL_CPU_RUNTIME}, "
      "but PyTorch is configured with MKLDNN_CPU_RUNTIME=${MKLDNN_CPU_RUNTIME}. "
      "Rebuild oneDNN with -DDNNL_CPU_RUNTIME=${MKLDNN_CPU_RUNTIME}, "
      "or pass -DMKLDNN_CPU_RUNTIME=${DNNL_CPU_RUNTIME}.")
  endif()
  if(USE_XPU AND NOT DNNL_GPU_RUNTIME STREQUAL "SYCL")
    message(FATAL_ERROR
      "USE_XPU=ON requires a system oneDNN built with -DDNNL_GPU_RUNTIME=SYCL. "
      "This oneDNN reports DNNL_GPU_RUNTIME=${DNNL_GPU_RUNTIME}. "
      "One installed libdnnl serves both the CPU and XPU paths.")
  endif()

  get_target_property(_raw DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
  set(_includes "")
  if(NOT _raw MATCHES "NOTFOUND")
    foreach(_inc IN LISTS _raw)
      if(_inc MATCHES "^\\$<")
        continue()
      endif()
      list(APPEND _includes "${_inc}")
      if(EXISTS "${_inc}/oneapi/dnnl/dnnl.hpp")
        # ideep includes <dnnl.hpp>. Some packages only install that header under oneapi/dnnl.
        list(APPEND _includes "${_inc}/oneapi/dnnl")
      endif()
    endforeach()
  endif()
  if(NOT _includes)
    message(FATAL_ERROR "DNNL::dnnl does not publish an include directory.")
  endif()

  set(_graph_header "")
  set(_graph_sycl_header "")
  foreach(_inc IN LISTS _includes)
    if(NOT _graph_header AND EXISTS "${_inc}/oneapi/dnnl/dnnl_graph.hpp")
      set(_graph_header "${_inc}/oneapi/dnnl/dnnl_graph.hpp")
    elseif(NOT _graph_header AND EXISTS "${_inc}/dnnl_graph.hpp")
      set(_graph_header "${_inc}/dnnl_graph.hpp")
    endif()
    if(NOT _graph_sycl_header AND EXISTS "${_inc}/oneapi/dnnl/dnnl_graph_sycl.hpp")
      set(_graph_sycl_header "${_inc}/oneapi/dnnl/dnnl_graph_sycl.hpp")
    elseif(NOT _graph_sycl_header AND EXISTS "${_inc}/dnnl_graph_sycl.hpp")
      set(_graph_sycl_header "${_inc}/dnnl_graph_sycl.hpp")
    endif()
  endforeach()

  set(_need_graph FALSE)
  if(NOT APPLE AND NOT WIN32 AND NOT BUILD_LITE_INTERPRETER)
    set(_need_graph TRUE)
  endif()
  if(_graph_header)
    set(_build_graph ON)
  else()
    set(_build_graph OFF)
  endif()
  if(_need_graph AND NOT _build_graph)
    message(FATAL_ERROR
      "System oneDNN ${dnnl_VERSION} has no graph API (dnnl_graph.hpp). "
      "Rebuild it with -DONEDNN_BUILD_GRAPH=ON.")
  endif()
  if(USE_XPU AND _build_graph AND NOT _graph_sycl_header)
    message(FATAL_ERROR
      "System oneDNN is missing dnnl_graph_sycl.hpp. "
      "Rebuild it with -DDNNL_GPU_RUNTIME=SYCL and -DONEDNN_BUILD_GRAPH=ON.")
  endif()

  set(_ideep_root "${PROJECT_SOURCE_DIR}/third_party/ideep")
  find_path(IDEEP_INCLUDE_DIR ideep.hpp PATHS "${_ideep_root}" PATH_SUFFIXES include NO_DEFAULT_PATH)
  if(NOT IDEEP_INCLUDE_DIR)
    message(FATAL_ERROR
      "USE_SYSTEM_ONEDNN is ON, but ideep headers were not found in ${_ideep_root}. "
      "PyTorch still compiles against ideep. Initialize the third_party/ideep submodule.")
  endif()

  set(_onednn_includes "${_includes}")
  list(APPEND _includes "${IDEEP_INCLUDE_DIR}")

  if(USE_XPU)
    add_library(xpu_mkldnn INTERFACE)
    target_link_libraries(xpu_mkldnn INTERFACE DNNL::dnnl)
    set(XPU_MKLDNN_LIBRARIES DNNL::dnnl PARENT_SCOPE)
    set(XPU_MKLDNN_INCLUDE "${_onednn_includes}" PARENT_SCOPE)
  endif()

  set(BUILD_ONEDNN_GRAPH ${_build_graph} PARENT_SCOPE)
  set(MKLDNN_INCLUDE_DIR "${_includes}" PARENT_SCOPE)
  set(MKLDNN_LIBRARIES DNNL::dnnl PARENT_SCOPE)
  set(MKLDNN_FOUND TRUE PARENT_SCOPE)
  set(MKLDNN_CPU_RUNTIME "${MKLDNN_CPU_RUNTIME}" PARENT_SCOPE)
  message(STATUS
    "Using system oneDNN ${dnnl_VERSION} "
    "(CPU runtime ${DNNL_CPU_RUNTIME}, GPU runtime ${DNNL_GPU_RUNTIME})")
  message(STATUS "Found MKL-DNN: TRUE")
endfunction()

IF(NOT MKLDNN_FOUND)
  SET(MKLDNN_LIBRARIES)
  SET(MKLDNN_INCLUDE_DIR)

  if(USE_SYSTEM_ONEDNN)
    _pytorch_use_system_onednn()
    return()
  endif()

  SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
  SET(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")

  if(USE_XPU) # Build oneDNN GPU library
    if(WIN32)
      # Windows
      set(DNNL_HOST_COMPILER "DEFAULT")
      set(DNNL_C_COMPILER "icx")
      set(SYCL_CXX_DRIVER "icx")
      set(DNNL_LIB_NAME "dnnl.lib")
    elseif(LINUX)
      # Linux
      # g++ is soft linked to /usr/bin/cxx, oneDNN would not treat it as an absolute path
      set(DNNL_HOST_COMPILER "g++")
      if("${XPU_SYCL_COMPILER}" MATCHES "icx")
        set(DNNL_C_COMPILER "icx")
        set(SYCL_CXX_DRIVER "icpx")
      elseif("${XPU_SYCL_COMPILER}" MATCHES "dpclang")
        set(DNNL_C_COMPILER "dpclang")
        set(SYCL_CXX_DRIVER "dpclang++")
      else()
        message(FATAL_ERROR "Unsupported SYCL compiler: ${XPU_SYCL_COMPILER}")
      endif()
      set(DNNL_LIB_NAME "libdnnl.a")
    else()
      MESSAGE(FATAL_ERROR "OneDNN for Intel GPU in PyTorch currently supports only Windows and Linux.
                           Detected system '${CMAKE_SYSTEM_NAME}' is not supported.")
    endif()

    if(XPU_MKLDNN_DIR_PREFIX)
      set(_xpu_mkldnn_prefix "${XPU_MKLDNN_DIR_PREFIX}")
    else()
      set(_xpu_mkldnn_prefix "${CMAKE_CURRENT_BINARY_DIR}/xpu_mkldnn_proj-prefix")
    endif()
    set(_xpu_mkldnn_src_dir "${_xpu_mkldnn_prefix}/src/xpu_mkldnn_proj")
    set(_xpu_mkldnn_build_dir "${_xpu_mkldnn_prefix}/src/xpu_mkldnn_proj-build")
    set(_xpu_mkldnn_lib "${_xpu_mkldnn_build_dir}/src/${DNNL_LIB_NAME}")

    if(EXISTS "${_xpu_mkldnn_lib}"
        AND IS_DIRECTORY "${_xpu_mkldnn_src_dir}/include"
        AND IS_DIRECTORY "${_xpu_mkldnn_build_dir}/include")
      message(STATUS "Reusing existing oneDNN XPU build: ${_xpu_mkldnn_lib}")
      add_custom_target(xpu_mkldnn_proj)
      set(XPU_MKLDNN_LIBRARIES "${_xpu_mkldnn_lib}")
      set(XPU_MKLDNN_INCLUDE "${_xpu_mkldnn_src_dir}/include" "${_xpu_mkldnn_build_dir}/include")
    else()
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
        GIT_TAG v${PYTORCH_BUNDLED_ONEDNN_VERSION}
        PREFIX "${_xpu_mkldnn_prefix}"
        BUILD_IN_SOURCE 0
        CMAKE_ARGS  -DCMAKE_C_COMPILER=${DNNL_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${SYCL_CXX_DRIVER}
        -DDNNL_GPU_RUNTIME=SYCL
        -DDNNL_CPU_RUNTIME=NONE
        -DDNNL_BUILD_TESTS=OFF
        -DDNNL_BUILD_EXAMPLES=OFF
        -DONEDNN_BUILD_GRAPH=ON
        -DDNNL_LIBRARY_TYPE=STATIC
        -DDNNL_DPCPP_HOST_COMPILER=${DNNL_HOST_COMPILER}
        -G ${CMAKE_GENERATOR}
        BUILD_COMMAND ${DNNL_MAKE_COMMAND}
        BUILD_BYPRODUCTS "${_xpu_mkldnn_build_dir}/src/${DNNL_LIB_NAME}"
        INSTALL_COMMAND ""
      )

      ExternalProject_Get_Property(xpu_mkldnn_proj SOURCE_DIR BINARY_DIR)
      set(XPU_MKLDNN_LIBRARIES ${BINARY_DIR}/src/${DNNL_LIB_NAME})
      set(XPU_MKLDNN_INCLUDE ${SOURCE_DIR}/include ${BINARY_DIR}/include)
    endif()
    # This target would be further linked to libtorch_xpu.so.
    # The libtorch_xpu.so would contain Conv&GEMM operators that depend on
    # oneDNN primitive implementations inside libdnnl.a.
    add_library(xpu_mkldnn INTERFACE)
    add_dependencies(xpu_mkldnn xpu_mkldnn_proj)
    target_link_libraries(xpu_mkldnn INTERFACE ${XPU_MKLDNN_LIBRARIES})
    target_include_directories(xpu_mkldnn INTERFACE ${XPU_MKLDNN_INCLUDE})
  endif()

  IF(NOT APPLE AND NOT WIN32 AND NOT BUILD_LITE_INTERPRETER)
    MESSAGE("-- Will build oneDNN Graph")
    SET(LLGA_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")
    SET(BUILD_ONEDNN_GRAPH ON)
    SET(ONEDNN_BUILD_GRAPH ON CACHE BOOL "" FORCE)
  ENDIF(NOT APPLE AND NOT WIN32 AND NOT BUILD_LITE_INTERPRETER)

  IF(EXISTS "${MKLDNN_ROOT}/include/oneapi/dnnl/dnnl_ukernel.hpp")
    IF(CPU_POWER OR CPU_RISCV)
      SET(DNNL_EXPERIMENTAL_UKERNEL OFF CACHE BOOL "" FORCE)
    ELSE()
      MESSAGE("-- Will build oneDNN UKERNEL")
      SET(DNNL_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)
    ENDIF()
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
