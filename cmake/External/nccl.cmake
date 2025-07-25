if(NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  if(USE_SYSTEM_NCCL)
    # NCCL_ROOT, NCCL_LIB_DIR, NCCL_INCLUDE_DIR will be accounted in the
    # following line.
    find_package(NCCL REQUIRED)
    if(NCCL_FOUND)
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
      target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    endif()
  else()
    torch_cuda_get_nvcc_gencode_flag(NVCC_GENCODE)
    string(REPLACE "-gencode;" "-gencode=" NVCC_GENCODE "${NVCC_GENCODE}")
    # this second replacement is needed when there are multiple archs
    string(REPLACE ";-gencode" " -gencode" NVCC_GENCODE "${NVCC_GENCODE}")

    if(DEFINED ENV{MAX_JOBS})
      set(MAX_JOBS "$ENV{MAX_JOBS}")
    else()
      include(ProcessorCount)
      ProcessorCount(NUM_HARDWARE_THREADS)
      # Assume 2 hardware threads per cpu core
      math(EXPR MAX_JOBS "${NUM_HARDWARE_THREADS} / 2")
      # ProcessorCount might return 0, set to a positive number
      if(MAX_JOBS LESS 2)
        set(MAX_JOBS 2)
      endif()
    endif()

    if("${CMAKE_GENERATOR}" MATCHES "Make")
      # Recursive make with jobserver for parallelism, and also put a load limit
      # here to avoid flaky OOM,
      # https://www.gnu.org/software/make/manual/html_node/Parallel.html
      set(MAKE_COMMAND "$(MAKE)" "-l${MAX_JOBS}")
    else()
      # Parallel build with CPU load limit to avoid oversubscription
      set(MAKE_COMMAND "make" "-j${MAX_JOBS}" "-l${MAX_JOBS}")
    endif()

    set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")
    ExternalProject_Add(
      nccl_external
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        ${MAKE_COMMAND} "CXX=${CMAKE_CXX_COMPILER}"
        "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "NVCC=${CUDA_NVCC_EXECUTABLE}"
        "NVCC_GENCODE=${NVCC_GENCODE}" "BUILDDIR=${__NCCL_BUILD_DIR}"
        "VERBOSE=0" "DEBUG=0"
      BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_static.a"
      INSTALL_COMMAND "")

    set(__NCCL_LIBRARY_DEP nccl_external)
    set(NCCL_LIBRARIES ${__NCCL_BUILD_DIR}/lib/libnccl_static.a)

    set(NCCL_FOUND TRUE)
    add_library(__caffe2_nccl INTERFACE)
    # The following old-style variables are set so that other libs, such as
    # Gloo, can still use it.
    set(NCCL_INCLUDE_DIRS ${__NCCL_BUILD_DIR}/include)
    add_dependencies(__caffe2_nccl ${__NCCL_LIBRARY_DEP})
    target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    # nccl includes calls to shm_open/shm_close and therefore must depend on
    # librt on Linux
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(__caffe2_nccl INTERFACE rt)
    endif()
  endif()
endif()
