if(NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  if(USE_SYSTEM_NCCL)
    # NCCL_ROOT, NCCL_LIB_DIR, NCCL_INCLUDE_DIR will be accounted in the following line.
    find_package(rccl REQUIRED)
    if(rccl_FOUND)
      message(STATUS "RCCL Found!")
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE roc::rccl)

      # Some RCCL packages ship nccl_device.h before it is compatible with host
      # C++ translation units. Probe the installed header instead of assuming
      # that header presence or version macros imply host compatibility.
      block()
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        set(rccl_device_header_test
          "${PROJECT_BINARY_DIR}/rccl_device_header_test.cc")
        file(WRITE ${rccl_device_header_test} ""
          "#include <nccl.h>\n"
          "#include <nccl_device.h>\n"
          )
        try_compile(RCCL_DEVICE_HEADER_HOST_COMPATIBLE
          ${PROJECT_BINARY_DIR} ${rccl_device_header_test}
          COMPILE_DEFINITIONS -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1
          LINK_LIBRARIES roc::rccl
          OUTPUT_VARIABLE rccl_device_header_compile_output)

        if(RCCL_DEVICE_HEADER_HOST_COMPATIBLE)
          target_compile_definitions(
            __caffe2_nccl INTERFACE RCCL_DEVICE_HEADER_HOST_COMPATIBLE)
          message(STATUS "RCCL device header supports host compilation")
        else()
          message(STATUS "RCCL device header does not support host compilation")
          message(VERBOSE "${rccl_device_header_compile_output}")
        endif()
      endblock()
    else()
      message(STATUS "RCCL NOT Found!")
    endif()
  else()
    message(STATUS "USE_SYSTEM_NCCL=OFF is not supported yet when using RCCL")
  endif()
endif()
