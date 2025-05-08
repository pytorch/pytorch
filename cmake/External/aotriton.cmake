macro(get_target_gpus_from_pytorch target_gpus)
   set(gfx90a_key MI200)
   set(gfx942_key MI300X)
   set(gfx1100_key Navi31)

   foreach(X IN LISTS PYTORCH_ROCM_ARCH)
       set(key ${X})
       string(APPEND key "_key")
       string(APPEND target_gpus ${${key}})
       string(APPEND target_gpus "|")
   endforeach()
endmacro()

if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_RELEASE_PAGE "0.9.2b")
  set(__AOTRITON_VER_LIST
      "0.9.2b"  # rocm6.2
      "0.9.2b"  # rocm6.3
      "0.9.2b"  # rocm6.4
      "0.9.2b_612896439f"  # rocm6.5 with gfx950
      )
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.2
      "manylinux_2_28"  # rocm6.3
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm6.5
      )
  set(__AOTRITON_ROCM_LIST
      "rocm6.2"
      "rocm6.3"
      "rocm6.4"
      "rocm6.5"
      )
  set(__AOTRITON_CI_COMMIT "612896439fb4f78509b1a566b5ef0a333e9585bb")    # source of rocm6.5 with gfx950
  set(__AOTRITON_SHA256_LIST
      "08d84f96f4c984179f80f517c0431c7511ee26bb0ce9bd05a827573ddd78cc79"  # rocm6.2
      "9094d59717e7e6eace9126ca100dd0e86510f07fc6c3a349569fc4e2d9056604"  # rocm6.3
      "41190202c2736d5ff75b13a3abc0fb52ebfbb67226cf85dc3de7699c7000db44"  # rocm6.4
      "c85da64d21510190277794455ef8bd3f2d543a6f2462140d3da27e1df0ab8f82"  # rocm6.5 with gfx950
      )
  set(__AOTRITON_Z "gz")

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  else()
    set(__AOTRITON_SYSTEM_ROCM "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}")
    list(GET __AOTRITON_ROCM_LIST 0 __AOTRITON_ROCM_DEFAULT_STR)
    # Initialize __AOTRITON_ROCM to lowest version, in case all builds > system's ROCM
    string(SUBSTRING ${__AOTRITON_ROCM_DEFAULT_STR} 4 -1 __AOTRITON_ROCM)
    foreach(AOTRITON_ROCM_BUILD_STR IN LISTS __AOTRITON_ROCM_LIST)
      # len("rocm") == 4
      string(SUBSTRING ${AOTRITON_ROCM_BUILD_STR} 4 -1 AOTRITON_ROCM_BUILD)
      # Find the last build that <= system's ROCM
      # Assume the list is from lower to higher
      if(AOTRITON_ROCM_BUILD VERSION_GREATER __AOTRITON_SYSTEM_ROCM)
        break()
      endif()
      set(__AOTRITON_ROCM ${AOTRITON_ROCM_BUILD})
    endforeach()
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_ROCM}" __AOTRITON_ROCM_INDEX)
    list(GET __AOTRITON_SHA256_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_SHA256)
    list(GET __AOTRITON_MANYLINUX_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_MANYLINUX)
    list(GET __AOTRITON_VER_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_VER)
    set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-rocm${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"
                                 "${__AOTRITON_RELEASE_PAGE}/${__AOTRITON_FILE}")
    ExternalProject_Add(aotriton_external
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so"
    )
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton from pre-compiled binary ${__AOTRITON_URL}.\
    Set env variables AOTRITON_INSTALL_FROM_SOURCE=1 to build from source.")
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
