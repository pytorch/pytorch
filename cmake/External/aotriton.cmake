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
  set(__AOTRITON_RELEASE_PAGE "0.10b")
  set(__AOTRITON_VER_LIST
      "0.10b"  # rocm6.3
      "0.10b"  # rocm6.4
      "0.10b"  # rocm6.5
      "0.10b"  # rocm7.0
      )
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.3
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm6.5
      "manylinux_2_28"  # rocm7.0
      )
  set(__AOTRITON_ROCM_LIST
      "rocm6.3"
      "rocm6.4"
      "rocm6.5"
      "rocm7.0"
      )
  set(__AOTRITON_CI_COMMIT "6fca155f4deeb8d9529326f7b69f350aeeb93477")    # source of rocm6.5 with gfx950
  set(__AOTRITON_SHA256_LIST
      "861cd9f7479eec943933c27cb86920247e5b5dd139bc7c1376c81808abb7d7fe"  # rocm6.3
      "acea7d811a2d3bbe718b6e07fc2a9f739e49eecd60b4b6a36fcb3fe8edf85d78"  # rocm6.4
      "7e29c325d5bd33ba896ddb106f5d4fc7d715274dca7fe937f724fffa82017838"  # rocm6.5
      "1e9b3dddf0c7fc07131c6f0f5266129e83ce2331f459fa2be8c63f4ae91b0f5b"  # rocm7.0
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
    message(STATUS "Using AOTriton from pre-compiled binary ${__AOTRITON_URL}.")
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
