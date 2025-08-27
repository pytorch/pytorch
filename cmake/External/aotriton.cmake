if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_VER "0.10b")
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
  set(__AOTRITON_CI_COMMIT "6fca155f4deeb8d9529326f7b69f350aeeb93477")
  set(__AOTRITON_SHA256_LIST
      "861cd9f7479eec943933c27cb86920247e5b5dd139bc7c1376c81808abb7d7fe"  # rocm6.3
      "acea7d811a2d3bbe718b6e07fc2a9f739e49eecd60b4b6a36fcb3fe8edf85d78"  # rocm6.4
      "7e29c325d5bd33ba896ddb106f5d4fc7d715274dca7fe937f724fffa82017838"  # rocm6.5
      "1e9b3dddf0c7fc07131c6f0f5266129e83ce2331f459fa2be8c63f4ae91b0f5b"  # rocm7.0
      )
  set(__AOTRITON_IMAGE_LIST
      "amd-gfx90a"
      "amd-gfx942"
      "amd-gfx950"
      "amd-gfx11xx"
      "amd-gfx120x"
     )
  set(__AOTRITON_IMAGE_SHA256_LIST
     "" # amd-gfx90a
     "" # amd-gfx942
     "" # amd-gfx950
     "" # amd-gfx11xx
     "" # amd-gfx120x
     )
  set(__AOTRITON_Z "gz")
  function(build_aotriton_from_source noimage project)
    ExternalProject_Add(${project}
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      PREFIX ${__AOTRITON_EXTERN_PREFIX}
      INSTALL_DIR ${__AOTRITON_INSTALL_DIR}
      -DAOTRITON_TARGET_ARCH:STRING=${PYTORCH_ROCM_ARCH}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DAOTRITON_GPU_BUILD_TIMEOUT=0
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NO_SHARED=OFF
      -DAOTRITON_NOIMAGE_MODE=${noimage}
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
    )
  endfunction()

  set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
  function(download_aotriton_runtime index project)
    list(GET __AOTRITON_ROCM_LIST ${index} __AOTRITON_ROCM)
    list(GET __AOTRITON_MANYLINUX_LIST ${index} __AOTRITON_MANYLINUX)
    list(GET __AOTRITON_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-rocm${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"  # @lint-ignore
                                 "${__AOTRITON_VER}/${__AOTRITON_FILE}")
    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_runtime
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/aotriton_runtime"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so"
    )
  endfunction()

  function(download_aotriton_image image project)
    list(FIND __AOTRITON_IMAGE_LIST ${image} index)
    list(GET __AOTRITON_IMAGE_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE "aotriton-${__AOTRITON_VER}-images-"
                                  "${image}.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"  # @lint-ignore
                                 "${__AOTRITON_VER}/${__AOTRITON_FILE}")
    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${image}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${image}"
      "${__AOTRITON_INSTALL_DIR}"
      BUILD_BYPRODUCTS
      "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images/${image}/__signature__"
    )
    message(STATUS "Download AOTriton pre-compiled GPU images from ${__AOTRITON_URL}.")
  endfunction()

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  elseif(DEFINED ENV{AOTRITON_INSTALL_FROM_SOURCE})
    build_aotriton_from_source(OFF aotriton_external)
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_EXTERN_PREFIX}")
  else()
    set(__AOTRITON_SYSTEM_ROCM "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_SYSTEM_ROCM}" __AOTRITON_RUNTIME_INDEX)
    if (${__AOTRITON_RUNTIME_INDEX} LESS 0)
      download_aotriton_runtime(${__AOTRITON_RUNTIME_INDEX} aotriton_runtime)
    else()
      build_aotriton_from_source(ON aotriton_runtime)
    endif()
    add_dependencies(__caffe2_aotriton aotriton_runtime)
    message(STATUS "Using AOTriton Runtime from pre-compiled binary ${__AOTRITON_URL}.\
    Set env variables AOTRITON_INSTALL_FROM_SOURCE=1 to build from source.")
    foreach(image ${__AOTRITON_IMAGE_LIST})
      string(SUBSTRING ${image} 7 -1 gfx_number)
      string(REPLACE "x" "." gfx_pattern ${gfx_number})
      foreach(target ${PYTORCH_ROCM_ARCH})
        if (target MATCHES ${gfx_pattern})
          download_aotriton_image(${image} aotriton_image_${gfx_number})
          add_dependencies(aotriton_runtime aotriton_image_${gfx_number})
          break()
        endif()
      endforeach()
    endforeach()
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
