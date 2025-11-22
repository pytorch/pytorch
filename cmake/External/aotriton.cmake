if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_VER "0.11b")
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.2
      "manylinux_2_28"  # rocm6.3
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm7.0
      "manylinux_2_28"  # rocm7.1
      )
  set(__AOTRITON_ROCM_LIST
      "rocm6.2"
      "rocm6.3"
      "rocm6.4"
      "rocm7.0"
      "rocm7.1"
      )
  set(__AOTRITON_CI_COMMIT "972223c501ffc22068bb035ac5d64cf54318d895")
  set(__AOTRITON_SHA256_LIST
      "6cae3d5de75ee205d22e088f7dfaab1227056d02ea67f29ccdbc09f2be4e8c8f"  # rocm6.2
      "72a153549ea20707331e8a1f1e3d1b8de2913f9d5af2b900c56235d578b57efe"  # rocm6.3
      "c7f319dd7448cbbbab81889dd8a37d47dbc25ebcbd89760f09e6a0904e556393"  # rocm6.4
      "a2a974e0ad929a5e5827c0f896c59bda4872459cbaf8dd8e0a00407f404491cf"  # rocm7.0
      "d4eb24c9f1a0cfedb35f9292efb41d16589cf5a4b98c3c0940181bbefc49d722"  # rocm7.1
      )
  set(__AOTRITON_IMAGE_LIST
      "amd-gfx90a"
      "amd-gfx942"
      "amd-gfx950"
      "amd-gfx11xx"
      "amd-gfx120x"
     )
  set(__AOTRITON_IMAGE_SHA256_LIST
     "c19a41c9480510ab32e6fb05e6ed0a3832d6b07634f050b836b760200befa735" # amd-gfx90a
     "3a06a99971dddb7703a30378f1c5d6b41468d926ea51821156d1b6857b985bc4" # amd-gfx942
     "27fc21f6761d57987a700436de8cf29cbdd9eeee91318dfed596eeb147d219ad" # amd-gfx950
     "ec134032087344176695505db659387374d1916adfee16f0db47dee38d9c8603" # amd-gfx11xx
     "fec05205747ff51649b1e151545267d5aa2037ba9d0338cad286882915b941b0" # amd-gfx120x
     )
  set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
  set(__AOTRITON_Z "gz")
  # Set the default __AOTRITON_LIB path
  if(NOT WIN32)
    set(__AOTRITON_LIB "lib/libaotriton_v2.so")
  else()
    set(__AOTRITON_LIB "lib/aotriton_v2.lib")
  endif()

  function(aotriton_build_windows_dependencies dlfcn-win32_external xz_external dlfcn-win32_DIR liblzma_DIR)
    # Windows-specific dependencies - build these first
    if(NOT noimage)
      message(FATAL_ERROR "noimage must be ON for Windows builds")
    endif()
    # Build dlfcn-win32
    set(__DLFCN_WIN32_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dlfcn-win32")
    set(__DLFCN_WIN32_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dlfcn-win32-install")

    ExternalProject_Add(${dlfcn-win32_external}
      GIT_REPOSITORY https://github.com/dlfcn-win32/dlfcn-win32.git
      GIT_TAG v1.4.2
      PREFIX ${__DLFCN_WIN32_PREFIX}
      INSTALL_DIR ${__DLFCN_WIN32_INSTALL_DIR}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${__DLFCN_WIN32_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTS=OFF
      BUILD_BYPRODUCTS
        "${__DLFCN_WIN32_INSTALL_DIR}/lib/dl.lib"
        "${__DLFCN_WIN32_INSTALL_DIR}/bin/dl.dll"
    )
    ExternalProject_Add_Step(${dlfcn-win32_external} copy_to_aotriton
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${__DLFCN_WIN32_INSTALL_DIR}/bin/dl.dll"
        "${__AOTRITON_INSTALL_DIR}/lib/"
      DEPENDEES install
    )
    set(${dlfcn-win32_DIR} "${__DLFCN_WIN32_INSTALL_DIR}/share/dlfcn-win32" CACHE PATH "Path to dlfcn-win32 CMake config" FORCE)

    # Build xz/liblzma
    set(__XZ_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/xz")
    set(__XZ_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/xz-install")

    ExternalProject_Add(${xz_external}
      GIT_REPOSITORY https://github.com/tukaani-project/xz.git
      GIT_TAG v5.8.1
      PREFIX ${__XZ_PREFIX}
      INSTALL_DIR ${__XZ_INSTALL_DIR}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${__XZ_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DENABLE_NLS=OFF
        -DXZ_TOOL_LZMAINFO=OFF
        -DXZ_TOOL_XZ=OFF
        -DXZ_TOOL_XZDEC=OFF
        -DXZ_TOOL_LZMADEC=OFF
      BUILD_BYPRODUCTS
        "${__XZ_INSTALL_DIR}/lib/lzma.lib"
        "${__XZ_INSTALL_DIR}/bin/liblzma.dll"
    )
    ExternalProject_Add_Step(${xz_external} copy_to_aotriton
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${__XZ_INSTALL_DIR}/bin/liblzma.dll"
        "${__AOTRITON_INSTALL_DIR}/lib/"
      DEPENDEES install
    )
    set(${liblzma_DIR} "${__XZ_INSTALL_DIR}/lib/cmake/liblzma" CACHE PATH "Path to xz/liblzma CMake config" FORCE)
  endfunction()

  function(aotriton_build_from_source noimage project)
    if(noimage)
      SET(RECURSIVE "OFF")
    else()
      SET(RECURSIVE "ON")
    endif()
    if(WIN32)
      message(STATUS "Building AOTriton Windows dependencies")
      aotriton_build_windows_dependencies(dlfcn-win32_external xz_external dlfcn-win32_DIR liblzma_DIR)
    endif()
    message(STATUS "PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}")

    ExternalProject_Add(${project}
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_SUBMODULES_RECURSE ${RECURSIVE}
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      PREFIX ${__AOTRITON_EXTERN_PREFIX}
      CMAKE_CACHE_ARGS
      -DAOTRITON_TARGET_ARCH:STRING=${PYTORCH_ROCM_ARCH}
      -DCMAKE_INSTALL_PREFIX:FILEPATH=${__AOTRITON_INSTALL_DIR}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DAOTRITON_GPU_BUILD_TIMEOUT=0
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NOIMAGE_MODE=${noimage}
      -DHIP_PLATFORM=amd
      $<$<BOOL:${WIN32}>:-Ddlfcn-win32_DIR=${dlfcn-win32_DIR}>
      $<$<BOOL:${WIN32}>:-Dliblzma_DIR=${liblzma_DIR}>
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
    )
    if(WIN32)
      add_dependencies(${project} dlfcn-win32_external xz_external)
    endif()
  endfunction()

  set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
  function(aotriton_download_runtime index project)
    list(GET __AOTRITON_ROCM_LIST ${index} __AOTRITON_ROCM)
    list(GET __AOTRITON_MANYLINUX_LIST ${index} __AOTRITON_MANYLINUX)
    list(GET __AOTRITON_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL
           "${__AOTRITON_BASE_URL}"
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
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}"
    )
    message(STATUS "Using AOTriton Runtime from pre-compiled binary ${__AOTRITON_URL}.\
    Set env variables AOTRITON_INSTALL_FROM_SOURCE=1 to build from source.")
  endfunction()

  function(aotriton_download_image image project)
    list(FIND __AOTRITON_IMAGE_LIST ${image} index)
    list(GET __AOTRITON_IMAGE_SHA256_LIST ${index} __AOTRITON_SHA256)

    string(CONCAT __AOTRITON_FILE
           "aotriton-${__AOTRITON_VER}-images-"
           "${image}.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL
           "${__AOTRITON_BASE_URL}"
           "${__AOTRITON_VER}/${__AOTRITON_FILE}")

    # Set up directories
    set(__AOTRITON_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_download-${image})
    set(__AOTRITON_EXTRACT_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_image-${image})
    set(__AOTRITON_INSTALL_SOURCE_DIR ${__AOTRITON_EXTRACT_DIR})
    set(__DOWNLOAD_NO_EXTRACT "")
    set(__BUILD_COMMANDS "")

    # On Windows, we need custom tar extraction with UTF-8 support
    if(WIN32)
      set(__DOWNLOAD_NO_EXTRACT "DOWNLOAD_NO_EXTRACT;TRUE")
      set(__BUILD_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory "${__AOTRITON_EXTRACT_DIR}"
        COMMAND tar --options hdrcharset=UTF-8 -xf "${__AOTRITON_DOWNLOAD_DIR}/${__AOTRITON_FILE}" -C "${__AOTRITON_EXTRACT_DIR}"
      )
      set(__AOTRITON_INSTALL_SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}/aotriton)
    endif()

    ExternalProject_Add(${project}
      URL "${__AOTRITON_URL}"
      URL_HASH SHA256=${__AOTRITON_SHA256}
      DOWNLOAD_DIR ${__AOTRITON_DOWNLOAD_DIR}
      ${__DOWNLOAD_NO_EXTRACT}
      SOURCE_DIR ${__AOTRITON_EXTRACT_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      ${__BUILD_COMMANDS}
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${__AOTRITON_INSTALL_SOURCE_DIR}"
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
    aotriton_build_from_source(OFF aotriton_external)
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_EXTERN_PREFIX}")
  else()
    set(__AOTRITON_SYSTEM_ROCM "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_SYSTEM_ROCM}" __AOTRITON_RUNTIME_INDEX)
    # Always build aotriton runtime from source on Windows due to lack of pre-built binaries
    if(${__AOTRITON_RUNTIME_INDEX} LESS 0 OR WIN32)
      message(STATUS "Cannot find AOTriton runtime for ROCM ${__AOTRITON_SYSTEM_ROCM}. \
      Build runtime from source")
      aotriton_build_from_source(ON aotriton_runtime)
    else()
      aotriton_download_runtime(${__AOTRITON_RUNTIME_INDEX} aotriton_runtime)
    endif()
    add_dependencies(__caffe2_aotriton aotriton_runtime)
    set(__AOTRITON_CHAINED_IMAGE "aotriton_runtime")
    foreach(image ${__AOTRITON_IMAGE_LIST})
      string(SUBSTRING ${image} 7 -1 gfx_pattern)
      string(REPLACE "x" "." gfx_regex ${gfx_pattern})
      foreach(target ${PYTORCH_ROCM_ARCH})
        if(target MATCHES ${gfx_regex})
          set(__AOTRITON_DOWNLOAD_TARGET aotriton_image_${gfx_pattern})
          aotriton_download_image(${image} ${__AOTRITON_DOWNLOAD_TARGET})
          add_dependencies(${__AOTRITON_CHAINED_IMAGE} ${__AOTRITON_DOWNLOAD_TARGET})
          set(__AOTRITON_CHAINED_IMAGE ${__AOTRITON_DOWNLOAD_TARGET})
          break()
        endif()
      endforeach()
    endforeach()
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE "${__AOTRITON_INSTALL_DIR}/${__AOTRITON_LIB}")
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
