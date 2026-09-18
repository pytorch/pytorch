if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_VER "0.14b")
  set(__AOTRITON_BUILD_VARIANTS "")
  set(__AOTRITON_MANYLINUX_LIST
      "manylinux_2_28"  # rocm6.4
      "manylinux_2_28"  # rocm7.0
      "manylinux_2_28"  # rocm7.1
      "manylinux_2_28"  # rocm7.2
      "manylinux_2_28"  # rocm7.14
      "manylinux_2_28"  # rocm7.15
      )
  set(__AOTRITON_ROCM_LIST
      "rocm6.4"
      "rocm7.0"
      "rocm7.1"
      "rocm7.2"
      "rocm7.14"
      "rocm7.15"
      )
  if(DEFINED ENV{PYTORCH_AOTRITON_COMMIT})
    set(__AOTRITON_CI_COMMIT "$ENV{PYTORCH_AOTRITON_COMMIT}")
  else()
    set(__AOTRITON_CI_COMMIT "44fa7def5bdfd87f687b21ee2fa283e133e727da")
  endif()
  # SHA256 for rocm6.4/7.0/7.1/7.2/7.15 are placeholders (0s): no such
  # artifact has been built for 0.14b yet, only rocm7.14 has.
  set(__AOTRITON_SHA256_LIST
      "0000000000000000000000000000000000000000000000000000000000000000"  # rocm6.4
      "0000000000000000000000000000000000000000000000000000000000000000"  # rocm7.0
      "0000000000000000000000000000000000000000000000000000000000000000"  # rocm7.1
      "0000000000000000000000000000000000000000000000000000000000000000"  # rocm7.2
      "bb1edc7090882650fe3c84881c0c217b7cf6a10a715191d96c28fd45723ba848"  # rocm7.14
      "0000000000000000000000000000000000000000000000000000000000000000"  # rocm7.15
      )
  set(__AOTRITON_IMAGE_LIST
      "amd-gfx90a"
      "amd-gfx942"
      "amd-gfx950"
      "amd-gfx110x"
      "amd-gfx115x"
      "amd-gfx120x"
      "amd-gfx1250"
     )
  set(__AOTRITON_IMAGE_SHA256_LIST
     "e5e891000401548a1f166584ed305c2ec4dd27536ec0fa254930e9f8446d1f8d" # amd-gfx90a
     "81e28934cc44f8354c7d33cf0a926e90c743b74fff962793ddf4f43bf90c0d18" # amd-gfx942
     "3fd6143379221836a15c8fc33d50f9b4832c2b3571a780ad948c3eb1497dbbd2" # amd-gfx950
     "65af418feb232c143d5bcea6a53b1aef1185dfcc68769c3fbe5fdfc689634805" # amd-gfx110x
     "a83695777a7e1a5ca47f3297ddeb0ba292717aecea9138c56b2317377d8a9ca1" # amd-gfx115x
     "489c81800acf874c8bacf2f85210cdef7bd92b106afa614f7a241caa96e434fc" # amd-gfx120x
     "fe4c72c18d91d5437c4f0f4bde9421a8c75b5dd4470ea3db403d0eeb4a81d322" # amd-gfx1250
     )
  if(USE_ASAN)
    set(__AOTRITON_BUILD_VARIANTS "+asan")
    set(__AOTRITON_MANYLINUX_LIST
        "manylinux_2_28"  # rocm7.14
        "manylinux_2_28"  # rocm7.15
        )
    # ASAN only supports rocm7.14
    set(__AOTRITON_ROCM_LIST
        "rocm7.14"
        "rocm7.15"
        )
    set(__AOTRITON_SHA256_LIST
        "3f5cfba6c42261a3e3b44022c66083ec859fcc98296faa4646b65373fead3448"  # rocm7.14+asan
        "7a7928d881d6341fc0b8ffb3ad7077f62438a8412ec57f97fb4b4dfbc73b3e64"  # rocm7.15+asan
        )
    # ASAN only supports gfx942+gfx950
    set(__AOTRITON_IMAGE_LIST
        "amd-gfx942"
        "amd-gfx950"
       )
    set(__AOTRITON_IMAGE_SHA256_LIST
       "563d2e4c41b367725c7b8e12fdfd04df4d1b1ff15947e011e2452818f3a43d26" # amd-gfx942+asan
       "b428dfe6eef7a1dcfac54ac2408dd136dbd622016331dc863e87ce9ea84c8054" # amd-gfx950+asan
       )
  endif()
  set(__AOTRITON_BASE_URL "$ENV{PYTORCH_AOTRITON_BASE_URL}")
  if(NOT __AOTRITON_BASE_URL)
    set(__AOTRITON_BASE_URL "https://github.com/ROCm/aotriton/releases/download/")  # @lint-ignore
  endif()
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
                                  "${__AOTRITON_VER}"
                                  "${__AOTRITON_BUILD_VARIANTS}-"
                                  "${__AOTRITON_MANYLINUX}"
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
           "aotriton-${__AOTRITON_VER}${__AOTRITON_BUILD_VARIANTS}-images-"
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
  # Install libaotriton_v2.so into the cmake install tree so it ends up in
  # site-packages/torch/lib/ when building with scikit-build-core.
  # aotriton's ExternalProject puts the library directly in the source tree
  # (${PROJECT_SOURCE_DIR}/torch/lib/) without a cmake install() rule, so it
  # is absent from the installed wheel and causes link failures in downstream
  # cmake builds (e.g., custom op builds) that link against installed torch.
  install(DIRECTORY "${__AOTRITON_INSTALL_DIR}/lib/"
    DESTINATION "lib"
    FILES_MATCHING PATTERN "libaotriton_v2*.so*"
  )
  # Install aotriton GPU kernel images (compressed ISA blobs) into the wheel.
  install(DIRECTORY "${__AOTRITON_INSTALL_DIR}/lib/aotriton.images"
    DESTINATION "lib"
    OPTIONAL
  )
endif() # __AOTRITON_INCLUDED
