if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_EXTERN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  # Note packages information may have versions skipped (due to no ABI breaks)
  # But they must be listed from lower version to higher version
  set(__AOTRITON_VER "0.13b")
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
    set(__AOTRITON_CI_COMMIT "6e00ef3e335b45dfb49065259533b59c68995bfe")
  endif()
  set(__AOTRITON_SHA256_LIST
      "2fafa80953d9a49bd20e794bb8c0e1646e8aa815be2fb161deaa849a47547b17"  # rocm6.4
      "7409f7c974cc79be731a419818bdb2ed6b8a3640fd40665baa76ec3c2a537204"  # rocm7.0
      "f061a997679d8529a7b196b0ffb39912145ede217e515e1ee9ef5673b56d9e41"  # rocm7.1
      "1cdeebb7ef61ab691fba1d81da919b9db5d8bef28269c892a30bd13a0495b7a0"  # rocm7.2
      "7a139797c16b002fd5d9bcd706d36dc9819bb108877150f8186da21d0590eaa6"  # rocm7.14
      "f024225d8b6063f7d95974e5957cb20893a1579a9a73b22b60426441331bc021"  # rocm7.15
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
     "a3d1a6868ce290ba8118618207093e785252eff4e18a64f495752cb5a03ffed6" # amd-gfx90a
     "ccdbc7e3d96839be4895ee004f21531cc55d590c9018937b9e314bba363b3927" # amd-gfx942
     "518fd072eb05948fc0a6c25a20832591c6406df865e3b691a2aeff3fd4c5ce1d" # amd-gfx950
     "efe773e7a2c8adc995d90ecd0daca2db801285445ee10432df2b29c67d5b11d2" # amd-gfx110x
     "1bc50e8aa8b6bda3410e92886ccca8fd45df3e60a6cbda9ffc58b2c541efd5c2" # amd-gfx115x
     "6a465dbc03148bba8a2d78c4c2a3cb83155eca00f4f7f749e676402d7660968c" # amd-gfx120x
     "4aaf71d6e510549d593757e5f88598df1e4a29cbcd91f70750ee8a76f65c027f" # amd-gfx1250
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
