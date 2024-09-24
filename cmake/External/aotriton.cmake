if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton/src")
  set(__AOTRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton/build")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)

  # AOTriton package information from GitHub Release Pages
  # Replaces .ci/docker/aotriton_version.txt
  set(__AOTRITON_VER "0.7b")
  list(JOIN
       "manylinux_2_17"  # rocm6.1
       "manylinux_2_17"  # rocm6.2
       ";"
       __AOTRITON_MANYLINUX_LIST)
  list(JOIN
       "rocm6.1"
       "rocm6.2"
       ";"
       __AOTRITON_ROCM_LIST)
  set(__AOTRITON_CI_INFO "9be04068c3c0857a4cfd17d7e39e71d0423ebac2")
  list(JOIN
       "006f4d982c9a9c768f31f0095128705fecb792136827e2456241fe79764de7a4"  # rocm6.1
       "3e9e1959d23b93d78a08fcc5f868125dc3854dece32fd9458be9ef4467982291"  # rocm6.2
       ";"
       __AOTRITON_SHA256_LIST)
  set(__AOTRITON_Z "gz")

  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    install(DIRECTORY
            $ENV{AOTRITON_INSTALLED_PREFIX}/lib
            $ENV{AOTRITON_INSTALLED_PREFIX}/include
            DESTINATION ${__AOTRITON_INSTALL_DIR})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  elseif(DEFINED ENV{AOTRITON_INSTALL_FROM_SOURCE})
    ExternalProject_Add(aotriton_external
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      SOURCE_DIR ${__AOTRITON_SOURCE_DIR}
      BINARY_DIR ${__AOTRITON_BUILD_DIR}
      PREFIX ${__AOTRITON_INSTALL_DIR}
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__AOTRITON_INSTALL_DIR}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NO_SHARED=OFF
      # CONFIGURE_COMMAND ""
      BUILD_COMMAND ""  # No build, install command will repeat the build process due to problems in the build system.
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
      # INSTALL_COMMAND ${MAKE_COMMAND} install
      )
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_SOURCE_DIR}")
  else()
    list(GET __AOTRITON_ROCM_LIST 0 __AOTRITON_ROCM_LOW_STR)
    list(GET __AOTRITON_ROCM_LIST -1 __AOTRITON_ROCM_HIGH_STR)
    # len("rocm") == 4
    string(SUBSTRING ${__AOTRITON_ROCM_LOW_STR} 4 -1 __AOTRITON_ROCM_LOW)
    string(SUBSTRING ${__AOTRITON_ROCM_HIGH_STR} 4 -1 __AOTRITON_ROCM_HIGH)
    set(__AOTRITON_SYSTEM_ROCM "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}")
    if(__AOTRITON_SYSTEM_ROCM VERSION_LESS __AOTRITON_ROCM_LOW)
      set(__AOTRITON_ROCM ${__AOTRITON_ROCM_LOW})
    elseif(__AOTRITON_SYSTEM_ROCM VERSION_GREATER __AOTRITON_ROCM_HIGH)
      set(__AOTRITON_ROCM ${__AOTRITON_ROCM_HIGH})
    else()
      set(__AOTRITON_ROCM ${__AOTRITON_SYSTEM_ROCM})
    endif()
    list(FIND __AOTRITON_ROCM_LIST "rocm${__AOTRITON_ROCM}" __AOTRITON_ROCM_INDEX)
    list(GET __AOTRITON_SHA256_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_SHA256)
    list(GET __AOTRITON_MANYLINUX_LIST ${__AOTRITON_ROCM_INDEX} __AOTRITON_MANYLINUX)
    set(__AOTRITON_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
    string(CONCAT __AOTRITON_FILE "aotriton-"
                                  "${__AOTRITON_VER}-${__AOTRITON_MANYLINUX}"
                                  "_${__AOTRITON_ARCH}-rocm${__AOTRITON_ROCM}"
                                  "-shared.tar.${__AOTRITON_Z}")
    string(CONCAT __AOTRITON_URL "https://github.com/ROCm/aotriton/releases/download/"
                                 "${__AOTRITON_VER}/${__AOTRITON_FILE}")
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
    Set env variable AOTRITON_INSTALL_FROM_SOURCE=1 to build from source.")
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.so)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
