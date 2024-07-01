if(NOT __AOTRITON_INCLUDED)
  set(__AOTRITON_INCLUDED TRUE)

  set(__AOTRITON_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton/src")
  set(__AOTRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton/build")
  set(__AOTRITON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  add_library(__caffe2_aotriton INTERFACE)
  # Note it is INSTALL"ED"
  if(DEFINED ENV{AOTRITON_INSTALLED_PREFIX})
    set(__AOTRITON_INSTALL_DIR "$ENV{AOTRITON_INSTALLED_PREFIX}")
    message(STATUS "Using Preinstalled AOTriton at ${__AOTRITON_INSTALL_DIR}")
  else()
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.ci/docker/aotriton_version.txt" __AOTRITON_CI_INFO)
    list(GET __AOTRITON_CI_INFO 3 __AOTRITON_CI_COMMIT)
    ExternalProject_Add(aotriton_external
      GIT_REPOSITORY https://github.com/ROCm/aotriton.git
      GIT_TAG ${__AOTRITON_CI_COMMIT}
      SOURCE_DIR ${__AOTRITON_SOURCE_DIR}
      BINARY_DIR ${__AOTRITON_BUILD_DIR}
      PREFIX ${__AOTRITON_INSTALL_DIR}
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__AOTRITON_INSTALL_DIR}
      -DAOTRITON_COMPRESS_KERNEL=OFF
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DAOTRITON_NO_PYTHON=ON
      -DAOTRITON_NO_SHARED=ON
      # CONFIGURE_COMMAND ""
      BUILD_COMMAND ""  # No build, install command will repeat the build process due to problems in the build system.
      BUILD_BYPRODUCTS "${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.a"
      USES_TERMINAL_DOWNLOAD TRUE
      USES_TERMINAL_CONFIGURE TRUE
      USES_TERMINAL_BUILD TRUE
      USES_TERMINAL_INSTALL TRUE
      # INSTALL_COMMAND ${MAKE_COMMAND} install
      )
    add_dependencies(__caffe2_aotriton aotriton_external)
    message(STATUS "Using AOTriton compiled from source directory ${__AOTRITON_SOURCE_DIR}")
  endif()
  target_link_libraries(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/lib/libaotriton_v2.a)
  target_include_directories(__caffe2_aotriton INTERFACE ${__AOTRITON_INSTALL_DIR}/include)
  set(AOTRITON_FOUND TRUE)
endif() # __AOTRITON_INCLUDED
