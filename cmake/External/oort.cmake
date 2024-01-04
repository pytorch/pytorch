if(NOT __OORT_INCLUDED)
  set(__OORT_INCLUDED TRUE)

  set(__OORT_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/oort/src")
  set(__OORT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/oort/build")
  set(__OORT_INSTALL_DIR "${PROJECT_SOURCE_DIR}/torch")
  ExternalProject_Add(oort_external
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/triton.git
    GIT_TAG 29e1252c1ac8e6a54deb883701e553e5b201a1ba
    SOURCE_DIR ${__OORT_SOURCE_DIR}
    SOURCE_SUBDIR mathaot
    BINARY_DIR ${__OORT_BUILD_DIR}
    PREFIX ${__OORT_INSTALL_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${__OORT_INSTALL_DIR}
    # CONFIGURE_COMMAND ""
    # BUILD_COMMAND ${MAKE_COMMAND}
    BUILD_BYPRODUCTS "${__OORT_INSTALL_DIR}/lib/liboort.a"
    # INSTALL_COMMAND ${MAKE_COMMAND} install
    )
  set(OORT_FOUND TRUE)
  add_library(__caffe2_oort INTERFACE)
  add_dependencies(__caffe2_oort oort_external)
  target_link_libraries(__caffe2_oort INTERFACE ${__OORT_INSTALL_DIR}/lib/liboort.a)
  target_include_directories(__caffe2_oort INTERFACE ${__OORT_INSTALL_DIR}/include)
endif() # __OORT_INCLUDED
