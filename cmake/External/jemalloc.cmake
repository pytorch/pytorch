if(NOT USE_JEMALLOC OR TARGET jemalloc::jemalloc)
  return()
endif()

if(MSVC)
  set(USE_JEMALLOC OFF)
  return()
endif()

find_program(AUTOCONF_EXE NAMES autoconf)
if(AUTOCONF_EXE STREQUAL "AUTOCONF_EXE-NOTFOUND")
  set(USE_JEMALLOC OFF)
  return()
endif()

find_program(MAKE_EXE NAMES make gmake)
if(MAKE_EXE STREQUAL "MAKE_EXE-NOTFOUND")
  set(USE_JEMALLOC OFF)
  return()
endif()

# Build jemalloc with autotools and make
set(JEMALLOC_ROOT "${Torch_SOURCE_DIR}/third_party/jemalloc")
ExternalProject_Add(jemalloc_setup
  SOURCE_DIR "${JEMALLOC_ROOT}"
  BUILD_IN_SOURCE ON
  CONFIGURE_COMMAND
    "${JEMALLOC_ROOT}/configure"
    # Required for dlopen support, see https://github.com/jemalloc/jemalloc/issues/937
    --disable-initial-exec-tls
    # Don't override system malloc
    --with-jemalloc-prefix=je_
    # Don't export je_ symbol names
    --without-export
  BUILD_COMMAND
    "${MAKE_EXE}"
    build_lib_static
    "CC=${CMAKE_C_COMPILER}"
    "CXX=${CMAKE_CXX_COMPILER}"
    VERBOSE=0
    -j$ENV{MAX_JOBS}
  BUILD_BYPRODUCTS
    "${JEMALLOC_ROOT}/lib/libjemalloc.a"
    "${JEMALLOC_ROOT}/lib/libjemalloc_pic.a"
    "${JEMALLOC_ROOT}/include/jemalloc.h"
  INSTALL_COMMAND ""
  )

ExternalProject_Add_Step(jemalloc_setup autoconf
  DEPENDERS configure
  DEPENDS "${JEMALLOC_ROOT}/configure.ac"
  BYPRODUCTS "${JEMALLOC_ROOT}/configure"
  WORKING_DIRECTORY "${JEMALLOC_ROOT}"
  COMMAND "${AUTOCONF_EXE}")

add_library(jemalloc::jemalloc INTERFACE IMPORTED GLOBAL)
add_dependencies(jemalloc::jemalloc jemalloc_setup)
set_target_properties(
  jemalloc::jemalloc PROPERTIES
  INTERFACE_LINK_LIBRARIES
    "${JEMALLOC_ROOT}/lib/libjemalloc_pic.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${JEMALLOC_ROOT}/include/")
