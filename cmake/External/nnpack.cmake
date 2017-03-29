if (__NNPACK_INCLUDED)
  return()
endif()
set(__NNPACK_INCLUDED TRUE)
 
if (NOT USE_NNPACK)
  return()
endif()

# try any external nnpack first
find_package(NNPACK)

if (NNPACK_FOUND)
  message(INFO "Found external NNPACK installation.")
  return()
endif()

##############################################################################
# Custom build rules to build nnpack, if external dependency is not found 
##############################################################################

set(NNPACK_PREFIX ${PROJECT_SOURCE_DIR}/third_party/NNPACK)

##############################################################################
# (1) MSVC - unsupported 
##############################################################################

if (MSVC)
  message(WARNING "NNPACK not supported on MSVC yet. Turn this warning off by USE_NNPACK=OFF.")
  set(USE_NNPACK OFF)
  return()
endif()

##############################################################################
# (2) Mobile platform - direct build
##############################################################################

if (ANDROID OR IOS)
  message(WARNING "NNPACK for mobile cmake support is wip")
  set(USE_NNPACK OFF)
  return()
endif()

##############################################################################
# (3) Linux/Mac: use PeachPy
##############################################################################

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(STATUS "Will try to build NNPACK from source. If anything fails, "
                 "follow the NNPACK prerequisite installation steps.")
  find_program(CAFFE2_CONFU_COMMAND confu)
  find_program(CAFFE2_NINJA_COMMAND ninja)
  if (CAFFE2_CONFU_COMMAND AND CAFFE2_NINJA_COMMAND)
    # Note: per Marat, there is no support for fPIC right now so we will need to
    # manually change it in build.ninja
    ExternalProject_Add(nnpack_external
        SOURCE_DIR ${NNPACK_PREFIX}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND confu setup
        COMMAND python ./configure.py
        COMMAND sed -ibuild.ninja.bak "s/cflags = /cflags = -fPIC /" build.ninja
        COMMAND sed -ibuild.ninja.bak "s/cxxflags = /cxxflags = -fPIC /" build.ninja
        COMMAND ninja nnpack
        INSTALL_COMMAND ""
        )

    set(NNPACK_FOUND TRUE)
    set(NNPACK_INCLUDE_DIRS
        ${NNPACK_PREFIX}/include
        ${NNPACK_PREFIX}/deps/pthreadpool/include)
    set(NNPACK_LIBRARIES ${NNPACK_PREFIX}/lib/libnnpack.a ${NNPACK_PREFIX}/lib/libpthreadpool.a)
    set(NNPACK_LIBRARY_DIRS ${NNPACK_PREFIX}/lib)
  
    list(APPEND external_project_dependencies nnpack_external)
  else()
    message(WARNING "NNPACK is chosen to be installed, but confu and ninja "
                    "that are needed by it are not installed. As a result "
                    "we won't build with NNPACK.")
    set(USE_NNPACK OFF)
  endif()
  return()
endif()

##############################################################################
# (3) Catch-all: not supported.
##############################################################################

message(WARNING "Unknown platform - I don't know how to build NNPACK. "
                "See cmake/External/nnpack.cmake for details.")
set(USE_NNPACK OFF)
