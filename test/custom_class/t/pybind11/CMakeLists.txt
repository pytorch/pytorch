# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

# CMake versions < 3.4.0 do not support try_compile/pthread checks without C as active language.
if(CMAKE_VERSION VERSION_LESS 3.4.0)
  project(pybind11)
else()
  project(pybind11 CXX)
endif()

# Check if pybind11 is being used directly or via add_subdirectory
set(PYBIND11_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PYBIND11_MASTER_PROJECT ON)
endif()

option(PYBIND11_INSTALL "Install pybind11 header files?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_TEST    "Build pybind11 test suite?"     ${PYBIND11_MASTER_PROJECT})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")

include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX ${PYTHON_MODULE_PREFIX} CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} CACHE INTERNAL "")
set(PYTHON_VERSION_MAJOR ${PYTHON_VERSION_MAJOR} CACHE INTERNAL "")
set(PYTHON_VERSION_MINOR ${PYTHON_VERSION_MINOR} CACHE INTERNAL "")

# NB: when adding a header don't forget to also add it to setup.py
set(PYBIND11_HEADERS
  include/pybind11/detail/class.h
  include/pybind11/detail/common.h
  include/pybind11/detail/descr.h
  include/pybind11/detail/init.h
  include/pybind11/detail/internals.h
  include/pybind11/detail/typeid.h
  include/pybind11/attr.h
  include/pybind11/buffer_info.h
  include/pybind11/cast.h
  include/pybind11/chrono.h
  include/pybind11/common.h
  include/pybind11/complex.h
  include/pybind11/options.h
  include/pybind11/eigen.h
  include/pybind11/embed.h
  include/pybind11/eval.h
  include/pybind11/functional.h
  include/pybind11/numpy.h
  include/pybind11/operators.h
  include/pybind11/pybind11.h
  include/pybind11/pytypes.h
  include/pybind11/stl.h
  include/pybind11/stl_bind.h
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       PYBIND11_HEADERS "${PYBIND11_HEADERS}")

if (PYBIND11_TEST)
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# extract project version from source
file(STRINGS "${PYBIND11_INCLUDE_DIR}/pybind11/detail/common.h" pybind11_version_defines
     REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(ver ${pybind11_version_defines})
  if (ver MATCHES "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
set(${PROJECT_NAME}_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})
message(STATUS "pybind11 v${${PROJECT_NAME}_VERSION}")

option (USE_PYTHON_INCLUDE_DIR "Install pybind11 headers in Python include directory instead of default installation prefix" OFF)
if (USE_PYTHON_INCLUDE_DIR)
    file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_INCLUDE_DIRS})
endif()

if(NOT (CMAKE_VERSION VERSION_LESS 3.0))  # CMake >= 3.0
  # Build an interface library target:
  add_library(pybind11 INTERFACE)
  add_library(pybind11::pybind11 ALIAS pybind11)  # to match exported target
  target_include_directories(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
                                                $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>
                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_options(pybind11 INTERFACE $<BUILD_INTERFACE:${PYBIND11_CPP_STANDARD}>)

  add_library(module INTERFACE)
  add_library(pybind11::module ALIAS module)
  if(NOT MSVC)
    target_compile_options(module INTERFACE -fvisibility=hidden)
  endif()
  target_link_libraries(module INTERFACE pybind11::pybind11)
  if(WIN32 OR CYGWIN)
    target_link_libraries(module INTERFACE $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
  elseif(APPLE)
    target_link_libraries(module INTERFACE "-undefined dynamic_lookup")
  endif()

  add_library(embed INTERFACE)
  add_library(pybind11::embed ALIAS embed)
  target_link_libraries(embed INTERFACE pybind11::pybind11 $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
endif()

if (PYBIND11_INSTALL)
  install(DIRECTORY ${PYBIND11_INCLUDE_DIR}/pybind11 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  # GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
  set(PYBIND11_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for pybind11Config.cmake")

  configure_package_config_file(tools/${PROJECT_NAME}Config.cmake.in
                                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
  # Remove CMAKE_SIZEOF_VOID_P from ConfigVersion.cmake since the library does
  # not depend on architecture specific settings or libraries.
  set(_PYBIND11_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
  unset(CMAKE_SIZEOF_VOID_P)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                   VERSION ${${PROJECT_NAME}_VERSION}
                                   COMPATIBILITY AnyNewerVersion)
  set(CMAKE_SIZEOF_VOID_P ${_PYBIND11_CMAKE_SIZEOF_VOID_P})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                tools/FindPythonLibsNew.cmake
                tools/pybind11Tools.cmake
          DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(NOT (CMAKE_VERSION VERSION_LESS 3.0))
    if(NOT PYBIND11_EXPORT_NAME)
      set(PYBIND11_EXPORT_NAME "${PROJECT_NAME}Targets")
    endif()

    install(TARGETS pybind11 module embed
            EXPORT "${PYBIND11_EXPORT_NAME}")
    if(PYBIND11_MASTER_PROJECT)
      install(EXPORT "${PYBIND11_EXPORT_NAME}"
              NAMESPACE "${PROJECT_NAME}::"
              DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
    endif()
  endif()
endif()
