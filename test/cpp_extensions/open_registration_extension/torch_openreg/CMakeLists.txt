cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_OPENREG CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/openreg)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_openreg/csrc)
