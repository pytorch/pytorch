cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_OPENREG CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib/:$ORIGIN/")

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_LIBDIR lib)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${PYTORCH_INSTALL_DIR}/include)

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/openreg)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_openreg/csrc)
