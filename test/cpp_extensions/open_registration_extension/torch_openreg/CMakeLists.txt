cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_OPENREG CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# ---[ Build Directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---[ RPATH.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib/:$ORIGIN/")

# ---[ Default Variables.
set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_LIBDIR lib)

# ---[ Find Packages
set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

# --[ CXX Flags.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_options(-fabi-version=11)
include_directories(${PYTORCH_INSTALL_DIR}/include)

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

# ---[ Variables.
set(OPENREG_LIB torch_openreg)
set(PYTHON_OPENREG_LIB torch_openreg_python)

add_subdirectory(${PROJECT_SOURCE_DIR}/backend)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_openreg/csrc)
