# Mandatory-ish cmake settings

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
# TODO: merge this into the entry above
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../src/TH/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules_CUDA_fix")
set(SUBPROJECT 1)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/public/utils.cmake")

# Polyfill for upstream FindCUDA
include(CMakeInitializeConfigs)

# if() recognizes numbers and boolean constants
cmake_policy(SET CMP0012 NEW)

# Disallow use of the LOCATION property for build targets.
# Avoid some cmake warnings.
if(POLICY CMP0026)
 cmake_policy(SET CMP0026 OLD)
endif()

if(UNIX)
  # prevent Unknown CMake command "check_function_exists".
  include(CheckFunctionExists)
endif()
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
include(CheckLibraryExists)

include(CompilerSettings)
include(CompilerRPATH)
include(CheckCXXSourceCompiles)
include(TestForGlibCXXUseC99) # Test for an GCC 5 bug on Ubuntu 17.10 and newer
include(TestTrivialProgramRuns)
include(TestForCpuid)
include(TestForGccEbxFpicBug)
include(TestForAtomics)

# TODO: refactor this once we understand it better
if(UNIX AND NOT APPLE)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   check_library_exists(rt clock_gettime "time.h" NEED_LIBRT)
   if(NEED_LIBRT)
     # So that the check_function_exists() tests work...
     set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   endif()
endif()

if(UNIX)
  set(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  check_function_exists(mmap HAVE_MMAP)
  if(HAVE_MMAP)
    add_definitions(-DHAVE_MMAP=1)
  endif()
  check_function_exists(shm_open HAVE_SHM_OPEN)
  if(HAVE_SHM_OPEN)
    add_definitions(-DHAVE_SHM_OPEN=1)
  endif(HAVE_SHM_OPEN)
  check_function_exists(shm_unlink HAVE_SHM_UNLINK)
  if(HAVE_SHM_UNLINK)
    add_definitions(-DHAVE_SHM_UNLINK=1)
  endif(HAVE_SHM_UNLINK)
  check_function_exists(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  if(HAVE_MALLOC_USABLE_SIZE)
    add_definitions(-DHAVE_MALLOC_USABLE_SIZE=1)
  endif(HAVE_MALLOC_USABLE_SIZE)
endif()

# Is __thread supported?
if(NOT MSVC)
  check_c_source_compiles("static __thread int x = 1; int main() { return x; }" C_HAS_THREAD)
else()
  check_c_source_compiles("static __declspec( thread ) int x = 1; int main() { return x; }" C_HAS_THREAD)
endif()
if(NOT C_HAS_THREAD)
  message(STATUS "Warning: __thread is not supported, generating thread-unsafe code")
else()
  add_definitions(-DTH_HAVE_THREAD)
endif()

SET(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
SET(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
SET(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")
