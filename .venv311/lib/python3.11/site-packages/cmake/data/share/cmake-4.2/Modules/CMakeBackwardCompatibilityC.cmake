# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if(NOT CMAKE_SKIP_COMPATIBILITY_TESTS)
  # Old CMake versions did not support OS X universal binaries anyway,
  # so just get through this with at least some size for the types.
  list(LENGTH CMAKE_OSX_ARCHITECTURES NUM_ARCHS)
  if(${NUM_ARCHS} GREATER 1)
    if(NOT DEFINED CMAKE_TRY_COMPILE_OSX_ARCHITECTURES)
      message(WARNING "This module does not work with OS X universal binaries.")
      set(__ERASE_CMAKE_TRY_COMPILE_OSX_ARCHITECTURES 1)
      list(GET CMAKE_OSX_ARCHITECTURES 0 CMAKE_TRY_COMPILE_OSX_ARCHITECTURES)
    endif()
  endif()

  include (CheckTypeSize)
  check_type_size(int      CMAKE_SIZEOF_INT)
  check_type_size(long     CMAKE_SIZEOF_LONG)
  check_type_size("void*"  CMAKE_SIZEOF_VOID_P)
  check_type_size(char     CMAKE_SIZEOF_CHAR)
  check_type_size(short    CMAKE_SIZEOF_SHORT)
  check_type_size(float    CMAKE_SIZEOF_FLOAT)
  check_type_size(double   CMAKE_SIZEOF_DOUBLE)

  include (CheckIncludeFile)
  check_include_file("limits.h"       CMAKE_HAVE_LIMITS_H)
  check_include_file("unistd.h"       CMAKE_HAVE_UNISTD_H)
  check_include_file("pthread.h"      CMAKE_HAVE_PTHREAD_H)

  include (CheckIncludeFiles)
  check_include_files("sys/types.h;sys/prctl.h"    CMAKE_HAVE_SYS_PRCTL_H)

  include (TestBigEndian)
  test_big_endian(CMAKE_WORDS_BIGENDIAN)
  include (FindX11)

  if("${X11_X11_INCLUDE_PATH}" STREQUAL "/usr/include")
    set (CMAKE_X_CFLAGS "" CACHE STRING "X11 extra flags.")
  else()
    set (CMAKE_X_CFLAGS "-I${X11_X11_INCLUDE_PATH}" CACHE STRING
         "X11 extra flags.")
  endif()
  set (CMAKE_X_LIBS "${X11_LIBRARIES}" CACHE STRING
       "Libraries and options used in X11 programs.")
  set (CMAKE_HAS_X "${X11_FOUND}" CACHE INTERNAL "Is X11 around.")

  include (FindThreads)

  set (CMAKE_THREAD_LIBS        "${CMAKE_THREAD_LIBS_INIT}" CACHE STRING
    "Thread library used.")

  set (CMAKE_USE_PTHREADS       "${CMAKE_USE_PTHREADS_INIT}" CACHE BOOL
     "Use the pthreads library.")

  set (CMAKE_USE_WIN32_THREADS  "${CMAKE_USE_WIN32_THREADS_INIT}" CACHE BOOL
       "Use the win32 thread library.")

  set (CMAKE_HP_PTHREADS        ${CMAKE_HP_PTHREADS_INIT} CACHE BOOL
     "Use HP pthreads.")

  if(__ERASE_CMAKE_TRY_COMPILE_OSX_ARCHITECTURES)
    set(CMAKE_TRY_COMPILE_OSX_ARCHITECTURES)
    set(__ERASE_CMAKE_TRY_COMPILE_OSX_ARCHITECTURES)
  endif()
endif()

mark_as_advanced(
CMAKE_HP_PTHREADS
CMAKE_THREAD_LIBS
CMAKE_USE_PTHREADS
CMAKE_USE_WIN32_THREADS
CMAKE_X_CFLAGS
CMAKE_X_LIBS
)
