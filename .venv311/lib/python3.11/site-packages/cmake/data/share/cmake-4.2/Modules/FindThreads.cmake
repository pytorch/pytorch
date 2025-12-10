# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindThreads
-----------

Finds and determines the thread library of the system for multithreading
support:

.. code-block:: cmake

  find_package(Threads [...])

Multithreading enables concurrent execution within a single program,
typically by creating multiple threads of execution.  Most commonly, this
is done using libraries such as POSIX Threads (``pthreads``) on Unix-like
systems or Windows threads on Windows.

This module abstracts the platform-specific differences and detects how to
enable thread support - whether it requires linking to a specific library,
adding compiler flags (like ``-pthread``), or both.  On some platforms,
threading is also implicitly available in default libraries without the
need to use additional flags or libraries.

This module is suitable for use in both C and C++ projects (and occasionally
other compiled languages) that rely on system-level threading APIs.

Using this module ensures that project builds correctly across different
platforms by handling the detection and setup of thread support in a
portable way.

C and C++ Language Standards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The C11 standard introduced a minimal cross-platform thread API via
``<threads.h>`` header file, and C++11 added ``<thread>`` header to the
standard library, offering high-level multithreading support.  These standard
headers allow writing portable threaded code at the language level, without
directly using platform-specific APIs like ``pthreads`` or Windows threads.

However, even with standard C11 or C++11 threads support available, there
may still be a need for platform-specific compiler or linker flags (e.g.,
``-pthread`` on Unix-like systems) for some applications.  This is where
FindThreads remains relevant - it ensures these flags and any required
libraries are correctly set up, even if not explicitly using system APIs.

In short:

* Use ``<thread>`` (C++11 and later) or ``<threads.h>`` (C11) in source code
  for portability and simpler syntax.

* Use ``find_package(Threads)`` in CMake project when application needs the
  traditional threading support and to ensure code compiles and links
  correctly across different platforms.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Threads::Threads``
  .. versionadded:: 3.1

  Target encapsulating the usage requirements to enable threading through
  flags or a threading library, if found.  This target is available if
  threads are detected as supported.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Threads_FOUND``
  Boolean indicating whether Threads is supported, either through a separate
  library or a standard library.
``CMAKE_THREAD_LIBS_INIT``
  The thread library to use.  This may be empty if the thread functions
  are provided by the system libraries and no special flags are needed
  to use them.
``CMAKE_USE_WIN32_THREADS_INIT``
  If the found thread library is the win32 one.
``CMAKE_USE_PTHREADS_INIT``
  If the found thread library is pthread compatible.
``CMAKE_HP_PTHREADS_INIT``
  If the found thread library is the HP thread library.

Variables Affecting Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module accepts the following variables before calling
``find_package(Threads)``:

``THREADS_PREFER_PTHREAD_FLAG``
  .. versionadded:: 3.1

  If the use of the ``-pthread`` compiler and linker flag is preferred then
  the caller can set this variable to boolean true.  The compiler flag can
  only be used with the imported target.  Use of both the imported target
  as well as this switch is highly recommended for new code.

  This variable has no effect if the system libraries provide the
  thread functions, i.e. when ``CMAKE_THREAD_LIBS_INIT`` will be empty.

Examples
^^^^^^^^

Finding Threads and linking the imported target to a project target:

.. code-block:: cmake

  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads)
  target_link_libraries(example PRIVATE Threads::Threads)
#]=======================================================================]

include (CheckLibraryExists)
set(Threads_FOUND FALSE)
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ${Threads_FIND_QUIETLY})

if(CMAKE_C_COMPILER_LOADED)
  include (CheckIncludeFile)
  include (CheckCSourceCompiles)
elseif(CMAKE_CXX_COMPILER_LOADED)
  include (CheckIncludeFileCXX)
  include (CheckCXXSourceCompiles)
else()
  message(FATAL_ERROR "FindThreads only works if either C or CXX language is enabled")
endif()

# simple pthread test code
set(PTHREAD_C_CXX_TEST_SOURCE [====[
#include <pthread.h>

static void* test_func(void* data)
{
  return data;
}

int main(void)
{
  pthread_t thread;
  pthread_create(&thread, NULL, test_func, NULL);
  pthread_detach(thread);
  pthread_cancel(thread);
  pthread_join(thread, NULL);
  pthread_atfork(NULL, NULL, NULL);
  pthread_exit(NULL);

  return 0;
}
]====])

# Internal helper macro.
# Do NOT even think about using it outside of this file!
macro(_threads_check_libc)
  if(NOT Threads_FOUND)
    if(CMAKE_C_COMPILER_LOADED)
      check_c_source_compiles("${PTHREAD_C_CXX_TEST_SOURCE}" CMAKE_HAVE_LIBC_PTHREAD)
    elseif(CMAKE_CXX_COMPILER_LOADED)
      check_cxx_source_compiles("${PTHREAD_C_CXX_TEST_SOURCE}" CMAKE_HAVE_LIBC_PTHREAD)
    endif()
    if(CMAKE_HAVE_LIBC_PTHREAD)
      set(CMAKE_THREAD_LIBS_INIT "")
      set(Threads_FOUND TRUE)
    endif()
  endif ()
endmacro()

# Internal helper macro.
# Do NOT even think about using it outside of this file!
macro(_threads_check_lib LIBNAME FUNCNAME VARNAME)
  if(NOT Threads_FOUND)
     check_library_exists(${LIBNAME} ${FUNCNAME} "" ${VARNAME})
     if(${VARNAME})
       set(CMAKE_THREAD_LIBS_INIT "-l${LIBNAME}")
       set(Threads_FOUND TRUE)
     endif()
  endif ()
endmacro()

# Internal helper macro.
# Do NOT even think about using it outside of this file!
macro(_threads_check_flag_pthread)
  if(NOT Threads_FOUND)
    # If we did not find -lpthreads, -lpthread, or -lthread, look for -pthread
    # except on compilers known to not have it.
    if(MSVC)
      # Compilers targeting the MSVC ABI do not have a -pthread flag.
      set(THREADS_HAVE_PTHREAD_ARG FALSE)
    elseif(NOT DEFINED THREADS_HAVE_PTHREAD_ARG)
      message(CHECK_START "Check if compiler accepts -pthread")
      if(CMAKE_C_COMPILER_LOADED)
        set(_threads_src CheckForPthreads.c)
      elseif(CMAKE_CXX_COMPILER_LOADED)
        set(_threads_src CheckForPthreads.cxx)
      endif()
      try_compile(THREADS_HAVE_PTHREAD_ARG
        SOURCE_FROM_FILE "${_threads_src}" "${CMAKE_CURRENT_LIST_DIR}/CheckForPthreads.c"
        CMAKE_FLAGS -DLINK_LIBRARIES:STRING=-pthread
        )

      unset(_threads_src)

      if(THREADS_HAVE_PTHREAD_ARG)
        set(Threads_FOUND TRUE)
        message(CHECK_PASS "yes")
      else()
        message(CHECK_FAIL "no")
      endif()

    endif()

    if(THREADS_HAVE_PTHREAD_ARG)
      set(Threads_FOUND TRUE)
      set(CMAKE_THREAD_LIBS_INIT "-pthread")
    endif()
  endif()
endmacro()

# Check if pthread functions are in normal C library.
# We list some pthread functions in PTHREAD_C_CXX_TEST_SOURCE test code.
# If the pthread functions already exist in C library, we could just use
# them instead of linking to the additional pthread library.
_threads_check_libc()

# Check for -pthread first if enabled. This is the recommended
# way, but not backwards compatible as one must also pass -pthread
# as compiler flag then.
if (THREADS_PREFER_PTHREAD_FLAG)
  _threads_check_flag_pthread()
endif ()

if(CMAKE_SYSTEM MATCHES "GHS-MULTI")
  _threads_check_lib(posix pthread_create CMAKE_HAVE_PTHREADS_CREATE)
endif()
_threads_check_lib(pthreads pthread_create CMAKE_HAVE_PTHREADS_CREATE)
_threads_check_lib(pthread  pthread_create CMAKE_HAVE_PTHREAD_CREATE)

if (NOT THREADS_PREFER_PTHREAD_FLAG)
  _threads_check_flag_pthread()
endif()

if(CMAKE_THREAD_LIBS_INIT OR CMAKE_HAVE_LIBC_PTHREAD)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(Threads_FOUND TRUE)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(CMAKE_USE_WIN32_THREADS_INIT 1)
  set(Threads_FOUND TRUE)
endif()

if(CMAKE_USE_PTHREADS_INIT)
  if(CMAKE_SYSTEM_NAME MATCHES "HP-UX")
    # Use libcma if it exists and can be used.  It provides more
    # symbols than the plain pthread library.  CMA threads
    # have actually been deprecated:
    #   http://docs.hp.com/en/B3920-90091/ch12s03.html#d0e11395
    #   http://docs.hp.com/en/947/d8.html
    # but we need to maintain compatibility here.
    # The CMAKE_HP_PTHREADS setting actually indicates whether CMA threads
    # are available.
    check_library_exists(cma pthread_attr_create "" CMAKE_HAVE_HP_CMA)
    if(CMAKE_HAVE_HP_CMA)
      set(CMAKE_THREAD_LIBS_INIT "-lcma")
      set(CMAKE_HP_PTHREADS_INIT 1)
      set(Threads_FOUND TRUE)
    endif()
    set(CMAKE_USE_PTHREADS_INIT 1)
  endif()

  if(CMAKE_SYSTEM MATCHES "OSF1-V")
    set(CMAKE_USE_PTHREADS_INIT 0)
    set(CMAKE_THREAD_LIBS_INIT )
  endif()

  if(CMAKE_SYSTEM MATCHES "CYGWIN_NT" OR CMAKE_SYSTEM MATCHES "MSYS_NT")
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(Threads_FOUND TRUE)
    set(CMAKE_THREAD_LIBS_INIT )
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
  endif()
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Threads DEFAULT_MSG Threads_FOUND)

if(Threads_FOUND AND NOT TARGET Threads::Threads)
  add_library(Threads::Threads INTERFACE IMPORTED)

  if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET Threads::Threads
                 PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler -pthread>"
                                                    "$<$<AND:$<NOT:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>,$<NOT:$<COMPILE_LANGUAGE:Swift>>>:-pthread>")
  endif()

  if(CMAKE_THREAD_LIBS_INIT)
    set_property(TARGET Threads::Threads PROPERTY INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endif()
