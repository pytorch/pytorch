find_package(Threads REQUIRED)
# For newer CMake, Threads::Threads is already defined. Otherwise, we will
# provide a backward compatible wrapper for Threads::Threads.
if(THREADS_FOUND AND NOT TARGET Threads::Threads)
  add_library(Threads::Threads INTERFACE IMPORTED)

  if(THREADS_HAVE_PTHREAD_ARG)
    set_property(
        TARGET Threads::Threads
        PROPERTY INTERFACE_COMPILE_OPTIONS "-pthread")
  endif()

  if(CMAKE_THREAD_LIBS_INIT)
    set_property(
        TARGET Threads::Threads
        PROPERTY INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endif()