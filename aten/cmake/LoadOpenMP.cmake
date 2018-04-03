# OpenMP support?
set(WITH_OPENMP ON CACHE BOOL "OpenMP support if available?")
if(APPLE AND CMAKE_COMPILER_IS_GNUCC)
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  message(STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
  if(DARWIN_VERSION GREATER 9)
    set(APPLE_OPENMP_SUCKS 1)
  endif()
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION)
  if(APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
    message(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
    message(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
    set(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
  endif()
endif()

if(WITH_OPENMP AND NOT CHECKED_OPENMP)
  find_package(OpenMP)
  set(CHECKED_OPENMP ON CACHE BOOL "already checked for OpenMP")

  # OPENMP_FOUND is not cached in FindOpenMP.cmake (all other variables are cached)
  # see https://github.com/Kitware/CMake/blob/master/Modules/FindOpenMP.cmake
  set(OPENMP_FOUND ${OPENMP_FOUND} CACHE BOOL "OpenMP Support found")
endif()

if(OPENMP_FOUND)
  message(STATUS "Compiling with OpenMP support")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
