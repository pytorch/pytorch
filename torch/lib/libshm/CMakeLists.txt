project(libshm C CXX)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)

if(NOT LIBSHM_INSTALL_LIB_SUBDIR)
  set(LIBSHM_INSTALL_LIB_SUBDIR "lib" CACHE PATH "libshm install library directory")
endif()

add_library(shm SHARED core.cpp)
if(HAVE_SOVERSION)
  set_target_properties(shm PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()

target_include_directories(shm PUBLIC
  ${TORCH_ROOT}/torch/lib # provides "libshm/libshm.h"
)

### Torch packages supposes libraries prefix is "lib"
set_target_properties(shm PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib"
  CXX_STANDARD 17)
target_link_libraries(shm PRIVATE torch_cpu)

if(UNIX AND NOT APPLE)
  include(CheckLibraryExists)
  find_package(Threads REQUIRED)
  # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
  check_library_exists(rt clock_gettime "time.h" NEED_LIBRT)
  if(NEED_LIBRT)
    target_link_libraries(shm PUBLIC rt)
  else()
    message(STATUS "Checking if rt requires pthread")
    # Sometimes, rt won't be available unless you also link against
    # pthreads.  In this case, the NEED_LIBRT test will fail, because
    # check_library_exists isn't going to build the C file with the
    # pthread file, and the build will fail, setting NEED_LIBRT to
    # false (this is TOTALLY BOGUS, this situation should be an error
    # situation, not a "oh, I guess rt is not supported", but it's
    # not too easy to distinguish between the two situations).  So,
    # if it fails, we try again, but this time also with a dependency
    # on pthread.  If it succeeds this time, we know we not only need
    # an rt dependency, but we also need pthread.
    #
    # BTW, this test looks for shm_open, because that's what we
    # really care about (not clock_gettime).  I didn't change the
    # site above though in case there was a reason we were testing
    # against clock_gettime. In principle, the choice of symbol you
    # test for shouldn't matter.
    set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
    check_library_exists(rt shm_open "sys/mman.h" NEED_RT_AND_PTHREAD)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NEED_RT_AND_PTHREAD)
      message(STATUS "Needs it, linking against pthread and rt")
      target_link_libraries(shm PUBLIC rt Threads::Threads)
    endif()
  endif()
endif()

add_executable(torch_shm_manager manager.cpp)
target_link_libraries(torch_shm_manager PRIVATE shm c10)
set_target_properties(torch_shm_manager PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib")

install(TARGETS shm LIBRARY DESTINATION ${LIBSHM_INSTALL_LIB_SUBDIR})
install(FILES libshm.h DESTINATION "include")
install(TARGETS torch_shm_manager DESTINATION "bin")
