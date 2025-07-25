project(my-project LANGUAGES C CXX)

# Build output setup
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin)

# TODO(voz): Fix hack below Start hack
list(APPEND policies_new CMP0079)

foreach(policy ${policies_new})
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()
# End hack

# ##############################################################################
# GTest
# ##############################################################################
project(googletest-git NONE)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
set(BUILD_GMOCK
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_GTEST
    ON
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ##############################################################################
# Tests
# ##############################################################################

# TODO(voz): This is a little assumptive of just this one test, rewrite with
# real dir includes
include_directories(${ATEN_INCLUDE})
add_executable(test_cpp_prefix test_cpp_prefix.cpp
                               ../../torchinductor/codegen/cpp_prefix.h)
target_link_libraries(test_cpp_prefix gtest_main)
add_test(NAME test_cpp_prefix COMMAND test_cpp_prefix)
