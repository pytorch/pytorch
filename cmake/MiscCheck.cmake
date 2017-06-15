INCLUDE(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++11")

# ---[ Check if the data type long and int32_t/int64_t overlap. 
CHECK_CXX_SOURCE_COMPILES(
    "#include <cstdint>

    template <typename T> void Foo();
    template<> void Foo<int32_t>() {}
    template<> void Foo<int64_t>() {}
    int main(int argc, char** argv) {
      Foo<long>();
      return 0;
    }" CAFFE2_LONG_IS_INT32_OR_64)

if (CAFFE2_LONG_IS_INT32_OR_64)
  message(STATUS "Does not need to define long separately.")
else()
  message(STATUS "Need to define long as a separate typeid.")
  add_definitions(-DCAFFE2_UNIQUE_LONG_TYPEMETA)
endif()

# ---[ Check if __builtin_cpu_supports is supported by the compiler
CHECK_CXX_SOURCE_COMPILES(
    "#include <iostream>

    int main(int argc, char** argv) {
      std::cout << __builtin_cpu_supports(\"avx2\") << std::endl;
      return 0;
    }" HAS_BUILTIN_CPU_SUPPORTS)
if (HAS_BUILTIN_CPU_SUPPORTS)
  message(STATUS "This compiler has builtin_cpu_supports feature.")
else()
  message(STATUS "This compiler does not have builtin_cpu_supports feature.")
  add_definitions(-DCAFFE2_NO_BUILTIN_CPU_SUPPORTS)
endif()

# Note(jiayq): on ubuntu 14.04, the default glog install uses ext/hash_set that
# is being deprecated. As a result, we will test if this is the environment we
# are building under. If yes, we will turn off deprecation warning for a
# cleaner build output.
CHECK_CXX_SOURCE_COMPILES(
    "#include <glog/stl_logging.h>
    int main(int argc, char** argv) {
      return 0;
    }" CAFFE2_NEED_TO_TURN_OFF_DEPRECATION_WARNING
    FAIL_REGEX ".*-Wno-deprecated.*")

if(NOT CAFFE2_NEED_TO_TURN_OFF_DEPRECATION_WARNING AND NOT MSVC)
  message(STATUS "Turning off deprecation warning due to glog.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
endif()

# ---[ If we are using msvc, set no warning flags
# Note(jiayq): if you are going to add a warning flag, check if this is
# totally necessary, and only add when you see fit. If it is needed due to
# a third party library (like Protobuf), mention it in the comment as
# "THIRD_PARTY_NAME related"
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  add_compile_options(
      /wd4018 # (3): Signed/unsigned mismatch
      /wd4065 # (3): switch with default but no case. Protobuf related.
      /wd4244 # (2/3/4): Possible loss of precision
      /wd4267 # (3): Conversion of size_t to smaller type. Possible loss of data.
      /wd4503 # (1): decorated name length exceeded, name was truncated. Eigen related.
      /wd4506 # (1): no definition for inline function. Protobuf related.
      /wd4554 # (3)： check operator precedence for possible error. Eigen related.
      /wd4800 # (3): Forcing non-boolean value to true or false.
      /wd4996 # (3): Use of a deprecated member
  )
endif()
