include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# ---[ If running on Ubuntu, check system version and compiler version.
if(EXISTS "/etc/os-release")
  execute_process(COMMAND
    "sed" "-ne" "s/^ID=\\([a-z]\\+\\)$/\\1/p" "/etc/os-release"
    OUTPUT_VARIABLE OS_RELEASE_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(COMMAND
    "sed" "-ne" "s/^VERSION_ID=\"\\([0-9\\.]\\+\\)\"$/\\1/p" "/etc/os-release"
    OUTPUT_VARIABLE OS_RELEASE_VERSION_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if(OS_RELEASE_ID STREQUAL "ubuntu")
    if(OS_RELEASE_VERSION_ID VERSION_GREATER "17.04")
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0.0")
          message(FATAL_ERROR
            "Please use GCC 6 or higher on Ubuntu 17.04 and higher. "
            "For more information, see: "
            "https://github.com/caffe2/caffe2/issues/1633"
            )
        endif()
      endif()
    endif()
  endif()
endif()

# ---[ Check if the data type long and int32_t/int64_t overlap.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
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
  set(CAFFE2_UNIQUE_LONG_TYPEMETA 1)
endif()
cmake_pop_check_state()

# ---[ Check if we want to turn off deprecated warning due to glog.
# Note(jiayq): on ubuntu 14.04, the default glog install uses ext/hash_set that
# is being deprecated. As a result, we will test if this is the environment we
# are building under. If yes, we will turn off deprecation warning for a
# cleaner build output.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
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
cmake_pop_check_state()

# ---[ Check if the compiler has AVX/AVX2 support. We only check AVX2.
cmake_push_check_state(RESET)
if (MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2")
endif()
CHECK_CXX_SOURCE_COMPILES(
    "#include <immintrin.h>
     int main() {
       __m256i a, b;
       a = _mm256_set1_epi8 (1);
       b = a;
       _mm256_add_epi8 (a,a);
       return 0;
     }" CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
if (CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  message(STATUS "Current compiler supports avx2 extention. Will build perfkernels.")
  # Currently MSVC seems to have a symbol not found error while linking (related
  # to source file order?). As a result we will currently disable the perfkernel
  # in msvc.
  # Also see CMakeLists.txt under caffe2/perfkernels.
  if (NOT MSVC)
    set(CAFFE2_PERF_WITH_AVX 1)
    set(CAFFE2_PERF_WITH_AVX2 1)
  endif()
endif()
cmake_pop_check_state()

# ---[ If we are using msvc, set no warning flags
# Note(jiayq): if you are going to add a warning flag, check if this is
# totally necessary, and only add when you see fit. If it is needed due to
# a third party library (like Protobuf), mention it in the comment as
# "THIRD_PARTY_NAME related"
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  add_compile_options(
      ##########################################
      # Third party related. Cannot remove.
      ##########################################
      /wd4065 # (3): switch with default but no case. Protobuf related.
      /wd4503 # (1): decorated name length exceeded, name was truncated.
              #      Eigen related.
      /wd4506 # (1): no definition for inline function. Protobuf related.
      /wd4554 # (3)： check operator precedence for possible error.
              # Eigen related.
      ##########################################
      # These are directly Caffe2 related.
      ##########################################
      /wd4018 # (3): Signed/unsigned mismatch. We've used it in many places of
              #      the code and it would be hard to correct all.
      /wd4244 # (2/3/4): Possible loss of precision. Various cases where we
              #      implicitly cast TIndex to int etc. Need further cleaning.
      /wd4267 # (3): Conversion of size_t to smaller type. Same reason as 4244.
      /wd4996 # (3): Use of deprecated POSIX functions. Since we develop
              #      mainly on Linux, this is ignored.
  )
  
  # Exception handing for compiler warining C4530, see
  # https://msdn.microsoft.com/en-us/library/2axwkyt4.aspx
  add_definitions("/EHsc")
endif()

# ---[ If we are building on ios, or building with opengl support, we will
# enable -mfpu=neon-fp16 for iOS Metal build. For Android, this fpu setting
# is going to be done with android-cmake by setting
#     -DANDROID_ABI="armeabi-v7a with NEON FP16"
# in the build command.
# Also, we will turn off deprecated-declarations
# due to protobuf.

if (IOS)
  add_definitions("-mfpu=neon-fp16")
  add_definitions("-Wno-deprecated-declarations")
endif()

# ---[ If we use asan, turn on the flags.
# TODO: This only works with new style gcc and clang (not the old -faddress-sanitizer).
# Change if necessary on old platforms.
if (USE_ASAN)
  set(CAFFE2_ASAN_FLAG "-fsanitize=address -fPIE -pie")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CAFFE2_ASAN_FLAG}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_ASAN_FLAG}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CAFFE2_ASAN_FLAG}")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${CAFFE2_ASAN_FLAG}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CAFFE2_ASAN_FLAG}")
endif()

# ---[ Create CAFFE2_BUILD_SHARED_LIBS for macros.h.in usage.
set(CAFFE2_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
