include(CheckCCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
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

# ---[ Check if std::exception_ptr is supported.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
CHECK_CXX_SOURCE_COMPILES(
    "#include <string>
    #include <exception>
    int main(int argc, char** argv) {
      std::exception_ptr eptr;
      try {
          std::string().at(1);
      } catch(...) {
          eptr = std::current_exception();
      }
    }" CAFFE2_EXCEPTION_PTR_SUPPORTED)

if (CAFFE2_EXCEPTION_PTR_SUPPORTED)
  message(STATUS "std::exception_ptr is supported.")
  set(CAFFE2_USE_EXCEPTION_PTR 1)
else()
  message(STATUS "std::exception_ptr is NOT supported.")
endif()
cmake_pop_check_state()

# ---[ Check for NUMA support
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
CHECK_CXX_SOURCE_COMPILES(
    "#include <numa.h>
    #include <numaif.h>

    int main(int argc, char** argv) {
    }" CAFFE2_IS_NUMA_AVAILABLE)

if (CAFFE2_IS_NUMA_AVAILABLE)
  message(STATUS "NUMA is available")
else()
  message(STATUS "NUMA is not available")
  set(CAFFE2_DISABLE_NUMA 1)
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

# ---[ Checks if compiler supports -fvisibility=hidden
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)
if (${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
  set(CAFFE2_VISIBILITY_FLAG "-fvisibility-inlines-hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_VISIBILITY_FLAG}")
endif()

# ---[ If we are using msvc, set no warning flags
# Note(jiayq): if you are going to add a warning flag, check if this is
# totally necessary, and only add when you see fit. If it is needed due to
# a third party library (like Protobuf), mention it in the comment as
# "THIRD_PARTY_NAME related"
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  add_compile_options(
      ##########################################
      # Protobuf related. Cannot remove.
      # This is directly copied from
      #     https://github.com/google/protobuf/blob/master/cmake/README.md
      ##########################################
      /wd4018 # 'expression' : signed/unsigned mismatch
      /wd4065 # (3): switch with default but no case.
      /wd4146 # unary minus operator applied to unsigned type, result still unsigned
      /wd4244 # Conversion from 'type1' to 'type2', possible loss of data.
      /wd4251 # 'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'
      /wd4267 # Conversion from 'size_t' to 'type', possible loss of data.
      /wd4305 # 'identifier' : truncation from 'type1' to 'type2'
      /wd4355 # 'this' : used in base member initializer list
      /wd4506 # (1): no definition for inline function. Protobuf related.
      /wd4661 # No suitable definition provided for explicit template instantiation request
      /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
      /wd4996 # 'function': was declared deprecated
      ##########################################
      # Third party related. Cannot remove.
      ##########################################
      /wd4141 # (1): inline used twice. google benchmark related.
      /wd4503 # (1): decorated name length exceeded, name was truncated.
              #      Eigen related.
      /wd4554 # (3): check operator precedence for possible error.
              # Eigen related.
      /wd4805 # (1): Unsafe mix of types in gtest/gtest.h. Gtest related.
      ##########################################
      # These are directly Caffe2 related. However, several are covered by
      # protobuf now. We leave them here for documentation purposes only.
      ##########################################
      #/wd4018 # (3): Signed/unsigned mismatch. We've used it in many places
      #        #      of the code and it would be hard to correct all.
      #/wd4244 # (2/3/4): Possible loss of precision. Various cases where we
      #        #      implicitly cast TIndex to int etc. Need cleaning.
      #/wd4267 # (3): Conversion of size_t to smaller type. Same as 4244.
      #/wd4996 # (3): Use of deprecated POSIX functions. Since we develop
      #        #      mainly on Linux, this is ignored.
      /wd4273 # (1): inconsistent dll linkage. This is related to the
              #      caffe2 FLAGS_* definition using dllimport in header and
              #      dllexport in cc file. The strategy is copied from gflags.
  )

  # Exception handing for compiler warining C4530, see
  # https://msdn.microsoft.com/en-us/library/2axwkyt4.aspx
  add_definitions("/EHsc")

  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049 /ignore:4217")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049 /ignore:4217")
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

# ---[ If we are building with ACL, we will enable neon-fp16.
if(USE_ACL)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^armv")
    # 32-bit ARM (armv7, armv7-a, armv7l, etc)
    set(ACL_ARCH "armv7a")
    # Compilers for 32-bit ARM need extra flags to enable NEON-FP16
    add_definitions("-mfpu=neon-fp16")

    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG(
        -mfp16-format=ieee CAFFE2_COMPILER_SUPPORTS_FP16_FORMAT)
    if (CAFFE2_COMPILER_SUPPORTS_FP16_FORMAT)
      add_definitions("-mfp16-format=ieee")
    endif()
  endif()
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

# ---[ Check if we will need to include the local Modules_CUDA_fix folder.
# Add your conditions here if needed.
if (MSVC)
  # We know that VS2017 needs the new FindCUDA functionality, so we will
  # simply enable it for the whole Windows build.
  set(CAFFE2_CMAKE_USE_LOCAL_FINDCUDA ON)
elseif (${CMAKE_VERSION} VERSION_LESS 3.7 AND ${USE_CUDA})
  set(CAFFE2_CMAKE_USE_LOCAL_FINDCUDA ON)
endif()

if (${CAFFE2_CMAKE_USE_LOCAL_FINDCUDA})
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix)
endif()

if (USE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    add_definitions("-march=native")
  else()
    message(
        WARNING
        "Your compiler does not support -march=native. Turn off this warning "
        "by setting -DUSE_NATIVE_ARCH=OFF.")
  endif()
endif()
