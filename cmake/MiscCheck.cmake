if (UNIX)
  # prevent Unknown CMake command "check_function_exists".
  include(CheckFunctionExists)
endif()
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
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

if (NOT INTERN_BUILD_MOBILE)
  # ---[ Check that our programs run.  This is different from the native CMake
  # compiler check, which just tests if the program compiles and links.  This is
  # important because with ASAN you might need to help the compiled library find
  # some dynamic libraries.
  cmake_push_check_state(RESET)
  CHECK_C_SOURCE_RUNS("
  int main() { return 0; }
  " COMPILER_WORKS)
  if (NOT COMPILER_WORKS)
    # Force cmake to retest next time around
    unset(COMPILER_WORKS CACHE)
    message(FATAL_ERROR
        "Could not run a simple program built with your compiler. "
        "If you are trying to use -fsanitize=address, make sure "
        "libasan is properly installed on your system (you can confirm "
        "if the problem is this by attempting to build and run a "
        "small program.)")
  endif()
  cmake_pop_check_state()
endif()

if (NOT INTERN_BUILD_MOBILE)
  # ---[ Check if certain std functions are supported. Sometimes
  # _GLIBCXX_USE_C99 macro is not defined and some functions are missing.
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "-std=c++14")
  CHECK_CXX_SOURCE_COMPILES("
  #include <cmath>
  #include <string>

  int main() {
    int a = std::isinf(3.0);
    int b = std::isnan(0.0);
    std::string s = std::to_string(1);

    return 0;
    }" SUPPORT_GLIBCXX_USE_C99)
  if (NOT SUPPORT_GLIBCXX_USE_C99)
    # Force cmake to retest next time around
    unset(SUPPORT_GLIBCXX_USE_C99 CACHE)
    message(FATAL_ERROR
        "The C++ compiler does not support required functions. "
        "This is very likely due to a known bug in GCC 5 "
        "(and maybe other versions) on Ubuntu 17.10 and newer. "
        "For more information, see: "
        "https://github.com/pytorch/pytorch/issues/5229")
  endif()
  cmake_pop_check_state()
endif()

# ---[ Check if std::exception_ptr is supported.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++14")
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

# ---[ Check if we want to turn off deprecated warning due to glog.
# Note(jiayq): on ubuntu 14.04, the default glog install uses ext/hash_set that
# is being deprecated. As a result, we will test if this is the environment we
# are building under. If yes, we will turn off deprecation warning for a
# cleaner build output.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++14")
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
if (NOT INTERN_BUILD_MOBILE)
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
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
      }" CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  if (CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
    message(STATUS "Current compiler supports avx2 extension. Will build perfkernels.")
    # Also see CMakeLists.txt under caffe2/perfkernels.
    set(CAFFE2_PERF_WITH_AVX 1)
    set(CAFFE2_PERF_WITH_AVX2 1)
  endif()
  cmake_pop_check_state()
endif()
# ---[ Check if the compiler has AVX512 support.
cmake_push_check_state(RESET)
if (MSVC)
  # We could've used MSVC's hidden option /arch:AVX512 that defines __AVX512F__,
  # __AVX512DQ__, and __AVX512VL__, and /arch:AVX512F that defines __AVX512F__.
  # But, we chose not to do that not to rely on hidden options.
  set(CMAKE_REQUIRED_FLAGS "/D__AVX512F__ /D__AVX512DQ__ /D__AVX512VL__")
else()
  # We only consider the case where all of avx512f, avx512dq, and avx512vl are
  # supported.
  # Platforms where avx512f is supported by not avx512dq and avx512vl as of
  # Jan 15 2019 : linux_manywheel_2.7mu_cpu_build and
  # linux_conda_3.7_cu100_build
  set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512dq -mavx512vl")
endif()
CHECK_CXX_SOURCE_COMPILES(
    "#if defined(_MSC_VER)
     #include <intrin.h>
     #else
     #include <immintrin.h>
     #endif
     // check avx512f
     __m512 addConstant(__m512 arg) {
       return _mm512_add_ps(arg, _mm512_set1_ps(1.f));
     }
     // check avx512dq
     __m512 andConstant(__m512 arg) {
       return _mm512_and_ps(arg, _mm512_set1_ps(1.f));
     }
     int main() {
       __m512i a = _mm512_set1_epi32(1);
       __m256i ymm = _mm512_extracti64x4_epi64(a, 0);
       ymm = _mm256_abs_epi64(ymm); // check avx512vl
       __mmask16 m = _mm512_cmp_epi32_mask(a, a, _MM_CMPINT_EQ);
       __m512i r = _mm512_andnot_si512(a, a);
     }" CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
if (CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
  message(STATUS "Current compiler supports avx512f extension. Will build fbgemm.")
  # Also see CMakeLists.txt under caffe2/perfkernels.
  set(CAFFE2_PERF_WITH_AVX512 1)
endif()
cmake_pop_check_state()

# ---[ Checks if compiler supports -fvisibility=hidden
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)
if (${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
  set(CAFFE2_VISIBILITY_FLAG "-fvisibility-inlines-hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_VISIBILITY_FLAG}")
endif()

# ---[ Checks if linker supports -rdynamic. `-rdynamic` tells linker
# -to add all (including unused) symbols into the dynamic symbol
# -table. We need this to get symbols when generating backtrace at
# -runtime.
check_cxx_compiler_flag("-rdynamic" COMPILER_SUPPORTS_RDYNAMIC)
if (${COMPILER_SUPPORTS_RDYNAMIC})
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
endif()

# ---[ If we are using msvc, set no warning flags
# Note(jiayq): if you are going to add a warning flag, check if this is
# totally necessary, and only add when you see fit. If it is needed due to
# a third party library (like Protobuf), mention it in the comment as
# "THIRD_PARTY_NAME related"
# From https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
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
      # These are directly ATen related. However, several are covered by
      # the above now. We leave them here for documentation purposes only.
      #/wd4267 # Conversion from 'size_t' to 'type', possible loss of data.
      /wd4522 # (3): 'class' : multiple assignment operators specified
      /wd4838 # (1): conversion from 'type_1' to 'type_2' requires a
              #      narrowing conversion
      #/wd4305 # 'identifier' : truncation from 'type1' to 'type2'
      #/wd4244 # Conversion from 'type1' to 'type2', possible loss of data.
      /wd4190 # (1): 'identifier1' has C-linkage specified, but returns UDT
              #      'identifier2' which is incompatible with C
      /wd4101 # (3): 'identifier' : unreferenced local variable
      #/wd4996 # (3): Use of deprecated POSIX functions. Since we develop
      #        #      mainly on Linux, this is ignored.
      /wd4275 # (2): non - DLL-interface classkey 'identifier' used as
              #      base for DLL-interface classkey 'identifier'
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

  # Make sure windows.h does not include additional headers.
  add_definitions("/DWIN32_LEAN_AND_MEAN")

  # Make sure windef.h does not define max/min macros.
  # Required by ATen among others.
  add_definitions("/DNOMINMAX")

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

if (USE_TSAN)
  set(CAFFE2_TSAN_FLAG "-fsanitize=thread -fPIE -pie")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CAFFE2_TSAN_FLAG}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_TSAN_FLAG}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CAFFE2_TSAN_FLAG}")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${CAFFE2_TSAN_FLAG}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CAFFE2_TSAN_FLAG}")
endif()

# ---[ Create CAFFE2_BUILD_SHARED_LIBS for macros.h.in usage.
set(CAFFE2_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

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
