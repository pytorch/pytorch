include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

set(CAFFE2_USE_EXCEPTION_PTR 1)

# ---[ Check if we want to turn off deprecated warning due to glog.
if(USE_GLOG)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "-std=c++17")
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
endif()

# ---[ Check if the compiler has AVX/AVX2 support. We only check AVX2.
if(NOT INTERN_BUILD_MOBILE)
  find_package(AVX) # checks AVX and AVX2
  if(CXX_AVX2_FOUND)
    message(STATUS "Current compiler supports avx2 extension. Will build perfkernels.")
    # Also see CMakeLists.txt under caffe2/perfkernels.
    set(CAFFE2_PERF_WITH_AVX 1)
    set(CAFFE2_PERF_WITH_AVX2 1)
  endif()
endif()
# ---[ Check if the compiler has AVX512 support.
cmake_push_check_state(RESET)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
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
if(CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
  message(STATUS "Current compiler supports avx512f extension. Will build fbgemm.")
endif()
cmake_pop_check_state()

# ---[ Checks if compiler supports -fvisibility=hidden
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)
if(${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
  set(CAFFE2_VISIBILITY_FLAG "-fvisibility-inlines-hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_VISIBILITY_FLAG}")
endif()

# ---[ Checks if linker supports -rdynamic. `-rdynamic` tells linker
# -to add all (including unused) symbols into the dynamic symbol
# -table. We need this to get symbols when generating backtrace at
# -runtime.
if(NOT MSVC)
  check_cxx_compiler_flag("-rdynamic" COMPILER_SUPPORTS_RDYNAMIC)
  if(${COMPILER_SUPPORTS_RDYNAMIC})
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  endif()
endif()

# ---[ If we are building on ios, or building with opengl support, we will
# enable -mfpu=neon-fp16 for iOS Metal build. For Android, this fpu setting
# is going to be done with android-cmake by setting
#     -DANDROID_ABI="armeabi-v7a with NEON FP16"
# in the build command.
# Also, we will turn off deprecated-declarations
# due to protobuf.

# ---[ Check if the compiler has SVE support.
find_package(ARM) # checks SVE
if(CXX_SVE_FOUND)
  message(STATUS "Compiler supports SVE extension. Will build perfkernels.")
  # Also see CMakeLists.txt under caffe2/perfkernels.
  add_compile_definitions(CAFFE2_PERF_WITH_SVE=1)
else()
  message(STATUS "Compiler does not support SVE extension. Will not build perfkernels.")
endif()

if(IOS AND (${IOS_ARCH} MATCHES "armv7*"))
  add_definitions("-mfpu=neon-fp16")
  add_definitions("-arch" ${IOS_ARCH})
  add_definitions("-Wno-deprecated-declarations")
endif()

# ---[ Create CAFFE2_BUILD_SHARED_LIBS for macros.h.in usage.
set(CAFFE2_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

if(USE_NATIVE_ARCH AND NOT MSVC)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_definitions("-march=native")
  else()
    message(
        WARNING
        "Your compiler does not support -march=native. Turn off this warning "
        "by setting -DUSE_NATIVE_ARCH=OFF.")
  endif()
endif()
