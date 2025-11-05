include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

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
