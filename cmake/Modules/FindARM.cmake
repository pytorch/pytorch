# Check ARM feature availability for SVE compile-time support.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    INCLUDE(CheckCXXSourceCompiles)

    SET(SVE_BF16_CODE "
      #include <arm_sve.h>
      #include <arm_neon.h>
      int main()
      {
        svfloat64_t a;
        a = svdup_n_f64(0);
        float32x4_t b = vdupq_n_f32(0);
        bfloat16x8_t c = vreinterpretq_bf16_f32(b);
        bfloat16x4_t d = vget_low_bf16(c);
        return 0;
      }
    ")

    SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=armv8-a+sve+bf16 -msve-vector-bits=256")
    CHECK_CXX_SOURCE_COMPILES("${SVE_BF16_CODE}" CXX_SVE256_FOUND)
    SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    # gcc 11.5 hits an internal compiler error on the SVE intrinsics in ATen's
    # vec headers, so keep SVE off and let aarch64 fall back to NEON.
    # See https://github.com/pytorch/pytorch/issues/162422
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "11.5.0"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.6.0")
      message(WARNING "gcc ${CMAKE_CXX_COMPILER_VERSION} miscompiles SVE intrinsics. "
        "Disabling SVE and falling back to NEON; use gcc 13 or newer for SVE. "
        "See https://github.com/pytorch/pytorch/issues/162422")
      set(CXX_SVE256_FOUND FALSE CACHE BOOL "SVE256 disabled for gcc 11.5" FORCE)
      set(CXX_SVE_FOUND FALSE CACHE BOOL "SVE disabled for gcc 11.5" FORCE)
    elseif(CXX_SVE256_FOUND)
      # Any compiler that supports SVE256 also supports SVE128
      set(CXX_SVE_FOUND TRUE CACHE BOOL "SVE available on host")
      message(STATUS "SVE support detected.")
    else()
      set(CXX_SVE_FOUND FALSE CACHE BOOL "SVE not available on host")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND NOT DEFINED ENV{BUILD_IGNORE_SVE_UNAVAILABLE})
        message(FATAL_ERROR "No SVE support on this machine. "
          "Set BUILD_IGNORE_SVE_UNAVAILABLE environment variable to ignore this error.")
      else()
        message(STATUS "No SVE support on this machine.")
      endif()
    endif()

    mark_as_advanced(CXX_SVE_FOUND CXX_SVE256_FOUND)
ENDIF()
