# Check ARM feature availability for ASIMD BF16 and SVE compile-time support.
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    INCLUDE(CheckCXXSourceCompiles)

    SET(ASIMD_BF16_CODE "
      #include <arm_neon.h>
      int main()
      {
        float32x4_t b = vdupq_n_f32(0);
        bfloat16x8_t c = vreinterpretq_bf16_f32(b);
        bfloat16x4_t d = vget_low_bf16(c);
        return 0;
      }
    ")

    SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    SET(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=armv8-a+bf16")
    CHECK_CXX_SOURCE_COMPILES("${ASIMD_BF16_CODE}" CXX_ASIMD_BF16_FOUND)
    SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if(CXX_ASIMD_BF16_FOUND)
      set(CXX_ASIMD_BF16_FOUND TRUE CACHE BOOL "ASIMD BF16 available on host")
      message(STATUS "ASIMD BF16 support detected.")
    else()
      set(CXX_ASIMD_BF16_FOUND FALSE CACHE BOOL "ASIMD BF16 not available on host")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND NOT DEFINED ENV{BUILD_IGNORE_ASIMD_BF16_UNAVAILABLE})
        message(FATAL_ERROR "No ASIMD BF16 support on this machine. "
          "Set BUILD_IGNORE_ASIMD_BF16_UNAVAILABLE environment variable to ignore this error.")
      else()
        message(STATUS "No ASIMD BF16 support on this machine.")
      endif()
    endif()

    mark_as_advanced(CXX_ASIMD_BF16_FOUND CXX_ASIMD_BF16_FOUND)

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

    if(CXX_SVE256_FOUND)
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
