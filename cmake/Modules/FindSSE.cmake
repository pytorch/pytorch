INCLUDE(CheckCSourceRuns)
INCLUDE(CheckCXXSourceRuns)

SET(SSE1_CODE "
  #include <xmmintrin.h>

  int main()
  {
    __m128 a;
    float vals[4] = {0,0,0,0};
    a = _mm_loadu_ps(vals);
    return 0;
  }")

SET(SSE2_CODE "
  #include <emmintrin.h>

  int main()
  {
    __m128d a;
    double vals[2] = {0,0};
    a = _mm_loadu_pd(vals);
    return 0;
  }")

SET(SSE3_CODE "
  #include <pmmintrin.h>

  int main( )
  {
    const int vals[4] = {0,0,0,0};
    __m128i a;
    a = _mm_lddqu_si128( (const __m128i*)vals );
    return 0;
  }")

SET(SSE4_1_CODE "
  #include <smmintrin.h>

  int main ()
  {
    __m128i a = {0,0,0,0}, b = {0,0,0,0};
    __m128i res = _mm_max_epi8(a, b);

    return 0;
  }
")

SET(SSE4_2_CODE "
  #include <nmmintrin.h>

  int main()
  {
    __m128i a = {0,0,0,0}, b = {0,0,0,0}, c = {0,0,0,0};
    c = _mm_cmpgt_epi64(a, b);
    return 0;
  }
")

SET(AVX_CODE "
  #include <immintrin.h>

  int main()
  {
    __m256 a;
    a = _mm256_set1_ps(0);
    return 0;
  }
")

SET(AVX2_CODE "
  #include <immintrin.h>

  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    return 0;
  }
")

MACRO(CHECK_SSE lang type flags)
  SET(__FLAG_I 1)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  FOREACH(__FLAG ${flags})
    IF(NOT ${lang}_${type}_FOUND)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      IF(lang STREQUAL "CXX")
        CHECK_CXX_SOURCE_RUNS("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      ELSE()
        CHECK_C_SOURCE_RUNS("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      ENDIF()
      IF(${lang}_HAS_${type}_${__FLAG_I})
        SET(${lang}_${type}_FOUND TRUE CACHE BOOL "${lang} ${type} support")
        SET(${lang}_${type}_FLAGS "${__FLAG}" CACHE STRING "${lang} ${type} flags")
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  IF(NOT ${lang}_${type}_FOUND)
    SET(${lang}_${type}_FOUND FALSE CACHE BOOL "${lang} ${type} support")
    SET(${lang}_${type}_FLAGS "" CACHE STRING "${lang} ${type} flags")
  ENDIF()

  MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)

ENDMACRO()

CHECK_SSE(C "SSE1" " ;-msse;/arch:SSE")
CHECK_SSE(C "SSE2" " ;-msse2;/arch:SSE2")
CHECK_SSE(C "SSE3" " ;-msse3;/arch:SSE3")
CHECK_SSE(C "SSE4_1" " ;-msse4.1;-msse4;/arch:SSE4")
CHECK_SSE(C "SSE4_2" " ;-msse4.2;-msse4;/arch:SSE4")
CHECK_SSE(C "AVX" " ;-mavx;/arch:AVX")
CHECK_SSE(C "AVX2" " ;-mavx2 -mfma;/arch:AVX2")

CHECK_SSE(CXX "SSE1" " ;-msse;/arch:SSE")
CHECK_SSE(CXX "SSE2" " ;-msse2;/arch:SSE2")
CHECK_SSE(CXX "SSE3" " ;-msse3;/arch:SSE3")
CHECK_SSE(CXX "SSE4_1" " ;-msse4.1;-msse4;/arch:SSE4")
CHECK_SSE(CXX "SSE4_2" " ;-msse4.2;-msse4;/arch:SSE4")
CHECK_SSE(CXX "AVX" " ;-mavx;/arch:AVX")
CHECK_SSE(CXX "AVX2" " ;-mavx2 -mfma;/arch:AVX2")
