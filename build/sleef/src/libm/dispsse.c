//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdint.h>
#include <assert.h>

#include "misc.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define CONST const
#else
#define CONST
#endif

#define IMPORT_IS_EXPORT
#include "sleef.h"

static int cpuSupportsSSE4_1() {
  static int ret = -1;
  if (ret == -1) {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 1, 0);
    ret = (reg[2] & (1 << 19)) != 0;
  }
  return ret;
}

static int cpuSupportsAVX2() {
  static int ret = -1;
  if (ret == -1) {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 7, 0);
    ret = (reg[1] & (1 << 5)) != 0;
  }
  return ret;
}

static int cpuSupportsFMA() {
  static int ret = -1;
  if (ret == -1) {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 1, 0);
    ret = (reg[2] & (1 << 12)) != 0;
  }
  return ret;
}

#define SUBST_IF_SSE4(funcsse4) if (cpuSupportsSSE4_1()) p = funcsse4;

#ifdef ENABLE_AVX2
#define SUBST_IF_AVX2(funcavx2) if (cpuSupportsAVX2() && cpuSupportsFMA()) p = funcavx2;
#else
#define SUBST_IF_AVX2(funcavx2)
#endif

/*
 * DISPATCH_R_X, DISPATCH_R_X_Y and DISPATCH_R_X_Y_Z are the macro for
 * defining dispatchers. R, X, Y and Z represent the data types of
 * return value, first argument, second argument and third argument,
 * respectively. vf, vi, i and p correspond to vector FP, vector
 * integer, scalar integer and scalar pointer types, respectively.
 *
 * The arguments for the macros are as follows:
 *   fptype   : FP type name
 *   funcname : Fundamental function name
 *   pfn      : Name of pointer of the function to the dispatcher
 *   dfn      : Name of the dispatcher function
 *   funcsse2 : Name of the SSE2 function
 *   funcsse4 : Name of the SSE4 function
 *   funcavx2 : Name of the AVX2 function
 */

#define DISPATCH_vf_vf(fptype, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC fptype dfn(fptype arg0) {			\
    fptype CONST VECTOR_CC (*p)(fptype arg0) = funcsse2;		\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vf(fptype, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1);	\
  static CONST VECTOR_CC fptype dfn(fptype arg0, fptype arg1) {	\
    fptype CONST VECTOR_CC (*p)(fptype arg0, fptype arg1) = funcsse2;	\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1) = dfn; \
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, fptype arg1) { return (*pfn)(arg0, arg1); }

#define DISPATCH_vf2_vf(fptype, fptype2, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC fptype2 (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC fptype2 dfn(fptype arg0) {			\
    fptype2 CONST VECTOR_CC (*p)(fptype arg0) = funcsse2;		\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC fptype2 (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC fptype2 funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vi(fptype, itype, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, itype arg1);	\
  static CONST VECTOR_CC fptype dfn(fptype arg0, itype arg1) {		\
    fptype CONST VECTOR_CC (*p)(fptype arg0, itype arg1) = funcsse2;	\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, itype arg1) = dfn;	\
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, itype arg1) { return (*pfn)(arg0, arg1); }

#define DISPATCH_vi_vf(fptype, itype, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC itype (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC itype dfn(fptype arg0) {			\
    itype CONST VECTOR_CC (*p)(fptype arg0) = funcsse2;		\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC itype (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC itype funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vf_vf(fptype, funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1, fptype arg2); \
  static CONST VECTOR_CC fptype dfn(fptype arg0, fptype arg1, fptype arg2) { \
    fptype CONST VECTOR_CC (*p)(fptype arg0, fptype arg1, fptype arg2) = funcsse2; \
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1, arg2);					\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1, fptype arg2) = dfn; \
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, fptype arg1, fptype arg2) { return (*pfn)(arg0, arg1, arg2); }

#define DISPATCH_i_i(funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST int (*pfn)(int arg0);					\
  static CONST int dfn(int arg0) {					\
    int CONST (*p)(int) = funcsse2;					\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST int (*pfn)(int arg0) = dfn;				\
  EXPORT CONST int funcName(int arg0) { return (*pfn)(arg0); }

#define DISPATCH_p_i(funcName, pfn, dfn, funcsse2, funcsse4, funcavx2) \
  static CONST void *(*pfn)(int arg0);					\
  static CONST void *dfn(int arg0) {					\
    CONST void *(*p)(int) = funcsse2;					\
    SUBST_IF_SSE4(funcsse4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST void *(*pfn)(int arg0) = dfn;				\
  EXPORT CONST void *funcName(int arg0) { return (*pfn)(arg0); }

//
DISPATCH_vf_vf(__m128d, Sleef_sind2_u35, pnt_sind2_u35, disp_sind2_u35, Sleef_sind2_u35sse2, Sleef_sind2_u35sse4, Sleef_sind2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_sinf4_u35, pnt_sinf4_u35, disp_sinf4_u35, Sleef_sinf4_u35sse2, Sleef_sinf4_u35sse4, Sleef_sinf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_cosd2_u35, pnt_cosd2_u35, disp_cosd2_u35, Sleef_cosd2_u35sse2, Sleef_cosd2_u35sse4, Sleef_cosd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_cosf4_u35, pnt_cosf4_u35, disp_cosf4_u35, Sleef_cosf4_u35sse2, Sleef_cosf4_u35sse4, Sleef_cosf4_u35avx2128)
DISPATCH_vf2_vf(__m128d, Sleef___m128d_2, Sleef_sincosd2_u35, pnt_sincosd2_u35, disp_sincosd2_u35, Sleef_sincosd2_u35sse2, Sleef_sincosd2_u35sse4, Sleef_sincosd2_u35avx2128)
DISPATCH_vf2_vf(__m128, Sleef___m128_2, Sleef_sincosf4_u35, pnt_sincosf4_u35, disp_sincosf4_u35, Sleef_sincosf4_u35sse2, Sleef_sincosf4_u35sse4, Sleef_sincosf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_tand2_u35, pnt_tand2_u35, disp_tand2_u35, Sleef_tand2_u35sse2, Sleef_tand2_u35sse4, Sleef_tand2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_tanf4_u35, pnt_tanf4_u35, disp_tanf4_u35, Sleef_tanf4_u35sse2, Sleef_tanf4_u35sse4, Sleef_tanf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_asind2_u35, pnt_asind2_u35, disp_asind2_u35, Sleef_asind2_u35sse2, Sleef_asind2_u35sse4, Sleef_asind2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_asinf4_u35, pnt_asinf4_u35, disp_asinf4_u35, Sleef_asinf4_u35sse2, Sleef_asinf4_u35sse4, Sleef_asinf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_acosd2_u35, pnt_acosd2_u35, disp_acosd2_u35, Sleef_acosd2_u35sse2, Sleef_acosd2_u35sse4, Sleef_acosd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_acosf4_u35, pnt_acosf4_u35, disp_acosf4_u35, Sleef_acosf4_u35sse2, Sleef_acosf4_u35sse4, Sleef_acosf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_atand2_u35, pnt_atand2_u35, disp_atand2_u35, Sleef_atand2_u35sse2, Sleef_atand2_u35sse4, Sleef_atand2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_atanf4_u35, pnt_atanf4_u35, disp_atanf4_u35, Sleef_atanf4_u35sse2, Sleef_atanf4_u35sse4, Sleef_atanf4_u35avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_atan2d2_u35, pnt_atan2d2_u35, disp_atan2d2_u35, Sleef_atan2d2_u35sse2, Sleef_atan2d2_u35sse4, Sleef_atan2d2_u35avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_atan2f4_u35, pnt_atan2f4_u35, disp_atan2f4_u35, Sleef_atan2f4_u35sse2, Sleef_atan2f4_u35sse4, Sleef_atan2f4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_logd2_u35, pnt_logd2_u35, disp_logd2_u35, Sleef_logd2_u35sse2, Sleef_logd2_u35sse4, Sleef_logd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_logf4_u35, pnt_logf4_u35, disp_logf4_u35, Sleef_logf4_u35sse2, Sleef_logf4_u35sse4, Sleef_logf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_cbrtd2_u35, pnt_cbrtd2_u35, disp_cbrtd2_u35, Sleef_cbrtd2_u35sse2, Sleef_cbrtd2_u35sse4, Sleef_cbrtd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_cbrtf4_u35, pnt_cbrtf4_u35, disp_cbrtf4_u35, Sleef_cbrtf4_u35sse2, Sleef_cbrtf4_u35sse4, Sleef_cbrtf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sind2_u10, pnt_sind2_u10, disp_sind2_u10, Sleef_sind2_u10sse2, Sleef_sind2_u10sse4, Sleef_sind2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_sinf4_u10, pnt_sinf4_u10, disp_sinf4_u10, Sleef_sinf4_u10sse2, Sleef_sinf4_u10sse4, Sleef_sinf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_cosd2_u10, pnt_cosd2_u10, disp_cosd2_u10, Sleef_cosd2_u10sse2, Sleef_cosd2_u10sse4, Sleef_cosd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_cosf4_u10, pnt_cosf4_u10, disp_cosf4_u10, Sleef_cosf4_u10sse2, Sleef_cosf4_u10sse4, Sleef_cosf4_u10avx2128)
DISPATCH_vf2_vf(__m128d, Sleef___m128d_2, Sleef_sincosd2_u10, pnt_sincosd2_u10, disp_sincosd2_u10, Sleef_sincosd2_u10sse2, Sleef_sincosd2_u10sse4, Sleef_sincosd2_u10avx2128)
DISPATCH_vf2_vf(__m128, Sleef___m128_2, Sleef_sincosf4_u10, pnt_sincosf4_u10, disp_sincosf4_u10, Sleef_sincosf4_u10sse2, Sleef_sincosf4_u10sse4, Sleef_sincosf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_tand2_u10, pnt_tand2_u10, disp_tand2_u10, Sleef_tand2_u10sse2, Sleef_tand2_u10sse4, Sleef_tand2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_tanf4_u10, pnt_tanf4_u10, disp_tanf4_u10, Sleef_tanf4_u10sse2, Sleef_tanf4_u10sse4, Sleef_tanf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_asind2_u10, pnt_asind2_u10, disp_asind2_u10, Sleef_asind2_u10sse2, Sleef_asind2_u10sse4, Sleef_asind2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_asinf4_u10, pnt_asinf4_u10, disp_asinf4_u10, Sleef_asinf4_u10sse2, Sleef_asinf4_u10sse4, Sleef_asinf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_acosd2_u10, pnt_acosd2_u10, disp_acosd2_u10, Sleef_acosd2_u10sse2, Sleef_acosd2_u10sse4, Sleef_acosd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_acosf4_u10, pnt_acosf4_u10, disp_acosf4_u10, Sleef_acosf4_u10sse2, Sleef_acosf4_u10sse4, Sleef_acosf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_atand2_u10, pnt_atand2_u10, disp_atand2_u10, Sleef_atand2_u10sse2, Sleef_atand2_u10sse4, Sleef_atand2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_atanf4_u10, pnt_atanf4_u10, disp_atanf4_u10, Sleef_atanf4_u10sse2, Sleef_atanf4_u10sse4, Sleef_atanf4_u10avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_atan2d2_u10, pnt_atan2d2_u10, disp_atan2d2_u10, Sleef_atan2d2_u10sse2, Sleef_atan2d2_u10sse4, Sleef_atan2d2_u10avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_atan2f4_u10, pnt_atan2f4_u10, disp_atan2f4_u10, Sleef_atan2f4_u10sse2, Sleef_atan2f4_u10sse4, Sleef_atan2f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_logd2_u10, pnt_logd2_u10, disp_logd2_u10, Sleef_logd2_u10sse2, Sleef_logd2_u10sse4, Sleef_logd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_logf4_u10, pnt_logf4_u10, disp_logf4_u10, Sleef_logf4_u10sse2, Sleef_logf4_u10sse4, Sleef_logf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_cbrtd2_u10, pnt_cbrtd2_u10, disp_cbrtd2_u10, Sleef_cbrtd2_u10sse2, Sleef_cbrtd2_u10sse4, Sleef_cbrtd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_cbrtf4_u10, pnt_cbrtf4_u10, disp_cbrtf4_u10, Sleef_cbrtf4_u10sse2, Sleef_cbrtf4_u10sse4, Sleef_cbrtf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_expd2_u10, pnt_expd2_u10, disp_expd2_u10, Sleef_expd2_u10sse2, Sleef_expd2_u10sse4, Sleef_expd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_expf4_u10, pnt_expf4_u10, disp_expf4_u10, Sleef_expf4_u10sse2, Sleef_expf4_u10sse4, Sleef_expf4_u10avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_powd2_u10, pnt_powd2_u10, disp_powd2_u10, Sleef_powd2_u10sse2, Sleef_powd2_u10sse4, Sleef_powd2_u10avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_powf4_u10, pnt_powf4_u10, disp_powf4_u10, Sleef_powf4_u10sse2, Sleef_powf4_u10sse4, Sleef_powf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sinhd2_u10, pnt_sinhd2_u10, disp_sinhd2_u10, Sleef_sinhd2_u10sse2, Sleef_sinhd2_u10sse4, Sleef_sinhd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_sinhf4_u10, pnt_sinhf4_u10, disp_sinhf4_u10, Sleef_sinhf4_u10sse2, Sleef_sinhf4_u10sse4, Sleef_sinhf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_coshd2_u10, pnt_coshd2_u10, disp_coshd2_u10, Sleef_coshd2_u10sse2, Sleef_coshd2_u10sse4, Sleef_coshd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_coshf4_u10, pnt_coshf4_u10, disp_coshf4_u10, Sleef_coshf4_u10sse2, Sleef_coshf4_u10sse4, Sleef_coshf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_tanhd2_u10, pnt_tanhd2_u10, disp_tanhd2_u10, Sleef_tanhd2_u10sse2, Sleef_tanhd2_u10sse4, Sleef_tanhd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_tanhf4_u10, pnt_tanhf4_u10, disp_tanhf4_u10, Sleef_tanhf4_u10sse2, Sleef_tanhf4_u10sse4, Sleef_tanhf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sinhd2_u35, pnt_sinhd2_u35, disp_sinhd2_u35, Sleef_sinhd2_u35sse2, Sleef_sinhd2_u35sse4, Sleef_sinhd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_sinhf4_u35, pnt_sinhf4_u35, disp_sinhf4_u35, Sleef_sinhf4_u35sse2, Sleef_sinhf4_u35sse4, Sleef_sinhf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_coshd2_u35, pnt_coshd2_u35, disp_coshd2_u35, Sleef_coshd2_u35sse2, Sleef_coshd2_u35sse4, Sleef_coshd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_coshf4_u35, pnt_coshf4_u35, disp_coshf4_u35, Sleef_coshf4_u35sse2, Sleef_coshf4_u35sse4, Sleef_coshf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_tanhd2_u35, pnt_tanhd2_u35, disp_tanhd2_u35, Sleef_tanhd2_u35sse2, Sleef_tanhd2_u35sse4, Sleef_tanhd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_tanhf4_u35, pnt_tanhf4_u35, disp_tanhf4_u35, Sleef_tanhf4_u35sse2, Sleef_tanhf4_u35sse4, Sleef_tanhf4_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_fastsinf4_u3500, pnt_fastsinf4_u3500, disp_fastsinf4_u3500, Sleef_fastsinf4_u3500sse2, Sleef_fastsinf4_u3500sse4, Sleef_fastsinf4_u3500avx2128)
DISPATCH_vf_vf(__m128, Sleef_fastcosf4_u3500, pnt_fastcosf4_u3500, disp_fastcosf4_u3500, Sleef_fastcosf4_u3500sse2, Sleef_fastcosf4_u3500sse4, Sleef_fastcosf4_u3500avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_fastpowf4_u3500, pnt_fastpowf4_u3500, disp_fastpowf4_u3500, Sleef_fastpowf4_u3500sse2, Sleef_fastpowf4_u3500sse4, Sleef_fastpowf4_u3500avx2128)
DISPATCH_vf_vf(__m128d, Sleef_asinhd2_u10, pnt_asinhd2_u10, disp_asinhd2_u10, Sleef_asinhd2_u10sse2, Sleef_asinhd2_u10sse4, Sleef_asinhd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_asinhf4_u10, pnt_asinhf4_u10, disp_asinhf4_u10, Sleef_asinhf4_u10sse2, Sleef_asinhf4_u10sse4, Sleef_asinhf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_acoshd2_u10, pnt_acoshd2_u10, disp_acoshd2_u10, Sleef_acoshd2_u10sse2, Sleef_acoshd2_u10sse4, Sleef_acoshd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_acoshf4_u10, pnt_acoshf4_u10, disp_acoshf4_u10, Sleef_acoshf4_u10sse2, Sleef_acoshf4_u10sse4, Sleef_acoshf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_atanhd2_u10, pnt_atanhd2_u10, disp_atanhd2_u10, Sleef_atanhd2_u10sse2, Sleef_atanhd2_u10sse4, Sleef_atanhd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_atanhf4_u10, pnt_atanhf4_u10, disp_atanhf4_u10, Sleef_atanhf4_u10sse2, Sleef_atanhf4_u10sse4, Sleef_atanhf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_exp2d2_u10, pnt_exp2d2_u10, disp_exp2d2_u10, Sleef_exp2d2_u10sse2, Sleef_exp2d2_u10sse4, Sleef_exp2d2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_exp2f4_u10, pnt_exp2f4_u10, disp_exp2f4_u10, Sleef_exp2f4_u10sse2, Sleef_exp2f4_u10sse4, Sleef_exp2f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_exp2d2_u35, pnt_exp2d2_u35, disp_exp2d2_u35, Sleef_exp2d2_u35sse2, Sleef_exp2d2_u35sse4, Sleef_exp2d2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_exp2f4_u35, pnt_exp2f4_u35, disp_exp2f4_u35, Sleef_exp2f4_u35sse2, Sleef_exp2f4_u35sse4, Sleef_exp2f4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_exp10d2_u10, pnt_exp10d2_u10, disp_exp10d2_u10, Sleef_exp10d2_u10sse2, Sleef_exp10d2_u10sse4, Sleef_exp10d2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_exp10f4_u10, pnt_exp10f4_u10, disp_exp10f4_u10, Sleef_exp10f4_u10sse2, Sleef_exp10f4_u10sse4, Sleef_exp10f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_exp10d2_u35, pnt_exp10d2_u35, disp_exp10d2_u35, Sleef_exp10d2_u35sse2, Sleef_exp10d2_u35sse4, Sleef_exp10d2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_exp10f4_u35, pnt_exp10f4_u35, disp_exp10f4_u35, Sleef_exp10f4_u35sse2, Sleef_exp10f4_u35sse4, Sleef_exp10f4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_expm1d2_u10, pnt_expm1d2_u10, disp_expm1d2_u10, Sleef_expm1d2_u10sse2, Sleef_expm1d2_u10sse4, Sleef_expm1d2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_expm1f4_u10, pnt_expm1f4_u10, disp_expm1f4_u10, Sleef_expm1f4_u10sse2, Sleef_expm1f4_u10sse4, Sleef_expm1f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_log10d2_u10, pnt_log10d2_u10, disp_log10d2_u10, Sleef_log10d2_u10sse2, Sleef_log10d2_u10sse4, Sleef_log10d2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_log10f4_u10, pnt_log10f4_u10, disp_log10f4_u10, Sleef_log10f4_u10sse2, Sleef_log10f4_u10sse4, Sleef_log10f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_log2d2_u10, pnt_log2d2_u10, disp_log2d2_u10, Sleef_log2d2_u10sse2, Sleef_log2d2_u10sse4, Sleef_log2d2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_log2f4_u10, pnt_log2f4_u10, disp_log2f4_u10, Sleef_log2f4_u10sse2, Sleef_log2f4_u10sse4, Sleef_log2f4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_log2d2_u35, pnt_log2d2_u35, disp_log2d2_u35, Sleef_log2d2_u35sse2, Sleef_log2d2_u35sse4, Sleef_log2d2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_log2f4_u35, pnt_log2f4_u35, disp_log2f4_u35, Sleef_log2f4_u35sse2, Sleef_log2f4_u35sse4, Sleef_log2f4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_log1pd2_u10, pnt_log1pd2_u10, disp_log1pd2_u10, Sleef_log1pd2_u10sse2, Sleef_log1pd2_u10sse4, Sleef_log1pd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_log1pf4_u10, pnt_log1pf4_u10, disp_log1pf4_u10, Sleef_log1pf4_u10sse2, Sleef_log1pf4_u10sse4, Sleef_log1pf4_u10avx2128)
DISPATCH_vf2_vf(__m128d, Sleef___m128d_2, Sleef_sincospid2_u05, pnt_sincospid2_u05, disp_sincospid2_u05, Sleef_sincospid2_u05sse2, Sleef_sincospid2_u05sse4, Sleef_sincospid2_u05avx2128)
DISPATCH_vf2_vf(__m128, Sleef___m128_2, Sleef_sincospif4_u05, pnt_sincospif4_u05, disp_sincospif4_u05, Sleef_sincospif4_u05sse2, Sleef_sincospif4_u05sse4, Sleef_sincospif4_u05avx2128)
DISPATCH_vf2_vf(__m128d, Sleef___m128d_2, Sleef_sincospid2_u35, pnt_sincospid2_u35, disp_sincospid2_u35, Sleef_sincospid2_u35sse2, Sleef_sincospid2_u35sse4, Sleef_sincospid2_u35avx2128)
DISPATCH_vf2_vf(__m128, Sleef___m128_2, Sleef_sincospif4_u35, pnt_sincospif4_u35, disp_sincospif4_u35, Sleef_sincospif4_u35sse2, Sleef_sincospif4_u35sse4, Sleef_sincospif4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sinpid2_u05, pnt_sinpid2_u05, disp_sinpid2_u05, Sleef_sinpid2_u05sse2, Sleef_sinpid2_u05sse4, Sleef_sinpid2_u05avx2128)
DISPATCH_vf_vf(__m128, Sleef_sinpif4_u05, pnt_sinpif4_u05, disp_sinpif4_u05, Sleef_sinpif4_u05sse2, Sleef_sinpif4_u05sse4, Sleef_sinpif4_u05avx2128)
DISPATCH_vf_vf(__m128d, Sleef_cospid2_u05, pnt_cospid2_u05, disp_cospid2_u05, Sleef_cospid2_u05sse2, Sleef_cospid2_u05sse4, Sleef_cospid2_u05avx2128)
DISPATCH_vf_vf(__m128, Sleef_cospif4_u05, pnt_cospif4_u05, disp_cospif4_u05, Sleef_cospif4_u05sse2, Sleef_cospif4_u05sse4, Sleef_cospif4_u05avx2128)
DISPATCH_vf_vf_vi(__m128d, __m128i, Sleef_ldexpd2, pnt_ldexpd2, disp_ldexpd2, Sleef_ldexpd2_sse2, Sleef_ldexpd2_sse4, Sleef_ldexpd2_avx2128)
DISPATCH_vi_vf(__m128d, __m128i, Sleef_ilogbd2, pnt_ilogbd2, disp_ilogbd2, Sleef_ilogbd2_sse2, Sleef_ilogbd2_sse4, Sleef_ilogbd2_avx2128)
DISPATCH_vf_vf_vf_vf(__m128d, Sleef_fmad2, pnt_fmad2, disp_fmad2, Sleef_fmad2_sse2, Sleef_fmad2_sse4, Sleef_fmad2_avx2128)
DISPATCH_vf_vf_vf_vf(__m128, Sleef_fmaf4, pnt_fmaf4, disp_fmaf4, Sleef_fmaf4_sse2, Sleef_fmaf4_sse4, Sleef_fmaf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sqrtd2, pnt_sqrtd2, disp_sqrtd2, Sleef_sqrtd2_sse2, Sleef_sqrtd2_sse4, Sleef_sqrtd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_sqrtf4, pnt_sqrtf4, disp_sqrtf4, Sleef_sqrtf4_sse2, Sleef_sqrtf4_sse4, Sleef_sqrtf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sqrtd2_u05, pnt_sqrtd2_u05, disp_sqrtd2_u05, Sleef_sqrtd2_u05sse2, Sleef_sqrtd2_u05sse4, Sleef_sqrtd2_u05avx2128)
DISPATCH_vf_vf(__m128, Sleef_sqrtf4_u05, pnt_sqrtf4_u05, disp_sqrtf4_u05, Sleef_sqrtf4_u05sse2, Sleef_sqrtf4_u05sse4, Sleef_sqrtf4_u05avx2128)
DISPATCH_vf_vf(__m128d, Sleef_sqrtd2_u35, pnt_sqrtd2_u35, disp_sqrtd2_u35, Sleef_sqrtd2_u35sse2, Sleef_sqrtd2_u35sse4, Sleef_sqrtd2_u35avx2128)
DISPATCH_vf_vf(__m128, Sleef_sqrtf4_u35, pnt_sqrtf4_u35, disp_sqrtf4_u35, Sleef_sqrtf4_u35sse2, Sleef_sqrtf4_u35sse4, Sleef_sqrtf4_u35avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_hypotd2_u05, pnt_hypotd2_u05, disp_hypotd2_u05, Sleef_hypotd2_u05sse2, Sleef_hypotd2_u05sse4, Sleef_hypotd2_u05avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_hypotf4_u05, pnt_hypotf4_u05, disp_hypotf4_u05, Sleef_hypotf4_u05sse2, Sleef_hypotf4_u05sse4, Sleef_hypotf4_u05avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_hypotd2_u35, pnt_hypotd2_u35, disp_hypotd2_u35, Sleef_hypotd2_u35sse2, Sleef_hypotd2_u35sse4, Sleef_hypotd2_u35avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_hypotf4_u35, pnt_hypotf4_u35, disp_hypotf4_u35, Sleef_hypotf4_u35sse2, Sleef_hypotf4_u35sse4, Sleef_hypotf4_u35avx2128)
DISPATCH_vf_vf(__m128d, Sleef_fabsd2, pnt_fabsd2, disp_fabsd2, Sleef_fabsd2_sse2, Sleef_fabsd2_sse4, Sleef_fabsd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_fabsf4, pnt_fabsf4, disp_fabsf4, Sleef_fabsf4_sse2, Sleef_fabsf4_sse4, Sleef_fabsf4_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_copysignd2, pnt_copysignd2, disp_copysignd2, Sleef_copysignd2_sse2, Sleef_copysignd2_sse4, Sleef_copysignd2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_copysignf4, pnt_copysignf4, disp_copysignf4, Sleef_copysignf4_sse2, Sleef_copysignf4_sse4, Sleef_copysignf4_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_fmaxd2, pnt_fmaxd2, disp_fmaxd2, Sleef_fmaxd2_sse2, Sleef_fmaxd2_sse4, Sleef_fmaxd2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_fmaxf4, pnt_fmaxf4, disp_fmaxf4, Sleef_fmaxf4_sse2, Sleef_fmaxf4_sse4, Sleef_fmaxf4_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_fmind2, pnt_fmind2, disp_fmind2, Sleef_fmind2_sse2, Sleef_fmind2_sse4, Sleef_fmind2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_fminf4, pnt_fminf4, disp_fminf4, Sleef_fminf4_sse2, Sleef_fminf4_sse4, Sleef_fminf4_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_fdimd2, pnt_fdimd2, disp_fdimd2, Sleef_fdimd2_sse2, Sleef_fdimd2_sse4, Sleef_fdimd2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_fdimf4, pnt_fdimf4, disp_fdimf4, Sleef_fdimf4_sse2, Sleef_fdimf4_sse4, Sleef_fdimf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_truncd2, pnt_truncd2, disp_truncd2, Sleef_truncd2_sse2, Sleef_truncd2_sse4, Sleef_truncd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_truncf4, pnt_truncf4, disp_truncf4, Sleef_truncf4_sse2, Sleef_truncf4_sse4, Sleef_truncf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_floord2, pnt_floord2, disp_floord2, Sleef_floord2_sse2, Sleef_floord2_sse4, Sleef_floord2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_floorf4, pnt_floorf4, disp_floorf4, Sleef_floorf4_sse2, Sleef_floorf4_sse4, Sleef_floorf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_ceild2, pnt_ceild2, disp_ceild2, Sleef_ceild2_sse2, Sleef_ceild2_sse4, Sleef_ceild2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_ceilf4, pnt_ceilf4, disp_ceilf4, Sleef_ceilf4_sse2, Sleef_ceilf4_sse4, Sleef_ceilf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_roundd2, pnt_roundd2, disp_roundd2, Sleef_roundd2_sse2, Sleef_roundd2_sse4, Sleef_roundd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_roundf4, pnt_roundf4, disp_roundf4, Sleef_roundf4_sse2, Sleef_roundf4_sse4, Sleef_roundf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_rintd2, pnt_rintd2, disp_rintd2, Sleef_rintd2_sse2, Sleef_rintd2_sse4, Sleef_rintd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_rintf4, pnt_rintf4, disp_rintf4, Sleef_rintf4_sse2, Sleef_rintf4_sse4, Sleef_rintf4_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_nextafterd2, pnt_nextafterd2, disp_nextafterd2, Sleef_nextafterd2_sse2, Sleef_nextafterd2_sse4, Sleef_nextafterd2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_nextafterf4, pnt_nextafterf4, disp_nextafterf4, Sleef_nextafterf4_sse2, Sleef_nextafterf4_sse4, Sleef_nextafterf4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_frfrexpd2, pnt_frfrexpd2, disp_frfrexpd2, Sleef_frfrexpd2_sse2, Sleef_frfrexpd2_sse4, Sleef_frfrexpd2_avx2128)
DISPATCH_vf_vf(__m128, Sleef_frfrexpf4, pnt_frfrexpf4, disp_frfrexpf4, Sleef_frfrexpf4_sse2, Sleef_frfrexpf4_sse4, Sleef_frfrexpf4_avx2128)
DISPATCH_vi_vf(__m128d, __m128i, Sleef_expfrexpd2, pnt_expfrexpd2, disp_expfrexpd2, Sleef_expfrexpd2_sse2, Sleef_expfrexpd2_sse4, Sleef_expfrexpd2_avx2128)
DISPATCH_vf_vf_vf(__m128d, Sleef_fmodd2, pnt_fmodd2, disp_fmodd2, Sleef_fmodd2_sse2, Sleef_fmodd2_sse4, Sleef_fmodd2_avx2128)
DISPATCH_vf_vf_vf(__m128, Sleef_fmodf4, pnt_fmodf4, disp_fmodf4, Sleef_fmodf4_sse2, Sleef_fmodf4_sse4, Sleef_fmodf4_avx2128)
DISPATCH_vf2_vf(__m128d, Sleef___m128d_2, Sleef_modfd2, pnt_modfd2, disp_modfd2, Sleef_modfd2_sse2, Sleef_modfd2_sse4, Sleef_modfd2_avx2128)
DISPATCH_vf2_vf(__m128, Sleef___m128_2, Sleef_modff4, pnt_modff4, disp_modff4, Sleef_modff4_sse2, Sleef_modff4_sse4, Sleef_modff4_avx2128)
DISPATCH_vf_vf(__m128d, Sleef_lgammad2_u10, pnt_lgammad2_u10, disp_lgammad2_u10, Sleef_lgammad2_u10sse2, Sleef_lgammad2_u10sse4, Sleef_lgammad2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_lgammaf4_u10, pnt_lgammaf4_u10, disp_lgammaf4_u10, Sleef_lgammaf4_u10sse2, Sleef_lgammaf4_u10sse4, Sleef_lgammaf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_tgammad2_u10, pnt_tgammad2_u10, disp_tgammad2_u10, Sleef_tgammad2_u10sse2, Sleef_tgammad2_u10sse4, Sleef_tgammad2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_tgammaf4_u10, pnt_tgammaf4_u10, disp_tgammaf4_u10, Sleef_tgammaf4_u10sse2, Sleef_tgammaf4_u10sse4, Sleef_tgammaf4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_erfd2_u10, pnt_erfd2_u10, disp_erfd2_u10, Sleef_erfd2_u10sse2, Sleef_erfd2_u10sse4, Sleef_erfd2_u10avx2128)
DISPATCH_vf_vf(__m128, Sleef_erff4_u10, pnt_erff4_u10, disp_erff4_u10, Sleef_erff4_u10sse2, Sleef_erff4_u10sse4, Sleef_erff4_u10avx2128)
DISPATCH_vf_vf(__m128d, Sleef_erfcd2_u15, pnt_erfcd2_u15, disp_erfcd2_u15, Sleef_erfcd2_u15sse2, Sleef_erfcd2_u15sse4, Sleef_erfcd2_u15avx2128)
DISPATCH_vf_vf(__m128, Sleef_erfcf4_u15, pnt_erfcf4_u15, disp_erfcf4_u15, Sleef_erfcf4_u15sse2, Sleef_erfcf4_u15sse4, Sleef_erfcf4_u15avx2128)
DISPATCH_i_i(Sleef_getIntf4, pnt_getIntf4, disp_getIntf4, Sleef_getIntf4_sse2, Sleef_getIntf4_sse4, Sleef_getIntf4_avx2128)
DISPATCH_i_i(Sleef_getIntd2, pnt_getIntd2, disp_getIntd2, Sleef_getIntd2_sse2, Sleef_getIntd2_sse4, Sleef_getIntd2_avx2128)
DISPATCH_p_i(Sleef_getPtrf4, pnt_getPtrf4, disp_getPtrf4, Sleef_getPtrf4_sse2, Sleef_getPtrf4_sse4, Sleef_getPtrf4_avx2128)
DISPATCH_p_i(Sleef_getPtrd2, pnt_getPtrd2, disp_getPtrd2, Sleef_getPtrd2_sse2, Sleef_getPtrd2_sse4, Sleef_getPtrd2_avx2128)
