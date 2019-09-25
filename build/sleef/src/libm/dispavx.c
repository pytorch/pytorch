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

static int cpuSupportsFMA4() {
  static int ret = -1;
  if (ret == -1) {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 0x80000001, 0);
    ret = (reg[3] & (1 << 16)) != 0;
  }
  return ret;
}

#ifdef ENABLE_FMA4
#define SUBST_IF_FMA4(funcfma4) if (cpuSupportsFMA4()) p = funcfma4;
#else
#define SUBST_IF_FMA4(funcfma4)
#endif

#ifdef ENABLE_AVX2
#define SUBST_IF_AVX2(funcavx2) if (cpuSupportsAVX2() && cpuSupportsFMA()) p = funcavx2;
#else
#define SUBST_IF_AVX2(funcavx2)
#endif

#define DISPATCH_vf_vf(fptype, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC fptype dfn(fptype arg0) {			\
    fptype CONST VECTOR_CC (*p)(fptype arg0) = funcavx;		\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vf(fptype, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1);	\
  static CONST VECTOR_CC fptype dfn(fptype arg0, fptype arg1) {	\
    fptype CONST VECTOR_CC (*p)(fptype arg0, fptype arg1) = funcavx;	\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1) = dfn;	\
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, fptype arg1) { return (*pfn)(arg0, arg1); }

#define DISPATCH_vf2_vf(fptype, fptype2, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC fptype2 (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC fptype2 dfn(fptype arg0) {			\
    fptype2 CONST VECTOR_CC (*p)(fptype arg0) = funcavx;		\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC fptype2 (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC fptype2 funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vi(fptype, itype, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, itype arg1);	\
  static CONST VECTOR_CC fptype dfn(fptype arg0, itype arg1) {		\
    fptype CONST VECTOR_CC (*p)(fptype arg0, itype arg1) = funcavx;	\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1);						\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, itype arg1) = dfn;	\
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, itype arg1) { return (*pfn)(arg0, arg1); }

#define DISPATCH_vi_vf(fptype, itype, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC itype (*pfn)(fptype arg0);			\
  static CONST VECTOR_CC itype dfn(fptype arg0) {			\
    itype CONST VECTOR_CC (*p)(fptype arg0) = funcavx;			\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST VECTOR_CC itype (*pfn)(fptype arg0) = dfn;		\
  EXPORT CONST VECTOR_CC itype funcName(fptype arg0) { return (*pfn)(arg0); }

#define DISPATCH_vf_vf_vf_vf(fptype, funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1, fptype arg2); \
  static CONST VECTOR_CC fptype dfn(fptype arg0, fptype arg1, fptype arg2) { \
    fptype CONST VECTOR_CC (*p)(fptype arg0, fptype arg1, fptype arg2) = funcavx; \
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0, arg1, arg2);					\
  }									\
  static CONST VECTOR_CC fptype (*pfn)(fptype arg0, fptype arg1, fptype arg2) = dfn; \
  EXPORT CONST VECTOR_CC fptype funcName(fptype arg0, fptype arg1, fptype arg2) { return (*pfn)(arg0, arg1, arg2); }

#define DISPATCH_i_i(funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST int (*pfn)(int arg0);					\
  static CONST int dfn(int arg0) {					\
    int CONST (*p)(int) = funcavx;					\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST int (*pfn)(int arg0) = dfn;				\
  EXPORT CONST int funcName(int arg0) { return (*pfn)(arg0); }

#define DISPATCH_p_i(funcName, pfn, dfn, funcavx, funcfma4, funcavx2) \
  static CONST void *(*pfn)(int arg0);					\
  static CONST void *dfn(int arg0) {					\
    CONST void *(*p)(int) = funcavx;					\
    SUBST_IF_FMA4(funcfma4);						\
    SUBST_IF_AVX2(funcavx2);						\
    pfn = p;								\
    return (*pfn)(arg0);						\
  }									\
  static CONST void *(*pfn)(int arg0) = dfn;				\
  EXPORT CONST void *funcName(int arg0) { return (*pfn)(arg0); }

//
DISPATCH_vf_vf(__m256d, Sleef_sind4_u35, pnt_sind4_u35, disp_sind4_u35, Sleef_sind4_u35avx, Sleef_sind4_u35fma4, Sleef_sind4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_sinf8_u35, pnt_sinf8_u35, disp_sinf8_u35, Sleef_sinf8_u35avx, Sleef_sinf8_u35fma4, Sleef_sinf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_cosd4_u35, pnt_cosd4_u35, disp_cosd4_u35, Sleef_cosd4_u35avx, Sleef_cosd4_u35fma4, Sleef_cosd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_cosf8_u35, pnt_cosf8_u35, disp_cosf8_u35, Sleef_cosf8_u35avx, Sleef_cosf8_u35fma4, Sleef_cosf8_u35avx2)
DISPATCH_vf2_vf(__m256d, Sleef___m256d_2, Sleef_sincosd4_u35, pnt_sincosd4_u35, disp_sincosd4_u35, Sleef_sincosd4_u35avx, Sleef_sincosd4_u35fma4, Sleef_sincosd4_u35avx2)
DISPATCH_vf2_vf(__m256, Sleef___m256_2, Sleef_sincosf8_u35, pnt_sincosf8_u35, disp_sincosf8_u35, Sleef_sincosf8_u35avx, Sleef_sincosf8_u35fma4, Sleef_sincosf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_tand4_u35, pnt_tand4_u35, disp_tand4_u35, Sleef_tand4_u35avx, Sleef_tand4_u35fma4, Sleef_tand4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_tanf8_u35, pnt_tanf8_u35, disp_tanf8_u35, Sleef_tanf8_u35avx, Sleef_tanf8_u35fma4, Sleef_tanf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_asind4_u35, pnt_asind4_u35, disp_asind4_u35, Sleef_asind4_u35avx, Sleef_asind4_u35fma4, Sleef_asind4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_asinf8_u35, pnt_asinf8_u35, disp_asinf8_u35, Sleef_asinf8_u35avx, Sleef_asinf8_u35fma4, Sleef_asinf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_acosd4_u35, pnt_acosd4_u35, disp_acosd4_u35, Sleef_acosd4_u35avx, Sleef_acosd4_u35fma4, Sleef_acosd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_acosf8_u35, pnt_acosf8_u35, disp_acosf8_u35, Sleef_acosf8_u35avx, Sleef_acosf8_u35fma4, Sleef_acosf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_atand4_u35, pnt_atand4_u35, disp_atand4_u35, Sleef_atand4_u35avx, Sleef_atand4_u35fma4, Sleef_atand4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_atanf8_u35, pnt_atanf8_u35, disp_atanf8_u35, Sleef_atanf8_u35avx, Sleef_atanf8_u35fma4, Sleef_atanf8_u35avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_atan2d4_u35, pnt_atan2d4_u35, disp_atan2d4_u35, Sleef_atan2d4_u35avx, Sleef_atan2d4_u35fma4, Sleef_atan2d4_u35avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_atan2f8_u35, pnt_atan2f8_u35, disp_atan2f8_u35, Sleef_atan2f8_u35avx, Sleef_atan2f8_u35fma4, Sleef_atan2f8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_logd4_u35, pnt_logd4_u35, disp_logd4_u35, Sleef_logd4_u35avx, Sleef_logd4_u35fma4, Sleef_logd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_logf8_u35, pnt_logf8_u35, disp_logf8_u35, Sleef_logf8_u35avx, Sleef_logf8_u35fma4, Sleef_logf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_cbrtd4_u35, pnt_cbrtd4_u35, disp_cbrtd4_u35, Sleef_cbrtd4_u35avx, Sleef_cbrtd4_u35fma4, Sleef_cbrtd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_cbrtf8_u35, pnt_cbrtf8_u35, disp_cbrtf8_u35, Sleef_cbrtf8_u35avx, Sleef_cbrtf8_u35fma4, Sleef_cbrtf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_sind4_u10, pnt_sind4_u10, disp_sind4_u10, Sleef_sind4_u10avx, Sleef_sind4_u10fma4, Sleef_sind4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_sinf8_u10, pnt_sinf8_u10, disp_sinf8_u10, Sleef_sinf8_u10avx, Sleef_sinf8_u10fma4, Sleef_sinf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_cosd4_u10, pnt_cosd4_u10, disp_cosd4_u10, Sleef_cosd4_u10avx, Sleef_cosd4_u10fma4, Sleef_cosd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_cosf8_u10, pnt_cosf8_u10, disp_cosf8_u10, Sleef_cosf8_u10avx, Sleef_cosf8_u10fma4, Sleef_cosf8_u10avx2)
DISPATCH_vf2_vf(__m256d, Sleef___m256d_2, Sleef_sincosd4_u10, pnt_sincosd4_u10, disp_sincosd4_u10, Sleef_sincosd4_u10avx, Sleef_sincosd4_u10fma4, Sleef_sincosd4_u10avx2)
DISPATCH_vf2_vf(__m256, Sleef___m256_2, Sleef_sincosf8_u10, pnt_sincosf8_u10, disp_sincosf8_u10, Sleef_sincosf8_u10avx, Sleef_sincosf8_u10fma4, Sleef_sincosf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_tand4_u10, pnt_tand4_u10, disp_tand4_u10, Sleef_tand4_u10avx, Sleef_tand4_u10fma4, Sleef_tand4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_tanf8_u10, pnt_tanf8_u10, disp_tanf8_u10, Sleef_tanf8_u10avx, Sleef_tanf8_u10fma4, Sleef_tanf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_asind4_u10, pnt_asind4_u10, disp_asind4_u10, Sleef_asind4_u10avx, Sleef_asind4_u10fma4, Sleef_asind4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_asinf8_u10, pnt_asinf8_u10, disp_asinf8_u10, Sleef_asinf8_u10avx, Sleef_asinf8_u10fma4, Sleef_asinf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_acosd4_u10, pnt_acosd4_u10, disp_acosd4_u10, Sleef_acosd4_u10avx, Sleef_acosd4_u10fma4, Sleef_acosd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_acosf8_u10, pnt_acosf8_u10, disp_acosf8_u10, Sleef_acosf8_u10avx, Sleef_acosf8_u10fma4, Sleef_acosf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_atand4_u10, pnt_atand4_u10, disp_atand4_u10, Sleef_atand4_u10avx, Sleef_atand4_u10fma4, Sleef_atand4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_atanf8_u10, pnt_atanf8_u10, disp_atanf8_u10, Sleef_atanf8_u10avx, Sleef_atanf8_u10fma4, Sleef_atanf8_u10avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_atan2d4_u10, pnt_atan2d4_u10, disp_atan2d4_u10, Sleef_atan2d4_u10avx, Sleef_atan2d4_u10fma4, Sleef_atan2d4_u10avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_atan2f8_u10, pnt_atan2f8_u10, disp_atan2f8_u10, Sleef_atan2f8_u10avx, Sleef_atan2f8_u10fma4, Sleef_atan2f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_logd4_u10, pnt_logd4_u10, disp_logd4_u10, Sleef_logd4_u10avx, Sleef_logd4_u10fma4, Sleef_logd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_logf8_u10, pnt_logf8_u10, disp_logf8_u10, Sleef_logf8_u10avx, Sleef_logf8_u10fma4, Sleef_logf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_cbrtd4_u10, pnt_cbrtd4_u10, disp_cbrtd4_u10, Sleef_cbrtd4_u10avx, Sleef_cbrtd4_u10fma4, Sleef_cbrtd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_cbrtf8_u10, pnt_cbrtf8_u10, disp_cbrtf8_u10, Sleef_cbrtf8_u10avx, Sleef_cbrtf8_u10fma4, Sleef_cbrtf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_expd4_u10, pnt_expd4_u10, disp_expd4_u10, Sleef_expd4_u10avx, Sleef_expd4_u10fma4, Sleef_expd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_expf8_u10, pnt_expf8_u10, disp_expf8_u10, Sleef_expf8_u10avx, Sleef_expf8_u10fma4, Sleef_expf8_u10avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_powd4_u10, pnt_powd4_u10, disp_powd4_u10, Sleef_powd4_u10avx, Sleef_powd4_u10fma4, Sleef_powd4_u10avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_powf8_u10, pnt_powf8_u10, disp_powf8_u10, Sleef_powf8_u10avx, Sleef_powf8_u10fma4, Sleef_powf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_sinhd4_u10, pnt_sinhd4_u10, disp_sinhd4_u10, Sleef_sinhd4_u10avx, Sleef_sinhd4_u10fma4, Sleef_sinhd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_sinhf8_u10, pnt_sinhf8_u10, disp_sinhf8_u10, Sleef_sinhf8_u10avx, Sleef_sinhf8_u10fma4, Sleef_sinhf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_coshd4_u10, pnt_coshd4_u10, disp_coshd4_u10, Sleef_coshd4_u10avx, Sleef_coshd4_u10fma4, Sleef_coshd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_coshf8_u10, pnt_coshf8_u10, disp_coshf8_u10, Sleef_coshf8_u10avx, Sleef_coshf8_u10fma4, Sleef_coshf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_tanhd4_u10, pnt_tanhd4_u10, disp_tanhd4_u10, Sleef_tanhd4_u10avx, Sleef_tanhd4_u10fma4, Sleef_tanhd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_tanhf8_u10, pnt_tanhf8_u10, disp_tanhf8_u10, Sleef_tanhf8_u10avx, Sleef_tanhf8_u10fma4, Sleef_tanhf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_sinhd4_u35, pnt_sinhd4_u35, disp_sinhd4_u35, Sleef_sinhd4_u35avx, Sleef_sinhd4_u35fma4, Sleef_sinhd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_sinhf8_u35, pnt_sinhf8_u35, disp_sinhf8_u35, Sleef_sinhf8_u35avx, Sleef_sinhf8_u35fma4, Sleef_sinhf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_coshd4_u35, pnt_coshd4_u35, disp_coshd4_u35, Sleef_coshd4_u35avx, Sleef_coshd4_u35fma4, Sleef_coshd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_coshf8_u35, pnt_coshf8_u35, disp_coshf8_u35, Sleef_coshf8_u35avx, Sleef_coshf8_u35fma4, Sleef_coshf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_tanhd4_u35, pnt_tanhd4_u35, disp_tanhd4_u35, Sleef_tanhd4_u35avx, Sleef_tanhd4_u35fma4, Sleef_tanhd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_tanhf8_u35, pnt_tanhf8_u35, disp_tanhf8_u35, Sleef_tanhf8_u35avx, Sleef_tanhf8_u35fma4, Sleef_tanhf8_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_fastsinf8_u3500, pnt_fastsinf8_u3500, disp_fastsinf8_u3500, Sleef_fastsinf8_u3500avx, Sleef_fastsinf8_u3500fma4, Sleef_fastsinf8_u3500avx2)
DISPATCH_vf_vf(__m256, Sleef_fastcosf8_u3500, pnt_fastcosf8_u3500, disp_fastcosf8_u3500, Sleef_fastcosf8_u3500avx, Sleef_fastcosf8_u3500fma4, Sleef_fastcosf8_u3500avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_fastpowf8_u3500, pnt_fastpowf8_u3500, disp_fastpowf8_u3500, Sleef_fastpowf8_u3500avx, Sleef_fastpowf8_u3500fma4, Sleef_fastpowf8_u3500avx2)
DISPATCH_vf_vf(__m256d, Sleef_asinhd4_u10, pnt_asinhd4_u10, disp_asinhd4_u10, Sleef_asinhd4_u10avx, Sleef_asinhd4_u10fma4, Sleef_asinhd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_asinhf8_u10, pnt_asinhf8_u10, disp_asinhf8_u10, Sleef_asinhf8_u10avx, Sleef_asinhf8_u10fma4, Sleef_asinhf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_acoshd4_u10, pnt_acoshd4_u10, disp_acoshd4_u10, Sleef_acoshd4_u10avx, Sleef_acoshd4_u10fma4, Sleef_acoshd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_acoshf8_u10, pnt_acoshf8_u10, disp_acoshf8_u10, Sleef_acoshf8_u10avx, Sleef_acoshf8_u10fma4, Sleef_acoshf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_atanhd4_u10, pnt_atanhd4_u10, disp_atanhd4_u10, Sleef_atanhd4_u10avx, Sleef_atanhd4_u10fma4, Sleef_atanhd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_atanhf8_u10, pnt_atanhf8_u10, disp_atanhf8_u10, Sleef_atanhf8_u10avx, Sleef_atanhf8_u10fma4, Sleef_atanhf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_exp2d4_u10, pnt_exp2d4_u10, disp_exp2d4_u10, Sleef_exp2d4_u10avx, Sleef_exp2d4_u10fma4, Sleef_exp2d4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_exp2f8_u10, pnt_exp2f8_u10, disp_exp2f8_u10, Sleef_exp2f8_u10avx, Sleef_exp2f8_u10fma4, Sleef_exp2f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_exp2d4_u35, pnt_exp2d4_u35, disp_exp2d4_u35, Sleef_exp2d4_u35avx, Sleef_exp2d4_u35fma4, Sleef_exp2d4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_exp2f8_u35, pnt_exp2f8_u35, disp_exp2f8_u35, Sleef_exp2f8_u35avx, Sleef_exp2f8_u35fma4, Sleef_exp2f8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_exp10d4_u10, pnt_exp10d4_u10, disp_exp10d4_u10, Sleef_exp10d4_u10avx, Sleef_exp10d4_u10fma4, Sleef_exp10d4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_exp10f8_u10, pnt_exp10f8_u10, disp_exp10f8_u10, Sleef_exp10f8_u10avx, Sleef_exp10f8_u10fma4, Sleef_exp10f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_exp10d4_u35, pnt_exp10d4_u35, disp_exp10d4_u35, Sleef_exp10d4_u35avx, Sleef_exp10d4_u35fma4, Sleef_exp10d4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_exp10f8_u35, pnt_exp10f8_u35, disp_exp10f8_u35, Sleef_exp10f8_u35avx, Sleef_exp10f8_u35fma4, Sleef_exp10f8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_expm1d4_u10, pnt_expm1d4_u10, disp_expm1d4_u10, Sleef_expm1d4_u10avx, Sleef_expm1d4_u10fma4, Sleef_expm1d4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_expm1f8_u10, pnt_expm1f8_u10, disp_expm1f8_u10, Sleef_expm1f8_u10avx, Sleef_expm1f8_u10fma4, Sleef_expm1f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_log10d4_u10, pnt_log10d4_u10, disp_log10d4_u10, Sleef_log10d4_u10avx, Sleef_log10d4_u10fma4, Sleef_log10d4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_log10f8_u10, pnt_log10f8_u10, disp_log10f8_u10, Sleef_log10f8_u10avx, Sleef_log10f8_u10fma4, Sleef_log10f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_log2d4_u10, pnt_log2d4_u10, disp_log2d4_u10, Sleef_log2d4_u10avx, Sleef_log2d4_u10fma4, Sleef_log2d4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_log2f8_u10, pnt_log2f8_u10, disp_log2f8_u10, Sleef_log2f8_u10avx, Sleef_log2f8_u10fma4, Sleef_log2f8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_log2d4_u35, pnt_log2d4_u35, disp_log2d4_u35, Sleef_log2d4_u35avx, Sleef_log2d4_u35fma4, Sleef_log2d4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_log2f8_u35, pnt_log2f8_u35, disp_log2f8_u35, Sleef_log2f8_u35avx, Sleef_log2f8_u35fma4, Sleef_log2f8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_log1pd4_u10, pnt_log1pd4_u10, disp_log1pd4_u10, Sleef_log1pd4_u10avx, Sleef_log1pd4_u10fma4, Sleef_log1pd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_log1pf8_u10, pnt_log1pf8_u10, disp_log1pf8_u10, Sleef_log1pf8_u10avx, Sleef_log1pf8_u10fma4, Sleef_log1pf8_u10avx2)
DISPATCH_vf2_vf(__m256d, Sleef___m256d_2, Sleef_sincospid4_u05, pnt_sincospid4_u05, disp_sincospid4_u05, Sleef_sincospid4_u05avx, Sleef_sincospid4_u05fma4, Sleef_sincospid4_u05avx2)
DISPATCH_vf2_vf(__m256, Sleef___m256_2, Sleef_sincospif8_u05, pnt_sincospif8_u05, disp_sincospif8_u05, Sleef_sincospif8_u05avx, Sleef_sincospif8_u05fma4, Sleef_sincospif8_u05avx2)
DISPATCH_vf2_vf(__m256d, Sleef___m256d_2, Sleef_sincospid4_u35, pnt_sincospid4_u35, disp_sincospid4_u35, Sleef_sincospid4_u35avx, Sleef_sincospid4_u35fma4, Sleef_sincospid4_u35avx2)
DISPATCH_vf2_vf(__m256, Sleef___m256_2, Sleef_sincospif8_u35, pnt_sincospif8_u35, disp_sincospif8_u35, Sleef_sincospif8_u35avx, Sleef_sincospif8_u35fma4, Sleef_sincospif8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_sinpid4_u05, pnt_sinpid4_u05, disp_sinpid4_u05, Sleef_sinpid4_u05avx, Sleef_sinpid4_u05fma4, Sleef_sinpid4_u05avx2)
DISPATCH_vf_vf(__m256, Sleef_sinpif8_u05, pnt_sinpif8_u05, disp_sinpif8_u05, Sleef_sinpif8_u05avx, Sleef_sinpif8_u05fma4, Sleef_sinpif8_u05avx2)
DISPATCH_vf_vf(__m256d, Sleef_cospid4_u05, pnt_cospid4_u05, disp_cospid4_u05, Sleef_cospid4_u05avx, Sleef_cospid4_u05fma4, Sleef_cospid4_u05avx2)
DISPATCH_vf_vf(__m256, Sleef_cospif8_u05, pnt_cospif8_u05, disp_cospif8_u05, Sleef_cospif8_u05avx, Sleef_cospif8_u05fma4, Sleef_cospif8_u05avx2)
DISPATCH_vf_vf_vi(__m256d, __m128i, Sleef_ldexpd4, pnt_ldexpd4, disp_ldexpd4, Sleef_ldexpd4_avx, Sleef_ldexpd4_fma4, Sleef_ldexpd4_avx2)
DISPATCH_vi_vf(__m256d, __m128i, Sleef_ilogbd4, pnt_ilogbd4, disp_ilogbd4, Sleef_ilogbd4_avx, Sleef_ilogbd4_fma4, Sleef_ilogbd4_avx2)
DISPATCH_vf_vf_vf_vf(__m256d, Sleef_fmad4, pnt_fmad4, disp_fmad4, Sleef_fmad4_avx, Sleef_fmad4_fma4, Sleef_fmad4_avx2)
DISPATCH_vf_vf_vf_vf(__m256, Sleef_fmaf8, pnt_fmaf8, disp_fmaf8, Sleef_fmaf8_avx, Sleef_fmaf8_fma4, Sleef_fmaf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_sqrtd4, pnt_sqrtd4, disp_sqrtd4, Sleef_sqrtd4_avx, Sleef_sqrtd4_fma4, Sleef_sqrtd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_sqrtf8, pnt_sqrtf8, disp_sqrtf8, Sleef_sqrtf8_avx, Sleef_sqrtf8_fma4, Sleef_sqrtf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_sqrtd4_u05, pnt_sqrtd4_u05, disp_sqrtd4_u05, Sleef_sqrtd4_u05avx, Sleef_sqrtd4_u05fma4, Sleef_sqrtd4_u05avx2)
DISPATCH_vf_vf(__m256, Sleef_sqrtf8_u05, pnt_sqrtf8_u05, disp_sqrtf8_u05, Sleef_sqrtf8_u05avx, Sleef_sqrtf8_u05fma4, Sleef_sqrtf8_u05avx2)
DISPATCH_vf_vf(__m256d, Sleef_sqrtd4_u35, pnt_sqrtd4_u35, disp_sqrtd4_u35, Sleef_sqrtd4_u35avx, Sleef_sqrtd4_u35fma4, Sleef_sqrtd4_u35avx2)
DISPATCH_vf_vf(__m256, Sleef_sqrtf8_u35, pnt_sqrtf8_u35, disp_sqrtf8_u35, Sleef_sqrtf8_u35avx, Sleef_sqrtf8_u35fma4, Sleef_sqrtf8_u35avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_hypotd4_u05, pnt_hypotd4_u05, disp_hypotd4_u05, Sleef_hypotd4_u05avx, Sleef_hypotd4_u05fma4, Sleef_hypotd4_u05avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_hypotf8_u05, pnt_hypotf8_u05, disp_hypotf8_u05, Sleef_hypotf8_u05avx, Sleef_hypotf8_u05fma4, Sleef_hypotf8_u05avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_hypotd4_u35, pnt_hypotd4_u35, disp_hypotd4_u35, Sleef_hypotd4_u35avx, Sleef_hypotd4_u35fma4, Sleef_hypotd4_u35avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_hypotf8_u35, pnt_hypotf8_u35, disp_hypotf8_u35, Sleef_hypotf8_u35avx, Sleef_hypotf8_u35fma4, Sleef_hypotf8_u35avx2)
DISPATCH_vf_vf(__m256d, Sleef_fabsd4, pnt_fabsd4, disp_fabsd4, Sleef_fabsd4_avx, Sleef_fabsd4_fma4, Sleef_fabsd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_fabsf8, pnt_fabsf8, disp_fabsf8, Sleef_fabsf8_avx, Sleef_fabsf8_fma4, Sleef_fabsf8_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_copysignd4, pnt_copysignd4, disp_copysignd4, Sleef_copysignd4_avx, Sleef_copysignd4_fma4, Sleef_copysignd4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_copysignf8, pnt_copysignf8, disp_copysignf8, Sleef_copysignf8_avx, Sleef_copysignf8_fma4, Sleef_copysignf8_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_fmaxd4, pnt_fmaxd4, disp_fmaxd4, Sleef_fmaxd4_avx, Sleef_fmaxd4_fma4, Sleef_fmaxd4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_fmaxf8, pnt_fmaxf8, disp_fmaxf8, Sleef_fmaxf8_avx, Sleef_fmaxf8_fma4, Sleef_fmaxf8_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_fmind4, pnt_fmind4, disp_fmind4, Sleef_fmind4_avx, Sleef_fmind4_fma4, Sleef_fmind4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_fminf8, pnt_fminf8, disp_fminf8, Sleef_fminf8_avx, Sleef_fminf8_fma4, Sleef_fminf8_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_fdimd4, pnt_fdimd4, disp_fdimd4, Sleef_fdimd4_avx, Sleef_fdimd4_fma4, Sleef_fdimd4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_fdimf8, pnt_fdimf8, disp_fdimf8, Sleef_fdimf8_avx, Sleef_fdimf8_fma4, Sleef_fdimf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_truncd4, pnt_truncd4, disp_truncd4, Sleef_truncd4_avx, Sleef_truncd4_fma4, Sleef_truncd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_truncf8, pnt_truncf8, disp_truncf8, Sleef_truncf8_avx, Sleef_truncf8_fma4, Sleef_truncf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_floord4, pnt_floord4, disp_floord4, Sleef_floord4_avx, Sleef_floord4_fma4, Sleef_floord4_avx2)
DISPATCH_vf_vf(__m256, Sleef_floorf8, pnt_floorf8, disp_floorf8, Sleef_floorf8_avx, Sleef_floorf8_fma4, Sleef_floorf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_ceild4, pnt_ceild4, disp_ceild4, Sleef_ceild4_avx, Sleef_ceild4_fma4, Sleef_ceild4_avx2)
DISPATCH_vf_vf(__m256, Sleef_ceilf8, pnt_ceilf8, disp_ceilf8, Sleef_ceilf8_avx, Sleef_ceilf8_fma4, Sleef_ceilf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_roundd4, pnt_roundd4, disp_roundd4, Sleef_roundd4_avx, Sleef_roundd4_fma4, Sleef_roundd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_roundf8, pnt_roundf8, disp_roundf8, Sleef_roundf8_avx, Sleef_roundf8_fma4, Sleef_roundf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_rintd4, pnt_rintd4, disp_rintd4, Sleef_rintd4_avx, Sleef_rintd4_fma4, Sleef_rintd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_rintf8, pnt_rintf8, disp_rintf8, Sleef_rintf8_avx, Sleef_rintf8_fma4, Sleef_rintf8_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_nextafterd4, pnt_nextafterd4, disp_nextafterd4, Sleef_nextafterd4_avx, Sleef_nextafterd4_fma4, Sleef_nextafterd4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_nextafterf8, pnt_nextafterf8, disp_nextafterf8, Sleef_nextafterf8_avx, Sleef_nextafterf8_fma4, Sleef_nextafterf8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_frfrexpd4, pnt_frfrexpd4, disp_frfrexpd4, Sleef_frfrexpd4_avx, Sleef_frfrexpd4_fma4, Sleef_frfrexpd4_avx2)
DISPATCH_vf_vf(__m256, Sleef_frfrexpf8, pnt_frfrexpf8, disp_frfrexpf8, Sleef_frfrexpf8_avx, Sleef_frfrexpf8_fma4, Sleef_frfrexpf8_avx2)
DISPATCH_vi_vf(__m256d, __m128i, Sleef_expfrexpd4, pnt_expfrexpd4, disp_expfrexpd4, Sleef_expfrexpd4_avx, Sleef_expfrexpd4_fma4, Sleef_expfrexpd4_avx2)
DISPATCH_vf_vf_vf(__m256d, Sleef_fmodd4, pnt_fmodd4, disp_fmodd4, Sleef_fmodd4_avx, Sleef_fmodd4_fma4, Sleef_fmodd4_avx2)
DISPATCH_vf_vf_vf(__m256, Sleef_fmodf8, pnt_fmodf8, disp_fmodf8, Sleef_fmodf8_avx, Sleef_fmodf8_fma4, Sleef_fmodf8_avx2)
DISPATCH_vf2_vf(__m256d, Sleef___m256d_2, Sleef_modfd4, pnt_modfd4, disp_modfd4, Sleef_modfd4_avx, Sleef_modfd4_fma4, Sleef_modfd4_avx2)
DISPATCH_vf2_vf(__m256, Sleef___m256_2, Sleef_modff8, pnt_modff8, disp_modff8, Sleef_modff8_avx, Sleef_modff8_fma4, Sleef_modff8_avx2)
DISPATCH_vf_vf(__m256d, Sleef_lgammad4_u10, pnt_lgammad4_u10, disp_lgammad4_u10, Sleef_lgammad4_u10avx, Sleef_lgammad4_u10fma4, Sleef_lgammad4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_lgammaf8_u10, pnt_lgammaf8_u10, disp_lgammaf8_u10, Sleef_lgammaf8_u10avx, Sleef_lgammaf8_u10fma4, Sleef_lgammaf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_tgammad4_u10, pnt_tgammad4_u10, disp_tgammad4_u10, Sleef_tgammad4_u10avx, Sleef_tgammad4_u10fma4, Sleef_tgammad4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_tgammaf8_u10, pnt_tgammaf8_u10, disp_tgammaf8_u10, Sleef_tgammaf8_u10avx, Sleef_tgammaf8_u10fma4, Sleef_tgammaf8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_erfd4_u10, pnt_erfd4_u10, disp_erfd4_u10, Sleef_erfd4_u10avx, Sleef_erfd4_u10fma4, Sleef_erfd4_u10avx2)
DISPATCH_vf_vf(__m256, Sleef_erff8_u10, pnt_erff8_u10, disp_erff8_u10, Sleef_erff8_u10avx, Sleef_erff8_u10fma4, Sleef_erff8_u10avx2)
DISPATCH_vf_vf(__m256d, Sleef_erfcd4_u15, pnt_erfcd4_u15, disp_erfcd4_u15, Sleef_erfcd4_u15avx, Sleef_erfcd4_u15fma4, Sleef_erfcd4_u15avx2)
DISPATCH_vf_vf(__m256, Sleef_erfcf8_u15, pnt_erfcf8_u15, disp_erfcf8_u15, Sleef_erfcf8_u15avx, Sleef_erfcf8_u15fma4, Sleef_erfcf8_u15avx2)
DISPATCH_i_i(Sleef_getIntf8, pnt_getIntf8, disp_getIntf8, Sleef_getIntf8_avx, Sleef_getIntf8_fma4, Sleef_getIntf8_avx2)
DISPATCH_i_i(Sleef_getIntd4, pnt_getIntd4, disp_getIntd4, Sleef_getIntd4_avx, Sleef_getIntd4_fma4, Sleef_getIntd4_avx2)
DISPATCH_p_i(Sleef_getPtrf8, pnt_getPtrf8, disp_getPtrf8, Sleef_getPtrf8_avx, Sleef_getPtrf8_fma4, Sleef_getPtrf8_avx2)
DISPATCH_p_i(Sleef_getPtrd4, pnt_getPtrd4, disp_getPtrd4, Sleef_getPtrd4_avx, Sleef_getPtrd4_fma4, Sleef_getPtrd4_avx2)
