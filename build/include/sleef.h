//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SLEEF_H__
#define __SLEEF_H__

#include <stddef.h>
#include <stdint.h>

#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define CONST const
#else
#define CONST
#endif

#if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)
#ifdef IMPORT_IS_EXPORT
#define IMPORT __declspec(dllexport)
#else // #ifdef IMPORT_IS_EXPORT
#define IMPORT __declspec(dllimport)
#if (defined(_MSC_VER))
#pragma comment(lib,"sleef.lib")
#endif // #if (defined(_MSC_VER))
#endif // #ifdef IMPORT_IS_EXPORT
#else // #if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)
#define IMPORT
#endif // #if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)

#if (defined(__GNUC__) || defined(__CLANG__)) && (defined(__i386__) || defined(__x86_64__))
#include <x86intrin.h>
#endif

#if (defined(_MSC_VER))
#include <intrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

//

#ifndef SLEEF_FP_ILOGB0
#define SLEEF_FP_ILOGB0 ((int)-2147483648)
#endif

#ifndef SLEEF_FP_ILOGBNAN
#define SLEEF_FP_ILOGBNAN ((int)2147483647)
#endif

//

IMPORT void *Sleef_malloc(size_t z);
IMPORT void Sleef_free(void *ptr);
IMPORT uint64_t Sleef_currentTimeMicros();

#if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)
IMPORT void Sleef_x86CpuID(int32_t out[4], uint32_t eax, uint32_t ecx);
#endif

//

#ifndef Sleef_double2_DEFINED
#define Sleef_double2_DEFINED
typedef struct {
  double x, y;
} Sleef_double2;
#endif

#ifndef Sleef_float2_DEFINED
#define Sleef_float2_DEFINED
typedef struct {
  float x, y;
} Sleef_float2;
#endif

#ifndef Sleef_longdouble2_DEFINED
#define Sleef_longdouble2_DEFINED
typedef struct {
  long double x, y;
} Sleef_longdouble2;
#endif

#if defined(ENABLEFLOAT128) && !defined(Sleef_quad2_DEFINED)
#define Sleef_quad2_DEFINED
typedef __float128 Sleef_quad;
typedef struct {
  __float128 x, y;
} Sleef_quad2;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

IMPORT CONST double Sleef_sin_u35(double);
IMPORT CONST double Sleef_cos_u35(double);
IMPORT CONST Sleef_double2 Sleef_sincos_u35(double);
IMPORT CONST double Sleef_tan_u35(double);
IMPORT CONST double Sleef_asin_u35(double);
IMPORT CONST double Sleef_acos_u35(double);
IMPORT CONST double Sleef_atan_u35(double);
IMPORT CONST double Sleef_atan2_u35(double, double);
IMPORT CONST double Sleef_log_u35(double);
IMPORT CONST double Sleef_cbrt_u35(double);
IMPORT CONST double Sleef_sin_u10(double);
IMPORT CONST double Sleef_cos_u10(double);
IMPORT CONST Sleef_double2 Sleef_sincos_u10(double);
IMPORT CONST double Sleef_tan_u10(double);
IMPORT CONST double Sleef_asin_u10(double);
IMPORT CONST double Sleef_acos_u10(double);
IMPORT CONST double Sleef_atan_u10(double);
IMPORT CONST double Sleef_atan2_u10(double, double);
IMPORT CONST double Sleef_log_u10(double);
IMPORT CONST double Sleef_cbrt_u10(double);
IMPORT CONST double Sleef_exp_u10(double);
IMPORT CONST double Sleef_pow_u10(double, double);
IMPORT CONST double Sleef_sinh_u10(double);
IMPORT CONST double Sleef_cosh_u10(double);
IMPORT CONST double Sleef_tanh_u10(double);
IMPORT CONST double Sleef_asinh_u10(double);
IMPORT CONST double Sleef_acosh_u10(double);
IMPORT CONST double Sleef_atanh_u10(double);
IMPORT CONST double Sleef_exp2_u10(double);
IMPORT CONST double Sleef_exp10_u10(double);
IMPORT CONST double Sleef_expm1_u10(double);
IMPORT CONST double Sleef_log10_u10(double);
IMPORT CONST double Sleef_log2_u10(double);
IMPORT CONST double Sleef_log1p_u10(double);
IMPORT CONST Sleef_double2 Sleef_sincospi_u05(double);
IMPORT CONST Sleef_double2 Sleef_sincospi_u35(double);
IMPORT CONST double Sleef_sinpi_u05(double);
IMPORT CONST double Sleef_cospi_u05(double);
IMPORT CONST double Sleef_ldexp(double, int);
IMPORT CONST int Sleef_ilogb(double);
IMPORT CONST double Sleef_fma(double, double, double);
IMPORT CONST double Sleef_sqrt(double);
IMPORT CONST double Sleef_sqrt_u05(double);
IMPORT CONST double Sleef_sqrt_u35(double);

IMPORT CONST double Sleef_hypot_u05(double, double);
IMPORT CONST double Sleef_hypot_u35(double, double);

IMPORT CONST double Sleef_fabs(double);
IMPORT CONST double Sleef_copysign(double, double);
IMPORT CONST double Sleef_fmax(double, double);
IMPORT CONST double Sleef_fmin(double, double);
IMPORT CONST double Sleef_fdim(double, double);
IMPORT CONST double Sleef_trunc(double);
IMPORT CONST double Sleef_floor(double);
IMPORT CONST double Sleef_ceil(double);
IMPORT CONST double Sleef_round(double);
IMPORT CONST double Sleef_rint(double);
IMPORT CONST double Sleef_nextafter(double, double);
IMPORT CONST double Sleef_frfrexp(double);
IMPORT CONST int Sleef_expfrexp(double);
IMPORT CONST double Sleef_fmod(double, double);
IMPORT CONST Sleef_double2 Sleef_modf(double);

IMPORT CONST double Sleef_lgamma_u10(double);
IMPORT CONST double Sleef_tgamma_u10(double);
IMPORT CONST double Sleef_erf_u10(double);
IMPORT CONST double Sleef_erfc_u15(double);

IMPORT CONST float Sleef_sinf_u35(float);
IMPORT CONST float Sleef_cosf_u35(float);
IMPORT CONST Sleef_float2 Sleef_sincosf_u35(float);
IMPORT CONST float Sleef_tanf_u35(float);
IMPORT CONST float Sleef_asinf_u35(float);
IMPORT CONST float Sleef_acosf_u35(float);
IMPORT CONST float Sleef_atanf_u35(float);
IMPORT CONST float Sleef_atan2f_u35(float, float);
IMPORT CONST float Sleef_logf_u35(float);
IMPORT CONST float Sleef_cbrtf_u35(float);
IMPORT CONST float Sleef_sinf_u10(float);
IMPORT CONST float Sleef_cosf_u10(float);
IMPORT CONST Sleef_float2 Sleef_sincosf_u10(float);
IMPORT CONST float Sleef_tanf_u10(float);
IMPORT CONST float Sleef_asinf_u10(float);
IMPORT CONST float Sleef_acosf_u10(float);
IMPORT CONST float Sleef_atanf_u10(float);
IMPORT CONST float Sleef_atan2f_u10(float, float);
IMPORT CONST float Sleef_logf_u10(float);
IMPORT CONST float Sleef_cbrtf_u10(float);
IMPORT CONST float Sleef_expf_u10(float);
IMPORT CONST float Sleef_powf_u10(float, float);
IMPORT CONST float Sleef_sinhf_u10(float);
IMPORT CONST float Sleef_coshf_u10(float);
IMPORT CONST float Sleef_tanhf_u10(float);
IMPORT CONST float Sleef_asinhf_u10(float);
IMPORT CONST float Sleef_acoshf_u10(float);
IMPORT CONST float Sleef_atanhf_u10(float);
IMPORT CONST float Sleef_exp2f_u10(float);
IMPORT CONST float Sleef_exp10f_u10(float);
IMPORT CONST float Sleef_expm1f_u10(float);
IMPORT CONST float Sleef_log10f_u10(float);
IMPORT CONST float Sleef_log2f_u10(float);
IMPORT CONST float Sleef_log1pf_u10(float);
IMPORT CONST Sleef_float2 Sleef_sincospif_u05(float);
IMPORT CONST Sleef_float2 Sleef_sincospif_u35(float);
IMPORT CONST float Sleef_sinpif_u05(float d);
IMPORT CONST float Sleef_cospif_u05(float d);
IMPORT CONST float Sleef_ldexpf(float, int);
IMPORT CONST int Sleef_ilogbf(float);
IMPORT CONST float Sleef_fmaf(float, float, float);
IMPORT CONST float Sleef_sqrtf(float);
IMPORT CONST float Sleef_sqrtf_u05(float);
IMPORT CONST float Sleef_sqrtf_u35(float);

IMPORT CONST float Sleef_hypotf_u05(float, float);
IMPORT CONST float Sleef_hypotf_u35(float, float);

IMPORT CONST float Sleef_fabsf(float);
IMPORT CONST float Sleef_copysignf(float, float);
IMPORT CONST float Sleef_fmaxf(float, float);
IMPORT CONST float Sleef_fminf(float, float);
IMPORT CONST float Sleef_fdimf(float, float);
IMPORT CONST float Sleef_truncf(float);
IMPORT CONST float Sleef_floorf(float);
IMPORT CONST float Sleef_ceilf(float);
IMPORT CONST float Sleef_roundf(float);
IMPORT CONST float Sleef_rintf(float);
IMPORT CONST float Sleef_nextafterf(float, float);
IMPORT CONST float Sleef_frfrexpf(float);
IMPORT CONST int Sleef_expfrexpf(float);
IMPORT CONST float Sleef_fmodf(float, float);
IMPORT CONST Sleef_float2 Sleef_modff(float);

IMPORT CONST float Sleef_lgammaf_u10(float);
IMPORT CONST float Sleef_tgammaf_u10(float);
IMPORT CONST float Sleef_erff_u10(float);
IMPORT CONST float Sleef_erfcf_u15(float);

IMPORT CONST Sleef_longdouble2 Sleef_sincospil_u05(long double);
IMPORT CONST Sleef_longdouble2 Sleef_sincospil_u35(long double);

#if defined(Sleef_quad2_DEFINED)
IMPORT CONST Sleef_quad2 Sleef_sincospiq_u05(Sleef_quad);
IMPORT CONST Sleef_quad2 Sleef_sincospiq_u35(Sleef_quad);
#endif
#ifdef __SSE2__
#define STRUCT_KEYWORD___SSE2__ struct

#ifndef Sleef___m128d_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128d x, y;
} Sleef___m128d_2;
#define Sleef___m128d_2_DEFINED
#endif

IMPORT CONST __m128d Sleef_sind2_u35(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u35(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u35(__m128d);
IMPORT CONST __m128d Sleef_tand2_u35(__m128d);
IMPORT CONST __m128d Sleef_asind2_u35(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u35(__m128d);
IMPORT CONST __m128d Sleef_atand2_u35(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u35(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u35(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u35(__m128d);
IMPORT CONST __m128d Sleef_sind2_u10(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u10(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u10(__m128d);
IMPORT CONST __m128d Sleef_tand2_u10(__m128d);
IMPORT CONST __m128d Sleef_asind2_u10(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u10(__m128d);
IMPORT CONST __m128d Sleef_atand2_u10(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u10(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u10(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u10(__m128d);
IMPORT CONST __m128d Sleef_expd2_u10(__m128d);
IMPORT CONST __m128d Sleef_powd2_u10(__m128d, __m128d);
IMPORT CONST __m128d Sleef_sinhd2_u10(__m128d);
IMPORT CONST __m128d Sleef_coshd2_u10(__m128d);
IMPORT CONST __m128d Sleef_tanhd2_u10(__m128d);
IMPORT CONST __m128d Sleef_asinhd2_u10(__m128d);
IMPORT CONST __m128d Sleef_acoshd2_u10(__m128d);
IMPORT CONST __m128d Sleef_atanhd2_u10(__m128d);
IMPORT CONST __m128d Sleef_exp2d2_u10(__m128d);
IMPORT CONST __m128d Sleef_exp10d2_u10(__m128d);
IMPORT CONST __m128d Sleef_expm1d2_u10(__m128d);
IMPORT CONST __m128d Sleef_log10d2_u10(__m128d);
IMPORT CONST __m128d Sleef_log2d2_u10(__m128d);
IMPORT CONST __m128d Sleef_log1pd2_u10(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u05(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u35(__m128d);
IMPORT CONST __m128d Sleef_sinpid2_u05(__m128d);
IMPORT CONST __m128d Sleef_cospid2_u05(__m128d);
IMPORT CONST __m128d Sleef_ldexpd2(__m128d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd2(__m128d);
IMPORT CONST __m128d Sleef_fmad2(__m128d, __m128d, __m128d);
IMPORT CONST __m128d Sleef_sqrtd2(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u05(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u35(__m128d);
IMPORT CONST __m128d Sleef_hypotd2_u05(__m128d, __m128d);
IMPORT CONST __m128d Sleef_hypotd2_u35(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fabsd2(__m128d);
IMPORT CONST __m128d Sleef_copysignd2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmaxd2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmind2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fdimd2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_truncd2(__m128d);
IMPORT CONST __m128d Sleef_floord2(__m128d);
IMPORT CONST __m128d Sleef_ceild2(__m128d);
IMPORT CONST __m128d Sleef_roundd2(__m128d);
IMPORT CONST __m128d Sleef_rintd2(__m128d);
IMPORT CONST __m128d Sleef_nextafterd2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_frfrexpd2(__m128d);
IMPORT CONST __m128i Sleef_expfrexpd2(__m128d);
IMPORT CONST __m128d Sleef_fmodd2(__m128d, __m128d);
IMPORT CONST Sleef___m128d_2 Sleef_modfd2(__m128d);
IMPORT CONST __m128d Sleef_lgammad2_u10(__m128d);
IMPORT CONST __m128d Sleef_tgammad2_u10(__m128d);
IMPORT CONST __m128d Sleef_erfd2_u10(__m128d);
IMPORT CONST __m128d Sleef_erfcd2_u15(__m128d);
IMPORT CONST int Sleef_getIntd2(int);
IMPORT CONST void *Sleef_getPtrd2(int);

#ifndef Sleef___m128_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128 x, y;
} Sleef___m128_2;
#define Sleef___m128_2_DEFINED
#endif

IMPORT CONST __m128 Sleef_sinf4_u35(__m128);
IMPORT CONST __m128 Sleef_cosf4_u35(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u35(__m128);
IMPORT CONST __m128 Sleef_tanf4_u35(__m128);
IMPORT CONST __m128 Sleef_asinf4_u35(__m128);
IMPORT CONST __m128 Sleef_acosf4_u35(__m128);
IMPORT CONST __m128 Sleef_atanf4_u35(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u35(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u35(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u35(__m128);
IMPORT CONST __m128 Sleef_sinf4_u10(__m128);
IMPORT CONST __m128 Sleef_cosf4_u10(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u10(__m128);
IMPORT CONST __m128 Sleef_tanf4_u10(__m128);
IMPORT CONST __m128 Sleef_asinf4_u10(__m128);
IMPORT CONST __m128 Sleef_acosf4_u10(__m128);
IMPORT CONST __m128 Sleef_atanf4_u10(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u10(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u10(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u10(__m128);
IMPORT CONST __m128 Sleef_expf4_u10(__m128);
IMPORT CONST __m128 Sleef_powf4_u10(__m128, __m128);
IMPORT CONST __m128 Sleef_sinhf4_u10(__m128);
IMPORT CONST __m128 Sleef_coshf4_u10(__m128);
IMPORT CONST __m128 Sleef_tanhf4_u10(__m128);
IMPORT CONST __m128 Sleef_asinhf4_u10(__m128);
IMPORT CONST __m128 Sleef_acoshf4_u10(__m128);
IMPORT CONST __m128 Sleef_atanhf4_u10(__m128);
IMPORT CONST __m128 Sleef_exp2f4_u10(__m128);
IMPORT CONST __m128 Sleef_exp10f4_u10(__m128);
IMPORT CONST __m128 Sleef_expm1f4_u10(__m128);
IMPORT CONST __m128 Sleef_log10f4_u10(__m128);
IMPORT CONST __m128 Sleef_log2f4_u10(__m128);
IMPORT CONST __m128 Sleef_log1pf4_u10(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u05(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u35(__m128);
IMPORT CONST __m128 Sleef_sinpif4_u05(__m128);
IMPORT CONST __m128 Sleef_cospif4_u05(__m128);
IMPORT CONST __m128 Sleef_fmaf4(__m128, __m128, __m128);
IMPORT CONST __m128 Sleef_sqrtf4(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u05(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u35(__m128);
IMPORT CONST __m128 Sleef_hypotf4_u05(__m128, __m128);
IMPORT CONST __m128 Sleef_hypotf4_u35(__m128, __m128);
IMPORT CONST __m128 Sleef_fabsf4(__m128);
IMPORT CONST __m128 Sleef_copysignf4(__m128, __m128);
IMPORT CONST __m128 Sleef_fmaxf4(__m128, __m128);
IMPORT CONST __m128 Sleef_fminf4(__m128, __m128);
IMPORT CONST __m128 Sleef_fdimf4(__m128, __m128);
IMPORT CONST __m128 Sleef_truncf4(__m128);
IMPORT CONST __m128 Sleef_floorf4(__m128);
IMPORT CONST __m128 Sleef_ceilf4(__m128);
IMPORT CONST __m128 Sleef_roundf4(__m128);
IMPORT CONST __m128 Sleef_rintf4(__m128);
IMPORT CONST __m128 Sleef_nextafterf4(__m128, __m128);
IMPORT CONST __m128 Sleef_frfrexpf4(__m128);
IMPORT CONST __m128 Sleef_fmodf4(__m128, __m128);
IMPORT CONST Sleef___m128_2 Sleef_modff4(__m128);
IMPORT CONST __m128 Sleef_lgammaf4_u10(__m128);
IMPORT CONST __m128 Sleef_tgammaf4_u10(__m128);
IMPORT CONST __m128 Sleef_erff4_u10(__m128);
IMPORT CONST __m128 Sleef_erfcf4_u15(__m128);
IMPORT CONST int Sleef_getIntf4(int);
IMPORT CONST void *Sleef_getPtrf4(int);
#endif
#ifdef __SSE2__
#define STRUCT_KEYWORD___SSE2__ struct

#ifndef Sleef___m128d_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128d x, y;
} Sleef___m128d_2;
#define Sleef___m128d_2_DEFINED
#endif

IMPORT CONST __m128d Sleef_sind2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u35sse2(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_tand2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_asind2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_atand2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u35sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_sind2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u10sse2(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_tand2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_asind2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_atand2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u10sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_expd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_powd2_u10sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_sinhd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_coshd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_tanhd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_asinhd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_acoshd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_atanhd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_exp2d2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_exp10d2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_expm1d2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_log10d2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_log2d2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_log1pd2_u10sse2(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u05sse2(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_sinpid2_u05sse2(__m128d);
IMPORT CONST __m128d Sleef_cospid2_u05sse2(__m128d);
IMPORT CONST __m128d Sleef_ldexpd2_sse2(__m128d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_fmad2_sse2(__m128d, __m128d, __m128d);
IMPORT CONST __m128d Sleef_sqrtd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u05sse2(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u35sse2(__m128d);
IMPORT CONST __m128d Sleef_hypotd2_u05sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_hypotd2_u35sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fabsd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_copysignd2_sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmaxd2_sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmind2_sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fdimd2_sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_truncd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_floord2_sse2(__m128d);
IMPORT CONST __m128d Sleef_ceild2_sse2(__m128d);
IMPORT CONST __m128d Sleef_roundd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_rintd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_nextafterd2_sse2(__m128d, __m128d);
IMPORT CONST __m128d Sleef_frfrexpd2_sse2(__m128d);
IMPORT CONST __m128i Sleef_expfrexpd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_fmodd2_sse2(__m128d, __m128d);
IMPORT CONST Sleef___m128d_2 Sleef_modfd2_sse2(__m128d);
IMPORT CONST __m128d Sleef_lgammad2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_tgammad2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_erfd2_u10sse2(__m128d);
IMPORT CONST __m128d Sleef_erfcd2_u15sse2(__m128d);
IMPORT CONST int Sleef_getIntd2_sse2(int);
IMPORT CONST void *Sleef_getPtrd2_sse2(int);

#ifndef Sleef___m128_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128 x, y;
} Sleef___m128_2;
#define Sleef___m128_2_DEFINED
#endif

IMPORT CONST __m128 Sleef_sinf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_cosf4_u35sse2(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_tanf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_asinf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_acosf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_atanf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u35sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_sinf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_cosf4_u10sse2(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_tanf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_asinf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_acosf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_atanf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u10sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_expf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_powf4_u10sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_sinhf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_coshf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_tanhf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_asinhf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_acoshf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_atanhf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_exp2f4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_exp10f4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_expm1f4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_log10f4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_log2f4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_log1pf4_u10sse2(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u05sse2(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_sinpif4_u05sse2(__m128);
IMPORT CONST __m128 Sleef_cospif4_u05sse2(__m128);
IMPORT CONST __m128 Sleef_fmaf4_sse2(__m128, __m128, __m128);
IMPORT CONST __m128 Sleef_sqrtf4_sse2(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u05sse2(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u35sse2(__m128);
IMPORT CONST __m128 Sleef_hypotf4_u05sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_hypotf4_u35sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_fabsf4_sse2(__m128);
IMPORT CONST __m128 Sleef_copysignf4_sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_fmaxf4_sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_fminf4_sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_fdimf4_sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_truncf4_sse2(__m128);
IMPORT CONST __m128 Sleef_floorf4_sse2(__m128);
IMPORT CONST __m128 Sleef_ceilf4_sse2(__m128);
IMPORT CONST __m128 Sleef_roundf4_sse2(__m128);
IMPORT CONST __m128 Sleef_rintf4_sse2(__m128);
IMPORT CONST __m128 Sleef_nextafterf4_sse2(__m128, __m128);
IMPORT CONST __m128 Sleef_frfrexpf4_sse2(__m128);
IMPORT CONST __m128 Sleef_fmodf4_sse2(__m128, __m128);
IMPORT CONST Sleef___m128_2 Sleef_modff4_sse2(__m128);
IMPORT CONST __m128 Sleef_lgammaf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_tgammaf4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_erff4_u10sse2(__m128);
IMPORT CONST __m128 Sleef_erfcf4_u15sse2(__m128);
IMPORT CONST int Sleef_getIntf4_sse2(int);
IMPORT CONST void *Sleef_getPtrf4_sse2(int);
#endif
#ifdef __SSE2__
#define STRUCT_KEYWORD___SSE2__ struct

#ifndef Sleef___m128d_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128d x, y;
} Sleef___m128d_2;
#define Sleef___m128d_2_DEFINED
#endif

IMPORT CONST __m128d Sleef_sind2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u35sse4(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_tand2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_asind2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_atand2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u35sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_sind2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u10sse4(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_tand2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_asind2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_atand2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u10sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_expd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_powd2_u10sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_sinhd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_coshd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_tanhd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_asinhd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_acoshd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_atanhd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_exp2d2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_exp10d2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_expm1d2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_log10d2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_log2d2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_log1pd2_u10sse4(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u05sse4(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_sinpid2_u05sse4(__m128d);
IMPORT CONST __m128d Sleef_cospid2_u05sse4(__m128d);
IMPORT CONST __m128d Sleef_ldexpd2_sse4(__m128d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_fmad2_sse4(__m128d, __m128d, __m128d);
IMPORT CONST __m128d Sleef_sqrtd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u05sse4(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u35sse4(__m128d);
IMPORT CONST __m128d Sleef_hypotd2_u05sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_hypotd2_u35sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fabsd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_copysignd2_sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmaxd2_sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmind2_sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fdimd2_sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_truncd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_floord2_sse4(__m128d);
IMPORT CONST __m128d Sleef_ceild2_sse4(__m128d);
IMPORT CONST __m128d Sleef_roundd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_rintd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_nextafterd2_sse4(__m128d, __m128d);
IMPORT CONST __m128d Sleef_frfrexpd2_sse4(__m128d);
IMPORT CONST __m128i Sleef_expfrexpd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_fmodd2_sse4(__m128d, __m128d);
IMPORT CONST Sleef___m128d_2 Sleef_modfd2_sse4(__m128d);
IMPORT CONST __m128d Sleef_lgammad2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_tgammad2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_erfd2_u10sse4(__m128d);
IMPORT CONST __m128d Sleef_erfcd2_u15sse4(__m128d);
IMPORT CONST int Sleef_getIntd2_sse4(int);
IMPORT CONST void *Sleef_getPtrd2_sse4(int);

#ifndef Sleef___m128_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128 x, y;
} Sleef___m128_2;
#define Sleef___m128_2_DEFINED
#endif

IMPORT CONST __m128 Sleef_sinf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_cosf4_u35sse4(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_tanf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_asinf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_acosf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_atanf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u35sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_sinf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_cosf4_u10sse4(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_tanf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_asinf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_acosf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_atanf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u10sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_expf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_powf4_u10sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_sinhf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_coshf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_tanhf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_asinhf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_acoshf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_atanhf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_exp2f4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_exp10f4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_expm1f4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_log10f4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_log2f4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_log1pf4_u10sse4(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u05sse4(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_sinpif4_u05sse4(__m128);
IMPORT CONST __m128 Sleef_cospif4_u05sse4(__m128);
IMPORT CONST __m128 Sleef_fmaf4_sse4(__m128, __m128, __m128);
IMPORT CONST __m128 Sleef_sqrtf4_sse4(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u05sse4(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u35sse4(__m128);
IMPORT CONST __m128 Sleef_hypotf4_u05sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_hypotf4_u35sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_fabsf4_sse4(__m128);
IMPORT CONST __m128 Sleef_copysignf4_sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_fmaxf4_sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_fminf4_sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_fdimf4_sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_truncf4_sse4(__m128);
IMPORT CONST __m128 Sleef_floorf4_sse4(__m128);
IMPORT CONST __m128 Sleef_ceilf4_sse4(__m128);
IMPORT CONST __m128 Sleef_roundf4_sse4(__m128);
IMPORT CONST __m128 Sleef_rintf4_sse4(__m128);
IMPORT CONST __m128 Sleef_nextafterf4_sse4(__m128, __m128);
IMPORT CONST __m128 Sleef_frfrexpf4_sse4(__m128);
IMPORT CONST __m128 Sleef_fmodf4_sse4(__m128, __m128);
IMPORT CONST Sleef___m128_2 Sleef_modff4_sse4(__m128);
IMPORT CONST __m128 Sleef_lgammaf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_tgammaf4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_erff4_u10sse4(__m128);
IMPORT CONST __m128 Sleef_erfcf4_u15sse4(__m128);
IMPORT CONST int Sleef_getIntf4_sse4(int);
IMPORT CONST void *Sleef_getPtrf4_sse4(int);
#endif
#ifdef __AVX__
#define STRUCT_KEYWORD___AVX__ struct

#ifndef Sleef___m256d_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256d x, y;
} Sleef___m256d_2;
#define Sleef___m256d_2_DEFINED
#endif

IMPORT CONST __m256d Sleef_sind4_u35(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u35(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u35(__m256d);
IMPORT CONST __m256d Sleef_tand4_u35(__m256d);
IMPORT CONST __m256d Sleef_asind4_u35(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u35(__m256d);
IMPORT CONST __m256d Sleef_atand4_u35(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u35(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u35(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u35(__m256d);
IMPORT CONST __m256d Sleef_sind4_u10(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u10(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u10(__m256d);
IMPORT CONST __m256d Sleef_tand4_u10(__m256d);
IMPORT CONST __m256d Sleef_asind4_u10(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u10(__m256d);
IMPORT CONST __m256d Sleef_atand4_u10(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u10(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u10(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u10(__m256d);
IMPORT CONST __m256d Sleef_expd4_u10(__m256d);
IMPORT CONST __m256d Sleef_powd4_u10(__m256d, __m256d);
IMPORT CONST __m256d Sleef_sinhd4_u10(__m256d);
IMPORT CONST __m256d Sleef_coshd4_u10(__m256d);
IMPORT CONST __m256d Sleef_tanhd4_u10(__m256d);
IMPORT CONST __m256d Sleef_asinhd4_u10(__m256d);
IMPORT CONST __m256d Sleef_acoshd4_u10(__m256d);
IMPORT CONST __m256d Sleef_atanhd4_u10(__m256d);
IMPORT CONST __m256d Sleef_exp2d4_u10(__m256d);
IMPORT CONST __m256d Sleef_exp10d4_u10(__m256d);
IMPORT CONST __m256d Sleef_expm1d4_u10(__m256d);
IMPORT CONST __m256d Sleef_log10d4_u10(__m256d);
IMPORT CONST __m256d Sleef_log2d4_u10(__m256d);
IMPORT CONST __m256d Sleef_log1pd4_u10(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u05(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u35(__m256d);
IMPORT CONST __m256d Sleef_sinpid4_u05(__m256d);
IMPORT CONST __m256d Sleef_cospid4_u05(__m256d);
IMPORT CONST __m256d Sleef_ldexpd4(__m256d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd4(__m256d);
IMPORT CONST __m256d Sleef_fmad4(__m256d, __m256d, __m256d);
IMPORT CONST __m256d Sleef_sqrtd4(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u05(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u35(__m256d);
IMPORT CONST __m256d Sleef_hypotd4_u05(__m256d, __m256d);
IMPORT CONST __m256d Sleef_hypotd4_u35(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fabsd4(__m256d);
IMPORT CONST __m256d Sleef_copysignd4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmaxd4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmind4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fdimd4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_truncd4(__m256d);
IMPORT CONST __m256d Sleef_floord4(__m256d);
IMPORT CONST __m256d Sleef_ceild4(__m256d);
IMPORT CONST __m256d Sleef_roundd4(__m256d);
IMPORT CONST __m256d Sleef_rintd4(__m256d);
IMPORT CONST __m256d Sleef_nextafterd4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_frfrexpd4(__m256d);
IMPORT CONST __m128i Sleef_expfrexpd4(__m256d);
IMPORT CONST __m256d Sleef_fmodd4(__m256d, __m256d);
IMPORT CONST Sleef___m256d_2 Sleef_modfd4(__m256d);
IMPORT CONST __m256d Sleef_lgammad4_u10(__m256d);
IMPORT CONST __m256d Sleef_tgammad4_u10(__m256d);
IMPORT CONST __m256d Sleef_erfd4_u10(__m256d);
IMPORT CONST __m256d Sleef_erfcd4_u15(__m256d);
IMPORT CONST int Sleef_getIntd4(int);
IMPORT CONST void *Sleef_getPtrd4(int);

#ifndef Sleef___m256_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256 x, y;
} Sleef___m256_2;
#define Sleef___m256_2_DEFINED
#endif

IMPORT CONST __m256 Sleef_sinf8_u35(__m256);
IMPORT CONST __m256 Sleef_cosf8_u35(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u35(__m256);
IMPORT CONST __m256 Sleef_tanf8_u35(__m256);
IMPORT CONST __m256 Sleef_asinf8_u35(__m256);
IMPORT CONST __m256 Sleef_acosf8_u35(__m256);
IMPORT CONST __m256 Sleef_atanf8_u35(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u35(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u35(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u35(__m256);
IMPORT CONST __m256 Sleef_sinf8_u10(__m256);
IMPORT CONST __m256 Sleef_cosf8_u10(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u10(__m256);
IMPORT CONST __m256 Sleef_tanf8_u10(__m256);
IMPORT CONST __m256 Sleef_asinf8_u10(__m256);
IMPORT CONST __m256 Sleef_acosf8_u10(__m256);
IMPORT CONST __m256 Sleef_atanf8_u10(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u10(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u10(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u10(__m256);
IMPORT CONST __m256 Sleef_expf8_u10(__m256);
IMPORT CONST __m256 Sleef_powf8_u10(__m256, __m256);
IMPORT CONST __m256 Sleef_sinhf8_u10(__m256);
IMPORT CONST __m256 Sleef_coshf8_u10(__m256);
IMPORT CONST __m256 Sleef_tanhf8_u10(__m256);
IMPORT CONST __m256 Sleef_asinhf8_u10(__m256);
IMPORT CONST __m256 Sleef_acoshf8_u10(__m256);
IMPORT CONST __m256 Sleef_atanhf8_u10(__m256);
IMPORT CONST __m256 Sleef_exp2f8_u10(__m256);
IMPORT CONST __m256 Sleef_exp10f8_u10(__m256);
IMPORT CONST __m256 Sleef_expm1f8_u10(__m256);
IMPORT CONST __m256 Sleef_log10f8_u10(__m256);
IMPORT CONST __m256 Sleef_log2f8_u10(__m256);
IMPORT CONST __m256 Sleef_log1pf8_u10(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u05(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u35(__m256);
IMPORT CONST __m256 Sleef_sinpif8_u05(__m256);
IMPORT CONST __m256 Sleef_cospif8_u05(__m256);
IMPORT CONST __m256 Sleef_fmaf8(__m256, __m256, __m256);
IMPORT CONST __m256 Sleef_sqrtf8(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u05(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u35(__m256);
IMPORT CONST __m256 Sleef_hypotf8_u05(__m256, __m256);
IMPORT CONST __m256 Sleef_hypotf8_u35(__m256, __m256);
IMPORT CONST __m256 Sleef_fabsf8(__m256);
IMPORT CONST __m256 Sleef_copysignf8(__m256, __m256);
IMPORT CONST __m256 Sleef_fmaxf8(__m256, __m256);
IMPORT CONST __m256 Sleef_fminf8(__m256, __m256);
IMPORT CONST __m256 Sleef_fdimf8(__m256, __m256);
IMPORT CONST __m256 Sleef_truncf8(__m256);
IMPORT CONST __m256 Sleef_floorf8(__m256);
IMPORT CONST __m256 Sleef_ceilf8(__m256);
IMPORT CONST __m256 Sleef_roundf8(__m256);
IMPORT CONST __m256 Sleef_rintf8(__m256);
IMPORT CONST __m256 Sleef_nextafterf8(__m256, __m256);
IMPORT CONST __m256 Sleef_frfrexpf8(__m256);
IMPORT CONST __m256 Sleef_fmodf8(__m256, __m256);
IMPORT CONST Sleef___m256_2 Sleef_modff8(__m256);
IMPORT CONST __m256 Sleef_lgammaf8_u10(__m256);
IMPORT CONST __m256 Sleef_tgammaf8_u10(__m256);
IMPORT CONST __m256 Sleef_erff8_u10(__m256);
IMPORT CONST __m256 Sleef_erfcf8_u15(__m256);
IMPORT CONST int Sleef_getIntf8(int);
IMPORT CONST void *Sleef_getPtrf8(int);
#endif
#ifdef __AVX__
#define STRUCT_KEYWORD___AVX__ struct

#ifndef Sleef___m256d_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256d x, y;
} Sleef___m256d_2;
#define Sleef___m256d_2_DEFINED
#endif

IMPORT CONST __m256d Sleef_sind4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u35avx(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_tand4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_asind4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_atand4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u35avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_sind4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u10avx(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_tand4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_asind4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_atand4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u10avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_expd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_powd4_u10avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_sinhd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_coshd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_tanhd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_asinhd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_acoshd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_atanhd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_exp2d4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_exp10d4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_expm1d4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_log10d4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_log2d4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_log1pd4_u10avx(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u05avx(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_sinpid4_u05avx(__m256d);
IMPORT CONST __m256d Sleef_cospid4_u05avx(__m256d);
IMPORT CONST __m256d Sleef_ldexpd4_avx(__m256d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd4_avx(__m256d);
IMPORT CONST __m256d Sleef_fmad4_avx(__m256d, __m256d, __m256d);
IMPORT CONST __m256d Sleef_sqrtd4_avx(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u05avx(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u35avx(__m256d);
IMPORT CONST __m256d Sleef_hypotd4_u05avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_hypotd4_u35avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fabsd4_avx(__m256d);
IMPORT CONST __m256d Sleef_copysignd4_avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmaxd4_avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmind4_avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fdimd4_avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_truncd4_avx(__m256d);
IMPORT CONST __m256d Sleef_floord4_avx(__m256d);
IMPORT CONST __m256d Sleef_ceild4_avx(__m256d);
IMPORT CONST __m256d Sleef_roundd4_avx(__m256d);
IMPORT CONST __m256d Sleef_rintd4_avx(__m256d);
IMPORT CONST __m256d Sleef_nextafterd4_avx(__m256d, __m256d);
IMPORT CONST __m256d Sleef_frfrexpd4_avx(__m256d);
IMPORT CONST __m128i Sleef_expfrexpd4_avx(__m256d);
IMPORT CONST __m256d Sleef_fmodd4_avx(__m256d, __m256d);
IMPORT CONST Sleef___m256d_2 Sleef_modfd4_avx(__m256d);
IMPORT CONST __m256d Sleef_lgammad4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_tgammad4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_erfd4_u10avx(__m256d);
IMPORT CONST __m256d Sleef_erfcd4_u15avx(__m256d);
IMPORT CONST int Sleef_getIntd4_avx(int);
IMPORT CONST void *Sleef_getPtrd4_avx(int);

#ifndef Sleef___m256_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256 x, y;
} Sleef___m256_2;
#define Sleef___m256_2_DEFINED
#endif

IMPORT CONST __m256 Sleef_sinf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_cosf8_u35avx(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_tanf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_asinf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_acosf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_atanf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u35avx(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_sinf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_cosf8_u10avx(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_tanf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_asinf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_acosf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_atanf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u10avx(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_expf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_powf8_u10avx(__m256, __m256);
IMPORT CONST __m256 Sleef_sinhf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_coshf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_tanhf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_asinhf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_acoshf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_atanhf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_exp2f8_u10avx(__m256);
IMPORT CONST __m256 Sleef_exp10f8_u10avx(__m256);
IMPORT CONST __m256 Sleef_expm1f8_u10avx(__m256);
IMPORT CONST __m256 Sleef_log10f8_u10avx(__m256);
IMPORT CONST __m256 Sleef_log2f8_u10avx(__m256);
IMPORT CONST __m256 Sleef_log1pf8_u10avx(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u05avx(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u35avx(__m256);
IMPORT CONST __m256 Sleef_sinpif8_u05avx(__m256);
IMPORT CONST __m256 Sleef_cospif8_u05avx(__m256);
IMPORT CONST __m256 Sleef_fmaf8_avx(__m256, __m256, __m256);
IMPORT CONST __m256 Sleef_sqrtf8_avx(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u05avx(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u35avx(__m256);
IMPORT CONST __m256 Sleef_hypotf8_u05avx(__m256, __m256);
IMPORT CONST __m256 Sleef_hypotf8_u35avx(__m256, __m256);
IMPORT CONST __m256 Sleef_fabsf8_avx(__m256);
IMPORT CONST __m256 Sleef_copysignf8_avx(__m256, __m256);
IMPORT CONST __m256 Sleef_fmaxf8_avx(__m256, __m256);
IMPORT CONST __m256 Sleef_fminf8_avx(__m256, __m256);
IMPORT CONST __m256 Sleef_fdimf8_avx(__m256, __m256);
IMPORT CONST __m256 Sleef_truncf8_avx(__m256);
IMPORT CONST __m256 Sleef_floorf8_avx(__m256);
IMPORT CONST __m256 Sleef_ceilf8_avx(__m256);
IMPORT CONST __m256 Sleef_roundf8_avx(__m256);
IMPORT CONST __m256 Sleef_rintf8_avx(__m256);
IMPORT CONST __m256 Sleef_nextafterf8_avx(__m256, __m256);
IMPORT CONST __m256 Sleef_frfrexpf8_avx(__m256);
IMPORT CONST __m256 Sleef_fmodf8_avx(__m256, __m256);
IMPORT CONST Sleef___m256_2 Sleef_modff8_avx(__m256);
IMPORT CONST __m256 Sleef_lgammaf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_tgammaf8_u10avx(__m256);
IMPORT CONST __m256 Sleef_erff8_u10avx(__m256);
IMPORT CONST __m256 Sleef_erfcf8_u15avx(__m256);
IMPORT CONST int Sleef_getIntf8_avx(int);
IMPORT CONST void *Sleef_getPtrf8_avx(int);
#endif
#ifdef __AVX__
#define STRUCT_KEYWORD___AVX__ struct

#ifndef Sleef___m256d_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256d x, y;
} Sleef___m256d_2;
#define Sleef___m256d_2_DEFINED
#endif

IMPORT CONST __m256d Sleef_sind4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u35fma4(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_tand4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_asind4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_atand4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u35fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_sind4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u10fma4(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_tand4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_asind4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_atand4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u10fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_expd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_powd4_u10fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_sinhd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_coshd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_tanhd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_asinhd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_acoshd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_atanhd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_exp2d4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_exp10d4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_expm1d4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_log10d4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_log2d4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_log1pd4_u10fma4(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u05fma4(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_sinpid4_u05fma4(__m256d);
IMPORT CONST __m256d Sleef_cospid4_u05fma4(__m256d);
IMPORT CONST __m256d Sleef_ldexpd4_fma4(__m256d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_fmad4_fma4(__m256d, __m256d, __m256d);
IMPORT CONST __m256d Sleef_sqrtd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u05fma4(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u35fma4(__m256d);
IMPORT CONST __m256d Sleef_hypotd4_u05fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_hypotd4_u35fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fabsd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_copysignd4_fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmaxd4_fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmind4_fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fdimd4_fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_truncd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_floord4_fma4(__m256d);
IMPORT CONST __m256d Sleef_ceild4_fma4(__m256d);
IMPORT CONST __m256d Sleef_roundd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_rintd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_nextafterd4_fma4(__m256d, __m256d);
IMPORT CONST __m256d Sleef_frfrexpd4_fma4(__m256d);
IMPORT CONST __m128i Sleef_expfrexpd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_fmodd4_fma4(__m256d, __m256d);
IMPORT CONST Sleef___m256d_2 Sleef_modfd4_fma4(__m256d);
IMPORT CONST __m256d Sleef_lgammad4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_tgammad4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_erfd4_u10fma4(__m256d);
IMPORT CONST __m256d Sleef_erfcd4_u15fma4(__m256d);
IMPORT CONST int Sleef_getIntd4_fma4(int);
IMPORT CONST void *Sleef_getPtrd4_fma4(int);

#ifndef Sleef___m256_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256 x, y;
} Sleef___m256_2;
#define Sleef___m256_2_DEFINED
#endif

IMPORT CONST __m256 Sleef_sinf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_cosf8_u35fma4(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_tanf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_asinf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_acosf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_atanf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u35fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_sinf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_cosf8_u10fma4(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_tanf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_asinf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_acosf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_atanf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u10fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_expf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_powf8_u10fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_sinhf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_coshf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_tanhf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_asinhf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_acoshf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_atanhf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_exp2f8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_exp10f8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_expm1f8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_log10f8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_log2f8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_log1pf8_u10fma4(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u05fma4(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_sinpif8_u05fma4(__m256);
IMPORT CONST __m256 Sleef_cospif8_u05fma4(__m256);
IMPORT CONST __m256 Sleef_fmaf8_fma4(__m256, __m256, __m256);
IMPORT CONST __m256 Sleef_sqrtf8_fma4(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u05fma4(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u35fma4(__m256);
IMPORT CONST __m256 Sleef_hypotf8_u05fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_hypotf8_u35fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_fabsf8_fma4(__m256);
IMPORT CONST __m256 Sleef_copysignf8_fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_fmaxf8_fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_fminf8_fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_fdimf8_fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_truncf8_fma4(__m256);
IMPORT CONST __m256 Sleef_floorf8_fma4(__m256);
IMPORT CONST __m256 Sleef_ceilf8_fma4(__m256);
IMPORT CONST __m256 Sleef_roundf8_fma4(__m256);
IMPORT CONST __m256 Sleef_rintf8_fma4(__m256);
IMPORT CONST __m256 Sleef_nextafterf8_fma4(__m256, __m256);
IMPORT CONST __m256 Sleef_frfrexpf8_fma4(__m256);
IMPORT CONST __m256 Sleef_fmodf8_fma4(__m256, __m256);
IMPORT CONST Sleef___m256_2 Sleef_modff8_fma4(__m256);
IMPORT CONST __m256 Sleef_lgammaf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_tgammaf8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_erff8_u10fma4(__m256);
IMPORT CONST __m256 Sleef_erfcf8_u15fma4(__m256);
IMPORT CONST int Sleef_getIntf8_fma4(int);
IMPORT CONST void *Sleef_getPtrf8_fma4(int);
#endif
#ifdef __AVX__
#define STRUCT_KEYWORD___AVX__ struct

#ifndef Sleef___m256d_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256d x, y;
} Sleef___m256d_2;
#define Sleef___m256d_2_DEFINED
#endif

IMPORT CONST __m256d Sleef_sind4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u35avx2(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_tand4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_asind4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_atand4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u35avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_sind4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_cosd4_u10avx2(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincosd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_tand4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_asind4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_acosd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_atand4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_atan2d4_u10avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_logd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_cbrtd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_expd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_powd4_u10avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_sinhd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_coshd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_tanhd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_asinhd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_acoshd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_atanhd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_exp2d4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_exp10d4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_expm1d4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_log10d4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_log2d4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_log1pd4_u10avx2(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u05avx2(__m256d);
IMPORT CONST Sleef___m256d_2 Sleef_sincospid4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_sinpid4_u05avx2(__m256d);
IMPORT CONST __m256d Sleef_cospid4_u05avx2(__m256d);
IMPORT CONST __m256d Sleef_ldexpd4_avx2(__m256d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_fmad4_avx2(__m256d, __m256d, __m256d);
IMPORT CONST __m256d Sleef_sqrtd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u05avx2(__m256d);
IMPORT CONST __m256d Sleef_sqrtd4_u35avx2(__m256d);
IMPORT CONST __m256d Sleef_hypotd4_u05avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_hypotd4_u35avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fabsd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_copysignd4_avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmaxd4_avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fmind4_avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_fdimd4_avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_truncd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_floord4_avx2(__m256d);
IMPORT CONST __m256d Sleef_ceild4_avx2(__m256d);
IMPORT CONST __m256d Sleef_roundd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_rintd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_nextafterd4_avx2(__m256d, __m256d);
IMPORT CONST __m256d Sleef_frfrexpd4_avx2(__m256d);
IMPORT CONST __m128i Sleef_expfrexpd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_fmodd4_avx2(__m256d, __m256d);
IMPORT CONST Sleef___m256d_2 Sleef_modfd4_avx2(__m256d);
IMPORT CONST __m256d Sleef_lgammad4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_tgammad4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_erfd4_u10avx2(__m256d);
IMPORT CONST __m256d Sleef_erfcd4_u15avx2(__m256d);
IMPORT CONST int Sleef_getIntd4_avx2(int);
IMPORT CONST void *Sleef_getPtrd4_avx2(int);

#ifndef Sleef___m256_2_DEFINED
typedef STRUCT_KEYWORD___AVX__ {
  __m256 x, y;
} Sleef___m256_2;
#define Sleef___m256_2_DEFINED
#endif

IMPORT CONST __m256 Sleef_sinf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_cosf8_u35avx2(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_tanf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_asinf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_acosf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_atanf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u35avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_sinf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_cosf8_u10avx2(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincosf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_tanf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_asinf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_acosf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_atanf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_atan2f8_u10avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_logf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_cbrtf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_expf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_powf8_u10avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_sinhf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_coshf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_tanhf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_asinhf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_acoshf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_atanhf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_exp2f8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_exp10f8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_expm1f8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_log10f8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_log2f8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_log1pf8_u10avx2(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u05avx2(__m256);
IMPORT CONST Sleef___m256_2 Sleef_sincospif8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_sinpif8_u05avx2(__m256);
IMPORT CONST __m256 Sleef_cospif8_u05avx2(__m256);
IMPORT CONST __m256 Sleef_fmaf8_avx2(__m256, __m256, __m256);
IMPORT CONST __m256 Sleef_sqrtf8_avx2(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u05avx2(__m256);
IMPORT CONST __m256 Sleef_sqrtf8_u35avx2(__m256);
IMPORT CONST __m256 Sleef_hypotf8_u05avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_hypotf8_u35avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_fabsf8_avx2(__m256);
IMPORT CONST __m256 Sleef_copysignf8_avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_fmaxf8_avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_fminf8_avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_fdimf8_avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_truncf8_avx2(__m256);
IMPORT CONST __m256 Sleef_floorf8_avx2(__m256);
IMPORT CONST __m256 Sleef_ceilf8_avx2(__m256);
IMPORT CONST __m256 Sleef_roundf8_avx2(__m256);
IMPORT CONST __m256 Sleef_rintf8_avx2(__m256);
IMPORT CONST __m256 Sleef_nextafterf8_avx2(__m256, __m256);
IMPORT CONST __m256 Sleef_frfrexpf8_avx2(__m256);
IMPORT CONST __m256 Sleef_fmodf8_avx2(__m256, __m256);
IMPORT CONST Sleef___m256_2 Sleef_modff8_avx2(__m256);
IMPORT CONST __m256 Sleef_lgammaf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_tgammaf8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_erff8_u10avx2(__m256);
IMPORT CONST __m256 Sleef_erfcf8_u15avx2(__m256);
IMPORT CONST int Sleef_getIntf8_avx2(int);
IMPORT CONST void *Sleef_getPtrf8_avx2(int);
#endif
#ifdef __SSE2__
#define STRUCT_KEYWORD___SSE2__ struct

#ifndef Sleef___m128d_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128d x, y;
} Sleef___m128d_2;
#define Sleef___m128d_2_DEFINED
#endif

IMPORT CONST __m128d Sleef_sind2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u35avx2128(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_tand2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_asind2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_atand2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u35avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_sind2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_cosd2_u10avx2128(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincosd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_tand2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_asind2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_acosd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_atand2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_atan2d2_u10avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_logd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_cbrtd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_expd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_powd2_u10avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_sinhd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_coshd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_tanhd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_asinhd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_acoshd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_atanhd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_exp2d2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_exp10d2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_expm1d2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_log10d2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_log2d2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_log1pd2_u10avx2128(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u05avx2128(__m128d);
IMPORT CONST Sleef___m128d_2 Sleef_sincospid2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_sinpid2_u05avx2128(__m128d);
IMPORT CONST __m128d Sleef_cospid2_u05avx2128(__m128d);
IMPORT CONST __m128d Sleef_ldexpd2_avx2128(__m128d, __m128i);
IMPORT CONST __m128i Sleef_ilogbd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_fmad2_avx2128(__m128d, __m128d, __m128d);
IMPORT CONST __m128d Sleef_sqrtd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u05avx2128(__m128d);
IMPORT CONST __m128d Sleef_sqrtd2_u35avx2128(__m128d);
IMPORT CONST __m128d Sleef_hypotd2_u05avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_hypotd2_u35avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fabsd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_copysignd2_avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmaxd2_avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fmind2_avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_fdimd2_avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_truncd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_floord2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_ceild2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_roundd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_rintd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_nextafterd2_avx2128(__m128d, __m128d);
IMPORT CONST __m128d Sleef_frfrexpd2_avx2128(__m128d);
IMPORT CONST __m128i Sleef_expfrexpd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_fmodd2_avx2128(__m128d, __m128d);
IMPORT CONST Sleef___m128d_2 Sleef_modfd2_avx2128(__m128d);
IMPORT CONST __m128d Sleef_lgammad2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_tgammad2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_erfd2_u10avx2128(__m128d);
IMPORT CONST __m128d Sleef_erfcd2_u15avx2128(__m128d);
IMPORT CONST int Sleef_getIntd2_avx2128(int);
IMPORT CONST void *Sleef_getPtrd2_avx2128(int);

#ifndef Sleef___m128_2_DEFINED
typedef STRUCT_KEYWORD___SSE2__ {
  __m128 x, y;
} Sleef___m128_2;
#define Sleef___m128_2_DEFINED
#endif

IMPORT CONST __m128 Sleef_sinf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_cosf4_u35avx2128(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_tanf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_asinf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_acosf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_atanf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u35avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_sinf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_cosf4_u10avx2128(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincosf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_tanf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_asinf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_acosf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_atanf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_atan2f4_u10avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_logf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_cbrtf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_expf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_powf4_u10avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_sinhf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_coshf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_tanhf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_asinhf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_acoshf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_atanhf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_exp2f4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_exp10f4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_expm1f4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_log10f4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_log2f4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_log1pf4_u10avx2128(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u05avx2128(__m128);
IMPORT CONST Sleef___m128_2 Sleef_sincospif4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_sinpif4_u05avx2128(__m128);
IMPORT CONST __m128 Sleef_cospif4_u05avx2128(__m128);
IMPORT CONST __m128 Sleef_fmaf4_avx2128(__m128, __m128, __m128);
IMPORT CONST __m128 Sleef_sqrtf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u05avx2128(__m128);
IMPORT CONST __m128 Sleef_sqrtf4_u35avx2128(__m128);
IMPORT CONST __m128 Sleef_hypotf4_u05avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_hypotf4_u35avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_fabsf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_copysignf4_avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_fmaxf4_avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_fminf4_avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_fdimf4_avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_truncf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_floorf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_ceilf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_roundf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_rintf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_nextafterf4_avx2128(__m128, __m128);
IMPORT CONST __m128 Sleef_frfrexpf4_avx2128(__m128);
IMPORT CONST __m128 Sleef_fmodf4_avx2128(__m128, __m128);
IMPORT CONST Sleef___m128_2 Sleef_modff4_avx2128(__m128);
IMPORT CONST __m128 Sleef_lgammaf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_tgammaf4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_erff4_u10avx2128(__m128);
IMPORT CONST __m128 Sleef_erfcf4_u15avx2128(__m128);
IMPORT CONST int Sleef_getIntf4_avx2128(int);
IMPORT CONST void *Sleef_getPtrf4_avx2128(int);
#endif
#ifdef __AVX512F__
#define STRUCT_KEYWORD___AVX512F__ struct

#ifndef Sleef___m512d_2_DEFINED
typedef STRUCT_KEYWORD___AVX512F__ {
  __m512d x, y;
} Sleef___m512d_2;
#define Sleef___m512d_2_DEFINED
#endif

IMPORT CONST __m512d Sleef_sind8_u35(__m512d);
IMPORT CONST __m512d Sleef_cosd8_u35(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincosd8_u35(__m512d);
IMPORT CONST __m512d Sleef_tand8_u35(__m512d);
IMPORT CONST __m512d Sleef_asind8_u35(__m512d);
IMPORT CONST __m512d Sleef_acosd8_u35(__m512d);
IMPORT CONST __m512d Sleef_atand8_u35(__m512d);
IMPORT CONST __m512d Sleef_atan2d8_u35(__m512d, __m512d);
IMPORT CONST __m512d Sleef_logd8_u35(__m512d);
IMPORT CONST __m512d Sleef_cbrtd8_u35(__m512d);
IMPORT CONST __m512d Sleef_sind8_u10(__m512d);
IMPORT CONST __m512d Sleef_cosd8_u10(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincosd8_u10(__m512d);
IMPORT CONST __m512d Sleef_tand8_u10(__m512d);
IMPORT CONST __m512d Sleef_asind8_u10(__m512d);
IMPORT CONST __m512d Sleef_acosd8_u10(__m512d);
IMPORT CONST __m512d Sleef_atand8_u10(__m512d);
IMPORT CONST __m512d Sleef_atan2d8_u10(__m512d, __m512d);
IMPORT CONST __m512d Sleef_logd8_u10(__m512d);
IMPORT CONST __m512d Sleef_cbrtd8_u10(__m512d);
IMPORT CONST __m512d Sleef_expd8_u10(__m512d);
IMPORT CONST __m512d Sleef_powd8_u10(__m512d, __m512d);
IMPORT CONST __m512d Sleef_sinhd8_u10(__m512d);
IMPORT CONST __m512d Sleef_coshd8_u10(__m512d);
IMPORT CONST __m512d Sleef_tanhd8_u10(__m512d);
IMPORT CONST __m512d Sleef_asinhd8_u10(__m512d);
IMPORT CONST __m512d Sleef_acoshd8_u10(__m512d);
IMPORT CONST __m512d Sleef_atanhd8_u10(__m512d);
IMPORT CONST __m512d Sleef_exp2d8_u10(__m512d);
IMPORT CONST __m512d Sleef_exp10d8_u10(__m512d);
IMPORT CONST __m512d Sleef_expm1d8_u10(__m512d);
IMPORT CONST __m512d Sleef_log10d8_u10(__m512d);
IMPORT CONST __m512d Sleef_log2d8_u10(__m512d);
IMPORT CONST __m512d Sleef_log1pd8_u10(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincospid8_u05(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincospid8_u35(__m512d);
IMPORT CONST __m512d Sleef_sinpid8_u05(__m512d);
IMPORT CONST __m512d Sleef_cospid8_u05(__m512d);
IMPORT CONST __m512d Sleef_ldexpd8(__m512d, __m256i);
IMPORT CONST __m256i Sleef_ilogbd8(__m512d);
IMPORT CONST __m512d Sleef_fmad8(__m512d, __m512d, __m512d);
IMPORT CONST __m512d Sleef_sqrtd8(__m512d);
IMPORT CONST __m512d Sleef_sqrtd8_u05(__m512d);
IMPORT CONST __m512d Sleef_sqrtd8_u35(__m512d);
IMPORT CONST __m512d Sleef_hypotd8_u05(__m512d, __m512d);
IMPORT CONST __m512d Sleef_hypotd8_u35(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fabsd8(__m512d);
IMPORT CONST __m512d Sleef_copysignd8(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fmaxd8(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fmind8(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fdimd8(__m512d, __m512d);
IMPORT CONST __m512d Sleef_truncd8(__m512d);
IMPORT CONST __m512d Sleef_floord8(__m512d);
IMPORT CONST __m512d Sleef_ceild8(__m512d);
IMPORT CONST __m512d Sleef_roundd8(__m512d);
IMPORT CONST __m512d Sleef_rintd8(__m512d);
IMPORT CONST __m512d Sleef_nextafterd8(__m512d, __m512d);
IMPORT CONST __m512d Sleef_frfrexpd8(__m512d);
IMPORT CONST __m256i Sleef_expfrexpd8(__m512d);
IMPORT CONST __m512d Sleef_fmodd8(__m512d, __m512d);
IMPORT CONST Sleef___m512d_2 Sleef_modfd8(__m512d);
IMPORT CONST __m512d Sleef_lgammad8_u10(__m512d);
IMPORT CONST __m512d Sleef_tgammad8_u10(__m512d);
IMPORT CONST __m512d Sleef_erfd8_u10(__m512d);
IMPORT CONST __m512d Sleef_erfcd8_u15(__m512d);
IMPORT CONST int Sleef_getIntd8(int);
IMPORT CONST void *Sleef_getPtrd8(int);

#ifndef Sleef___m512_2_DEFINED
typedef STRUCT_KEYWORD___AVX512F__ {
  __m512 x, y;
} Sleef___m512_2;
#define Sleef___m512_2_DEFINED
#endif

IMPORT CONST __m512 Sleef_sinf16_u35(__m512);
IMPORT CONST __m512 Sleef_cosf16_u35(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincosf16_u35(__m512);
IMPORT CONST __m512 Sleef_tanf16_u35(__m512);
IMPORT CONST __m512 Sleef_asinf16_u35(__m512);
IMPORT CONST __m512 Sleef_acosf16_u35(__m512);
IMPORT CONST __m512 Sleef_atanf16_u35(__m512);
IMPORT CONST __m512 Sleef_atan2f16_u35(__m512, __m512);
IMPORT CONST __m512 Sleef_logf16_u35(__m512);
IMPORT CONST __m512 Sleef_cbrtf16_u35(__m512);
IMPORT CONST __m512 Sleef_sinf16_u10(__m512);
IMPORT CONST __m512 Sleef_cosf16_u10(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincosf16_u10(__m512);
IMPORT CONST __m512 Sleef_tanf16_u10(__m512);
IMPORT CONST __m512 Sleef_asinf16_u10(__m512);
IMPORT CONST __m512 Sleef_acosf16_u10(__m512);
IMPORT CONST __m512 Sleef_atanf16_u10(__m512);
IMPORT CONST __m512 Sleef_atan2f16_u10(__m512, __m512);
IMPORT CONST __m512 Sleef_logf16_u10(__m512);
IMPORT CONST __m512 Sleef_cbrtf16_u10(__m512);
IMPORT CONST __m512 Sleef_expf16_u10(__m512);
IMPORT CONST __m512 Sleef_powf16_u10(__m512, __m512);
IMPORT CONST __m512 Sleef_sinhf16_u10(__m512);
IMPORT CONST __m512 Sleef_coshf16_u10(__m512);
IMPORT CONST __m512 Sleef_tanhf16_u10(__m512);
IMPORT CONST __m512 Sleef_asinhf16_u10(__m512);
IMPORT CONST __m512 Sleef_acoshf16_u10(__m512);
IMPORT CONST __m512 Sleef_atanhf16_u10(__m512);
IMPORT CONST __m512 Sleef_exp2f16_u10(__m512);
IMPORT CONST __m512 Sleef_exp10f16_u10(__m512);
IMPORT CONST __m512 Sleef_expm1f16_u10(__m512);
IMPORT CONST __m512 Sleef_log10f16_u10(__m512);
IMPORT CONST __m512 Sleef_log2f16_u10(__m512);
IMPORT CONST __m512 Sleef_log1pf16_u10(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincospif16_u05(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincospif16_u35(__m512);
IMPORT CONST __m512 Sleef_sinpif16_u05(__m512);
IMPORT CONST __m512 Sleef_cospif16_u05(__m512);
IMPORT CONST __m512 Sleef_fmaf16(__m512, __m512, __m512);
IMPORT CONST __m512 Sleef_sqrtf16(__m512);
IMPORT CONST __m512 Sleef_sqrtf16_u05(__m512);
IMPORT CONST __m512 Sleef_sqrtf16_u35(__m512);
IMPORT CONST __m512 Sleef_hypotf16_u05(__m512, __m512);
IMPORT CONST __m512 Sleef_hypotf16_u35(__m512, __m512);
IMPORT CONST __m512 Sleef_fabsf16(__m512);
IMPORT CONST __m512 Sleef_copysignf16(__m512, __m512);
IMPORT CONST __m512 Sleef_fmaxf16(__m512, __m512);
IMPORT CONST __m512 Sleef_fminf16(__m512, __m512);
IMPORT CONST __m512 Sleef_fdimf16(__m512, __m512);
IMPORT CONST __m512 Sleef_truncf16(__m512);
IMPORT CONST __m512 Sleef_floorf16(__m512);
IMPORT CONST __m512 Sleef_ceilf16(__m512);
IMPORT CONST __m512 Sleef_roundf16(__m512);
IMPORT CONST __m512 Sleef_rintf16(__m512);
IMPORT CONST __m512 Sleef_nextafterf16(__m512, __m512);
IMPORT CONST __m512 Sleef_frfrexpf16(__m512);
IMPORT CONST __m512 Sleef_fmodf16(__m512, __m512);
IMPORT CONST Sleef___m512_2 Sleef_modff16(__m512);
IMPORT CONST __m512 Sleef_lgammaf16_u10(__m512);
IMPORT CONST __m512 Sleef_tgammaf16_u10(__m512);
IMPORT CONST __m512 Sleef_erff16_u10(__m512);
IMPORT CONST __m512 Sleef_erfcf16_u15(__m512);
IMPORT CONST int Sleef_getIntf16(int);
IMPORT CONST void *Sleef_getPtrf16(int);
#endif
#ifdef __AVX512F__
#define STRUCT_KEYWORD___AVX512F__ struct

#ifndef Sleef___m512d_2_DEFINED
typedef STRUCT_KEYWORD___AVX512F__ {
  __m512d x, y;
} Sleef___m512d_2;
#define Sleef___m512d_2_DEFINED
#endif

IMPORT CONST __m512d Sleef_sind8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_cosd8_u35avx512f(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincosd8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_tand8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_asind8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_acosd8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_atand8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_atan2d8_u35avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_logd8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_cbrtd8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_sind8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_cosd8_u10avx512f(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincosd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_tand8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_asind8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_acosd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_atand8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_atan2d8_u10avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_logd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_cbrtd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_expd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_powd8_u10avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_sinhd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_coshd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_tanhd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_asinhd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_acoshd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_atanhd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_exp2d8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_exp10d8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_expm1d8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_log10d8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_log2d8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_log1pd8_u10avx512f(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincospid8_u05avx512f(__m512d);
IMPORT CONST Sleef___m512d_2 Sleef_sincospid8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_sinpid8_u05avx512f(__m512d);
IMPORT CONST __m512d Sleef_cospid8_u05avx512f(__m512d);
IMPORT CONST __m512d Sleef_ldexpd8_avx512f(__m512d, __m256i);
IMPORT CONST __m256i Sleef_ilogbd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_fmad8_avx512f(__m512d, __m512d, __m512d);
IMPORT CONST __m512d Sleef_sqrtd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_sqrtd8_u05avx512f(__m512d);
IMPORT CONST __m512d Sleef_sqrtd8_u35avx512f(__m512d);
IMPORT CONST __m512d Sleef_hypotd8_u05avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_hypotd8_u35avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fabsd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_copysignd8_avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fmaxd8_avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fmind8_avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_fdimd8_avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_truncd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_floord8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_ceild8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_roundd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_rintd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_nextafterd8_avx512f(__m512d, __m512d);
IMPORT CONST __m512d Sleef_frfrexpd8_avx512f(__m512d);
IMPORT CONST __m256i Sleef_expfrexpd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_fmodd8_avx512f(__m512d, __m512d);
IMPORT CONST Sleef___m512d_2 Sleef_modfd8_avx512f(__m512d);
IMPORT CONST __m512d Sleef_lgammad8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_tgammad8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_erfd8_u10avx512f(__m512d);
IMPORT CONST __m512d Sleef_erfcd8_u15avx512f(__m512d);
IMPORT CONST int Sleef_getIntd8_avx512f(int);
IMPORT CONST void *Sleef_getPtrd8_avx512f(int);

#ifndef Sleef___m512_2_DEFINED
typedef STRUCT_KEYWORD___AVX512F__ {
  __m512 x, y;
} Sleef___m512_2;
#define Sleef___m512_2_DEFINED
#endif

IMPORT CONST __m512 Sleef_sinf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_cosf16_u35avx512f(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincosf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_tanf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_asinf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_acosf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_atanf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_atan2f16_u35avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_logf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_cbrtf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_sinf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_cosf16_u10avx512f(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincosf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_tanf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_asinf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_acosf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_atanf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_atan2f16_u10avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_logf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_cbrtf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_expf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_powf16_u10avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_sinhf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_coshf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_tanhf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_asinhf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_acoshf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_atanhf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_exp2f16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_exp10f16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_expm1f16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_log10f16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_log2f16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_log1pf16_u10avx512f(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincospif16_u05avx512f(__m512);
IMPORT CONST Sleef___m512_2 Sleef_sincospif16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_sinpif16_u05avx512f(__m512);
IMPORT CONST __m512 Sleef_cospif16_u05avx512f(__m512);
IMPORT CONST __m512 Sleef_fmaf16_avx512f(__m512, __m512, __m512);
IMPORT CONST __m512 Sleef_sqrtf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_sqrtf16_u05avx512f(__m512);
IMPORT CONST __m512 Sleef_sqrtf16_u35avx512f(__m512);
IMPORT CONST __m512 Sleef_hypotf16_u05avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_hypotf16_u35avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_fabsf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_copysignf16_avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_fmaxf16_avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_fminf16_avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_fdimf16_avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_truncf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_floorf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_ceilf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_roundf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_rintf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_nextafterf16_avx512f(__m512, __m512);
IMPORT CONST __m512 Sleef_frfrexpf16_avx512f(__m512);
IMPORT CONST __m512 Sleef_fmodf16_avx512f(__m512, __m512);
IMPORT CONST Sleef___m512_2 Sleef_modff16_avx512f(__m512);
IMPORT CONST __m512 Sleef_lgammaf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_tgammaf16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_erff16_u10avx512f(__m512);
IMPORT CONST __m512 Sleef_erfcf16_u15avx512f(__m512);
IMPORT CONST int Sleef_getIntf16_avx512f(int);
IMPORT CONST void *Sleef_getPtrf16_avx512f(int);
#endif
#ifdef __cplusplus
}
#endif

#undef IMPORT
#endif // #ifndef __SLEEF_H__
