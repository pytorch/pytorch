#include <c10/util/Half.h>
#include <torch/csrc/jit/tensorexpr/intrinsic_symbols.h>
#include <cmath>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

__m128 Sleef_acosf4_u10(__m128);
__m128 Sleef_asinf4_u10(__m128);
__m128 Sleef_atanf4_u10(__m128);
__m128 Sleef_cosf4_u10(__m128);
__m128 Sleef_sinf4_u10(__m128);
__m128 Sleef_tanf4_u10(__m128);
__m128 Sleef_coshf4_u10(__m128);
__m128 Sleef_sinhf4_u10(__m128);
__m128 Sleef_tanhf4_u10(__m128);
__m128 Sleef_erff4_u10(__m128);
__m128 Sleef_erfcf4_u15(__m128);
__m128 Sleef_expf4_u10(__m128);
__m128 Sleef_expm1f4_u10(__m128);
__m128 Sleef_logf4_u10(__m128);
__m128 Sleef_log2f4_u10(__m128);
__m128 Sleef_log10f4_u10(__m128);
__m128 Sleef_log1pf4_u10(__m128);
__m128 Sleef_sqrtf4_u05(__m128);
__m128 Sleef_fabsf4(__m128);
__m128 Sleef_floorf4(__m128);
__m128 Sleef_ceilf4(__m128);
__m128 Sleef_truncf4(__m128);
__m128 Sleef_roundf4(__m128);
__m128 Sleef_lgammaf4_u10(__m128);
__m128 Sleef_atan2f4_u10(__m128, __m128);
__m128 Sleef_powf4_u10(__m128, __m128);
__m128 Sleef_fmodf4(__m128, __m128);

#if !defined(_MSC_VER)
__m256 Sleef_acosf8_u10(__m256);
__m256 Sleef_asinf8_u10(__m256);
__m256 Sleef_atanf8_u10(__m256);
__m256 Sleef_cosf8_u10(__m256);
__m256 Sleef_sinf8_u10(__m256);
__m256 Sleef_tanf8_u10(__m256);
__m256 Sleef_coshf8_u10(__m256);
__m256 Sleef_sinhf8_u10(__m256);
__m256 Sleef_tanhf8_u10(__m256);
__m256 Sleef_erff8_u10(__m256);
__m256 Sleef_erfcf8_u15(__m256);
__m256 Sleef_expf8_u10(__m256);
__m256 Sleef_expm1f8_u10(__m256);
__m256 Sleef_logf8_u10(__m256);
__m256 Sleef_log2f8_u10(__m256);
__m256 Sleef_log10f8_u10(__m256);
__m256 Sleef_log1pf8_u10(__m256);
__m256 Sleef_sqrtf8_u05(__m256);
__m256 Sleef_fabsf8(__m256);
__m256 Sleef_floorf8(__m256);
__m256 Sleef_ceilf8(__m256);
__m256 Sleef_truncf8(__m256);
__m256 Sleef_roundf8(__m256);
__m256 Sleef_lgammaf8_u10(__m256);
__m256 Sleef_atan2f8_u10(__m256, __m256);
__m256 Sleef_powf8_u10(__m256, __m256);
__m256 Sleef_fmodf8(__m256, __m256);
#endif

__m128d Sleef_acosd2_u10(__m128d);
__m128d Sleef_asind2_u10(__m128d);
__m128d Sleef_atand2_u10(__m128d);
__m128d Sleef_cosd2_u10(__m128d);
__m128d Sleef_sind2_u10(__m128d);
__m128d Sleef_tand2_u10(__m128d);
__m128d Sleef_coshd2_u10(__m128d);
__m128d Sleef_sinhd2_u10(__m128d);
__m128d Sleef_tanhd2_u10(__m128d);
__m128d Sleef_erfd2_u10(__m128d);
__m128d Sleef_erfcd2_u15(__m128d);
__m128d Sleef_expd2_u10(__m128d);
__m128d Sleef_expm1d2_u10(__m128d);
__m128d Sleef_logd2_u10(__m128d);
__m128d Sleef_log2d2_u10(__m128d);
__m128d Sleef_log10d2_u10(__m128d);
__m128d Sleef_log1pd2_u10(__m128d);
__m128d Sleef_sqrtd2_u05(__m128d);
__m128d Sleef_fabsd2(__m128d);
__m128d Sleef_floord2(__m128d);
__m128d Sleef_ceild2(__m128d);
__m128d Sleef_truncd2(__m128d);
__m128d Sleef_roundd2(__m128d);
__m128d Sleef_lgammad2_u10(__m128d);
__m128d Sleef_atan2d2_u10(__m128d, __m128d);
__m128d Sleef_powd2_u10(__m128d, __m128d);
__m128d Sleef_fmodd2(__m128d, __m128d);

#if !defined(_MSC_VER)
__m256d Sleef_acosd4_u10(__m256d);
__m256d Sleef_asind4_u10(__m256d);
__m256d Sleef_atand4_u10(__m256d);
__m256d Sleef_cosd4_u10(__m256d);
__m256d Sleef_sind4_u10(__m256d);
__m256d Sleef_tand4_u10(__m256d);
__m256d Sleef_coshd4_u10(__m256d);
__m256d Sleef_sinhd4_u10(__m256d);
__m256d Sleef_tanhd4_u10(__m256d);
__m256d Sleef_erfd4_u10(__m256d);
__m256d Sleef_erfcd4_u15(__m256d);
__m256d Sleef_expd4_u10(__m256d);
__m256d Sleef_expm1d4_u10(__m256d);
__m256d Sleef_logd4_u10(__m256d);
__m256d Sleef_log2d4_u10(__m256d);
__m256d Sleef_log10d4_u10(__m256d);
__m256d Sleef_log1pd4_u10(__m256d);
__m256d Sleef_sqrtd4_u05(__m256d);
__m256d Sleef_fabsd4(__m256d);
__m256d Sleef_floord4(__m256d);
__m256d Sleef_ceild4(__m256d);
__m256d Sleef_truncd4(__m256d);
__m256d Sleef_roundd4(__m256d);
__m256d Sleef_lgammad4_u10(__m256d);
__m256d Sleef_atan2d4_u10(__m256d, __m256d);
__m256d Sleef_powd4_u10(__m256d, __m256d);
__m256d Sleef_fmodd4(__m256d, __m256d);
#endif

#ifdef __cplusplus
}
#endif

namespace torch {
namespace jit {
namespace tensorexpr {

c10::ArrayRef<SymbolAddress> getIntrinsicSymbols() {
  static SymbolAddress symbolAddresses[] = {
    {"log10f", &log10f},
    {"log1pf", &log1pf},
    {"logf", &logf},
    {"log2f", &log2f},
    {"expf", &expf},
    {"erff", &erff},
    {"cosf", &cosf},
    {"sinf", &sinf},
    {"tanf", &tanf},
    {"acosf", &acosf},
    {"asinf", &asinf},
    {"atanf", &atanf},
    {"coshf", &coshf},
    {"sinhf", &sinhf},
    {"tanhf", &tanhf},
    {"sqrtf", &sqrtf},
    {"fabsf", &fabsf},
    {"floorf", &floorf},
    {"ceilf", &ceilf},
    {"roundf", &roundf},
    {"truncf", &truncf},
    {"atan2f", &atan2f},
    {"fmodf", &fmodf},
    {"remainderf", &remainderf},

    // float -> half & half -> float conversions
    {"__gnu_h2f_ieee", &c10::detail::fp16_ieee_to_fp32_value},
    {"__gnu_f2h_ieee", &c10::detail::fp16_ieee_from_fp32_value},

    // FP32 Sleef functions -- SSE
    {"Sleef_acosf4", &Sleef_acosf4_u10},
    {"Sleef_asinf4", &Sleef_asinf4_u10},
    {"Sleef_atanf4", &Sleef_atanf4_u10},
    {"Sleef_cosf4", &Sleef_cosf4_u10},
    {"Sleef_sinf4", &Sleef_sinf4_u10},
    {"Sleef_tanf4", &Sleef_tanf4_u10},
    {"Sleef_coshf4", &Sleef_coshf4_u10},
    {"Sleef_sinhf4", &Sleef_sinhf4_u10},
    {"Sleef_tanhf4", &Sleef_tanhf4_u10},
    {"Sleef_erff4", &Sleef_erff4_u10},
    {"Sleef_erfcf4", &Sleef_erfcf4_u15},
    {"Sleef_expf4", &Sleef_expf4_u10},
    {"Sleef_expm1f4", &Sleef_expm1f4_u10},
    {"Sleef_logf4", &Sleef_logf4_u10},
    {"Sleef_log2f4", &Sleef_log2f4_u10},
    {"Sleef_log10f4", &Sleef_log10f4_u10},
    {"Sleef_log1pf4", &Sleef_log1pf4_u10},
    {"Sleef_sqrtf4", &Sleef_sqrtf4_u05},
    {"Sleef_fabsf4", &Sleef_fabsf4},
    {"Sleef_floorf4", &Sleef_floorf4},
    {"Sleef_ceilf4", &Sleef_ceilf4},
    {"Sleef_truncf4", &Sleef_truncf4},
    {"Sleef_roundf4", &Sleef_roundf4},
    {"Sleef_lgammaf4", &Sleef_lgammaf4_u10},
    {"Sleef_atan2f4", &Sleef_atan2f4_u10},
    {"Sleef_powf4", &Sleef_powf4_u10},
    {"Sleef_fmodf4", &Sleef_fmodf4},

  // FP32 Sleef functions -- AVX2
#if !defined(_MSC_VER)
    {"Sleef_acosf8", &Sleef_acosf8_u10},
    {"Sleef_asinf8", &Sleef_asinf8_u10},
    {"Sleef_atanf8", &Sleef_atanf8_u10},
    {"Sleef_cosf8", &Sleef_cosf8_u10},
    {"Sleef_sinf8", &Sleef_sinf8_u10},
    {"Sleef_tanf8", &Sleef_tanf8_u10},
    {"Sleef_coshf8", &Sleef_coshf8_u10},
    {"Sleef_sinhf8", &Sleef_sinhf8_u10},
    {"Sleef_tanhf8", &Sleef_tanhf8_u10},
    {"Sleef_erff8", &Sleef_erff8_u10},
    {"Sleef_erfcf8", &Sleef_erfcf8_u15},
    {"Sleef_expf8", &Sleef_expf8_u10},
    {"Sleef_expm1f8", &Sleef_expm1f8_u10},
    {"Sleef_logf8", &Sleef_logf8_u10},
    {"Sleef_log2f8", &Sleef_log2f8_u10},
    {"Sleef_log10f8", &Sleef_log10f8_u10},
    {"Sleef_log1pf8", &Sleef_log1pf8_u10},
    {"Sleef_sqrtf8", &Sleef_sqrtf8_u05},
    {"Sleef_fabsf8", &Sleef_fabsf8},
    {"Sleef_floorf8", &Sleef_floorf8},
    {"Sleef_ceilf8", &Sleef_ceilf8},
    {"Sleef_truncf8", &Sleef_truncf8},
    {"Sleef_roundf8", &Sleef_roundf8},
    {"Sleef_lgammaf8", &Sleef_lgammaf8_u10},
    {"Sleef_atan2f8", &Sleef_atan2f8_u10},
    {"Sleef_powf8", &Sleef_powf8_u10},
    {"Sleef_fmodf8", &Sleef_fmodf8},
#endif

    // FP64 Sleef functions -- SSE
    {"Sleef_acosd2", &Sleef_acosd2_u10},
    {"Sleef_asind2", &Sleef_asind2_u10},
    {"Sleef_atand2", &Sleef_atand2_u10},
    {"Sleef_cosd2", &Sleef_cosd2_u10},
    {"Sleef_sind2", &Sleef_sind2_u10},
    {"Sleef_tand2", &Sleef_tand2_u10},
    {"Sleef_coshd2", &Sleef_coshd2_u10},
    {"Sleef_sinhd2", &Sleef_sinhd2_u10},
    {"Sleef_tanhd2", &Sleef_tanhd2_u10},
    {"Sleef_erfd2", &Sleef_erfd2_u10},
    {"Sleef_erfcd2", &Sleef_erfcd2_u15},
    {"Sleef_expd2", &Sleef_expd2_u10},
    {"Sleef_expm1d2", &Sleef_expm1d2_u10},
    {"Sleef_logd2", &Sleef_logd2_u10},
    {"Sleef_log2d2", &Sleef_log2d2_u10},
    {"Sleef_log10d2", &Sleef_log10d2_u10},
    {"Sleef_log1pd2", &Sleef_log1pd2_u10},
    {"Sleef_sqrtd2", &Sleef_sqrtd2_u05},
    {"Sleef_fabsd2", &Sleef_fabsd2},
    {"Sleef_floord2", &Sleef_floord2},
    {"Sleef_ceild2", &Sleef_ceild2},
    {"Sleef_truncd2", &Sleef_truncd2},
    {"Sleef_roundd2", &Sleef_roundd2},
    {"Sleef_lgammad2", &Sleef_lgammad2_u10},
    {"Sleef_atan2d2", &Sleef_atan2d2_u10},
    {"Sleef_powd2", &Sleef_powd2_u10},
    {"Sleef_fmodd2", &Sleef_fmodd2},

  // FP64 Sleef functions -- AVX2
#if !defined(_MSC_VER)
    {"Sleef_acosd4", &Sleef_acosd4_u10},
    {"Sleef_asind4", &Sleef_asind4_u10},
    {"Sleef_atand4", &Sleef_atand4_u10},
    {"Sleef_cosd4", &Sleef_cosd4_u10},
    {"Sleef_sind4", &Sleef_sind4_u10},
    {"Sleef_tand4", &Sleef_tand4_u10},
    {"Sleef_coshd4", &Sleef_coshd4_u10},
    {"Sleef_sinhd4", &Sleef_sinhd4_u10},
    {"Sleef_tanhd4", &Sleef_tanhd4_u10},
    {"Sleef_erfd4", &Sleef_erfd4_u10},
    {"Sleef_erfcd4", &Sleef_erfcd4_u15},
    {"Sleef_expd4", &Sleef_expd4_u10},
    {"Sleef_expm1d4", &Sleef_expm1d4_u10},
    {"Sleef_logd4", &Sleef_logd4_u10},
    {"Sleef_log2d4", &Sleef_log2d4_u10},
    {"Sleef_log10d4", &Sleef_log10d4_u10},
    {"Sleef_log1pd4", &Sleef_log1pd4_u10},
    {"Sleef_sqrtd4", &Sleef_sqrtd4_u05},
    {"Sleef_fabsd4", &Sleef_fabsd4},
    {"Sleef_floord4", &Sleef_floord4},
    {"Sleef_ceild4", &Sleef_ceild4},
    {"Sleef_truncd4", &Sleef_truncd4},
    {"Sleef_roundd4", &Sleef_roundd4},
    {"Sleef_lgammad4", &Sleef_lgammad4_u10},
    {"Sleef_atan2d4", &Sleef_atan2d4_u10},
    {"Sleef_powd4", &Sleef_powd4_u10},
    {"Sleef_fmodd4", &Sleef_fmodd4},
#endif
  };
  return c10::ArrayRef<SymbolAddress>(symbolAddresses);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
