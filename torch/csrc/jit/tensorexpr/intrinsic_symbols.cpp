#ifdef TORCH_ENABLE_LLVM
#include <c10/util/Half.h>
#include <torch/csrc/jit/tensorexpr/intrinsic_symbols.h>
#include <cmath>

#if !defined(_MSC_VER) && defined(__x86_64__)
#include <x86intrin.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // !defined(_MSC_VER) && defined(__x86_64__)

namespace torch::jit::tensorexpr {

c10::ArrayRef<SymbolAddress> getIntrinsicSymbols() {
  static SymbolAddress symbolAddresses[] = {
    {"log10f", reinterpret_cast<void*>(&log10f)},
    {"log1pf", reinterpret_cast<void*>(&log1pf)},
    {"logf", reinterpret_cast<void*>(&logf)},
    {"log2f", reinterpret_cast<void*>(&log2f)},
    {"expf", reinterpret_cast<void*>(&expf)},
    {"erff", reinterpret_cast<void*>(&erff)},
    {"cosf", reinterpret_cast<void*>(&cosf)},
    {"sinf", reinterpret_cast<void*>(&sinf)},
    {"tanf", reinterpret_cast<void*>(&tanf)},
    {"acosf", reinterpret_cast<void*>(&acosf)},
    {"asinf", reinterpret_cast<void*>(&asinf)},
    {"atanf", reinterpret_cast<void*>(&atanf)},
    {"coshf", reinterpret_cast<void*>(&coshf)},
    {"sinhf", reinterpret_cast<void*>(&sinhf)},
    {"tanhf", reinterpret_cast<void*>(&tanhf)},
    {"sqrtf", reinterpret_cast<void*>(&sqrtf)},
    {"fabsf", reinterpret_cast<void*>(&fabsf)},
    {"floorf", reinterpret_cast<void*>(&floorf)},
    {"ceilf", reinterpret_cast<void*>(&ceilf)},
    {"roundf", reinterpret_cast<void*>(&roundf)},
    {"truncf", reinterpret_cast<void*>(&truncf)},
    {"atan2f", reinterpret_cast<void*>(&atan2f)},
    {"fmodf", reinterpret_cast<void*>(&fmodf)},
    {"remainderf", reinterpret_cast<void*>(&remainderf)},

    // float -> half & half -> float conversions
    {"__gnu_h2f_ieee",
     reinterpret_cast<void*>(&c10::detail::fp16_ieee_to_fp32_value)},
    {"__gnu_f2h_ieee",
     reinterpret_cast<void*>(&c10::detail::fp16_ieee_from_fp32_value)},

#if !defined(_MSC_VER) && defined(__x86_64__)
    // FP32 Sleef functions -- SSE
    {"Sleef_acosf4", reinterpret_cast<void*>(&Sleef_acosf4_u10)},
    {"Sleef_asinf4", reinterpret_cast<void*>(&Sleef_asinf4_u10)},
    {"Sleef_atanf4", reinterpret_cast<void*>(&Sleef_atanf4_u10)},
    {"Sleef_cosf4", reinterpret_cast<void*>(&Sleef_cosf4_u10)},
    {"Sleef_sinf4", reinterpret_cast<void*>(&Sleef_sinf4_u10)},
    {"Sleef_tanf4", reinterpret_cast<void*>(&Sleef_tanf4_u10)},
    {"Sleef_coshf4", reinterpret_cast<void*>(&Sleef_coshf4_u10)},
    {"Sleef_sinhf4", reinterpret_cast<void*>(&Sleef_sinhf4_u10)},
    {"Sleef_tanhf4", reinterpret_cast<void*>(&Sleef_tanhf4_u10)},
    {"Sleef_erff4", reinterpret_cast<void*>(&Sleef_erff4_u10)},
    {"Sleef_erfcf4", reinterpret_cast<void*>(&Sleef_erfcf4_u15)},
    {"Sleef_expf4", reinterpret_cast<void*>(&Sleef_expf4_u10)},
    {"Sleef_expm1f4", reinterpret_cast<void*>(&Sleef_expm1f4_u10)},
    {"Sleef_logf4", reinterpret_cast<void*>(&Sleef_logf4_u10)},
    {"Sleef_log2f4", reinterpret_cast<void*>(&Sleef_log2f4_u10)},
    {"Sleef_log10f4", reinterpret_cast<void*>(&Sleef_log10f4_u10)},
    {"Sleef_log1pf4", reinterpret_cast<void*>(&Sleef_log1pf4_u10)},
    {"Sleef_sqrtf4", reinterpret_cast<void*>(&Sleef_sqrtf4_u05)},
    {"Sleef_fabsf4", reinterpret_cast<void*>(&Sleef_fabsf4)},
    {"Sleef_floorf4", reinterpret_cast<void*>(&Sleef_floorf4)},
    {"Sleef_ceilf4", reinterpret_cast<void*>(&Sleef_ceilf4)},
    {"Sleef_truncf4", reinterpret_cast<void*>(&Sleef_truncf4)},
    {"Sleef_roundf4", reinterpret_cast<void*>(&Sleef_roundf4)},
    {"Sleef_lgammaf4", reinterpret_cast<void*>(&Sleef_lgammaf4_u10)},
    {"Sleef_atan2f4", reinterpret_cast<void*>(&Sleef_atan2f4_u10)},
    {"Sleef_powf4", reinterpret_cast<void*>(&Sleef_powf4_u10)},
    {"Sleef_fmodf4", reinterpret_cast<void*>(&Sleef_fmodf4)},

    // FP32 Sleef functions -- AVX2
    {"Sleef_acosf8", reinterpret_cast<void*>(&Sleef_acosf8_u10)},
    {"Sleef_asinf8", reinterpret_cast<void*>(&Sleef_asinf8_u10)},
    {"Sleef_atanf8", reinterpret_cast<void*>(&Sleef_atanf8_u10)},
    {"Sleef_cosf8", reinterpret_cast<void*>(&Sleef_cosf8_u10)},
    {"Sleef_sinf8", reinterpret_cast<void*>(&Sleef_sinf8_u10)},
    {"Sleef_tanf8", reinterpret_cast<void*>(&Sleef_tanf8_u10)},
    {"Sleef_coshf8", reinterpret_cast<void*>(&Sleef_coshf8_u10)},
    {"Sleef_sinhf8", reinterpret_cast<void*>(&Sleef_sinhf8_u10)},
    {"Sleef_tanhf8", reinterpret_cast<void*>(&Sleef_tanhf8_u10)},
    {"Sleef_erff8", reinterpret_cast<void*>(&Sleef_erff8_u10)},
    {"Sleef_erfcf8", reinterpret_cast<void*>(&Sleef_erfcf8_u15)},
    {"Sleef_expf8", reinterpret_cast<void*>(&Sleef_expf8_u10)},
    {"Sleef_expm1f8", reinterpret_cast<void*>(&Sleef_expm1f8_u10)},
    {"Sleef_logf8", reinterpret_cast<void*>(&Sleef_logf8_u10)},
    {"Sleef_log2f8", reinterpret_cast<void*>(&Sleef_log2f8_u10)},
    {"Sleef_log10f8", reinterpret_cast<void*>(&Sleef_log10f8_u10)},
    {"Sleef_log1pf8", reinterpret_cast<void*>(&Sleef_log1pf8_u10)},
    {"Sleef_sqrtf8", reinterpret_cast<void*>(&Sleef_sqrtf8_u05)},
    {"Sleef_fabsf8", reinterpret_cast<void*>(&Sleef_fabsf8)},
    {"Sleef_floorf8", reinterpret_cast<void*>(&Sleef_floorf8)},
    {"Sleef_ceilf8", reinterpret_cast<void*>(&Sleef_ceilf8)},
    {"Sleef_truncf8", reinterpret_cast<void*>(&Sleef_truncf8)},
    {"Sleef_roundf8", reinterpret_cast<void*>(&Sleef_roundf8)},
    {"Sleef_lgammaf8", reinterpret_cast<void*>(&Sleef_lgammaf8_u10)},
    {"Sleef_atan2f8", reinterpret_cast<void*>(&Sleef_atan2f8_u10)},
    {"Sleef_powf8", reinterpret_cast<void*>(&Sleef_powf8_u10)},
    {"Sleef_fmodf8", reinterpret_cast<void*>(&Sleef_fmodf8)},

    // FP64 Sleef functions -- SSE
    {"Sleef_acosd2", reinterpret_cast<void*>(&Sleef_acosd2_u10)},
    {"Sleef_asind2", reinterpret_cast<void*>(&Sleef_asind2_u10)},
    {"Sleef_atand2", reinterpret_cast<void*>(&Sleef_atand2_u10)},
    {"Sleef_cosd2", reinterpret_cast<void*>(&Sleef_cosd2_u10)},
    {"Sleef_sind2", reinterpret_cast<void*>(&Sleef_sind2_u10)},
    {"Sleef_tand2", reinterpret_cast<void*>(&Sleef_tand2_u10)},
    {"Sleef_coshd2", reinterpret_cast<void*>(&Sleef_coshd2_u10)},
    {"Sleef_sinhd2", reinterpret_cast<void*>(&Sleef_sinhd2_u10)},
    {"Sleef_tanhd2", reinterpret_cast<void*>(&Sleef_tanhd2_u10)},
    {"Sleef_erfd2", reinterpret_cast<void*>(&Sleef_erfd2_u10)},
    {"Sleef_erfcd2", reinterpret_cast<void*>(&Sleef_erfcd2_u15)},
    {"Sleef_expd2", reinterpret_cast<void*>(&Sleef_expd2_u10)},
    {"Sleef_expm1d2", reinterpret_cast<void*>(&Sleef_expm1d2_u10)},
    {"Sleef_logd2", reinterpret_cast<void*>(&Sleef_logd2_u10)},
    {"Sleef_log2d2", reinterpret_cast<void*>(&Sleef_log2d2_u10)},
    {"Sleef_log10d2", reinterpret_cast<void*>(&Sleef_log10d2_u10)},
    {"Sleef_log1pd2", reinterpret_cast<void*>(&Sleef_log1pd2_u10)},
    {"Sleef_sqrtd2", reinterpret_cast<void*>(&Sleef_sqrtd2_u05)},
    {"Sleef_fabsd2", reinterpret_cast<void*>(&Sleef_fabsd2)},
    {"Sleef_floord2", reinterpret_cast<void*>(&Sleef_floord2)},
    {"Sleef_ceild2", reinterpret_cast<void*>(&Sleef_ceild2)},
    {"Sleef_truncd2", reinterpret_cast<void*>(&Sleef_truncd2)},
    {"Sleef_roundd2", reinterpret_cast<void*>(&Sleef_roundd2)},
    {"Sleef_lgammad2", reinterpret_cast<void*>(&Sleef_lgammad2_u10)},
    {"Sleef_atan2d2", reinterpret_cast<void*>(&Sleef_atan2d2_u10)},
    {"Sleef_powd2", reinterpret_cast<void*>(&Sleef_powd2_u10)},
    {"Sleef_fmodd2", reinterpret_cast<void*>(&Sleef_fmodd2)},

    // FP64 Sleef functions -- AVX2
    {"Sleef_acosd4", reinterpret_cast<void*>(&Sleef_acosd4_u10)},
    {"Sleef_asind4", reinterpret_cast<void*>(&Sleef_asind4_u10)},
    {"Sleef_atand4", reinterpret_cast<void*>(&Sleef_atand4_u10)},
    {"Sleef_cosd4", reinterpret_cast<void*>(&Sleef_cosd4_u10)},
    {"Sleef_sind4", reinterpret_cast<void*>(&Sleef_sind4_u10)},
    {"Sleef_tand4", reinterpret_cast<void*>(&Sleef_tand4_u10)},
    {"Sleef_coshd4", reinterpret_cast<void*>(&Sleef_coshd4_u10)},
    {"Sleef_sinhd4", reinterpret_cast<void*>(&Sleef_sinhd4_u10)},
    {"Sleef_tanhd4", reinterpret_cast<void*>(&Sleef_tanhd4_u10)},
    {"Sleef_erfd4", reinterpret_cast<void*>(&Sleef_erfd4_u10)},
    {"Sleef_erfcd4", reinterpret_cast<void*>(&Sleef_erfcd4_u15)},
    {"Sleef_expd4", reinterpret_cast<void*>(&Sleef_expd4_u10)},
    {"Sleef_expm1d4", reinterpret_cast<void*>(&Sleef_expm1d4_u10)},
    {"Sleef_logd4", reinterpret_cast<void*>(&Sleef_logd4_u10)},
    {"Sleef_log2d4", reinterpret_cast<void*>(&Sleef_log2d4_u10)},
    {"Sleef_log10d4", reinterpret_cast<void*>(&Sleef_log10d4_u10)},
    {"Sleef_log1pd4", reinterpret_cast<void*>(&Sleef_log1pd4_u10)},
    {"Sleef_sqrtd4", reinterpret_cast<void*>(&Sleef_sqrtd4_u05)},
    {"Sleef_fabsd4", reinterpret_cast<void*>(&Sleef_fabsd4)},
    {"Sleef_floord4", reinterpret_cast<void*>(&Sleef_floord4)},
    {"Sleef_ceild4", reinterpret_cast<void*>(&Sleef_ceild4)},
    {"Sleef_truncd4", reinterpret_cast<void*>(&Sleef_truncd4)},
    {"Sleef_roundd4", reinterpret_cast<void*>(&Sleef_roundd4)},
    {"Sleef_lgammad4", reinterpret_cast<void*>(&Sleef_lgammad4_u10)},
    {"Sleef_atan2d4", reinterpret_cast<void*>(&Sleef_atan2d4_u10)},
    {"Sleef_powd4", reinterpret_cast<void*>(&Sleef_powd4_u10)},
    {"Sleef_fmodd4", reinterpret_cast<void*>(&Sleef_fmodd4)},
#endif
  };
  return c10::ArrayRef<SymbolAddress>(symbolAddresses);
}

} // namespace torch::jit::tensorexpr
#endif // TORCH_ENABLE_LLVM
