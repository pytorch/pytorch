#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(CAFFE2_PERF_WITH_SVE128)
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#include <cfloat>
#include <cmath>

#include "c10/macros/Macros.h"

/// Select `svlog` accuracy:
/// - 0: original.
/// - 1: more accurate, similar performance.
/// - 2: very high accuracy, a bit lower speed.
#define SVLOG_ACCURACY 2

/// Handle special cases in `svexp`:
/// - 0: original.
/// - 1: use clamp, better performance.
/// - 2: no special case handling.
#define SVEXP_SPECIAL_CLAMP 1

#if SVLOG_ACCURACY == 2
static inline svfloat32_t svlog(svfloat32_t x) {
  const svbool_t ptrue = svptrue_b8();

  svint32_t u = svreinterpret_s32(x) - 0x3F2AAAAB;

  svfloat32_t r = svreinterpret_f32((u & 0x007FFFFF) + 0x3F2AAAAB) - 1.0f;
  svfloat32_t n = svcvt_f32_x(ptrue, u >> 23);
  asm("" : "+w"(r)); // NOTE: can improve instruction scheduling.

  svfloat32_t r2 = r * r;
  svfloat32_t p = -0x1.4F9934p-3f + r * 0x1.5A9AA2p-3f;
  svfloat32_t q = -0x1.00187Cp-2f + r * 0x1.961348p-3f;
  svfloat32_t y = -0x1.FFFFC8p-2f + r * 0x1.555D7Cp-2f;
  return (r + n * 0x1.62E43p-1f) +
         (y + (q + (p + -0x1.3E737Cp-3f * r2) * r2) * r2) * r2;
}
#elif SVLOG_ACCURACY == 1
static inline svfloat32_t svlog(svfloat32_t x) {
  const svbool_t ptrue = svptrue_b8();

  svint32_t u = svreinterpret_s32(x) - 0x3F2AAAAB;

  svfloat32_t r = svreinterpret_f32((u & 0x007FFFFF) + 0x3F2AAAAB) - 1.0f;
  svfloat32_t n = svcvt_f32_x(ptrue, u >> 23);
  asm("" : "+w"(r)); // NOTE: can improve instruction scheduling.

  svfloat32_t r2 = r * r;
  svfloat32_t A = -0x1.923814p-3f + r * 0x1.689E5Ep-3f;
  svfloat32_t B = -0x1.FC0968p-3f + r * 0x1.93BF0Cp-3f;
  svfloat32_t C = -0x1.000478p-1f + r * 0x1.556906p-2f;

  return (r + n * 0x1.62E43p-1f) + (C + (B + A * r2) * r2) * r2;
}
#elif SVLOG_ACCURACY == 0
static inline svfloat32_t svlog(svfloat32_t x) {
  const svbool_t ptrue = svptrue_b8();

  svint32_t u = svsra_n_s32(svdup_n_s32(-127), svreinterpret_s32(x), 23);

  svfloat32_t n = svcvt_f32_x(ptrue, u);
  svfloat32_t r = svreinterpret_f32(svreinterpret_s32(x) - (u << 23));

  svfloat32_t D = -0.165253549814f + r * 0.0141278216615f;
  svfloat32_t C = -2.47071170807f + r * 0.844007015228f;
  svfloat32_t B = -5.68692588806f + r * 4.58445882797f;
  svfloat32_t A = -2.29561495781f + r * 5.17591238022f;

  svfloat32_t r2 = r * r;
  return (A + n * 0.6931471805f) + (B + (C + D * r2) * r2) * r2;
}
#endif

static inline svfloat32_t svexp(svfloat32_t x) {
  // Clamp interval set to prevent denormals!
  const svfloat32_t max_input = svdup_n_f32(88.722839f);
  const svfloat32_t min_input = svdup_n_f32(-87.33654f);
  const svfloat32_t shift = svdup_n_f32(0x1.0000FEp+23f);
  const svbool_t ptrue = svptrue_b8();

#if SVEXP_SPECIAL_CLAMP == 1
  x = svmax_x(ptrue, svmin_x(ptrue, x, max_input), min_input);
#endif

  svfloat32_t z = svmla_n_f32_x(ptrue, shift, x, 0x1.715476p+0f);
  svfloat32_t n = z - shift;
  svfloat32_t scale = svreinterpret_f32(svreinterpret_u32(z) << 23);

  svfloat32_t r_hi = x - n * 0x1.62E400p-1f;
  svfloat32_t r = r_hi - n * 0x1.7F7D1Cp-20f;
  svfloat32_t r2 = r * r;

  svfloat32_t C = 0x1.573E2Ep-5f + r * 0x1.0E4020p-7f;
  svfloat32_t B = 0x1.FFFDB6p-2f + r * 0x1.555E66p-3f;
  svfloat32_t A = r * 0x1.FFFFECp-1f;

  svfloat32_t poly = scale + (A + (B + C * r2) * r2) * scale;

#if SVEXP_SPECIAL_CLAMP == 0
  const svfloat32_t inf = svdup_n_f32(std::numeric_limits<float>::infinity());
  poly = svsel_f32(svcmplt_f32(ptrue, x, min_input), svdup_n_f32(0.0f), poly);
  poly = svsel_f32(svcmpgt_f32(ptrue, x, max_input), inf, poly);
#endif

  return poly;
}

static inline svfloat32_t compute_batch_box_cox_vec_sve128_float(
    svfloat32_t lambda1_v,
    svfloat32_t lambda2_v,
    svfloat32_t data_v,
    svfloat32_t k_eps) {
  const svbool_t ptrue = svptrue_b8();

  svfloat32_t lnData = svlog(svmax_x(ptrue, data_v + lambda2_v, k_eps));
  svbool_t predNZ = svcmpne_n_f32(ptrue, lambda1_v, 0.0f);
  if (C10_LIKELY(svptest_any(predNZ, predNZ))) {
    svfloat32_t lambda1_r = svdivr_f32_m(predNZ, lambda1_v, svdup_n_f32(1.0f));
    svfloat32_t pow = svexp(lnData * lambda1_v);
    lnData = svsel_f32(predNZ, lambda1_r, lnData);
    lnData = svnmsb_f32_m(predNZ, lnData, pow, lnData);
  }
  return lnData;
}

template <typename T>
void compute_batch_box_cox_vec_sve128(
    std::size_t N,
    std::size_t D,
    const T* data_ptr,
    const T* __restrict lambda1_ptr,
    const T* __restrict lambda2_ptr,
    T* output_ptr);

template <>
void compute_batch_box_cox_vec_sve128(
    std::size_t N,
    std::size_t D,
    const float *data_ptr,
    const float *__restrict lambda1_ptr,
    const float *__restrict lambda2_ptr,
    float *output_ptr) {
  const svfloat32_t k_eps = svdup_n_f32(static_cast<float>(1e-6));

  std::size_t remainder = D % 4;
  std::size_t loopBound = D - remainder;
  svbool_t remainderPred = svwhilelt_b32_u64(0, remainder);

  for (; C10_LIKELY(N > 0); --N) {
    for (std::size_t j = 0; C10_LIKELY(j != loopBound);
         j += 4, data_ptr += 4, output_ptr += 4) {
      svfloat32_t lambda1_v =
          svset_neonq(svundef_f32(), vld1q_f32(lambda1_ptr + j));
      svfloat32_t lambda2_v =
          svset_neonq(svundef_f32(), vld1q_f32(lambda2_ptr + j));
      svfloat32_t data_v = svset_neonq(svundef_f32(), vld1q_f32(data_ptr));
      svfloat32_t result = compute_batch_box_cox_vec_sve128_float(
          lambda1_v, lambda2_v, data_v, k_eps);
      vst1q_f32(output_ptr, svget_neonq(result));
    }
    if (C10_LIKELY(remainder > 0)) {
      svfloat32_t lambda1_v = svld1_f32(remainderPred, lambda1_ptr + loopBound);
      svfloat32_t lambda2_v = svld1_f32(remainderPred, lambda2_ptr + loopBound);
      svfloat32_t data_v = svld1_f32(remainderPred, data_ptr);
      svfloat32_t result = compute_batch_box_cox_vec_sve128_float(
          lambda1_v, lambda2_v, data_v, k_eps);
      svst1_f32(remainderPred, output_ptr, result);
      data_ptr += remainder;
      output_ptr += remainder;
    }
  }
}

namespace caffe2::details {

template <typename T>
void compute_batch_box_cox__sve128(
    std::size_t N,
    std::size_t D,
    const T* self_data,
    const T* __restrict lambda1_data,
    const T* __restrict lambda2_data,
    T* output_data) {
  compute_batch_box_cox_vec_sve128<T>(
      N, D, self_data, lambda1_data, lambda2_data, output_data);
}

// Vectorized version specializations for float and double
template void compute_batch_box_cox__sve128<float>(
    std::size_t N,
    std::size_t D,
    const float* self_data,
    const float* __restrict lambda1_data,
    const float* __restrict lambda2_data,
    float* output_data);

} // namespace caffe2::details

#endif // __aarch64__ && __ARM_FEATURE_SVE && CAFFE2_PERF_WITH_SVE128
