#if defined(CPU_CAPABILITY_SVE128) && defined(CAFFE2_PERF_WITH_SVE128)
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>

// Log and exp approximations inspired from ACL implementation

inline float32x4_t vtaylor_polyq_for_log_f32(float32x4_t x) {
  const float32x4_t log_tab_1 = vdupq_n_f32(-2.29561495781f);
  const float32x4_t log_tab_2 = vdupq_n_f32(-2.47071170807f);
  const float32x4_t log_tab_3 = vdupq_n_f32(-5.68692588806f);
  const float32x4_t log_tab_4 = vdupq_n_f32(-0.165253549814f);
  const float32x4_t log_tab_5 = vdupq_n_f32(5.17591238022f);
  const float32x4_t log_tab_6 = vdupq_n_f32(0.844007015228f);
  const float32x4_t log_tab_7 = vdupq_n_f32(4.58445882797f);
  const float32x4_t log_tab_8 = vdupq_n_f32(0.0141278216615f);

  float32x4_t A = vmlaq_f32(log_tab_1, log_tab_5, x);
  float32x4_t B = vmlaq_f32(log_tab_3, log_tab_7, x);
  float32x4_t C = vmlaq_f32(log_tab_2, log_tab_6, x);
  float32x4_t x2 = vmulq_f32(x, x);
  float32x4_t D = svget_neonq(svmad_f32_x(
      svptrue_b8(),
      svset_neonq(svundef_f32(), x),
      svset_neonq(svundef_f32(), log_tab_8),
      svset_neonq(svundef_f32(), log_tab_4)));
  float32x4_t x4 = vmulq_f32(x2, x2);
  float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
  return res;
}

inline float32x4_t vlogq_f32(float32x4_t x) {
  const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)

  // Extract exponent
  int32x4_t m = svget_neonq(svsub_n_s32_x(
      svptrue_b8(),
      svset_neonq(
          svundef_s32(),
          vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23))),
      127));
  float32x4_t val = vreinterpretq_f32_s32(
      vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

  // Polynomial Approximation
  float32x4_t poly = vtaylor_polyq_for_log_f32(val);

  // Reconstruct
  poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

  return poly;
}

inline float32x4_t vexpq_f32(float32x4_t x) {
  const auto c1 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3f7ffff6)));
  const auto c2 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3efffedb)));
  const auto c3 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3e2aaf33)));
  const auto c4 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3d2b9f17)));
  const auto c5 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3c072010)));

  const auto shift = vreinterpretq_f32_u32(
      svget_neonq(svdup_n_u32(0x4b00007f))); // 2^23 + 127 = 0x1.0000fep23f
  const auto inv_ln2 = vreinterpretq_f32_u32(
      svget_neonq(svdup_n_u32(0x3fb8aa3b))); // 1 / ln(2) = 0x1.715476p+0f
  const auto neg_ln2_hi = vreinterpretq_f32_u32(svget_neonq(
      svdup_n_u32(0xbf317200))); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
  const auto neg_ln2_lo = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(
      0xb5bfbe8e))); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

  const auto inf = svdup_n_f32(std::numeric_limits<float>::infinity());
  const auto max_input = svdup_n_f32(88.37f); // Approximately ln(2^127.5)
  const auto zero = svdup_n_f32(0.f);
  const auto min_input = svdup_n_f32(-86.64f); // Approximately ln(2^-125)

  // Range reduction:
  //   e^x = 2^n * e^r
  // where:
  //   n = floor(x / ln(2))
  //   r = x - n * ln(2)
  //
  // By adding x / ln(2) with 2^23 + 127 (shift):
  //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127
  //   forces decimal part
  //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n)
  //     + 127 will occupy the whole fraction part of z in FP32 format.
  //     Subtracting 2^23 + 127 (shift) from z will result in the integer part
  //     of x / ln(2) (i.e. n) because the decimal part has been pushed out and
  //     lost.
  //   * The addition of 127 makes the FP32 fraction part of z ready to be used
  //   as the exponent
  //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
  const auto z = vfmaq_f32(shift, x, inv_ln2);
  const auto n = z - shift;
  const auto scale =
      vreinterpretq_f32_u32(vreinterpretq_u32_f32(z) << 23); // 2^n

  // The calculation of n * ln(2) is done using 2 steps to achieve accuracy
  // beyond FP32. This outperforms longer Taylor series (3-4 tabs) both in term
  // of accuracy and performance.
  const auto r_hi = vfmaq_f32(x, n, neg_ln2_hi);
  const auto r = vfmaq_f32(r_hi, n, neg_ln2_lo);

  // Compute the truncated Taylor series of e^r.
  //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
  const auto r2 = r * r;

  const auto p1 = c1 * r;
  const auto p23 = vfmaq_f32(c2, c3, r);
  const auto p45 = vfmaq_f32(c4, c5, r);
  const auto p2345 = vfmaq_f32(p23, p45, r2);
  const auto p12345 = vfmaq_f32(p1, p2345, r2);

  auto poly = svset_neonq(svundef_f32(), vfmaq_f32(scale, p12345, scale));

  // Handle underflow and overflow.
  poly = svsel_f32(
      svcmplt_f32(svptrue_b8(), svset_neonq(svundef_f32(), x), min_input),
      zero,
      poly);
  poly = svsel_f32(
      svcmpgt_f32(svptrue_b8(), svset_neonq(svundef_f32(), x), max_input),
      inf,
      poly);

  return svget_neonq(poly);
}

// ln(x) = log2(x) * ln(2)
// pow(x, n) = exp(n * ln(x))
inline float32x4_t compute_batch_box_cox_vec_sve128_float(
    svfloat32_t lambda1_v,
    svfloat32_t lambda2_v,
    svfloat32_t data_v,
    svfloat32_t k_eps) {
  float32x4_t sum_v = vaddq_f32(svget_neonq(data_v), svget_neonq(lambda2_v));
  svbool_t predNZ = svcmpne_n_f32(svptrue_b8(), lambda1_v, 0.0f);
  sum_v = vmaxq_f32(sum_v, svget_neonq(k_eps));
  svfloat32_t lnData = svset_neonq(svundef_f32(), vlogq_f32(sum_v));
  if (__builtin_expect(svptest_any(predNZ, predNZ), 1)) {
    float32x4_t pow = vmulq_f32(svget_neonq(lnData), svget_neonq(lambda1_v));
    svfloat32_t lambda1_r = svdivr_f32_m(predNZ, lambda1_v, svdup_n_f32(1.0f));
    pow = vexpq_f32(pow);
    lnData = svsel_f32(predNZ, lambda1_r, lnData);
    lnData =
        svnmsb_f32_m(predNZ, lnData, svset_neonq(svundef_f32(), pow), lnData);
  }
  return svget_neonq(lnData);
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
    const float* data_ptr,
    const float* __restrict lambda1_ptr,
    const float* __restrict lambda2_ptr,
    float* output_ptr) {
  svfloat32_t k_eps = svdup_n_f32(static_cast<float>(1e-6));

  std::size_t remainder = D % 4;
  std::size_t loopBound = D - remainder;
  svbool_t remainderPred = svwhilelt_b32_u64(0, remainder);

  for (; __builtin_expect(N > 0, 1); --N) {
    for (std::size_t j = 0; __builtin_expect(j != loopBound, 1);
         j += 4, data_ptr += 4, output_ptr += 4) {
      svfloat32_t lambda1_v =
          svset_neonq(svundef_f32(), vld1q_f32(lambda1_ptr + j));
      svfloat32_t lambda2_v =
          svset_neonq(svundef_f32(), vld1q_f32(lambda2_ptr + j));
      svfloat32_t data_v = svset_neonq(svundef_f32(), vld1q_f32(data_ptr));
      float32x4_t result = compute_batch_box_cox_vec_sve128_float(
          lambda1_v, lambda2_v, data_v, k_eps);
      vst1q_f32(output_ptr, result);
    }
    if (__builtin_expect(remainder > 0, 1)) {
      svfloat32_t lambda1_v = svld1_f32(remainderPred, lambda1_ptr + loopBound);
      svfloat32_t lambda2_v = svld1_f32(remainderPred, lambda2_ptr + loopBound);
      svfloat32_t data_v = svld1_f32(remainderPred, data_ptr);
      float32x4_t result = compute_batch_box_cox_vec_sve128_float(
          lambda1_v, lambda2_v, data_v, k_eps);
      svst1_f32(remainderPred, output_ptr, svset_neonq(svundef_f32(), result));
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

#endif // CAFFE2_PERF_USE_MKL && CPU_CAPABILITY_SVE128
