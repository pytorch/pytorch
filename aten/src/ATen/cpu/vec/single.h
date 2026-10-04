#pragma once

#include <cstdint>
#include <limits>

namespace at::vec {
inline namespace CPU_CAPABILITY {

using Vf = Vectorized<float>;
using Vi = Vectorized<int32_t>;
using Vu = Vectorized<uint32_t>;

// |x| compared against y, for y >= 0. aarch64 has a fused absolute compare
// (FACGE/FACGT, and their operand-swapped FACLE/FACLT spellings) that takes the
// magnitude of both operands for free, so the abs costs nothing; x86 has no
// equivalent at any width, and materialises |x| with an AND before an ordinary
// compare. Only worth reaching for where |x| is wanted ONLY by the compare --
// where the same |x| also feeds arithmetic, the AND/FABS is paid anyway and
// these save nothing.
//
// PRECONDITION: y >= 0. The aarch64 forms compare against |y|, so a negative y
// silently means something different there than it does on x86.
#if defined(CPU_CAPABILITY_SVE256)
#define AT_VEC_DEFINE_ABS_CMP(NAME, OP, NEON, SVE)                                            \
  inline Vectorized<float> NAME(Vectorized<float> x, Vectorized<float> y) {                   \
    return svreinterpret_f32_u32(svdup_n_u32_z(SVE##_f32(svptrue_b32(), x, y), 0xFFFFFFFFu)); \
  }                                                                                           \
  inline Vectorized<double> NAME(Vectorized<double> x, Vectorized<double> y) {                \
    return svreinterpret_f64_u64(svdup_n_u64_z(SVE##_f64(svptrue_b64(), x, y), UINT64_MAX));  \
  }
#elif defined(__aarch64__)
#define AT_VEC_DEFINE_ABS_CMP(NAME, OP, NEON, SVE)                             \
  inline Vectorized<float> NAME(Vectorized<float> x, Vectorized<float> y) {    \
    return Vectorized<float>(vreinterpretq_f32_u32(NEON##q_f32(x, y)));        \
  }                                                                            \
  inline Vectorized<double> NAME(Vectorized<double> x, Vectorized<double> y) { \
    return Vectorized<double>(vreinterpretq_f64_u64(NEON##q_f64(x, y)));       \
  }
#else
#define AT_VEC_DEFINE_ABS_CMP(NAME, OP, NEON, SVE)                             \
  inline Vectorized<float> NAME(Vectorized<float> x, Vectorized<float> y) {    \
    return x.abs() OP y;                                                       \
  }                                                                            \
  inline Vectorized<double> NAME(Vectorized<double> x, Vectorized<double> y) { \
    return x.abs() OP y;                                                       \
  }
#endif

AT_VEC_DEFINE_ABS_CMP(ltAbs, <, vcalt, svaclt)
AT_VEC_DEFINE_ABS_CMP(leAbs, <=, vcale, svacle)
AT_VEC_DEFINE_ABS_CMP(gtAbs, >, vcagt, svacgt)
AT_VEC_DEFINE_ABS_CMP(geAbs, >=, vcage, svacge)
#undef AT_VEC_DEFINE_ABS_CMP

// Whole-vector predicates: "does EVERY lane compare true against y". A NaN lane
// always answers no, in all four directions, and the kernels below depend on
// that to keep a NaN out of a fast path that would turn it into a finite value.
//
// The two shapes are not interchangeable, so do not unify them. On x86 a splat
// compare answers this in 5 instructions (AVX512) or 6 (AVX2) against 18 and 16
// for a horizontal reduce: neither ISA has a horizontal min/max, so the reduce
// is a shuffle ladder, and MAXPS/MINPS return their SECOND operand when either
// input is NaN -- the ladder does not merely drop a NaN, it displaces whatever
// the NaN was compared against, so a real extremum can fall out of it and the
// NaN answer needs a separate unordered compare on top. aarch64 is the other
// way round: FMAXV/FMINV are single instructions that already propagate NaN.
#if defined(CPU_CAPABILITY_AVX512)
#define AT_VEC_DEFINE_ALL_CMP(NAME, OP, PRED, RED)                   \
  inline bool NAME(Vectorized<float> x, float y) {                   \
    return _mm512_cmp_ps_mask(x, _mm512_set1_ps(y), PRED) == 0xFFFF; \
  }                                                                  \
  inline bool NAME(Vectorized<double> x, double y) {                 \
    return _mm512_cmp_pd_mask(x, _mm512_set1_pd(y), PRED) == 0xFF;   \
  }
#elif defined(CPU_CAPABILITY_AVX2)
#define AT_VEC_DEFINE_ALL_CMP(NAME, OP, PRED, RED)               \
  inline bool NAME(Vectorized<float> x, float y) {               \
    const __m256 m = _mm256_cmp_ps(x, _mm256_set1_ps(y), PRED);  \
    return _mm256_movemask_ps(m) == 0xFF;                        \
  }                                                              \
  inline bool NAME(Vectorized<double> x, double y) {             \
    const __m256d m = _mm256_cmp_pd(x, _mm256_set1_pd(y), PRED); \
    return _mm256_movemask_pd(m) == 0xF;                         \
  }
#elif defined(CPU_CAPABILITY_SVE256)
#define AT_VEC_DEFINE_ALL_CMP(NAME, OP, PRED, RED)   \
  inline bool NAME(Vectorized<float> x, float y) {   \
    return sv##RED##v_f32(svptrue_b32(), x) OP y;    \
  }                                                  \
  inline bool NAME(Vectorized<double> x, double y) { \
    return sv##RED##v_f64(svptrue_b64(), x) OP y;    \
  }
#elif defined(__aarch64__)
#define AT_VEC_DEFINE_ALL_CMP(NAME, OP, PRED, RED)   \
  inline bool NAME(Vectorized<float> x, float y) {   \
    return v##RED##vq_f32(x) OP y;                   \
  }                                                  \
  inline bool NAME(Vectorized<double> x, double y) { \
    return v##RED##vq_f64(x) OP y;                   \
  }
#else
#define AT_VEC_DEFINE_ALL_CMP(NAME, OP, PRED, RED)         \
  inline bool NAME(Vectorized<float> x, float y) {         \
    __at_align__ float tmp[Vectorized<float>::size()];     \
    x.store(tmp);                                          \
    for (int i = 0; i < Vectorized<float>::size(); i++) {  \
      if (!(tmp[i] OP y))                                  \
        return false;                                      \
    }                                                      \
    return true;                                           \
  }                                                        \
  inline bool NAME(Vectorized<double> x, double y) {       \
    __at_align__ double tmp[Vectorized<double>::size()];   \
    x.store(tmp);                                          \
    for (int i = 0; i < Vectorized<double>::size(); i++) { \
      if (!(tmp[i] OP y))                                  \
        return false;                                      \
    }                                                      \
    return true;                                           \
  }
#endif

AT_VEC_DEFINE_ALL_CMP(allLt, <, _CMP_LT_OQ, max)
AT_VEC_DEFINE_ALL_CMP(allLe, <=, _CMP_LE_OQ, max)
AT_VEC_DEFINE_ALL_CMP(allGt, >, _CMP_GT_OQ, min)
AT_VEC_DEFINE_ALL_CMP(allGe, >=, _CMP_GE_OQ, min)
#undef AT_VEC_DEFINE_ALL_CMP

// Per-lane mask: is EITHER operand a NaN. A mask like operator!=, not a
// whole-vector answer like the predicates above.
//
// Reach for this rather than minimum(x, y) != minimum(x, y), which is the
// tempting spelling because min propagates NaN. On x86 that one costs five
// instructions where this costs one: ATen's minimum() already runs the
// unordered compare internally to get its NaN propagation, ORs the mask into a
// min nobody wants, and then the self-compare re-derives the same mask. Writing
// it (x != x) | (y != y) is fine on x86 -- clang folds that pair back into the
// single unordered compare -- but not on aarch64, which has no unordered
// compare at all; there FMAX propagates NaN, so folding the operands first is
// three instructions against four.
//
// When the caller also wants the min itself, neither applies: fold both tests
// onto the min instead, the way igamma's `bad` mask does.
#if defined(CPU_CAPABILITY_AVX512)
inline Vectorized<float> isAnyNaN(Vectorized<float> x, Vectorized<float> y) {
  return _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(x, y, _CMP_UNORD_Q), -1));
}
inline Vectorized<double> isAnyNaN(Vectorized<double> x, Vectorized<double> y) {
  return _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(x, y, _CMP_UNORD_Q), -1));
}
#elif defined(CPU_CAPABILITY_AVX2)
inline Vectorized<float> isAnyNaN(Vectorized<float> x, Vectorized<float> y) {
  return _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
}
inline Vectorized<double> isAnyNaN(Vectorized<double> x, Vectorized<double> y) {
  return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
}
#elif defined(CPU_CAPABILITY_SVE256)
inline Vectorized<float> isAnyNaN(Vectorized<float> x, Vectorized<float> y) {
  return svreinterpret_f32_u32(svdup_n_u32_z(svcmpuo_f32(svptrue_b32(), x, y), 0xFFFFFFFFu));
}
inline Vectorized<double> isAnyNaN(Vectorized<double> x, Vectorized<double> y) {
  return svreinterpret_f64_u64(svdup_n_u64_z(svcmpuo_f64(svptrue_b64(), x, y), UINT64_MAX));
}
#elif defined(__aarch64__)
inline Vectorized<float> isAnyNaN(Vectorized<float> x, Vectorized<float> y) {
  const Vectorized<float> m = maximum(x, y);
  return m != m;
}
inline Vectorized<double> isAnyNaN(Vectorized<double> x, Vectorized<double> y) {
  const Vectorized<double> m = maximum(x, y);
  return m != m;
}
#else
inline Vectorized<float> isAnyNaN(Vectorized<float> x, Vectorized<float> y) {
  return (x != x) | (y != y);
}
inline Vectorized<double> isAnyNaN(Vectorized<double> x, Vectorized<double> y) {
  return (x != x) | (y != y);
}
#endif

inline Vectorized<float> Vectorized<float>::ceil() const {
#if defined(CPU_CAPABILITY_AVX512)
  return _mm512_ceil_ps(values);
#elif defined(CPU_CAPABILITY_AVX2)
  return _mm256_ceil_ps(values);
#elif defined(CPU_CAPABILITY_SVE256)
  return svrintp_f32_x(ptrue, values);
#elif defined(__aarch64__)
  return vrndpq_f32(values);
#else
  return map(at::native::ceil_impl);
#endif
}

inline Vectorized<float> Vectorized<float>::floor() const {
#if defined(CPU_CAPABILITY_AVX512)
  return _mm512_floor_ps(values);
#elif defined(CPU_CAPABILITY_AVX2)
  return _mm256_floor_ps(values);
#elif defined(CPU_CAPABILITY_SVE256)
  return svrintm_f32_x(ptrue, values);
#elif defined(__aarch64__)
  return vrndmq_f32(values);
#else
  return map(at::native::floor_impl);
#endif
}

inline Vectorized<float> Vectorized<float>::round() const {
#if defined(CPU_CAPABILITY_AVX512)
  return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(CPU_CAPABILITY_AVX2)
  return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(CPU_CAPABILITY_SVE256)
  return svrintn_f32_x(ptrue, values);
#elif defined(__aarch64__)
  return vrndnq_f32(values);
#else
  return map(at::native::round_impl);
#endif
}

// Max ULP: 1.55
template <bool CheckInput, bool HandleSubNormals>
Vectorized<float> logimpl(Vectorized<float> x) {
  const Vi kBias(0x3F2FE200);
  Vi kOff = kBias;
  const Vf ln2(0x1.62E43p-1f);

  const Vu isSubnormal = cast<uint32_t>(x) < Vu(0x00800000);

  Vf z = x;

  if constexpr (HandleSubNormals) {
    z = Vf::blendv(z, z * Vf(0x1p25f), cast<float>(isSubnormal));
    kOff = Vi::blendv(kBias, kBias + (25 << 23), cast<int32_t>(isSubnormal));
  }

  const Vi u = cast<int32_t>(z) - kBias;
  const Vi s = cast<int32_t>(z) - kOff;

  Vf w = ln2;

  if constexpr (CheckInput) {
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf nan(std::numeric_limits<float>::quiet_NaN());

    w = Vf::blendv(nan, w, x > Vf(0.0f));
    w = Vf::blendv(w, inf, x == inf);
    w = Vf::blendv(w, inf | x, x == Vf(0.0f));
  }

  const Vf v = cast<float>((u & Vi(0x007FFFFF)) + kBias);
  const Vf r = v - Vf(1.0f);

  const Vf n = convert<float>(s >> Vi(23));

  const Vf r2 = r * r;
  const Vf r4 = r2 * r2;

  const Vf a = fmadd(r, Vf(0x1.555660p-2f), Vf(-0x1.FFFFE8p-2f));
  const Vf b = fmadd(r, Vf(0x1.992B80p-3f), Vf(-0x1.000F60p-2f));
  const Vf c = fmadd(r, Vf(0x1.2A401Ap-3f), Vf(-0x1.5125B0p-3f));
  const Vf d = fmadd(r, Vf(0x1.C39E3Ap-4f), Vf(-0x1.34CAEAp-3f));

  // (a + b*r2) + (c + d*r2)*r4
  const Vf p = fmadd(fmadd(d, r2, c), r4, fmadd(b, r2, a));

  // (r + n*ln2) + p*r2
  return fmadd(p, r2, fmadd(n, w, r));
}

inline Vectorized<float> Vectorized<float>::log() const {
  return logimpl<true, true>(*this);
}

// Max ULP: 1.02
template <bool CheckInput>
Vectorized<float> expimpl(Vectorized<float> xin) {
  // Saturation points: above max_input every result overflows to +Inf, below
  // min_input every result rounds to +0. They also bound |n| so that
  // n * ln2_hi stays exact. clamp() propagates NaN on every CPU capability,
  // clamp_min/clamp_max do not.
  const Vf max_input(88.722839f); // > ln(FLT_MAX)
  const Vf min_input(-104.0f); // < ln(2^-150)
  // 1.5*2^23 forces the sum into the binade whose ULP is 1, so the add rounds
  // x*log2(e) to an integer; n is recovered by subtracting it back.
  const Vf shift(0x1.8p+23f);

  Vf x = xin;
  if constexpr (CheckInput) {
    x = clamp(xin, min_input, max_input);
  }

  const Vf z = fmadd(x, Vf(0x1.715476p+0f), shift);
  const Vf n = z - shift;

  // With this shift the bit pattern of z is exactly 0x4B400000 + n over the
  // whole clamped range, so n comes back as an integer without an FP convert.
  // n spans [-150, 128], which does not fit one exponent field; splitting it
  // in half puts both parts in [-75, 64], so s1 and s2 are always normal even
  // when the product underflows to a subnormal or overflows to +Inf.
  const Vi ni = cast<int32_t>(z) - Vi(0x4B400000);
  const Vi n1 = ni >> Vi(1);
  const Vi n2 = ni - n1;
  const Vf s1 = cast<float>((n1 + Vi(127)) << Vi(23));
  const Vf s2 = cast<float>((n2 + Vi(127)) << Vi(23));

  const Vf r_hi = fnmadd(n, Vf(0x1.62E400p-1f), x);
  const Vf r = fnmadd(n, Vf(0x1.7F7D1Cp-20f), r_hi);
  const Vf r2 = r * r;

  // y = exp(r) - 1 = r + r^2 * S(r), S minimax of degree 4. Pinning the
  // linear coefficient to exactly 1 removes a multiply, which pays for the
  // extra degree: same instruction count as the degree-5 fit with a free
  // linear coefficient, whose approximation error alone is worth ~1 ULP
  // here. What remains is evaluation rounding: at the worst input, 0.04 ULP
  // of the 1.02 is the polynomial.
  const Vf u = fmadd(r, Vf(0x1.555492p-3f), Vf(0x1.fffffcp-2f));
  const Vf v = fmadd(r, Vf(0x1.1239d6p-7f), Vf(0x1.5558f2p-5f));
  const Vf w = fmadd(r2, Vf(0x1.6a2448p-10f), v);
  const Vf y = fmadd(fmadd(w, r2, u), r2, r);

  // s1 is a power of two and s1*(1+poly) never leaves the normal range, so
  // folding the +1 into this FMA is exact -- it rounds in the same place the
  // separate (poly + 1) would have. Only the multiply by s2 rounds again,
  // which is what lets the result reach into the subnormals.
  //
  // Below min_input the result is +0 either way, but reaching it by letting
  // s1*s2 underflow raises the underflow exception, which costs an x86
  // microcode assist on every such vector. An exact +0 multiplier does not.
  if constexpr (CheckInput) {
    return fmadd(y, s1, s1) * (s2 & (x != min_input));
  } else {
    return fmadd(y, s1, s1) * s2;
  }
}

inline Vectorized<float> Vectorized<float>::exp() const {
  return expimpl<true>(*this);
}

// log2(m) * y, then 2^t, in double, for both halves of pow's float vector.
//
// m = 1 + r is pow's reduced mantissa in [1/sqrt2, sqrt2), so s = r/(2+r) is
// in [-0.1716, 0.1716] and log2(m) = s * L(s^2), L(z) = 2/ln2 + z P(z). The
// constant is pinned rather than fitted: an unpinned minimax puts its
// equioscillation error at s = 0, where |t| is largest relative to log2(m),
// and that measured 10x the wrong-rounding rate. P is a degree-5 minimax for
// relative error, 2^-44.9; |t| <= 130 then bounds t's absolute error near
// 2^-37.9, far below what the one rounding to float can see.
//
// 2^f = 1 + f Q(f) over f in [-1/2, 1/2], again with the constant pinned so
// that 2^0 is exact. Degree 8 is 2^-39.8; degree 7 is 2^-32.0, which is
// 0.18% wrong roundings instead of 0.0008% for 3.6% less time.
inline Vectorized<double> pow_core_d(Vectorized<double> r, Vectorized<double> n, Vectorized<double> y) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;
  constexpr double kL[5] = {
      0x1.ec709dc4c43a1p-1, 0x1.2776c2ecfd29ep-1, 0x1.a619f8d088957p-2, 0x1.479c64760df6p-2, 0x1.215b5e6a08285p-2};
  constexpr double kE[8] = {0x1.62e42fef7a78ap-1,
                            0x1.ebfbdff7c5a76p-3,
                            0x1.c6b08defc26fcp-5,
                            0x1.3b2ab7f3b3738p-7,
                            0x1.5d872975b43bcp-10,
                            0x1.4306c115d7f0dp-13,
                            0x1.00ee07266373ap-16,
                            0x1.66348851f3bb9p-20};

  const Vd s = r / (Vd(2.0) + r);
  const Vd z = s * s;
  Vd p(kL[4]);
  for (int i = 3; i >= 0; --i) {
    p = fmadd(p, z, Vd(kL[i]));
  }
  const Vd l2 = fmadd(s, fmadd(z, p, Vd(0x1.71547652b82fep+1)), n);

  // Every finite result lies within 2^[-150, 128]. The clamp keeps 2^k a
  // normal double, so the narrow below does the overflow, the underflow and
  // the subnormal rounding, each in a single rounding.
  const Vd t = clamp(y * l2, Vd(-160.0), Vd(130.0));

  const Vd shift(0x1.8p52);
  const Vd kd = t + shift;
  const Vd f = t - (kd - shift);
  Vd q(kE[7]);
  for (int i = 6; i >= 0; --i) {
    q = fmadd(q, f, Vd(kE[i]));
  }
  const Vd e = fmadd(f, q, Vd(1.0));
  // kd's low mantissa bits hold k in two's complement; shifted into the
  // exponent field, the add scales e by 2^k.
  return cast<double>(cast<int64_t>(e) + (cast<int64_t>(kd) << Vl(52)));
}

// Max ULP: 0.5000, 0.0008% not correctly rounded
inline Vectorized<float> Vectorized<float>::pow(const Vectorized<float>& n) const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(VectorizedN<double, kR>::size() == Vf::size());
  static_assert(kR == 2); // the double section below is written out for both halves

  const Vf x = *this;
  const Vf ax = x.abs();
  const Vf one(1.0f);
  const Vf zero(0.0f);
  const Vf inf(std::numeric_limits<float>::infinity());

  // |x| = 2^e * m with m in [1/sqrt2, sqrt2), done in float where it is
  // exact. r = m - 1 is exact by Sterbenz. Subnormals are scaled by 2^25 first.
  const Vi kBias(0x3F3504F3);
  const Vi isSubnormal = cast<int32_t>(ax) < Vi(0x00800000);
  const Vf zx = Vf::blendv(ax, ax * Vf(0x1p25f), cast<float>(isSubnormal));
  const Vi kOff = Vi::blendv(kBias, kBias + Vi(25 << 23), isSubnormal);
  const Vi u = cast<int32_t>(zx) - kBias;
  const Vf r = cast<float>((u & Vi(0x007FFFFF)) + kBias) - one;
  Vf e = convert<float>((cast<int32_t>(zx) - kOff) >> Vi(23));
  // log2|x| for the special bases rides on e: 0 -> -inf, inf -> +inf.
  e = Vf::blendv(e, -inf, ax == zero);
  e = Vf::blendv(e, inf, ax == inf);

  const VectorizedN<double, kR> rd = convert<double, kR, float, 1>(r);
  const VectorizedN<double, kR> ed = convert<double, kR, float, 1>(e);
  const VectorizedN<double, kR> yd = convert<double, kR, float, 1>(n);
  VectorizedN<double, kR> md;
  md[0] = pow_core_d(rd[0], ed[0], yd[0]);
  md[1] = pow_core_d(rd[1], ed[1], yd[1]);
  // The clamp need not propagate NaN, so NaN is put back in float.
  Vf m = convert<float, 1, double, kR>(md) | ((x != x) | (n != n));

  // |x| == 1 has magnitude exactly 1 for every n, including n = +-Inf where
  // log|x| * n would otherwise be 0 * Inf = NaN. This also covers x == +1, so
  // no separate x == 1 fixup is needed at the end.
  m = Vf::blendv(m, one, ax == one);

  // trunc(n) == n is also true for +-Inf, which is what the sign rules want.
  const Vf n_int = n == n.trunc();
  const Vf n_odd =
      n_int & (n.abs() < Vf(0x1p24f)) & cast<float>((convert_to_int_of_same_size<float>(n) & Vi(1)) != Vi(0));
  // Sign bit, not x < 0: pow(-0, odd) must come out negative.
  const Vf x_signed = cast<float>(cast<int32_t>(x) < Vi(0));
  const Vf x_neg_finite = (x < zero) & (ax != inf);

  const Vf signMask = cast<float>(cast<int32_t>(n_odd & x_signed) << Vi(31));

  Vf res = m ^ signMask;
  // A negative finite base with a non-integer exponent is a domain error.
  res = res | (x_neg_finite & ~n_int);
  // pow(x, +-0) == 1 for every x, including NaN.
  return Vf::blendv(res, one, n == zero);
}

// Max ULP: 1.57
template <bool CheckInput, bool HandleSubNormals>
Vectorized<float> log10impl(Vectorized<float> x) {
  // 0.666667f. Unlike log()'s bias this is 2/3, which puts r in [-1/3, 1/3]:
  // symmetric, so it minimises max|r|. The coefficients below are fitted
  // against this bias and are not interchangeable with logimpl's.
  const Vi kBias(0x3F2AAAAB);
  Vi kOff = kBias;
  // log10(2) split so that n * kL2Hi is EXACT: kL2Hi has 16 mantissa bits and
  // |n| <= 149 needs 8, so the product fits in 24. kL2Lo carries the rest.
  const Vf kL2Hi(0x1.3441p-2f);
  const Vf kL2Lo(0x1.A8503Ep-21f);
  const Vf invLn10(0x1.BCB7B2p-2f);

  const Vu isSubnormal = cast<uint32_t>(x) < Vu(0x00800000);

  Vf z = x;

  if constexpr (HandleSubNormals) {
    // x * 2^25 is exact for a subnormal (<= 23 significant bits, and
    // 2^-149 * 2^25 = 2^-124 is normal), so the reduction stays exact.
    // Folding the 25 into kOff rather than subtracting it from n keeps the
    // correction on the integer side, where it is free.
    z = Vf::blendv(z, z * Vf(0x1p25f), cast<float>(isSubnormal));
    kOff = Vi::blendv(kBias, kBias + Vi(25 << 23), cast<int32_t>(isSubnormal));
  }

  const Vi u = cast<int32_t>(z) - kBias;
  const Vi s = cast<int32_t>(z) - kOff;

  const Vf v = cast<float>((u & Vi(0x007FFFFF)) + kBias);
  const Vf r = v - Vf(1.0f);

  Vf w = kL2Hi;

  if constexpr (CheckInput) {
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf nan(std::numeric_limits<float>::quiet_NaN());

    w = Vf::blendv(nan, w, x > Vf(0.0f));
    w = Vf::blendv(w, inf, x == inf);
    w = Vf::blendv(w, inf | x, x == Vf(0.0f));
  }

  const Vf n = convert<float>(s >> Vi(23));

  const Vf r2 = r * r;

  // p ~ (log10(1+r) - r/ln(10)) / r^2 on r in [-1/3, 1/3], order 9 overall.
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel its rounding errors. Retune
  // (do not reuse) if the evaluation order or the reconstruction changes.
  const Vf a = fmadd(r, Vf(0x1.2879CAp-3f), Vf(-0x1.BCB7A2p-3f));
  const Vf b = fmadd(r, Vf(0x1.640982p-4f), Vf(-0x1.BCD446p-4f));
  const Vf c = fmadd(r, Vf(0x1.F0E56Cp-5f), Vf(-0x1.246F1Cp-4f));
  const Vf d = fmadd(r, Vf(0x1.F5F7B2p-5f), Vf(-0x1.0FC88Cp-4f));

  // a + r2*(b + r2*(c + r2*d))
  const Vf p = fmadd(r2, fmadd(r2, fmadd(r2, d, c), b), a);

  // n*kL2Hi + (r*invLn10 + n*kL2Lo + p*r2).
  //
  // rh is the rounded product and rl its EXACT residual -- a product of two
  // floats splits exactly into hi + lo, so nothing is lost to that multiply.
  // The accumulator then starts from the tiny residual instead of from rh, so
  // the n*kL2Lo and p*r2 adds round at ~0.02 scale rather than at result
  // scale. rh is folded in last, and the closing fmadd is exact in n*w,
  // leaving a single rounding at result magnitude.
  const Vf rh = r * invLn10;
  const Vf rl = fnmadd(r, invLn10, rh); // rh - r*invLn10
  Vf acc = fmsub(n, kL2Lo, rl);
  acc = fmadd(p, r2, acc);
  acc = acc + rh;
  return fmadd(n, w, acc);
}

inline Vectorized<float> Vectorized<float>::log10() const {
  return log10impl<true, true>(*this);
}

// Max ULP: 1.68
template <bool CheckInput, bool HandleSubNormals>
Vectorized<float> log2impl(Vectorized<float> x) {
  // Same 2/3 bias as log10impl: r in [-1/3, 1/3], symmetric, minimising max|r|.
  const Vi kBias(0x3F2AAAAB);
  Vi kOff = kBias;
  // log2(2^n) == n exactly, so unlike log() and log10() there is no hi/lo
  // split of the reconstruction constant and no n*kL2Lo term. w is simply 1
  // for ordinary inputs; it exists only to carry the special-value results
  // out through n * w.
  const Vf invLn2(0x1.715476p+0f);

  const Vu isSubnormal = cast<uint32_t>(x) < Vu(0x00800000);

  Vf z = x;

  if constexpr (HandleSubNormals) {
    z = Vf::blendv(z, z * Vf(0x1p25f), cast<float>(isSubnormal));
    kOff = Vi::blendv(kBias, kBias + Vi(25 << 23), cast<int32_t>(isSubnormal));
  }

  const Vi u = cast<int32_t>(z) - kBias;
  const Vi s = cast<int32_t>(z) - kOff;

  const Vf v = cast<float>((u & Vi(0x007FFFFF)) + kBias);
  const Vf r = v - Vf(1.0f);

  Vf w(1.0f);

  if constexpr (CheckInput) {
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf nan(std::numeric_limits<float>::quiet_NaN());

    w = Vf::blendv(nan, w, x > Vf(0.0f));
    w = Vf::blendv(w, inf, x == inf);
    w = Vf::blendv(w, inf | x, x == Vf(0.0f));
  }

  const Vf n = convert<float>(s >> Vi(23));

  const Vf r2 = r * r;

  // p ~ (log2(1+r) - r/ln(2)) / r^2 on r in [-1/3, 1/3], order 9 overall.
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel its rounding errors. Retune
  // (do not reuse) if the evaluation order or the reconstruction changes.
  const Vf a = fmadd(r, Vf(0x1.EC7036p-2f), Vf(-0x1.715458p-1f));
  const Vf b = fmadd(r, Vf(0x1.279B56p-2f), Vf(-0x1.7170FEp-2f));
  const Vf c = fmadd(r, Vf(0x1.9DCEDCp-3f), Vf(-0x1.E527ACp-3f));
  const Vf d = fmadd(r, Vf(0x1.9F2D7Ep-3f), Vf(-0x1.C64004p-3f));

  // a + r2*(b + r2*(c + r2*d))
  const Vf p = fmadd(r2, fmadd(r2, fmadd(r2, d, c), b), a);

  // n*w + (r*invLn2 + p*r2).
  //
  // rh is the rounded product and rl its EXACT residual, so the leading term
  // loses nothing to that multiply. p*r2 is accumulated onto the tiny residual
  // first and rh folded in last, leaving one rounding at result magnitude
  // before the closing fmadd, which is itself exact in n*w.
  const Vf rh = r * invLn2;
  const Vf rl = fmsub(r, invLn2, rh); // r*invLn2 - rh
  Vf acc = fmadd(p, r2, rl);
  acc = acc + rh;
  return fmadd(n, w, acc);
}

inline Vectorized<float> Vectorized<float>::log2() const {
  return log2impl<true, true>(*this);
}

// Max ULP: 1.34
template <bool CheckInput>
Vectorized<float> log1pimpl(Vectorized<float> x) {
  // 1 + x = 2^k * (1 + r), biased by 0.75 so that r lands in [-0.25, 0.5].
  // k is picked from 1 + x but the scaling is applied to x itself, so r is
  // built out of x's own bits and the low bits of a small x are never lost to
  // the +1. That is the whole reason log1p is not just log(1 + x).
  const Vi kBias(0x3F400000);
  const Vi kExpMask(static_cast<int32_t>(0xFF800000));
  const Vf ln2(0x1.62E43p-1f);

  const Vi k = (cast<int32_t>(x + Vf(1.0f)) - kBias) & kExpMask;

  // 4 * 2^-k, not 2^-k: k reaches 128 at FLT_MAX, and the spare factor of 4
  // keeps s a normal float there. 0.25 * s recovers 2^-k exactly.
  const Vf s = cast<float>(Vi(0x40800000) - k);
  const Vf r = cast<float>(cast<int32_t>(x) - k) + fmadd(Vf(0.25f), s, Vf(-1.0f));

  // As in logimpl, w is the reconstruction constant for ordinary inputs and
  // the carrier for every special value, which leaves the fast path branchless.
  Vf w = ln2;

  if constexpr (CheckInput) {
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf nan(std::numeric_limits<float>::quiet_NaN());

    // x < -1 is a domain error. Phrased as >= so NaN falls through to NaN too.
    // n * nan is nan for any n, so this does not depend on k being non-zero.
    w = Vf::blendv(nan, w, x >= Vf(-1.0f));
    // k is 128 at +Inf, so n * w carries the infinity out.
    w = Vf::blendv(w, inf, x == inf);
    // x == -1 needs no blend: the reduction drives r to a large negative value
    // and the polynomial reaches -Inf on its own.
  }

  const Vf n = convert<float>(k >> Vi(23));

  // log1p(r) = r - r^2/2 + c0*r^3 + ... + c6*r^9 on [-0.25, 0.5]. The 1 and
  // -1/2 are not stored: -1/2 rides in q and the 1 in the closing
  // reconstruction.
  //
  // NOT a plain minimax fit: seeded from a weighted Remez fit, then tuned
  // against the max ULP of this exact evaluation sequence, so the coefficients
  // also absorb some of its rounding. Retune (do not reuse) if the evaluation
  // order or the reconstruction changes. Truncating a degree-10 fit to these
  // terms instead of refitting costs 1810 ULP.
  //
  // Unlike the degree-10 form, approximation error is a real contributor here
  // (0.679 ULP-equivalent, against 0.0965 at degree 10). That buys back one
  // FMA for +0.08 ULP, because degree 10's worst case was reduction-limited
  // rather than approximation-limited.
  const Vf r2 = r * r;
  const Vf q = fmadd(r, Vf(0x1.5555AEp-2f), Vf(-0.5f));
  const Vf a = fmadd(r, Vf(-0x1.57D5E8p-3f), Vf(0x1.993CC4p-3f));
  const Vf b = fmadd(r, Vf(-0x1.F031DCp-4f), Vf(0x1.313E1p-3f));

  // The top term is a bare constant, not a fmadd pair -- this is the whole
  // saving over the degree-10 form.
  Vf p = fmadd(fmadd(Vf(0x1.C773DEp-5f), r2, b), r2, a);
  p = fmadd(p, r, Vf(-0x1.FFECB6p-3f));

  // q is folded in HERE rather than added after the r + r2*p below. Deferring
  // it would round r + r^4*P at the magnitude of r, and the -r^2/2 it still
  // owes can carry the result into the next lower binade, making that rounding
  // worth 2 ULP of the result. Everything inside one r + r2*(...) leaves a
  // single rounding at result magnitude, and costs one FMA less.
  p = fmadd(p, r2, q);

  // n * w last, unlike log(), which folds it into the polynomial's base. When
  // k == 0 the polynomial is the entire result, so it has to be the term that
  // rounds last; reconstructing in log()'s order costs 0.42 ULP here.
  Vf res = fmadd(n, w, fmadd(p, r2, r));

  if constexpr (CheckInput) {
    // log1p(-0) is -0, but the reduction yields +0 (-0.0 + 0.0 == +0.0).
    // log1p agrees in sign with x for every other input, so OR-ing x's sign
    // bit in is a no-op there.
    res = res | (x & Vf(-0.0f));
  }
  return res;
}

inline Vectorized<float> Vectorized<float>::log1p() const {
  return log1pimpl<true>(*this);
}

// Max ULP: 1.75
inline Vectorized<float> Vectorized<float>::exp2() const {
  // Saturation points: above max_input every result overflows to +Inf, below
  // min_input every result rounds to +0. They also bound |n| so the exponent
  // split below stays in range. clamp() propagates NaN on every CPU
  // capability, clamp_min/clamp_max do not.
  const Vf max_input(129.0f); // > log2(FLT_MAX)
  const Vf min_input(-151.0f); // < log2(2^-150)
  // 1.5*2^23 forces the sum into the binade whose ULP is 1, so the add rounds
  // x to an integer; n is recovered by subtracting it back.
  const Vf shift(0x1.8p+23f);

  const Vf x = clamp(*this, min_input, max_input);

  // Unlike exp() there is no x*log2(e) scaling: x is already the exponent.
  const Vf z = x + shift;
  const Vf n = z - shift;

  // With this shift the bit pattern of z is exactly 0x4B400000 + n over the
  // whole clamped range, so n comes back as an integer without an FP convert.
  // n spans [-151, 129], which does not fit one exponent field; splitting it
  // in half puts both parts in [-76, 65], so s1 and s2 are always normal even
  // when the product underflows to a subnormal or overflows to +Inf.
  const Vi ni = cast<int32_t>(z) - Vi(0x4B400000);
  const Vi n1 = ni >> Vi(1);
  const Vi n2 = ni - n1;
  const Vf s1 = cast<float>((n1 + Vi(127)) << Vi(23));
  const Vf s2 = cast<float>((n2 + Vi(127)) << Vi(23));

  // Exact: |r| <= 1/2 and r is a multiple of ulp(x), so it needs no more
  // precision than x did. There is no hi/lo reduction constant to split here,
  // which is what makes exp2 cheaper than exp.
  const Vf r = x - n;
  const Vf r2 = r * r;

  // y = 2^r - 1 over r in [-1/2, 1/2], degree 5 with no constant term.
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel its rounding errors. Retune
  // (do not reuse) if the evaluation order or the reconstruction changes.
  const Vf p = fmadd(r, Vf(0x1.59F82Ep-10f), Vf(0x1.3CF216p-7f));
  const Vf q = fmadd(r, Vf(0x1.C6BC94p-5f), Vf(0x1.EBF9A2p-3f));
  // The product that gets rounded is the small one; letting the dominant
  // ln2*r term into the closing fmadd unrounded instead is worth 0.2 ULP.
  // It costs one extra step of dependency chain but no extra instruction.
  const Vf t = fmadd(p, r2, q) * r2;
  const Vf y = fmadd(r, Vf(0x1.62E422p-1f), t);

  // s1 is a power of two and s1*(1+y) never leaves the normal range, so
  // folding the +1 into this FMA is exact -- it rounds in the same place the
  // separate (y + 1) would have. Only the multiply by s2 rounds again, which
  // is what lets the result reach into the subnormals.
  //
  // See exp(): masking s2 to an exact +0 below min_input avoids the
  // underflow exception, and so the x86 microcode assist, without changing
  // any result.
  return fmadd(y, s1, s1) * (s2 & (x != min_input));
}

// Max ULP: 1.29
inline Vectorized<float> Vectorized<float>::expm1() const {
  // Saturation points: at or above max_input the result overflows to +Inf; at
  // or below min_input e^x - 1 has already rounded to -1 (which happens by
  // x = -18, long before here). min_input also keeps n >= -127, so n + 127 is
  // non-negative and the exponent build needs no clamp: n = -127 gives
  // scale = 0, which yields exactly -1. clamp() propagates NaN on every CPU
  // capability, clamp_min/clamp_max do not.
  const Vf max_input(89.0f);
  const Vf min_input(-88.0f);
  // 1.5*2^23 forces the sum into the binade whose ULP is 1, so the add rounds
  // x*log2(e) to an integer; n is recovered by subtracting it back.
  const Vf shift(0x1.8p+23f);
  const Vf one(1.0f);

  const Vf x = clamp(*this, min_input, max_input);

  const Vf z = fmadd(x, Vf(0x1.715476p+0f), shift);
  const Vf n = z - shift;
  // With this shift the bit pattern of z is exactly 0x4B400000 + n over the
  // whole clamped range, so n comes back as an integer without an FP convert.
  const Vi ni = cast<int32_t>(z) - Vi(0x4B400000);

  // x - n*ln2_hi is exact: ln2_hi has 9 zero low mantissa bits and |n| <= 128,
  // so the product fits in 24. ln2_lo carries the rest of ln2.
  const Vf r_hi = fnmadd(n, Vf(0x1.62E400p-1f), x);
  const Vf r = fnmadd(n, Vf(0x1.7F7D1Cp-20f), r_hi);
  const Vf r2 = r * r;

  // e^r - 1 = r + r^2 * P(r) over r in [-ln2/2, ln2/2]. The linear
  // coefficient is pinned to exactly 1, which is what lets the reconstruction
  // below carry the dominant term without a rounding of its own. Attaching c4
  // inside the r^2 factor is the same polynomial at the same dependency depth
  // as p01 + r2*p23 + r4*c4, without ever forming r4.
  //
  // NOT a minimax fit: the approximation error is only 0.26 ULP, so these are
  // tuned against the max ULP of this exact evaluation sequence and cancel its
  // rounding instead. Retune (do not reuse) if the order or the
  // reconstruction changes.
  const Vf p01 = fmadd(r, Vf(0x1.55547p-3f), Vf(0x1.FFFFFAp-2f));
  const Vf p23 = fmadd(r, Vf(0x1.124C18p-7f), Vf(0x1.5558A8p-5f));
  const Vf q = fmadd(r2, Vf(0x1.6AE3AAp-10f), p23);
  const Vf P = fmadd(r2, q, p01);
  const Vf t = r2 * P;

  // scale = 2^n is exact and so is r*scale, so the dominant term contributes
  // no rounding and both roundings land at result magnitude. Forming (r + t)
  // first instead rounds at polynomial magnitude, which scale then amplifies
  // through the (scale - 1) cancellation -- worth ~0.32 ULP.
  const Vf scale = cast<float>((ni + Vi(127)) << Vi(23));
  const Vf small = fmadd(t, scale, fmadd(r, scale, scale - one));

  // For n >= 32 the -1 is below 0.002 ULP and 2^n may not fit a float.
  // 2^(n-32) * (1 + poly) scaled back by 2^32 overflows to +Inf on its own.
  const Vf poly = r + t;
  const Vf s2 = cast<float>((ni + Vi(95)) << Vi(23));
  const Vf big = fmadd(poly, s2, s2) * Vf(0x1p32f);

  return Vf::blendv(small, big, n >= Vf(32.0f));
}

// ---------------------------------------------------------------------------
// Scalar sin for the |x| >= 0x1p20 tail, where the three-term Cody-Waite
// reduction in sinimpl runs out of pi. Specialised from llvm-libc's sinf
// (libc/src/__support/math/sinf.h + sincosf_utils.h): its |x| <= pi/16,
// |x| < 2^-12, x == 0 and hard-coded 0x1.33333p13 branches all sit below 2^20
// and are unreachable here, so they are gone. What remains was verified
// bit-identical to upstream over every input in the domain, and is correctly
// rounded (0.5 ULP) throughout -- the tail is more accurate than the body.

// sin(k * pi/32) for k = 0..63. cos(k * pi/32) is the same table read at
// k + 16, since cos(t) = sin(t + pi/2).
inline constexpr double kSinKPiOver32[64] = {
    0x0.0000000000000p+0,  0x1.917a6bc29b42cp-4,  0x1.8f8b83c69a60bp-3,  0x1.294062ed59f06p-2,  0x1.87de2a6aea963p-2,
    0x1.e2b5d3806f63bp-2,  0x1.1c73b39ae68c8p-1,  0x1.44cf325091dd6p-1,  0x1.6a09e667f3bcdp-1,  0x1.8bc806b151741p-1,
    0x1.a9b66290ea1a3p-1,  0x1.c38b2f180bdb1p-1,  0x1.d906bcf328d46p-1,  0x1.e9f4156c62ddap-1,  0x1.f6297cff75cbp-1,
    0x1.fd88da3d12526p-1,  0x1.0000000000000p+0,  0x1.fd88da3d12526p-1,  0x1.f6297cff75cbp-1,   0x1.e9f4156c62ddap-1,
    0x1.d906bcf328d46p-1,  0x1.c38b2f180bdb1p-1,  0x1.a9b66290ea1a3p-1,  0x1.8bc806b151741p-1,  0x1.6a09e667f3bcdp-1,
    0x1.44cf325091dd6p-1,  0x1.1c73b39ae68c8p-1,  0x1.e2b5d3806f63bp-2,  0x1.87de2a6aea963p-2,  0x1.294062ed59f06p-2,
    0x1.8f8b83c69a60bp-3,  0x1.917a6bc29b42cp-4,  0x0.0000000000000p+0,  -0x1.917a6bc29b42cp-4, -0x1.8f8b83c69a60bp-3,
    -0x1.294062ed59f06p-2, -0x1.87de2a6aea963p-2, -0x1.e2b5d3806f63bp-2, -0x1.1c73b39ae68c8p-1, -0x1.44cf325091dd6p-1,
    -0x1.6a09e667f3bcdp-1, -0x1.8bc806b151741p-1, -0x1.a9b66290ea1a3p-1, -0x1.c38b2f180bdb1p-1, -0x1.d906bcf328d46p-1,
    -0x1.e9f4156c62ddap-1, -0x1.f6297cff75cbp-1,  -0x1.fd88da3d12526p-1, -0x1.0000000000000p+0, -0x1.fd88da3d12526p-1,
    -0x1.f6297cff75cbp-1,  -0x1.e9f4156c62ddap-1, -0x1.d906bcf328d46p-1, -0x1.c38b2f180bdb1p-1, -0x1.a9b66290ea1a3p-1,
    -0x1.8bc806b151741p-1, -0x1.6a09e667f3bcdp-1, -0x1.44cf325091dd6p-1, -0x1.1c73b39ae68c8p-1, -0x1.e2b5d3806f63bp-2,
    -0x1.87de2a6aea963p-2, -0x1.294062ed59f06p-2, -0x1.8f8b83c69a60bp-3, -0x1.917a6bc29b42cp-4,
};

// Digits of 32/pi, as an unevaluated sum of doubles.
inline constexpr double kThirtyTwoOverPi[5] = {0x1.45f306dc9c883p+3,
                                               -0x1.6b01ec5417056p-51,
                                               -0x1.6447e493ad4cep-105,
                                               0x1.e21c820ff28b2p-159,
                                               -0x1.508510ea79237p-214};

// Round to nearest, ties to even, INDEPENDENT of the current FP rounding mode.
// The independence is load-bearing: the reduction below assumes |y| <= 0.5, and
// a mode-following rint() would return a k one off under FE_UPWARD/FE_DOWNWARD,
// evaluating the polynomials outside the interval they are fitted on. Unlike
// llvm-libc this does not #error without SSE4.1 -- the generic arm is a real
// fallback, because this header is also compiled for CPU_CAPABILITY_DEFAULT.
inline double sin_nearest_integer(double x) {
#if defined(__x86_64__) && defined(__SSE4_1__)
  __m128d v = _mm_set_sd(x);
  return _mm_cvtsd_f64(_mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#elif defined(__aarch64__) && defined(__ARM_FP)
  double r;
  __asm__("frintn %d0, %d1" : "=w"(r) : "w"(x));
  return r;
#else
  // Add-and-subtract 2^52 lands the value in the binade whose ULP is 1, so the
  // add rounds to an integer; the residual test repairs a non-default mode.
  if (x < 0x1p53 && x > -0x1p53) {
    double r = x < 0 ? (x - 0x1.0p52) + 0x1.0p52 : (x + 0x1.0p52) - 0x1.0p52;
    double diff = x - r;
    if (C10_UNLIKELY(diff > 0.5))
      return r + 1.0;
    if (C10_UNLIKELY(diff < -0.5))
      return r - 1.0;
    return r;
  }
  return x;
#endif
}

inline double sin_clear_low_bits(double v, uint64_t mask) {
  uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  u &= mask;
  std::memcpy(&v, &u, sizeof v);
  return v;
}

// k = round(x * 32/pi), y = x * 32/pi - k, with |y| <= 0.5.
inline int64_t sin_reduce_small(double x, double& y) {
  double kd = sin_nearest_integer(x * kThirtyTwoOverPi[0]);
  y = std::fma(x, kThirtyTwoOverPi[0], -kd);
  y = std::fma(x, kThirtyTwoOverPi[1], y);
  return static_cast<int64_t>(kd);
}

// Same, for |x| >= 2^45, where the leading digits of 32/pi no longer suffice.
// Only the low 6 unit bits of x * 32/pi matter, because sin is 2pi-periodic and
// 64 steps of pi/32 is exactly 2pi, so the higher digits are discarded.
inline int64_t sin_reduce_large(double x, int x_exp, double& y) {
  if (x_exp < 99) {
    double prod_hi = sin_clear_low_bits(x * kThirtyTwoOverPi[0], (x_exp < 55) ? ~0xfffULL : ~0ULL);
    double k_hi = sin_nearest_integer(prod_hi);
    double truncated = std::fma(x, kThirtyTwoOverPi[0], -k_hi);
    double k_lo = sin_nearest_integer(std::fma(x, kThirtyTwoOverPi[1], truncated));
    y = std::fma(x, kThirtyTwoOverPi[1], truncated - k_lo);
    y = std::fma(x, kThirtyTwoOverPi[2], y);
    y = std::fma(x, kThirtyTwoOverPi[3], y);
    return static_cast<int64_t>(k_lo);
  }
  double prod_hi = sin_clear_low_bits(x * kThirtyTwoOverPi[1], (x_exp < 110) ? ~0xfffULL : ~0ULL);
  double k_hi = sin_nearest_integer(prod_hi);
  double truncated = std::fma(x, kThirtyTwoOverPi[1], -k_hi);
  double k_lo = sin_nearest_integer(std::fma(x, kThirtyTwoOverPi[2], truncated));
  y = std::fma(x, kThirtyTwoOverPi[2], truncated - k_lo);
  y = std::fma(x, kThirtyTwoOverPi[3], y);
  y = std::fma(x, kThirtyTwoOverPi[4], y);
  return static_cast<int64_t>(k_lo);
}

// PRECONDITION: |x| > 0x1p20f, or x is Inf/NaN.
inline float sin_large_scalar(float x) {
  uint32_t x_bits;
  std::memcpy(&x_bits, &x, sizeof x_bits);
  const uint32_t x_abs = x_bits & 0x7fffffffU;

  if (C10_UNLIKELY(x_abs >= 0x7f800000U)) {
    return x + std::numeric_limits<float>::quiet_NaN();
  }

  const double xd = static_cast<double>(x);
  double y;
  const int64_t k = (x_abs < 0x56000000U) // 2^45
      ? sin_reduce_small(xd, y)
      : sin_reduce_large(xd, static_cast<int>((x_abs >> 23) & 0xff) - 127, y);

  // sin(x) = sin((k + y)*pi/32)
  //        = sin(y*pi/32)*cos(k*pi/32) + cos(y*pi/32)*sin(k*pi/32)
  //        = (y*cos_k)*P + ((y^2*sin_k)*Q + sin_k)
  // for sin(t) = y*P and cos(t)-1 = y^2*Q, t = y*pi/32. Both table products
  // are formed BEFORE the polynomials rather than after: y and the table
  // entries are ready while P and Q are still evaluating, so this takes an
  // fmul off each branch of the dependency chain at the same operation count.
  const double sin_k = kSinKPiOver32[k & 63];
  const double cos_k = kSinKPiOver32[(k + 16) & 63];
  const double ysq = y * y;
  const double y_cos_k = y * cos_k;
  const double ysq_sin_k = ysq * sin_k;

  // Taylor, truncated to what a float result can see -- NOT to what upstream
  // needed. |y| <= 0.5 bounds t by pi/64, where the next term of each series
  // is worth 2^-38.4 (sin) and 2^-35.6 (cos) of the result. The t^4 cosine
  // term below does not pass the same test: it is worth 2^-22, about 4 ULP.
  const double P = std::fma(ysq, std::fma(ysq, 0x1.466bc624f2776p-24, -0x1.4abbce625abb1p-13), 0x1.921fb54442d18p-4);
  const double Q = std::fma(ysq, 0x1.03c1f070c2e27p-18, -0x1.3bd3cc9be430bp-8);

  return static_cast<float>(std::fma(y_cos_k, P, std::fma(ysq_sin_k, Q, sin_k)));
}
// ---------------------------------------------------------------------------

// Max ULP: 1.89
template <bool CheckRange>
Vectorized<float> sinimpl(Vectorized<float> x) {
  // pi as an unevaluated sum of three floats. n * kPi1 is exact for every n
  // the reduction produces -- both are multiples of ulp(kPi1) and their
  // difference is below 2 -- so only the kPi2 and kPi3 steps round. A fourth
  // pi term is not worth adding: what limits r is those two roundings, not
  // truncation of pi, and adding one changes the max ULP by exactly zero.
  const Vf invPi(0x1.45f306p-2f);
  const Vf kPi1(0x1.921fb6p+1f);
  const Vf kPi2(-0x1.777a5cp-24f);
  const Vf kPi3(-0x1.ee59dap-49f);
  // 1.5*2^23 forces the sum into the binade whose ULP is 1, so this fmadd
  // rounds x/pi straight to an integer and n recovers it exactly. |x| < 2^20
  // keeps |x*invPi| under 2^19, well inside that binade -- the same bound the
  // range check below enforces, so the two are not independent.
  const Vf shift(0x1.8p+23f);
  const Vf z = fmadd(x, invPi, shift);
  const Vf n = z - shift;

  // With this shift the bit pattern of z is exactly 0x4B400000 + n over the
  // whole valid range. Only the parity of n matters -- sin(x) = (-1)^n *
  // sin(x - n*pi) -- and that is just the low bit of z, so no float-to-int
  // convert is needed.
  const Vf sign = cast<float>(cast<int32_t>(z) << Vi(31));

  Vf r = fnmadd(n, kPi1, x);
  r = fnmadd(n, kPi2, r);
  r = fnmadd(n, kPi3, r);

  const Vf r2 = r * r;
  const Vf r3 = r2 * r;

  // sin(r) = r + r^3 * P(r^2) over |r| <= 1.6006, the widest reduced argument
  // this reduction can produce. Rounding the product once inside the fmadd
  // above, with ties to even, gives a slightly wider r than rounding it to a
  // float first and then to an integer with ties away from zero would; these
  // coefficients are fitted to that wider interval and are NOT interchangeable
  // with a set fitted to the narrower one.
  //
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel some of its rounding. Retune (do
  // not reuse) if the reduction or the evaluation order changes -- a plain
  // minimax fit of the same degree scores 2.60 here, and this reduction paired
  // with coefficients fitted to the narrower interval scores 2.17. The tuned
  // search is also multi-modal, so a local search seeded at one good point
  // will report convergence well short of the best available set.
  //
  // Degree is already at the sweet spot: dropping a term costs 21.7 ULP,
  // adding one buys 0.02 for an extra fma, and a further one buys nothing at
  // all. Approximation error is a small part of the total; the rest is
  // rounding in the evaluation, which no choice of coefficient can remove.
  Vf y = fmadd(r2, Vf(0x1.5d1d5ep-19f), Vf(-0x1.9f67c2p-13f));
  y = fmadd(r2, y, Vf(0x1.110ecap-7f));
  y = fmadd(r2, y, Vf(-0x1.55554ep-3f));
  // r enters through the addend, unrounded, so the dominant term costs no
  // rounding of its own and only one rounding lands at result magnitude.
  y = fmadd(r3, y, r);

  y = y ^ sign;

  // sin(-0) is -0, and the reduction cannot carry that through: n is zero, and
  // the correction steps then compute -0 - (-0) == +0 because kPi2 and kPi3
  // are negative. x is the answer for both zeros, and this is the only input
  // where the result is zero at all -- the error bound above rules out r
  // collapsing to zero for any other x -- so no other case is disturbed.
  y = Vf::blendv(y, x, x == Vf(0.0f));

  bool inRange = allLe(x.abs(), 0x1p20f);
  if (C10_LIKELY(!CheckRange || inRange)) {
    return y;
  } else {
    // |x| > 2^20 needs a Payne-Hanek reduction: three pi terms stop carrying
    // enough of pi once n gets large, and x*invPi leaves the binade the shift
    // above relies on. Rare enough to be worth a branch, so the vector path
    // pays only the compare. Phrased as < so that Inf and NaN take the scalar
    // path too, which returns NaN for both.
    __at_align__ float xs[Vf::size()];
    __at_align__ float ys[Vf::size()];
    x.store(xs);
    y.store(ys);
    for (int i = 0; i < Vf::size(); ++i) {
      if (!(std::fabs(xs[i]) <= 0x1p20f)) {
        ys[i] = sin_large_scalar(xs[i]);
      }
    }
    return Vf::loadu(ys);
  }
}

inline Vectorized<float> Vectorized<float>::sin() const {
  return sinimpl<true>(*this);
}

// PRECONDITION: |x| > 0x1p20f, or x is Inf/NaN.
// Reuses sin's Payne-Hanek reduction and table unchanged -- only the
// reconstruction differs. Max ULP 0.5005, 0.0037% not correctly rounded.
inline float cos_large_scalar(float x) {
  uint32_t x_bits;
  std::memcpy(&x_bits, &x, sizeof x_bits);
  const uint32_t x_abs = x_bits & 0x7fffffffU;

  if (C10_UNLIKELY(x_abs >= 0x7f800000U)) {
    return x + std::numeric_limits<float>::quiet_NaN();
  }

  const double xd = static_cast<double>(x);
  double y;
  const int64_t k = (x_abs < 0x56000000U) // 2^45
      ? sin_reduce_small(xd, y)
      : sin_reduce_large(xd, static_cast<int>((x_abs >> 23) & 0xff) - 127, y);

  // cos(x) = cos((k + y)*pi/32)
  //        = cos_k*cos(y*pi/32) - sin_k*sin(y*pi/32)
  //        = (y^2*cos_k)*Q + (-(y*sin_k)*P + cos_k)
  // for sin(t) = y*P and cos(t)-1 = y^2*Q, t = y*pi/32. cos_k is the sin
  // table read at k + 16, exactly as in sin_large_scalar, and the table
  // products are formed ahead of the polynomials for the reason given there.
  const double sin_k = kSinKPiOver32[k & 63];
  const double cos_k = kSinKPiOver32[(k + 16) & 63];
  const double ysq = y * y;
  const double ysq_cos_k = ysq * cos_k;
  const double y_sin_k = y * sin_k;

  const double P = std::fma(ysq, std::fma(ysq, 0x1.466bc624f2776p-24, -0x1.4abbce625abb1p-13), 0x1.921fb54442d18p-4);
  const double Q = std::fma(ysq, 0x1.03c1f070c2e27p-18, -0x1.3bd3cc9be430bp-8);

  // cos_k still enters through the addend of both FMAs, so the dominant term
  // is never rounded on its own; near the zeros of cos it is the small one.
  return static_cast<float>(std::fma(ysq_cos_k, Q, std::fma(-y_sin_k, P, cos_k)));
}

// Max ULP: 1.98
template <bool CheckRange>
Vectorized<float> cosimpl(Vectorized<float> x) {
  // Same three-float pi as sinimpl: n * kPi1 is exact for every n this
  // reduction produces, so only the kPi2 and kPi3 steps round. A fourth term
  // changes the max ULP by exactly zero -- what limits r is those roundings,
  // not truncation of pi.
  const Vf invPi(0x1.45f306p-2f);
  const Vf kPi1(0x1.921fb6p+1f);
  const Vf kPi2(-0x1.777a5cp-24f);
  const Vf kPi3(-0x1.ee59dap-49f);
  const Vf half(0x1p-1f);
  const Vf shift(0x1.8p+23f);

  // cos(x) = (-1)^n * sin(x - (n - 1/2)*pi) with n = round(x/pi + 1/2).
  // Reducing about the half-integers is what keeps the polynomial on sin:
  // reducing to multiples of pi would need cos, which cancels catastrophically
  // as r approaches +-pi/2.
  const Vf t = fmadd(x, invPi, half);

  // Round to nearest, ties away from zero: add copysign(0.5, t), truncate.
  // NOT Vectorized<float>::round() -- that is ties-to-even on every
  // capability, and ties-to-even widens the reduced argument to 1.662, past
  // the interval these coefficients are fitted on, which costs 1.7 ULP.
  //
  // The add rounds up at some t just below a midpoint, so this is not exactly
  // FRINTA (they differ on 0.55% of inputs). That is harmless: picking the
  // next n gives the equally valid decomposition r - pi with the parity
  // flipped, and the two negations cancel. Verified exhaustively -- max |r|,
  // the max ULP and its worst-case input are all identical either way.
  const Vf n0 = (t + ((t & Vf(-0.0f)) | half)).trunc();

  // n0 is integral with |n0| < 2^19, so this add is exact and lands it in the
  // binade whose ULP is 1: the low bit of the sum is the parity of n0, which
  // is the only part of it that matters. No float-to-int convert needed.
  const Vf sign = cast<float>(cast<int32_t>(n0 + shift) << Vi(31));

  const Vf n = n0 - half;
  Vf r = fnmadd(n, kPi1, x);
  r = fnmadd(n, kPi2, r);
  r = fnmadd(n, kPi3, r);

  const Vf r2 = r * r;
  const Vf r3 = r2 * r;
  const Vf r4 = r2 * r2;

  // sin(r) = r + r^3 * P(r^2) over |r| <= 1.5866, the widest reduced argument
  // this reduction produces -- 1% past pi/2, because t is itself rounded
  // before it is rounded to an integer.
  //
  // Estrin, not Horner: two independent chains cut the dependency depth by
  // one fmadd at the cost of forming r4. NOT a minimax fit, and NOT the same
  // coefficients a Horner arrangement wants -- these are tuned against the
  // max ULP of this exact evaluation sequence, so they also cancel some of
  // its rounding. Horner with its own tuned set reaches 1.81; the extra
  // rounding Estrin introduces is not recoverable by any choice of
  // coefficient. Retune (do not reuse) if the order or reduction changes:
  // the search is multi-modal and the optimum sits thousands of coefficient
  // ulps out in c2/c3, far outside any local perturbation.
  const Vf a = fmadd(r2, Vf(0x1.110edap-7f), Vf(-0x1.555552p-3f));
  const Vf b = fmadd(r2, Vf(0x1.5c95e8p-19f), Vf(-0x1.9f615p-13f));
  const Vf p = fmadd(r4, b, a);
  // r enters through the addend, unrounded, so the dominant term costs no
  // rounding of its own and only one rounding lands at result magnitude.
  Vf y = fmadd(r3, p, r);

  y = y ^ sign;

  // No zero fixup: cos is even and cos(+-0) already comes out exactly 1.0,
  // unlike sin, whose signed zero the reduction cannot carry through.

  bool inRange = allLe(x.abs(), 0x1p20f);
  if (C10_LIKELY(!CheckRange || inRange)) {
    return y;
  } else {
    // |x| > 2^20 needs a Payne-Hanek reduction: three pi terms stop carrying
    // enough of pi once n gets large. Rare enough to be worth a branch, so
    // the vector path pays only the compare. Phrased as <= so that Inf and
    // NaN take the scalar path too, which returns NaN for both.
    __at_align__ float xs[Vf::size()];
    __at_align__ float ys[Vf::size()];
    x.store(xs);
    y.store(ys);
    for (int i = 0; i < Vf::size(); ++i) {
      if (!(std::fabs(xs[i]) <= 0x1p20f)) {
        ys[i] = cos_large_scalar(xs[i]);
      }
    }
    return Vf::loadu(ys);
  }
}

inline Vectorized<float> Vectorized<float>::cos() const {
  return cosimpl<true>(*this);
}

inline Vectorized<float> Vectorized<float>::copysign(const Vectorized<float>& y) const {
  return this->abs() | (y & Vf(-0.0f));
}

// PRECONDITION: |x| > 0x1p16f, or x is Inf/NaN.
// Reuses sin's Payne-Hanek reduction and table unchanged -- only the
// reconstruction differs. Max ULP 0.5004, 0.0007% not correctly rounded.
inline float tan_large_scalar(float x) {
  uint32_t x_bits;
  std::memcpy(&x_bits, &x, sizeof x_bits);
  const uint32_t x_abs = x_bits & 0x7fffffffU;

  if (C10_UNLIKELY(x_abs >= 0x7f800000U)) {
    return x + std::numeric_limits<float>::quiet_NaN();
  }

  const double xd = static_cast<double>(x);
  double y;
  const int64_t k = (x_abs < 0x56000000U) // 2^45
      ? sin_reduce_small(xd, y)
      : sin_reduce_large(xd, static_cast<int>((x_abs >> 23) & 0xff) - 127, y);

  const double sin_k = kSinKPiOver32[k & 63];
  const double cos_k = kSinKPiOver32[(k + 16) & 63];
  const double ysq = y * y;

  // Truncated to what a float result can see, as in sin_large_scalar. NOT
  // reassociated the way sin and cos are: they fold the table product into
  // the outer FMA because each polynomial feeds one reconstruction, but here
  // each feeds two, so hoisting duplicates the multiplies rather than
  // removing them -- 1.09x against 1.16x for this form.
  const double sin_y =
      y * std::fma(ysq, std::fma(ysq, 0x1.466bc624f2776p-24, -0x1.4abbce625abb1p-13), 0x1.921fb54442d18p-4);
  const double cosm1_y = ysq * std::fma(ysq, 0x1.03c1f070c2e27p-18, -0x1.3bd3cc9be430bp-8);

  // Same two reconstructions sin_large_scalar and cos_large_scalar use, then
  // one double divide. Near a pole the quotient is large and the numerator
  // carries it; near a zero of tan the denominator does. Both are formed in
  // double from a reduction good to well past float precision, so the divide
  // has 29 bits of headroom over the 24 the result needs.
  const double s = std::fma(sin_y, cos_k, std::fma(cosm1_y, sin_k, sin_k));
  const double c = std::fma(cos_k, cosm1_y, std::fma(-sin_k, sin_y, cos_k));
  return static_cast<float>(s / c);
}

// Max ULP: 1.96
template <bool CheckRange>
Vectorized<float> tanimpl(Vectorized<float> x) {
  // -pi/2 as an unevaluated sum of three floats, negated so that every
  // reduction step below is an fmadd. A fourth term changes the max ULP by
  // exactly zero, here as in sinimpl: what limits the reduced argument is the
  // rounding, not truncation of pi.
  const Vf invPi2(0x1.45f306p-1f); // 2/pi
  const Vf kPi1(-0x1.921fb6p+0f);
  const Vf kPi2(0x1.777a5cp-25f);
  const Vf kPi3(0x1.ee59dap-50f);
  // 1.5*2^23 forces the sum into the binade whose ULP is 1, so this fmadd
  // rounds x*2/pi straight to an integer and n recovers it exactly.
  const Vf shift(0x1.8p+23f);
  const Vf one(1.0f);

  const Vf z = fmadd(x, invPi2, shift);
  const Vf n = z - shift;

  // tan(x) is tan(r) on even quadrants and -cot(r) on odd ones, so only the
  // parity of n matters. With this shift that is the low bit of z's bit
  // pattern; shifting it up and back down arithmetically broadcasts it to a
  // full mask, so no float-to-int convert is needed.
  const Vf odd = cast<float>((cast<int32_t>(z) << Vi(31)) >> Vi(31));

  // r + rlo = x - n*(pi/2) as an unevaluated sum.
  //
  // r1 is EXACT, which is the whole reason a single Fast2Sum suffices here
  // instead of a full double-float reduction: n*kPi1 is a multiple of 2^-23
  // (kPi1 has exponent 0), x is a multiple of 2^-23 or coarser whenever n != 0
  // (n != 0 implies |x| >= 0.785), and |r1| < 1, so the exact difference needs
  // at most 24 bits. Only the kPi2 and kPi3 steps round.
  //
  // p1l, the exact residual of the n*kPi2 product, and the kPi3 term are both
  // load-bearing -- dropping either costs ~750 ULP, not a fraction of one.
  const Vf r1 = fmadd(n, kPi1, x);
  const Vf p1 = n * kPi2;
  const Vf p1l = fmsub(n, kPi2, p1);
  const Vf r = r1 + p1;
  Vf rlo = (r1 - r) + p1;
  rlo = rlo + p1l;
  rlo = fmadd(n, kPi3, rlo);

  // tan is odd, so tan(r) is evaluated unsigned and the quadrant sign is
  // folded into the reciprocal below. That is three instructions cheaper than
  // negating r and rlo up front, and bit-identical.
  const Vf r2 = r * r;
  const Vf r4 = r2 * r2;
  const Vf r8 = r4 * r4;
  // tan(r) = r + r^3 * P(r^2) over |r| <= 0.78802, the widest reduced argument
  // this reduction produces for |x| < 2^16. That bound, not pi/4, is what the
  // fit interval has to be -- see the range note below.
  //
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel some of its rounding. Retune (do
  // not reuse) if the reduction, the evaluation order, the reconstruction or
  // the range threshold changes -- a plain minimax fit on the same interval
  // scores 2.25 here. Score candidates against the whole domain or a hard set:
  // the maximising inputs are rare enough that a uniform stride-64 subsample
  // misreports the score by half an ULP.
  const Vf e01 = fmadd(Vf(0x1.112ee4p-3f), r2, Vf(0x1.5554ep-2f));
  const Vf e23 = fmadd(Vf(0x1.916862p-6f), r2, Vf(0x1.b563a6p-5f));
  const Vf e45 = fmadd(Vf(0x1.35be96p-7f), r2, Vf(0x1.8a0024p-9f));
  const Vf p = fmadd(e45, r8, fmadd(e23, r4, e01));

  // tan(r + rlo) = tan(r) + rlo*sec^2(r). sec^2 only has to be right to about
  // 20% -- the correction it scales is itself under one ULP -- so 1 + r^2 is
  // enough and a closer approximation buys 0.03.
  const Vf sec2 = one + r2;
  // Both small terms are summed before r enters, so the dominant term costs no
  // rounding of its own and only one rounding lands at result magnitude.
  const Vf small = fmadd(r * r2, p, rlo * sec2);
  const Vf y = r + small;

  // Odd quadrant. A residual-corrected reciprocal, -1/y + ylo/y^2 with ylo the
  // exact Fast2Sum residual of the add above, would buy another 0.15 ULP by
  // removing the amplification of y's own rounding -- the result and y sit in
  // mismatched binades near a pole, so half an ULP of y becomes a whole ULP of
  // the answer. It costs four more instructions and ~17% throughput, and is
  // not needed to stay under 2.
  const Vf ncot = Vf(-1.0f) / y;

  Vf res = Vf::blendv(y, ncot, odd);

  // tan(-0) is -0, and the reduction cannot carry that through: n is zero, and
  // the correction steps then compute -0 + 0*kPi2 == +0 because kPi2 and kPi3
  // are positive. x is the answer for both zeros, and this is the only input
  // whose result is zero at all, so no other case is disturbed.
  res = Vf::blendv(res, x, x == Vf(0.0f));

  // The threshold is 2^16, not sin's and cos's 2^20, because the coefficients
  // above are fitted to the widest |r| this reduction produces *for this
  // range*. n = rint(x * float(2/pi)) drifts as |x| grows, so max|r| creeps
  // past pi/4 -- 0.78671 at 2^15, 0.78802 at 2^16, 0.79590 at 2^18, 0.82745 at
  // 2^20 -- and a wider range runs the polynomial past the end of its interval,
  // where a degree-5 error explodes (this set scores 12.6 at 2^20).
  //
  // That is a fit-interval limit, not a reduction limit. The reduction is flat
  // over the whole span: a correctly-rounded y from the (r,rlo) pair scores
  // 1.4954 at every range up to 2^20, and an exactly-evaluated polynomial
  // scores 1.95 to 2.05. Refitting on each range's own max|r| gives 1.84 at
  // scores 1.95 to 2.05. Refitting on each range's own max|r| gives 1.84 at
  // 2^15, 1.96 at 2^16, 2.01 at 2^17, 2.24 at 2^18 and 2.38 at 2^20, so 2^16
  // is the widest threshold that still clears 2 ULP at this degree. A fourth
  // pi term moves none of these numbers.
  bool inRange = allLe(x.abs(), 0x1p16f);
  if (C10_LIKELY(!CheckRange || inRange)) {
    return res;
  } else {
    // Phrased as <= so that Inf and NaN take the scalar path too, which
    // returns NaN for both.
    __at_align__ float xs[Vf::size()];
    __at_align__ float ys[Vf::size()];
    x.store(xs);
    res.store(ys);
    for (int i = 0; i < Vf::size(); ++i) {
      if (!(std::fabs(xs[i]) <= 0x1p16f)) {
        ys[i] = tan_large_scalar(xs[i]);
      }
    }
    return Vf::loadu(ys);
  }
}

inline Vectorized<float> Vectorized<float>::tan() const {
  return tanimpl<true>(*this);
}

// Max ULP: 1.21
inline Vectorized<float> Vectorized<float>::acos() const {
  const Vf x = *this;
  const Vf half(0.5f);
  const Vf kPiOver2(0x1.921fb6p+0f);
  const Vf kPi(0x1.921fb6p+1f);

  const Vf ax = x.abs();
  const Vf leHalf = ax <= half;

  // Two branches, one polynomial. Both go through asin(z) = z + z*w*P(w):
  //   |x| <= 1/2   w = x^2,       z = |x|
  //   |x| >  1/2   w = (1-|x|)/2, z = sqrt(w),  using
  //                acos(x) = 2*asin(sqrt((1-|x|)/2)), reflected below for x < 0.
  //
  // Splitting at exactly 1/2 is what minimises the interval the polynomial has
  // to cover: x^2 and (1-x)/2 cross there, so w <= 1/4 on both sides and any
  // other split point makes one side wider. On the sqrt branch, which is the
  // one that sets the error, w is exact -- 0.5*|x| is a scaling and the
  // subtraction is Sterbenz for |x| in [1/2, 1] -- so w costs no rounding.
  const Vf w = Vf::blendv(fnmadd(ax, half, half), ax * ax, leHalf);
  const Vf z = Vf::blendv(w.sqrt(), ax, leHalf);

  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence, so the coefficients also cancel some of its rounding. A plain
  // weighted Remez fit of the same degree scores 1.32 here. Retune (do not
  // reuse) if the split, the evaluation order or the reconstruction changes.
  // Score candidates over the whole domain or a hard set -- the maximising
  // inputs are vanishingly rare (162 of 2.1e9 inputs are above 1.25 ULP), so a
  // subsample does not see them at all.
  Vf p = fmadd(w, Vf(0x1.4397d8p-5f), Vf(0x1.ab2ddp-6f));
  p = fmadd(w, p, Vf(0x1.70aadcp-5f));
  p = fmadd(w, p, Vf(0x1.33322ap-4f));
  p = fmadd(w, p, Vf(0x1.555508p-3f));
  // z enters through the addend, unrounded, so the dominant term costs no
  // rounding of its own and only one rounding lands at result magnitude.
  p = fmadd(z * w, p, z);

  // p >= 0 by construction (z, w >= 0 and every coefficient is positive), so
  // OR-ing in the sign bit is copysign(p, x) without the abs. asin is odd, and
  // this is what lets both branches share one reconstruction below.
  const Vf y = p | (x & Vf(-0.0f));

  // acos(x) = pi/2 - sign(x)*asin(|x|)   , |x| <= 1/2
  //         = 2*asin(z)                  , x >  1/2
  //         = pi - 2*asin(z)             , x < -1/2
  // all three of which are add + mul*y once the sign lives in y.
  const Vf isNeg = x < Vf(0.0f);
  const Vf add = Vf::blendv(kPi & isNeg, kPiOver2, leHalf);
  const Vf mul = Vf::blendv(Vf(2.0f), Vf(-1.0f), leHalf);

  // No large-argument path and no special-case fixups, unlike sin/cos/tan: the
  // domain is [-1, 1], and outside it w goes negative so the sqrt already
  // produces the NaN that acos owes. acos(1) = 0 and acos(-1) = pi come out of
  // the same expression with z = 0. The only deviation from scalar acosf is the
  // sign bit of the NaN returned for x < -1, which IEEE leaves unspecified.
  return fmadd(mul, y, add);
}

// Max ULP: 1.59
inline Vectorized<float> Vectorized<float>::asin() const {
  const Vf x = *this;
  const Vf half(0.5f);
  const Vf kPiOver2(0x1.921fb6p+0f);
  const Vf kPiOver2Lo(-0x1.777a5cp-25f);

  const Vf ax = x.abs();
  const Vf ltHalf = ax < half;

  const Vf w = Vf::blendv(fnmadd(ax, half, half), ax * ax, ltHalf);
  const Vf z = Vf::blendv(w.sqrt(), ax, ltHalf);

  Vf p = fmadd(w, Vf(0x1.3cb7d8p-5f), Vf(0x1.b051dp-6f));
  p = fmadd(w, p, Vf(0x1.70cbdcp-5f));
  p = fmadd(w, p, Vf(0x1.3322bap-4f));
  p = fmadd(w, p, Vf(0x1.555588p-3f));
  p = fmadd(z * w, p, z);

  const Vf large = fnmadd(p, Vf(2.0f), kPiOver2) + kPiOver2Lo;
  const Vf y = Vf::blendv(large, p, ltHalf);
  return y | (x & Vf(-0.0f));
}

// Max ULP: 1.60
inline Vectorized<float> Vectorized<float>::atan() const {
  const Vf x = *this;
  const Vf one(1.0f);
  const Vf negOne(-1.0f);
  const Vf signBit(-0.0f);
  const Vf kPiOver2(0x1.921fb6p+0f);

  const Vf sign = x & signBit;

  // atan(x)           for |x| <= 1
  // atan(-1/x) + pi/2 for x > 1,  atan(-1/x) - pi/2 for x < -1
  // so |z| <= 1 either way and one polynomial covers both.  NaN compares
  // false here and falls through the unreduced side, where it propagates.
  const Vf red = x.abs() > one;
  const Vf z = Vf::blendv(x, negOne / x, red);
  const Vf shift = (kPiOver2 ^ sign) & red;

  const Vf z2 = z * z;
  const Vf z3 = z * z2;
  const Vf z4 = z2 * z2;
  const Vf z8 = z4 * z4;

  // atan(z) = z + z^3 * P(z^2) over the whole of |z| <= 1, which is why this
  // needs degree 7: degree 6 is 2.27 ULP of approximation error on its own,
  // and degree 8 does not help (the polynomial stops binding once the
  // reconstruction is the limit).
  //
  // Estrin, not Horner: Horner is two ops shorter but 15% slower here,
  // because a seven-deep chain does not fit in the shadow of the divide.
  //
  // NOT a minimax fit. Tuned against the max ULP of this exact evaluation
  // sequence, and in particular the coefficients absorb the bias of the
  // single-word pi/2 -- which is why this beats even an exactly evaluated
  // polynomial (1.77) and beats carrying pi/2 in two words (1.60) for free.
  // A plain fpminimax fit of the same degree scores 2.11.  Retune (do not
  // reuse) if the reduction, the evaluation order or the shift changes.
  //
  // The fit is constrained so that atan(1) stays correctly rounded: at
  // |x| = 1 the kernel collapses to fl(1 + sum(c)), so the constraint is a
  // half-ulp window on the coefficient sum, maintained during the search by
  // re-solving c7 (the finest coefficient) after every move.
  const Vf p01 = fmadd(z2, Vf(0x1.997854p-3f), Vf(-0x1.5554c8p-2f));
  const Vf p23 = fmadd(z2, Vf(0x1.b4de18p-4f), Vf(-0x1.23098p-3f));
  const Vf p45 = fmadd(z2, Vf(0x1.61fb2p-5f), Vf(-0x1.3554bap-4f));
  const Vf p67 = fmadd(z2, Vf(0x1.7ecc8ap-9f), Vf(-0x1.0c2894p-6f));
  const Vf p03 = fmadd(z4, p23, p01);
  const Vf p47 = fmadd(z4, p67, p45);
  const Vf y = fmadd(z8, p47, p03);

  // No large-argument path and no special-case fixups: atan(+-inf) = +-pi/2
  // falls out with z = -+0.
  //
  // One inherited quirk, also present in the AOR kernel this comes from:
  //   atan(-0) returns +0, not -0.  The tail computes z^3*y + (shift + z)
  //   with z = -0 and y < 0, so the product is +0 and +0 + -0 is +0.  Fixing
  //   it needs the whole function folded onto |x| with the sign reapplied at
  //   the end, which costs ~8% throughput.
  return fmadd(z3, y, shift + z);
}

// Max ULP: 1.71
inline Vectorized<float> Vectorized<float>::atan2(const Vectorized<float>& b) const {
  const Vf y = *this;
  const Vf x = b;
  const Vf signBit(-0.0f);
  const Vf one(1.0f);
  const Vf zero(0.0f);
  const Vf kPi(0x1.921fb6p+1f);
  // pi - (float)pi.  The quadrant term is a multiple of pi/2 and near the
  // worst case it partially cancels against atan(z), so the float pi's 0.37
  // ulp arrives amplified; carrying the low word is worth 0.4 ULP.
  const Vf kPiLo(-0x1.777a5cp-24f);

  const Vf sign_xy = (x ^ y) & signBit;
  const Vf ax = x.abs();
  const Vf ay = y.abs();
  const Vf aygtax = ay > ax;

  // Reduce to |z| <= 1.  den is max(|x|,|y|) and |num| is min(|x|,|y|), which
  // is what makes the two degenerate quotients cheap to spot below.
  const Vf num = Vf::blendv(ay, ax ^ signBit, aygtax);
  Vf den = Vf::blendv(ax, ay, aygtax);

  // den == 0 means both arguments are zero, i.e. 0/0.  Dividing by one
  // instead yields z = +-0, and the reconstruction then gives +-0 for
  // atan2(+-0, +0) and +-pi for atan2(+-0, -0), which is what IEEE wants.
  den = Vf::blendv(den, one, den == zero);
  Vf z = num / den;
  // num == den means both are infinite (num is then +inf and so is den), i.e.
  // inf/inf.  z = 1 gives atan(1) = pi/4, which the quadrant term turns into
  // +-pi/4 or +-3pi/4.  The test also fires when |x| == |y| is finite, where
  // the divide already produced exactly 1, so that case is a no-op.  A NaN
  // argument makes den NaN and the compare false, so NaN still propagates.
  z = Vf::blendv(z, one, num == den);

  // signbit(x), NOT x < 0.  They disagree at x = -0, where the result's sign
  // comes from the sign bit but the quadrant would not, and atan2(y, -0)
  // would come out with the wrong sign.  Same op count.
  const Vf xneg = cast<float>(cast<int32_t>(x) >> Vi(31));
  const Vf shift = (Vf(-1.0f) & xneg) + (Vf(0.5f) & aygtax);

  const Vf z2 = z * z;
  const Vf z3 = z2 * z;
  const Vf z4 = z2 * z2;
  const Vf z8 = z4 * z4;

  // atan(z) = z + z^3 * P(z^2) over the whole of |z| <= 1, hence degree 7:
  // degree 6 is 2.27 ULP of approximation error on its own, and degree 8 does
  // not help once the reconstruction is the limit.  Estrin, not Horner --
  // Horner is two ops shorter but loses on both axes here.
  //
  // NOT a minimax fit: tuned against the max ULP of this exact evaluation
  // sequence.  The fit also pins atan2(a, a) = pi/4 and atan2(a, -a) = 3pi/4,
  // which are exact for the original coefficients and would otherwise be
  // given up; at |z| = 1 every power of z is 1, so that pin is a half-ulp
  // window on the coefficient sum, held during the search by re-solving c7.
  const Vf p01 = fmadd(z2, Vf(0x1.997964p-3f), Vf(-0x1.5554e8p-2f));
  const Vf p23 = fmadd(z2, Vf(0x1.b4deb8p-4f), Vf(-0x1.230b4p-3f));
  const Vf p45 = fmadd(z2, Vf(0x1.61eefp-5f), Vf(-0x1.355042p-4f));
  const Vf p67 = fmadd(z2, Vf(0x1.7e9236p-9f), Vf(-0x1.0c15f4p-6f));
  const Vf poly = fmadd(z8, fmadd(z4, p67, p45), fmadd(z4, p23, p01));

  // Form atan(z) first and add the quadrant term after.  The original order
  // (quadrant term first, polynomial second) rounds a cancelling sum at
  // result magnitude; this order does not, and is worth 0.5 ULP -- but only
  // once the two-word pi above is in place, since otherwise the constant's
  // bias dominates and the reassociation buys exactly nothing.
  const Vf at = fmadd(z3, poly, z);
  Vf ret = fmadd(shift, kPi, at);
  ret = fmadd(shift, kPiLo, ret);
  return ret ^ sign_xy;
}

inline Vectorized<float> asinh_small(Vectorized<float> x) {
  const Vf w = x * x;
  const Vf w2 = w * w;
  const Vf w4 = w2 * w2;
  const Vf p01 = fmadd(w, Vf(0x1.332ffcp-4f), Vf(-0x1.555544p-3f));
  const Vf p23 = fmadd(w, Vf(0x1.ec9b18p-6f), Vf(-0x1.6d5dap-5f));
  const Vf p45 = fmadd(w, Vf(0x1.bdeb02p-7f), Vf(-0x1.582c68p-6f));
  const Vf p67 = fmadd(w, Vf(0x1.3bf29ep-9f), Vf(-0x1.cdaa62p-8f));
  const Vf p03 = fmadd(w2, p23, p01);
  const Vf p47 = fmadd(w2, p67, p45);
  const Vf p = fmadd(w4, fmadd(w4, Vf(-0x1.996ddep-12f), p47), p03);

  // Signed x, not |x|: asinh is odd and x + x^3*P(x^2) is odd term by term.
  // The OR is a signed-zero repair and only that -- P(0) = -1/6 is negative,
  // so at x = -0 the product is +0 and the addend's -0 is lost to +0 + -0.
  // Every additive rearrangement has that sign algebra; x * (1 + w*P) escapes
  // it but costs 0.77 -> 1.17 ULP by rounding the dominant term. For x != 0
  // the OR is a no-op, the fmadd already carries the sign.
  return fmadd(x * w, p, x) | (x & Vf(-0.0f));
}

// Max ULP: 1.57
inline Vectorized<float> Vectorized<float>::asinh() const {
  const Vf x = *this;
  const Vf one(1.0f);
  const Vf ax = x.abs();

  if (C10_LIKELY(allLe(ax, 1.0f))) {
    return asinh_small(x);
  }

  const Vf t = one + fmadd(ax, ax, one).sqrt();
  const Vf y = fmadd(ax, ax / t, ax);

  const Vf inf(std::numeric_limits<float>::infinity());
  const Vf big = cast<float>(cast<uint32_t>(ax) >= Vu(0x5f800000u));
  const Vf xy = Vf::blendv(y, ax, big);
  const Vf addend = Vf::blendv(ax, Vf(0x1.62E43p-1f), ltAbs(x, inf)) & big;
  // OR rather than XOR: log1p of a non-negative argument is non-negative, so
  // the two agree, and this shares the `x & -0.0f` that asinh_small forms.
  const Vf large = (log1pimpl<false>(xy) + addend) | (x & Vf(-0.0f));

  if (allGe(ax, 1.0f)) {
    return large;
  }
  return Vf::blendv(large, asinh_small(x), ltAbs(x, one));
}

// acosh(1+d) = sqrt(2d + d^2 * W(d)) for x in [1, 2].
//
// acosh is ill-conditioned in x near 1 -- the relative condition number is
// 1/(2d) -- but d = x - 1 is EXACT there by Sterbenz, and in d the condition
// number is 1/2.  So the whole problem goes away by never forming anything
// but d.  The shipped log1p route instead spends two roundings building
// (x-1)(x+1) before a sqrt that only halves them, which is why it reaches
// 3.2 ULP just above 1.
//
// The sqrt is LAST on purpose.  Writing this as sqrt(2d) * P(d) puts the sqrt
// one binade above the result wherever P scales across a power of two, which
// doubles its rounding cost and pins the form at 1.48 ULP no matter the
// degree.  Here the sqrt rounds in the result's own binade.  2d is exact and
// is kept out of the FMA's rounding for the same reason.
//
// NOT a minimax fit: tuned against the max ULP of this exact evaluation
// sequence.  acosh(1) = 0 needs no pin -- d = 0 drives the radicand to zero.
inline Vectorized<float> acosh_lower(Vectorized<float> x) {
  const Vf d = x - Vf(1.0f);
  const Vf d2 = d * d;
  const Vf d4 = d2 * d2;
  const Vf p01 = fmadd(d, Vf(0x1.6c15dp-4f), Vf(-0x1.555554p-2f));
  const Vf p23 = fmadd(d, Vf(0x1.49faf2p-7f), Vf(-0x1.d3e1e4p-6f));
  const Vf p45 = fmadd(d, Vf(0x1.16499p-10f), Vf(-0x1.d4912ep-9f));
  const Vf p03 = fmadd(d2, p23, p01);
  const Vf p = fmadd(d4, fmadd(d2, Vf(-0x1.718642p-13f), p45), p03);
  return fmadd(d2, p, d + d).sqrt();
}

// acosh(x) = log(x + sqrt(x^2 - 1)) for x > 2, on a private copy of logimpl.
//
// The copy exists for the reconstruction, not the polynomial.  n reaches 128
// here, so logimpl's single-word ln2 and its "add everything to n * ln2" order
// round the result once at magnitude ~88 and flatten the small terms: that
// alone costs 1.96 ULP and 25% wrong roundings.  Splitting ln2 in two is worth
// almost nothing by itself (1.90); what matters is accumulating r, n * ln2_lo
// and p * r^2 FIRST and adding the exact n * ln2_hi LAST.  ln2_hi has 9
// trailing zero mantissa bits, so n * ln2_hi is exact for every n in range.
//
// The coefficients are refit for this sequence and are NOT interchangeable
// with logimpl's -- log() keeps its own.
inline Vectorized<float> acosh_upper(Vectorized<float> x) {
  const Vf one(1.0f);
  const Vf inf(std::numeric_limits<float>::infinity());

  // At x >= 2^64 the square overflows, and there acosh(x) = log(2x) to well
  // under half an ulp. Doubling the argument is just one more in the exponent,
  // so this rides the exact n * ln2_hi term instead of a trailing + ln2, which
  // is what drops that band from 1.35 to 0.51 ULP.
  const Vf big = cast<float>(cast<uint32_t>(x) >= Vu(0x5f800000u));
  const Vf a = Vf::blendv(x + fmsub(x, x, one).sqrt(), x, big);

  const Vi u = cast<int32_t>(a) - Vi(0x3F2FE200);
  const Vf v = cast<float>((u & Vi(0x007FFFFF)) + Vi(0x3F2FE200));
  const Vf r = v - one;
  const Vf n = convert<float>(u >> Vi(23)) + (one & big);

  const Vf r2 = r * r;
  const Vf r4 = r2 * r2;
  const Vf pa = fmadd(r, Vf(0x1.555656p-2f), Vf(-0x1.ffffe6p-2f));
  const Vf pb = fmadd(r, Vf(0x1.992b8ep-3f), Vf(-0x1.000f84p-2f));
  const Vf pc = fmadd(r, Vf(0x1.2a3fcep-3f), Vf(-0x1.5124f6p-3f));
  const Vf pd = fmadd(r, Vf(0x1.c39eb2p-4f), Vf(-0x1.34ca3p-3f));
  const Vf p = fmadd(fmadd(pd, r2, pc), r4, fmadd(pb, r2, pa));

  const Vf lg = fmadd(n, Vf(0x1.62E400p-1f), fmadd(p, r2, fmadd(n, Vf(0x1.7F7D1Cp-20f), r)));

  // The reduction is integer bit surgery and does not propagate inf or NaN,
  // so carry them out here. Finite x contributes nothing.
  return lg + (x & ~(x < inf));
}

// Max ULP: 1.34
inline Vectorized<float> Vectorized<float>::acosh() const {
  const Vf x = *this;
  const Vf one(1.0f);
  const Vf two(2.0f);

  // acosh is defined on [1, inf). Phrased as x >= 1 rather than x < 1 so NaN
  // falls through to NaN as well.
  if (C10_LIKELY(allLe(x, 2.0f))) {
    return acosh_lower(x) | (x < one);
  }

  const Vf hi = acosh_upper(x);

  if (allGe(x, 2.0f)) {
    return hi;
  }
  return Vf::blendv(hi, acosh_lower(x), x < two) | (x < one);
}

inline Vectorized<float> atanh_small(Vectorized<float> x) {
  const Vf w = x * x;
  Vf p = Vf(0x1.4290bp-3f);
  p = fmadd(p, w, Vf(0x1.09f83ap-4f));
  p = fmadd(p, w, Vf(0x1.d5ce7p-4f));
  p = fmadd(p, w, Vf(0x1.24205p-3f));
  p = fmadd(p, w, Vf(0x1.999c1ep-3f));
  p = fmadd(p, w, Vf(0x1.555554p-2f));
  return fmadd(x * w, p, x);
}

// Max ULP: 1.17
inline Vectorized<float> Vectorized<float>::atanh() const {
  const Vf x = *this;
  const Vf ax = x.abs();

  // Most inputs land here, and this path never touches log1p or a divide.
  if (C10_LIKELY(allLe(ax, 0.5f))) {
    return atanh_small(x);
  }

  // atanh(x) = 1/2 * log1p(2|x| / (1 - |x|)), with the sign folded into the
  // 1/2 so the closing multiply is exact (it is a power of two). 2|x| is
  // exact and 1 - |x| is exact for every |x| >= 0.5, so the argument carries
  // only the divide's rounding.
  const Vf halfsign = Vf(0.5f) ^ (x ^ ax);
  const Vf r = (ax + ax) / (Vf(1.0f) - ax);

  // log1pimpl<true> supplies every special value on its own, so there is no
  // separate special-case path: |x| = 1 gives r = +Inf and log1p returns Inf;
  // |x| > 1 makes 1 - |x| negative, so r < -1 and log1p returns NaN; +-Inf
  // gives r = NaN, and NaN propagates. The sign then comes from halfsign.
  const Vf hi = halfsign * log1pimpl<true>(r);

  if (allGe(ax, 0.5f)) {
    return hi;
  }
  return Vf::blendv(hi, atanh_small(x), ax < Vf(0.5f));
}

inline Vectorized<float> sinh_small(Vectorized<float> x) {
  const Vf w = x * x;
  Vf p = Vf(0x1.78ab4cp-19f);
  p = fmadd(p, w, Vf(0x1.a00e62p-13f));
  p = fmadd(p, w, Vf(0x1.1110fep-7f));
  p = fmadd(p, w, Vf(0x1.555556p-3f));
  return fmadd(x * w, p, x);
}

// Max ULP: 1.55
inline Vectorized<float> Vectorized<float>::sinh() const {
  // 88.38: above this e^|x| overflows. 89.42: above this sinh itself does.
  constexpr float kExpm1Bound = 0x1.61814ap+6f;
  constexpr float kInfBound = 0x1.65a9fap+6f;

  const Vf x = *this;
  const Vf one(1.0f);
  const Vf ax = x.abs();

  if (C10_LIKELY(allLe(ax, 1.0f))) {
    return sinh_small(x);
  }

  // With t = e^|x| - 1, t + t/(t+1) = e^|x| - e^-|x| = 2 sinh(|x|). This is an
  // identity, not an approximation -- all the error here is expm1's plus these
  // three roundings. Folding the sign into the 1/2 keeps the closing multiply
  // exact.
  const Vf halfsign = Vf(0.5f) ^ (x ^ ax);
  const Vf t = ax.expm1();
  Vf hi = halfsign * (t + t / (t + one));

  // Phrased as !allLt rather than allGe so that a NaN lane also enters here:
  // a NaN is false in both directions, so with allGe the guard would be false
  // and an Inf lane sharing the vector would fall through to inf + inf/inf.
  if (C10_UNLIKELY(!allLt(ax, kExpm1Bound))) {
    // Between the two bounds e^|x| overflows but sinh does not, so reduce:
    // sinh(x) = e^x/2 = e^(|x|-S) * (e^S/2) to well under half an ulp.
    //
    // S = 8, and the constant is e^8/2 -- NOT cosh(8). AOR uses cosh(S), which
    // is larger by a factor (1 + e^-2S); at S = 9 that rounds to the same
    // float, but it is 2.79 ULP at S = 8 and overflows outright at S <= 6, so
    // using cosh(S) forces S = 9. With the correct constant S = 8 is better
    // (1.55 against 1.59). |x| - 8 is exact: both sit in the binade [64, 128).
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf big = Vf::blendv((ax - Vf(8.0f)).expm1() * Vf(0x1.749ea8p+10f), inf, ax > Vf(kInfBound)) | (x ^ ax);
    hi = Vf::blendv(hi, big, ax >= Vf(kExpm1Bound));
  }

  if (allGe(ax, 1.0f)) {
    return hi;
  }
  return Vf::blendv(hi, sinh_small(x), ax < one);
}

// Max ULP: 1.77
inline Vectorized<float> Vectorized<float>::cosh() const {
  // 86.64: e^|x| overflows above this. 89.42: cosh itself overflows above it.
  constexpr float kExpBound = 0x1.5a92d8p+6f;
  constexpr float kInfBound = 0x1.65a9fap+6f;

  const Vf one(1.0f);
  const Vf half(0.5f);
  const Vf ax = abs(); // even, so there is no sign to carry anywhere

  // e^|x| as 1 + expm1 rather than exp. expm1 is tuned to 1.29 ULP against
  // exp's 1.96, and that difference flows straight through: 2.39 -> 1.76 for
  // one extra add.
  //
  // There is deliberately NO small-|x| polynomial here, unlike sinh. cosh
  // amplifies the exponential's relative error by tanh(|x|), so near zero the
  // error is suppressed rather than exposed -- |x| <= 2^-12 measures 0.25 ULP
  // with zero wrong roundings. A series would only buy accuracy where there is
  // none to buy, and would cost a branch and a blend everywhere else.
  const Vf t = one + ax.expm1();

  // Do NOT reassociate this. Fed an exactly-rounded exp the form floors at
  // ~1.02 ULP, so it is already near optimal. Making the 1 explicit
  // (fma(0.5, e, 0.5 + 0.5/(1+e))) looks like it should save the rounding of
  // 1 + e, but at the inputs that matter 1 sits exactly at the half-ulp tie of
  // e, and the rewritten form lands on the same tie -- measured identical max
  // and a worse wrong-rounding rate, for one more op.
  Vf y = fmadd(half, t, half / t);

  // The residual 1.76 lives at |x| ~ 17, and is that same 1 + e tie. Skipping
  // the +1 above 24*ln2 = 16.64 (where 1 drops below half an ulp of e^|x|)
  // fixes it -- 1.76 -> 1.29 -- but costs a compare and a blend on every lane,
  // about 8%. Left out deliberately; add it if accuracy outranks throughput.

  // Phrased as !allLt so a NaN lane also enters here: a NaN is false in both
  // directions, and with allGe the guard would be false, leaving an Inf lane
  // sharing the vector to fall through to the fast path.
  if (C10_UNLIKELY(!allLt(ax, kExpBound))) {
    // Between the bounds e^|x| overflows but cosh does not, so reduce:
    // cosh(x) ~= e^x/2 = e^(|x|-9) * cosh(9) to well under half an ulp.
    // Unlike sinh, S = 8 with e^8/2 is WORSE here (1.5951 against 1.5870),
    // so this keeps AOR's 9. |x| - 9 is exact in the binade [64, 128).
    const Vf inf(std::numeric_limits<float>::infinity());
    const Vf big = Vf::blendv((ax - Vf(9.0f)).expm1() * Vf(0x1.fa7158p+11f), inf, ax > Vf(kInfBound));
    y = Vf::blendv(y, big, ax >= Vf(kExpBound));
  }
  return y;
}

inline Vectorized<float> tanh_small(Vectorized<float> ax) {
  const Vf w = ax * ax;
  Vf p = Vf(-0x1.c702ep-8f);
  p = fmadd(p, w, Vf(0x1.5fc862p-6f));
  p = fmadd(p, w, Vf(-0x1.b9d1f6p-5f));
  p = fmadd(p, w, Vf(0x1.11106ap-3f));
  p = fmadd(p, w, Vf(-0x1.555554p-2f));
  return fmadd(ax * w, p, ax);
}

// Max ULP: 1.50
inline Vectorized<float> Vectorized<float>::tanh() const {
  constexpr float kOneBound = 0x1.205966p+3f; // 9.01: tanh rounds to +-1 above

  const Vf x = *this;
  const Vf ax = x.abs();
  const Vf sign = x ^ ax;

  // EVERYTHING runs on |x| and the sign goes back on at the end. Two separate
  // reasons, both load-bearing -- do not "simplify" this away:
  //
  //  1. tanh = q/(q+2) amplifies q's relative error by 2/(q+2). That is <= 1
  //     while q >= 0, but rises to 2 as q -> -1, i.e. for negative x. Running
  //     on signed x measures 3.51 ULP against 1.50 here.
  //  2. P's leading coefficient is negative, so fma(x*w, P, x) with x = -0
  //     computes (-0)*(-1/3) = +0 and then +0 + -0 = +0. The sign dies in the
  //     polynomial. (The AOR kernel has exactly this bug: tanh(-0) = +0.)
  if (C10_LIKELY(allLe(ax, 0.5f))) {
    return tanh_small(ax) ^ sign;
  }

  // q = e^(2|x|) - 1, and 2|x| is exact, so expm1 sees no rounding of its own.
  const Vf q = (ax + ax).expm1();
  Vf hi = q / (q + Vf(2.0f));

  // Phrased as !allLt so a NaN lane also enters here: a NaN is false in both
  // directions, and with allGe the guard would be false, leaving an Inf lane
  // sharing the vector on q = Inf and hence Inf/(Inf+2) = NaN.
  if (C10_UNLIKELY(!allLt(ax, kOneBound))) {
    hi = Vf::blendv(hi, Vf(1.0f), ax > Vf(kOneBound));
  }
  hi = hi ^ sign;

  if (allGe(ax, 0.5f)) {
    return hi;
  }
  return Vf::blendv(hi, tanh_small(ax) ^ sign, ax < Vf(0.5f));
}

inline Vectorized<float> erf_small(Vectorized<float> ax) {
  constexpr float kChi = 0x1.20dd76p+0f; // 2/sqrt(pi), hi
  constexpr float kClo = -0x1.f7ac92p-25f; // 2/sqrt(pi), lo
  const Vf w = ax * ax;
  Vf q = Vf(0x1.5f7ba4p-14f);
  q = fmadd(q, w, Vf(-0x1.abeb98p-11f));
  q = fmadd(q, w, Vf(0x1.551656p-8f));
  q = fmadd(q, w, Vf(-0x1.b81a32p-6f));
  q = fmadd(q, w, Vf(0x1.ce2ebcp-4f));
  q = fmadd(q, w, Vf(-0x1.812746p-2f));
  return fmadd(ax, Vf(kChi), fmadd(ax * w, q, ax * Vf(kClo)));
}

// erf on (1, 3.9375], as a plain polynomial in x - 2.46875.
//
// No exp and no table. erf lies in [0.843, 1) here, so its ULP is constant at
// 2^-24 and an ordinary absolute-error minimax fit is already ULP-optimal --
// which is why this needs no weighting, unlike the exp(-x^2)*S(x) form it
// replaces (there the admissible error in S spans 6e-8 to 0.24 across the
// interval, and an unweighted fit of S wastes four degrees).
//
// Horner, not Estrin: the coefficients are tuned for this rounding order, and
// the same fit evaluated by Estrin measures 3.72 ULP. Estrin is ~15% faster
// and would need its own refit to be usable.
//
// Centred on the interval midpoint, and FIT on the normalised (x-c)/h before
// folding h back into the coefficients: fitting in d directly is ill
// conditioned enough to cost 1.16 -> 1.21 ULP.
//
// Degree 12, chosen for throughput. Degree 14 would give 1.23 ULP overall
// instead of 1.47 for 3-8% more time; degree 13 measures WORSE than 12, so
// past here the evaluation's own rounding, not the approximation, sets the
// error and the ordering stops being monotone.
inline Vectorized<float> erf_mid(Vectorized<float> ax) {
  const Vf d = ax - Vf(2.46875f);
  Vf p = Vf(-0x1.31cbb4p-17f);
  p = fmadd(p, d, Vf(0x1.9a7cp-16f));
  p = fmadd(p, d, Vf(0x1.c4fb64p-15f));
  p = fmadd(p, d, Vf(-0x1.3780b2p-12f));
  p = fmadd(p, d, Vf(0x1.819136p-12f));
  p = fmadd(p, d, Vf(0x1.0040a8p-11f));
  p = fmadd(p, d, Vf(-0x1.80a29p-9f));
  p = fmadd(p, d, Vf(0x1.b3578cp-8f));
  p = fmadd(p, d, Vf(-0x1.3aecd8p-7f));
  p = fmadd(p, d, Vf(0x1.37048p-7f));
  p = fmadd(p, d, Vf(-0x1.9bb534p-8f));
  p = fmadd(p, d, Vf(0x1.4d741cp-9f));
  p = fmadd(p, d, Vf(0x1.ffc102p-1f));
  return p;
}

// Max ULP: 1.48
inline Vectorized<float> Vectorized<float>::erf() const {
  // 3.9375 = 4 - 8/128: above it erf(|x|) rounds to 1. This clamp also covers
  // Inf, which erf_mid would otherwise turn into garbage.
  constexpr float kOneBound = 3.9375f;

  const Vf x = *this;
  const Vf ax = x.abs();
  const Vf sign = x ^ ax; // odd; both arms run on |x|

  // Branchless: both arms are always evaluated. Guarding the small-|x| arm
  // with a reduce_max is 2.9x faster when every lane is under 1, but 3%
  // slower when none are, and it makes the cost input-dependent. This form
  // runs at a flat 1.47 ns/element whatever the distribution, and is
  // bit-identical to the branched version over all 2^32 inputs.
  //
  // Lanes discarded by a blend can be Inf or NaN -- erf_small squares its
  // argument, and erf_mid is evaluated far outside its fit interval. Both
  // blends select on the INPUT, so none of that reaches the result, and
  // having no branch means no lane's result depends on its neighbours.
  //
  // Blend order matters: the 1.0f must be consumed by the SECOND blend. BSL
  // clobbers its mask operand, so putting the loop-invariant constant last
  // lets it stay in a register while the first blend works on two live
  // computed values. Same op count, same dependency depth, 8% faster.
  const Vf hi = Vf::blendv(erf_mid(ax), erf_small(ax), ax <= Vf(1.0f));
  return Vf::blendv(hi, Vf(1.0f), ax > Vf(kOneBound)) ^ sign;
}

inline Vectorized<float> erfc_small_f(Vectorized<float> ax) {
  const Vf u = fmadd(ax, Vf(2.0f), Vf(-1.0f));
  Vf p(0x1.5a061ep-20f);
  p = fmadd(p, u, Vf(0x1.e7cecp-19f));
  p = fmadd(p, u, Vf(-0x1.45bf24p-15f));
  p = fmadd(p, u, Vf(-0x1.5d9006p-15f));
  p = fmadd(p, u, Vf(0x1.99c7f4p-11f));
  p = fmadd(p, u, Vf(-0x1.e067e6p-13f));
  p = fmadd(p, u, Vf(-0x1.76f1a8p-7f));
  p = fmadd(p, u, Vf(0x1.2bf558p-6f));
  p = fmadd(p, u, Vf(0x1.c1efc8p-4f));
  p = fmadd(p, u, Vf(-0x1.c1efcap-2f));
  p = fmadd(p, u, Vf(0x1.eb0214p-2f));
  return p;
}

inline Vectorized<double> exp_neg_d(Vectorized<double> y) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;
  const Vd n = (y * Vd(-0x1.71547652b82fep+0)).round();
  Vd r = fnmsub(n, Vd(0x1.62e42fefa0000p-1), y); // -(n*ln2_hi) - y
  r = fnmadd(n, Vd(0x1.cf79abc9e3b3ap-40), r); // -(n*ln2_lo) + r
  Vd p(0x1.a01a01a01a01ap-13);
  p = fmadd(p, r, Vd(0x1.6c16c16c16c17p-10));
  p = fmadd(p, r, Vd(0x1.1111111111111p-7));
  p = fmadd(p, r, Vd(0x1.5555555555555p-5));
  p = fmadd(p, r, Vd(0x1.5555555555555p-3));
  p = fmadd(p, r, Vd(0x1.0p-1));
  p = fmadd(p, r, Vd(1.0));
  // One multiply, not the split-in-half scaling a float kernel needs: the
  // double result bottoms out near 1e-45, nowhere near double's subnormals,
  // so 2^n is always a normal double and the product is exact.
  const Vd scale = cast<double>((convert<int64_t>(n) + Vl(1023)) << Vl(52l));
  return fmadd(p, r, Vd(1.0)) * scale;
}

// erfc(|x|) = e^(-x^2) * H(|x|)/|x| on [1, 10.0625], H(x) = erfc(x)e^(x^2)x.
//
// H is fitted in x, not in 1/x or 1/x^2: H is entire in x, whereas the
// asymptotic series in 1/x^2 has an essential singularity at infinity and
// converges only algebraically (a degree-10 fit there measures 12.6 ULP).
//
// Split at 3.5 because H has a knee near x=1 and is flat by x=4; one fit
// across the whole span wastes its degree on that scale disparity. 3.5 was
// scanned, not guessed -- at this degree 3.0 gives 3.59 ULP and 4.0 gives
// 2.41.
//
// Evaluated on every lane, including |x| < 1 where it divides by a tiny
// value. That yields +-inf, never NaN, so the blend below cannot leak a bad
// lane into a good one.
inline Vectorized<double> erfc_tail_d(Vectorized<double> t, Vectorized<double> rc) {
  using Vd = Vectorized<double>;
  const Vd hi = t > Vd(3.5);
  const Vd a = Vd::blendv(Vd(0x1.999999999999ap-1), Vd(0x1.3813813813814p-2), hi);
  const Vd b = Vd::blendv(Vd(-0x1.ccccccccccccdp+0), Vd(-0x1.0888888888889p+1), hi);
  const Vd u = fmadd(t, a, b);
  Vd p = Vd::blendv(Vd(-0x1.73d353e63f1aap-15), Vd(-0x1.8481b4b793c7ap-15), hi);
  p = fmadd(p, u, Vd::blendv(Vd(0x1.ed49405c0c5c5p-14), Vd(0x1.7edb6dbe632c8p-14), hi));
  p = fmadd(p, u, Vd::blendv(Vd(-0x1.a1f3121aa341bp-13), Vd(-0x1.112a3c9d5afbep-14), hi));
  p = fmadd(p, u, Vd::blendv(Vd(0x1.118ef41f903c8p-11), Vd(0x1.31b4b5dd3188cp-13), hi));
  p = fmadd(p, u, Vd::blendv(Vd(-0x1.6a8a07e2a7049p-10), Vd(-0x1.a7f14b1e5457dp-12), hi));
  p = fmadd(p, u, Vd::blendv(Vd(0x1.a48290fc3a97fp-9), Vd(0x1.8f0ef0313ffb1p-11), hi));
  p = fmadd(p, u, Vd::blendv(Vd(-0x1.c949d5a0206c2p-8), Vd(-0x1.634e673ebc0f7p-10), hi));
  p = fmadd(p, u, Vd::blendv(Vd(0x1.cf20a80cb87e4p-7), Vd(0x1.3829180eca803p-9), hi));
  p = fmadd(p, u, Vd::blendv(Vd(-0x1.a67bf3b1d7edfp-6), Vd(-0x1.fda0fe08d5741p-9), hi));
  p = fmadd(p, u, Vd::blendv(Vd(0x1.479b17b130c14p-5), Vd(0x1.6d92f5203a389p-8), hi));
  p = fmadd(p, u, Vd::blendv(Vd(0x1.0a3668041da65p-1), Vd(0x1.1dd24de0f8b4fp-1), hi));
  return exp_neg_d(t * t) * (p * rc);
}

// Max ULP: 1.77
inline Vectorized<float> Vectorized<float>::erfc() const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(VectorizedN<double, kR>::size() == Vf::size());
  static_assert(kR == 2); // the tail below is written out for both halves

  using Vd = Vectorized<double>;
  const Vf x = *this;
  const Vf ax = x.abs();

  // Both arms run on |x|, so the double section carries no sign work at all.
  //
  // erfc(-x) = 2 - erfc(x) is therefore applied in float, after the narrow.
  // That is a second rounding, but a cheap one HERE: on this arm erfc(|x|)
  // <= 0.157, so rounding it costs 0.063 ULP against a result near 2. It
  // measures 80169 extra wrong roundings out of 4.28e9 and no change in max
  // ULP, for no measurable time either way.
  //
  // Do NOT copy this to a kernel whose small arm is also in double: there the
  // same move costs 51404380 extra wrong roundings (0.17% -> 1.37%), because
  // the operand runs up to 1.0 rather than 0.157 and Sterbenz never applies.

  // This is the ONLY part of the tail that survives in float. Dropping the H
  // polynomial to float measures 2.34 ULP, the exp Taylor 2.24, both of them
  // 3.51 -- against a 2.0 budget. Every factor that multiplies into the
  // result contributes its full relative error, and float's is too coarse.
  const Vf rcf = Vf(1.0f) / ax;

  const VectorizedN<double, kR> ad = convert<double, kR, float, 1>(ax);
  const VectorizedN<double, kR> rd = convert<double, kR, float, 1>(rcf);
  const Vf sp = erfc_small_f(ax);
  // Written out rather than looped, for the reason i0 and digamma document:
  // a RUNTIME index into VectorizedN forces its backing array out of
  // registers. Only 1.8% here -- clang unrolls a two-trip loop by itself, so
  // the index is already constant-folded -- but it is free and it stops the
  // one remaining instance of the pattern from being copied.
  VectorizedN<double, kR> td;
  td[0] = erfc_tail_d(ad[0], rd[0] * (Vd(2.0) - ad[0] * rd[0]));
  td[1] = erfc_tail_d(ad[1], rd[1] * (Vd(2.0) - ad[1] * rd[1]));

  const Vf tp = convert<float, 1, double, kR>(td);

  const Vf neg = x < Vf(0.0f); // false for -0.0, which is what erfc wants

  Vf y = Vf::blendv(tp, sp, ax < Vf(1.0f));

  y = Vf::blendv(y, Vf(2.0f) - y, neg);

  // Branchless. NaN reaches neither compare, falls through to the tail arm
  // and propagates -- verified on all 16777214 NaN patterns.

  return Vf::blendv(y, Vf(2.0f) & neg, ax > Vf(10.0625f));
}

// erf(|x|) for erfinv's Newton residual ONLY, specialised to the range
// erfinv's polynomial start can actually supply on the lanes whose result
// survives: it is consumed only where |y| <= 0.7, so |x| <= erfinv(0.7) =
// 0.7329. Lanes outside that range are overwritten at the end of erfinv, so
// this does not need to be right there.
//
// Same shape as erf_small, but the correction polynomial is refitted on the
// w = x^2 range this needs, w <= 0.5371. That reaches the same 2.86e-8 one
// degree lower -- at this width the float coefficients, not the degree, set
// the error. erf()'s other arm (erf_mid, |x| > 1) and its |x| > 3.9375
// saturation are both unreachable here, so neither is evaluated.
//
// The coefficients are then walked by integer ulps against erfinv's own
// measured max ULP rather than against this function's fit error. This is the
// only error source on the erf arm that is not a final rounding, and that arm
// amplifies its RELATIVE error by ya/(|x| erf'(|x|)), which reaches 1.45 at
// the ya = 0.7 crossover -- so this function, not the start and not the
// exponential, is what sets erfinv's max ULP. The walk is worth 1.80 -> 1.76.
// Degree is at its floor: dropping a coefficient does not refit back.
//
// w is a parameter because erfinv already needs x^2 for the exponential.
inline Vf erf_small_erfinv(Vf ax, Vf w) {
  constexpr float kChi = 0x1.20dd76p+0f; // 2/sqrt(pi), hi
  constexpr float kClo = -0x1.f7acc2p-25f; // 2/sqrt(pi), lo
  Vf q = Vf(-0x1.73ea08p-11f);
  q = fmadd(q, w, Vf(0x1.520a52p-8f));
  q = fmadd(q, w, Vf(-0x1.b7f8e4p-6f));
  q = fmadd(q, w, Vf(0x1.ce2e48p-4f));
  q = fmadd(q, w, Vf(-0x1.812746p-2f));
  return fmadd(ax, Vf(kChi), fmadd(ax * w, q, ax * Vf(kClo)));
}

// K(x) = sqrt(pi)/2 * erfc(x) * e^(x^2), on the whole range erfinv's erfc arm
// can ask for, [erfinv(0.7), erfinv(1-2^-24)] = [0.7329, 3.8325].
//
// K is what erfinv's erfc arm actually wants -- see the Newton step there --
// so no erfc value is ever formed, and with it go the general erfc's double
// tail, its separate small-|x| arm, the blend between them and the divide by
// x. K is entire in x (erfc and e^(x^2) both are), so one polynomial covers
// the span that erfc itself needs split at 1 and again at 3.5.
//
// What reaches the result is the ABSOLUTE error of K, so the fit weight is
// 1/x: at the top of the range the relative accuracy needed is x/K = 30 times
// looser than at the bottom. The coefficients are then walked by integer ulps
// against erfinv's measured erfc-arm max ULP. That objective is not
// interchangeable with K's own fit error, and minimising the fit error is
// actively wrong: doing that at degree 13 halves the weighted fit error and
// moves the measured arm the wrong way, 1.60 -> 1.73 on the same inputs,
// because what lands on the result is the fit error convolved with the
// exponential's and the final rounding. Walking it end to end is affordable
// because |y| in (0.7, 1) is only 5.03M inputs, so every candidate is scored
// on every input it can ever see.
//
// Degree 11 tuned that way measures 1.67 ulp on the arm against 1.89 for a
// plain degree-13 minimax. Degree 10 cannot get there (2.32).
inline Vf erfcx_erfinv(Vf ax) {
  const Vf t = fmadd(ax, Vf(0x1.4a5ba4p-1f), Vf(-0x1.790d8cp+0f));
  Vf p(-0x1.f18922p-14f);
  p = fmadd(p, t, Vf(0x1.e4630cp-13f));
  p = fmadd(p, t, Vf(-0x1.dd65ep-13f));
  p = fmadd(p, t, Vf(0x1.6b619cp-11f));
  p = fmadd(p, t, Vf(-0x1.f440eep-10f));
  p = fmadd(p, t, Vf(0x1.09d7c6p-8f));
  p = fmadd(p, t, Vf(-0x1.16a3b4p-7f));
  p = fmadd(p, t, Vf(0x1.1fc5b4p-6f));
  p = fmadd(p, t, Vf(-0x1.1ceddep-5f));
  p = fmadd(p, t, Vf(0x1.0dbe5p-4f));
  p = fmadd(p, t, Vf(-0x1.e63d6cp-4f));
  p = fmadd(p, t, Vf(0x1.9e3b84p-3f));
  return p;
}

// scale * e^x for erfinv's Newton step, where x = |erfinv(y)|^2 is in
// [0, 14.68] and scale is a compile-time constant.
//
// Two things that general-purpose expimpl cannot assume hold here, and the
// erfc arm needs both: n is in [0, 22], so 2^n needs neither the two-way
// split nor a clamp, and folding scale into that exact power of two is exact,
// which removes the rounding a separate multiply by scale would cost. The
// core carries one term more than expimpl's, whose degree 5 measures 3.14 ulp
// over this argument range against 1.46 here; erfinv's erfc arm multiplies
// that error by K/|x|, up to 0.62, so expimpl's would be 1.95 ulp on its own.
// The leading coefficient is exactly 1, so the r term is exact.
//
// Lanes with |y| >= 1 reach here with a garbage or infinite x and leave with
// a garbage scale; they are overwritten at the end of erfinv, and no lane's
// result depends on its neighbours.
inline Vf exp_erfinv(Vf x, Vf scale) {
  const Vf shift(0x1.8p+23f);
  const Vf z = fmadd(x, Vf(0x1.715476p+0f), shift);
  const Vf n = z - shift;
  const Vi ni = cast<int32_t>(z) - Vi(0x4B400000);
  const Vf s = cast<float>((ni + Vi(127)) << Vi(23)) * scale;
  const Vf r_hi = fnmadd(n, Vf(0x1.62E400p-1f), x);
  const Vf r = fnmadd(n, Vf(0x1.7F7D1Cp-20f), r_hi);
  const Vf r2 = r * r;
  const Vf b = fmadd(r, Vf(0x1.555482p-3f), Vf(0x1.fffffcp-2f));
  const Vf c = fmadd(r, Vf(0x1.123ae6p-7f), Vf(0x1.55582ep-5f));
  const Vf y = fmadd(fmadd(fmadd(Vf(0x1.6a87bp-10f), r2, c), r2, b), r2, r);
  return fmadd(y, s, s);
}

// erfinv(y): a polynomial start in w = -log((1-y)(1+y)), then ONE Newton step.
//
// NOT a port of ATen's calc_erfinv (Math.h). That one measures 127 ULP over
// all 2^32, and vectorising it faithfully measures 271158. The failure is
// structural: its Newton step forms std::erf(x) - y, and as |y| -> 1 both
// terms approach 1, so the subtraction is catastrophic cancellation -- at
// y = 0.99999994 the true difference is 6e-8, the same size as erf's own
// rounding. libm's erff happens to return exactly y for most of those
// inputs, so the scalar silently keeps its initial estimate; any other erf
// returns something slightly different and the correction, divided by
// erf'(x) = 7.7e-7, lands 0.11 away from the root.
//
// Fixed here by computing the tail residual as (1-|y|) - erfc(|x|), which is
// the same quantity with no cancellation: 1-|y| is exact by Sterbenz for
// |y| >= 0.5, and erfc is small and relatively accurate there.
//
// Max ULP: 1.76 (against 127.16 for the shipped scalar), 11.69% !=CR.

inline Vectorized<float> Vectorized<float>::erfinv() const {
  constexpr float kSqrtPiOver2 = 0.886226925452758013649083741670572591f;
  const Vf y = *this;

  // w = -log((1-y)(1+y)). Written as the product, not 1-y*y: both factors
  // are exact-ish and the cancellation of 1 - y^2 near |y| = 1 disappears.
  //
  // On y, not |y|: negating y exchanges the two factors, and each is the same
  // rounding either way, so the product is bit for bit what (1-|y|)(1+|y|)
  // gives -- but the abs stays off the log's dependency chain, and the log is
  // the long pole here.
  const Vf f1 = Vf(1.0f) - y;
  const Vf f2 = Vf(1.0f) + y;
  const Vf w = -logimpl<false, false>(f1 * f2);

  // erfinv(y)/y is smooth in w below 5 and in sqrt(w) above it, because
  // erfinv ~ sqrt(w) asymptotically. One blended Horner covers both.
  //
  // Degree is NOT what the max ULP needs -- the Newton step below absorbs the
  // start almost entirely, and a degree-4 start still finishes at 1.80, as
  // does a start that is 38 ULP wrong on its own. Nor, despite appearances,
  // is degree what the wrong-rounding RATE needs: a plain minimax fit at this
  // degree measures 27.7% !=CR against 11.65% at degree 10, but walking these
  // coefficients a few ulps against the measured rate brings degree 7 back to
  // 11.69%. The rate is set by a few-ulp bias in the coefficients, not by the
  // degree. Degree 6 is where it stops coming back (21.6%).
  //
  // The variable is load-bearing for the same reason: the identical degree-10
  // fit measures 11.65% expressed in w - 2.5 and 19.67% expressed in a
  // [-1, 1] normalisation of it, purely from where the Horner's roundings
  // land. Do not "clean this up" into a normalised argument.
  const Vf lo = w < Vf(5.0f);
  const Vf v = Vf::blendv(w.sqrt() - Vf(3.0f), w - Vf(2.5f), lo);
  Vf p = Vf::blendv(Vf(-0x1.0d55bep-15f), Vf(0x1.52e89p-22f), lo);
  p = fmadd(p, v, Vf::blendv(Vf(0x1.1ae80ap-10f), Vf(-0x1.ab9a2cp-19f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(-0x1.cf749cp-9f), Vf(-0x1.12c1b6p-18f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(0x1.801794p-8f), Vf(0x1.c7c96p-13f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(-0x1.f6174ap-8f), Vf(-0x1.48e4dcp-10f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(0x1.34af6p-7f), Vf(-0x1.11b20ap-8f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(0x1.006de2p+0f), Vf(0x1.f91f22p-3f), lo));
  p = fmadd(p, v, Vf::blendv(Vf(0x1.6a9efep+1f), Vf(0x1.805c5ep+0f), lo));
  Vf x = p * y;

  // ONE Newton step. Two measures 1.67 against 1.78 for 52% more time -- not
  // worth it once the start is this good.
  //
  // Both arms want 1/erf'(|x|) = sqrt(pi)/2 * e^(x^2) and neither wants
  // e^(-x^2), so the step needs exactly one exponential and no division.
  //
  // |x| < erfinv(1-2^-24) < 3.84 here, so x*x is in [0, 15]; the |y| >= 1
  // lanes are overwritten below. Squaring rounds, and e^(fl(x*x)) is
  // therefore off by up to 0.5 ulp(x*x) relative -- correcting that with the
  // exact residual of the square measures no change in either the max or the
  // rate, so it is not done.
  const Vf ax = x.abs();
  const Vf xsgn = x ^ ax;
  // x, not ax: squaring cancels the sign, so this is ax * ax bit for bit but
  // does not wait on the abs.
  const Vf s = x * x;
  const Vf r = exp_erfinv(s, Vf(kSqrtPiOver2));

  // The residual must be formed differently on each side, and neither form
  // works on the other: erf(x) - y cancels as |y| -> 1, and (1-|y|) - erfc
  // cancels for |y| < 0.5 where erfc(x) is itself near 1-|y|.
  // Both want |x|, and erf is odd, so the sign is taken once above.
  //
  // erf arm: x - (erf(|x|)^sign(x) - y) * r. The cancelling subtraction
  // happens BEFORE r multiplies it, so r scales a quantity that is already
  // ~1e-7 of the result and its own error is second order.
  const Vf f = (erf_small_erfinv(ax, s) ^ xsgn) - y;
  const Vf xe = fnmadd(f, r, x);

  // Declared next to their first use; nothing above needs either of them.
  const Vf ya = y.abs();
  const Vf sgn = y & Vf(-0.0f);

  // erfc arm: Newton on erfc(|x|) = 1-|y| rearranges to
  //     |x1| = |x| + K(|x|) - (1-|y|) * r,   K = sqrt(pi)/2 erfc(x) e^(x^2)
  // which needs no erfc value at all. Here there is no cancellation to hide
  // behind: r multiplies a term of K's size, so r's relative error lands on
  // the result amplified by K/|x| -- 0.62 at the bottom of this arm, which is
  // why exp_erfinv exists rather than a call to expimpl.
  //
  // |x| + (K - t*r), not (|x| + K) - t*r: the second rounds its intermediate
  // at ulp(|x| + K), worth 1.34 ULP of the result at the bottom of the arm.
  // 1 - |y| is just the smaller of the two factors the log already used, so
  // this reuses them instead of subtracting again. Exact for |y| >= 0.5 by
  // Sterbenz, as the subtraction was.
  const Vf t = minimum(f1, f2);
  const Vf xc = (ax + fnmadd(t, r, erfcx_erfinv(ax))) ^ sgn;

  x = Vf::blendv(xc, xe, ya <= Vf(0.7f));

  // The log barely matters: its error is absorbed by Newton -- perturbing it
  // by 32 ULP still measures 1.82. erf_small_erfinv is what sets 1.76.
  //
  // t is 1 - |y|, so t == 0 IS |y| == 1, and t is already in hand. Comparing
  // against zero also needs no constant register.
  //
  // NaN reaches neither compare below and propagates out of the polynomial.
  const Vf inf(std::numeric_limits<float>::infinity());
  x = Vf::blendv(x, inf | sgn, t == Vf(0.0f));
  return x | (ya > Vf(1.0f));
}

// hypot(x, y) = sqrt(x^2 + y^2), with no special-case path.
//
// The problem hypot has is range, not accuracy: x^2 + y^2 overflows for
// |x| > 2^64 and flushes to zero for |x| < 2^-75, so the usual shape is a
// fast path plus a scaled fallback behind a branch. AOR's AdvSIMD kernel does
// that and measures 1.2061 ULP, but its cost swings 1.07 -> 1.64 ns/element
// depending on whether any lane in the vector trips the fallback.
//
// Here both inputs are first scaled by an exact power of two, 2^-k, where 2^k
// is the exponent of max(|x|,|y|). Scaling by a power of two is exact, and it
// moves the squares into a range where nothing can overflow or underflow, so
// the fallback disappears entirely and the cost is flat at 1.07 ns whatever
// the inputs. What is left is arithmetically AOR's fast path applied to every
// input -- same 1.2061 ULP, at the same worst case.
//
// The clamp bounds are load-bearing and are not round numbers by accident:
// 2^-k must itself be a normal float so that ONE multiply does each
// direction. k ranges over [-149, 127], and 2^-127 is subnormal, so clamping
// max(|x|,|y|) into [2^-126, 2^126] is what keeps the scale representable.
// The two saturating ends are precisely the cases AOR needs its scaled path
// for: at the top the scaled square reaches 32 rather than 4 (still finite),
// and at the bottom a subnormal input scales up into the normal range.
//
// Max ULP: 1.21, 1.51% !=CR. Widening to double instead is correctly rounded
// (0.5000 ULP, zero wrong roundings measured) and also needs no special path,
// but costs 20% more -- take that one if accuracy outranks throughput here.
inline Vectorized<float> Vectorized<float>::hypot(const Vectorized<float>& b) const {
  const Vf x = *this;
  const Vf ax = x.abs();
  const Vf ay = b.abs();

  const Vf amax = clamp(maximum(ax, ay), Vf(0x1p-126f), Vf(0x1p+126f));
  const Vi e = cast<int32_t>(amax) & Vi(0x7F800000);
  const Vf down = cast<float>(Vi(0x7F000000) - e); // 2^-k
  const Vf up = cast<float>(e); // 2^k

  const Vf xs = x * down;
  const Vf ys = b * down;
  const Vf r = fmadd(ys, ys, xs * xs).sqrt() * up;

  // hypot(+-inf, NaN) is +inf, but the NaN reaches the fma first and wins, so
  // it has to be forced. It cannot be read off amax: ATen's maximum() is the
  // IEEE 754-201X operation, which PROPAGATES NaN, so maximum(inf, NaN) is
  // NaN rather than inf. Test the inputs instead.
  //
  // Everything else falls out without a fixup. NaN alone propagates through
  // the fma; a plain infinity survives the scaling (inf * 2^-126 = inf) and
  // comes back as inf; overflow to infinity happens naturally in the closing
  // multiply; and a subnormal result rounds once there.
  const Vf inf(std::numeric_limits<float>::infinity());
  return Vf::blendv(r, inf, (ax == inf) | (ay == inf));
}

inline Vectorized<double> fmod_core_d(Vectorized<double> r, Vectorized<double> d) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;

  // One reciprocal for the whole loop, nudged TOWARD ZERO by one ulp. That
  // makes q an underestimate, never an overestimate, so r can never go
  // negative; the cost is that q is occasionally one short, which the closing
  // conditional subtract below cleans up.
  Vd rcp = Vd(1.0) / d;
  rcp = cast<double>(cast<int64_t>(rcp) - Vl(1));

  const Vl er = (cast<int64_t>(r) >> Vl(52)) & Vl(0x7FF);
  const Vl ed = (cast<int64_t>(d) >> Vl(52)) & Vl(0x7FF);
  const Vl d0 = er - ed;

  auto minuend = Vl(28);

  Vd m;

  for (int i = 0; i < 10; i++) {
    // k is the schedule, not a measurement: r drops by at least 28 binades a
    // step, so the exponent never has to be re-read inside the loop.
    const Vl k = d0 - minuend;
    // Clamped with double maximum/minimum rather than an integer compare.
    // Vectorized<int64_t> arithmetic and shifts are vec128-specialised, but
    // its compares are not everywhere, and this form also measures 11%
    // faster. k >= -276 keeps both exponent fields in range unclamped.
    const Vd p2 = maximum(cast<double>((k + Vl(1023)) << Vl(52)), Vd(1.0));
    const Vd p2i = minimum(cast<double>((Vl(1023) - k) << Vl(52)), Vd(1.0));

    const Vd ds = d * p2;
    // p2i is a power of two, so scaling the reciprocal is exact -- do NOT
    // write this as rcp / p2, which puts a divide inside the loop and costs
    // 22%.
    const Vd q = (r * (rcp * p2i)).trunc();
    r = fnmadd(q, ds, r);

    m = r - d;

    // Vector-wide: leave once every lane has converged. Lanes that converged
    // earlier simply compute q = 0 and are unchanged, so the per-lane result
    // does not depend on its neighbours. A NaN lane is false in every
    // direction, so allLt never fires on it: NaN just runs the full ten
    // iterations and propagates.
    if (allLt(m, 0.0)) {
      break;
    }

    minuend += Vl(28);
  }
  // The toward-zero reciprocal can leave the remainder one multiple short.
  return Vd::blendv(r, m, r >= d);
}

inline Vectorized<float> Vectorized<float>::fmod(const Vectorized<float>& b) const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(VectorizedN<double, kR>::size() == Vf::size());

  const Vf x = *this;
  const Vf ax = x.abs();
  const Vf ay = b.abs();

  const VectorizedN<double, kR> rd = convert<double, kR, float, 1>(ax);
  const VectorizedN<double, kR> dd = convert<double, kR, float, 1>(ay);
  VectorizedN<double, kR> md;
  md[0] = fmod_core_d(rd[0], dd[0]);
  md[1] = fmod_core_d(rd[1], dd[1]);
  Vf m = convert<float, 1, double, kR>(md);

  // The loop runs on |x|, so the result is non-negative and takes x's sign.
  m = m | (x & Vf(-0.0f));

  // |x| < |y| covers x = +-0 as well, and returns it signed, which is what
  // fmod owes. y = +-inf with x finite returns x. y = 0 and |x| = inf are the
  // two domain errors. NaN reaches none of these compares and propagates out
  // of the loop on its own.
  m = Vf::blendv(m, x, ax < ay);
  const Vf bad = (ax - ax) | (ay == Vf(0.0f));
  return m | bad;
}

inline Vectorized<float> Vectorized<float>::nextafter(const Vectorized<float>& b) const {
  using Vi = Vectorized<int32_t>;
  const Vf x = *this;
  const Vf y = b;

  const Vi xi = cast<int32_t>(x);
  // The sign has to come from the BIT PATTERN, not from x < 0: -0 is negative
  // but does not compare so, and -0 is one of the cases that has to work.
  const Vf smask = cast<float>(xi >> Vi(31));
  // Incrementing the raw pattern always moves AWAY from zero, whichever side
  // of zero x is on; toward zero is a decrement. Which one we want is
  // (y > x) XOR signbit(x).
  const Vf away = (y > x) ^ smask;
  const Vi step = Vi::blendv(Vi(-1), Vi(1), cast<int32_t>(away));
  Vf r = cast<float>(xi + step);

  // x = +-0 is the one input where the magnitude step is wrong: +0 stepping
  // down and -0 stepping up both have to CROSS zero, and the raw pattern
  // instead walks off into 0xFFFFFFFF / 0x7FFFFFFF, which are NaN. The answer
  // is the smallest subnormal carrying y's sign.
  r = Vf::blendv(r, Vf(0x1p-149f) ^ (y & Vf(-0.0f)), x == Vf(0.0f));

  // x == y returns y. This also settles (+0, -0) and (-0, +0), where the
  // comparison is true and the sign of the result must come from y.
  r = Vf::blendv(r, y, x == y);

  // The arithmetic cannot carry NaN on its own: stepping a NaN pattern
  // usually stays NaN, but 0x7F800001 - 1 lands exactly on infinity, and
  // nextafter(finite, NaN) has to be NaN too. So the test is load-bearing.
  return r | isAnyNaN(x, y);
}

inline Vectorized<double> lgamma_log_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;

  // x = 2^e * m. Subtracting the bit pattern of 1/sqrt(2) before the shift
  // moves the split point from 2 to sqrt(2), so m lands in [1/sqrt2, sqrt2)
  // and |r| below is 0.1716 rather than 1/3 -- three fewer terms.
  const Vl xi = cast<int64_t>(x);
  const Vl e = (xi - Vl(0x3FE6A09E667F3BCDLL)) >> Vl(52);
  const Vd m = cast<double>(xi - (e << Vl(52)));
  // (double)e with no int-to-float convert: e is in [-150, 130] here, so
  // e+1023 drops into the mantissa field of 2^52 and one subtract peels it
  // back off. Both operands are within a factor of two, so it is exact.
  const Vd ef = cast<double>((e + Vl(1023)) | Vl(0x4330000000000000LL)) - Vd(0x1.00000000003FFp+52);

  // A real divide, not a reciprocal estimate refined by Newton. The estimate
  // needs three Newton steps to reach double, which is seven more ops on the
  // FMA pipes; the divider is otherwise idle, so the divide measures FASTER
  // (9.9 ns against 11.4) as well as being exact.
  const Vd r = (m - Vd(1.0)) / (m + Vd(1.0));
  const Vd s = r * r;
  Vd p = Vd(0x1.919813103260ep-4);
  p = fmadd(p, s, Vd(0x1.c6208cdc2c275p-4));
  p = fmadd(p, s, Vd(0x1.2494396056bc9p-3));
  p = fmadd(p, s, Vd(0x1.99999628a5c18p-3));
  p = fmadd(p, s, Vd(0x1.5555555672df3p-2));
  p = fmadd(p, s, Vd(0x1.fffffffffff10p-1));

  // ln2 has to be two words. |e| reaches 150, and a single-double ln2 leaves
  // 150 * ln2 * 2^-53 = 1.1e-14 of absolute error against a 2.6e-14 budget
  // at the negative zeros. llvm-libc uses one word because it never routes
  // those zeros through its log.
  //
  // r == 0 exactly when m == 1, so log(1) is exactly 0 and the exact zeros
  // at x = 1 and x = 2 survive.
  const Vd q = fmadd(Vd(2.0), r * p, ef * Vd(0x1.62e42fefa39efp-1));
  return fmadd(ef, Vd(0x1.abc9e3b39803fp-56), q);
}

inline Vectorized<double> lgamma_core_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd one(1.0), zero(0.0);
  const Vd inf(std::numeric_limits<double>::infinity());

  // Everything is reduced to z > 0. Negative x goes through the reflection
  // lgamma(x) = log(pi) - lgamma(1-x) - log|sin(pi x)| at the end.
  const Vd neg = x < zero;
  const Vd z = Vd::blendv(x, one - x, neg);
  const Vd big = z >= Vd(4.0);

  // z < 4: shift into y in [1,2) and carry the ratio Gamma(z)/Gamma(y) as a
  // product. n is -1, 0, 1 or 2, so the product is at most two factors.
  // Threshold 4 is a measured optimum: below it the Stirling polynomial
  // grows faster than the product loop shrinks, and 2 is impossible because
  // it would put z = 2 on the Stirling path, where lgamma(2) comes out 1e-16
  // instead of exactly 0 -- against a zero reference that reads 1.2e29 ULP.
  const Vd n = z.floor() - one;
  const Vd y = z - n;
  Vd prod = Vd::blendv(one, z - one, Vd(1.0) <= n);
  prod = prod * Vd::blendv(one, z - Vd(2.0), Vd(2.0) <= n);
  // z < 1 is the one case that divides rather than multiplies: there
  // lgamma(z) = lgamma(z+1) - log(z), so the log enters with a minus sign.
  // It only arises for x > 0, since x < 0 gives z = 1-x >= 1.
  const Vd lo = n < zero;
  const Vd arg = Vd::blendv(prod, z, lo);
  const Vd sigma = Vd::blendv(one, Vd(-1.0), lo);

  // |sin(pi x)| for the reflection, as pi*|t|*sinc(t^2) with t = x - round(x).
  // The factored form is what keeps relative accuracy as x approaches an
  // integer, where |sin| goes to zero and the result does not.
  //
  // Round to nearest built from trunc, the same idiom cos() uses above:
  // Vectorized<double>::round() is vrndiq_f64, which follows the DYNAMIC
  // rounding mode, and under round-toward-zero t leaves [-0.5, 0.5] and
  // walks off the end of the fit. Only x < 0 lanes are used, so trunc(x-0.5)
  // is the copysign form specialised to that sign. x - 0.5 is exact for
  // every float |x| < 2^23, and every float above that is an integer, which
  // the pole test below catches.
  //
  // It must be round-to-nearest and not floor: llvm-libc reduces with
  // frac = x - floor(x), which is fine for it because it only reflects
  // |x| >= 3.373, but here x = -5.55e-17 would need frac = 1 - 5.55e-17,
  // that rounds to exactly 1, and sin(pi x) collapses to 0. Round-to-nearest
  // returns t = x untouched and keeps every bit.
  const Vd t = x - (x - Vd(0.5)).trunc();
  const Vd u = t * t;
  Vd sc = Vd(-0x1.c3375664130e7p-18);
  sc = fmadd(sc, u, Vd(0x1.370f8790931cdp-13));
  sc = fmadd(sc, u, Vd(-0x1.3380ab9698318p-9));
  sc = fmadd(sc, u, Vd(0x1.ac6802f4365a0p-6));
  sc = fmadd(sc, u, Vd(-0x1.86a8e46c178bbp-3));
  sc = fmadd(sc, u, Vd(0x1.9f9cb402affa8p-1));
  sc = fmadd(sc, u, Vd(-0x1.a51a66253069bp+0));
  sc = fmadd(sc, u, Vd(0x1.fffffffffffffp-1));
  const Vd sn = Vd::blendv(one, Vd(0x1.921fb54442d18p+1) * t.abs() * sc, neg);

  // Two logs, and they cannot be folded into one. Below the threshold the
  // product and the sine multiply, so log(prod * |sin|) covers both. Above
  // it the Stirling term needs log(z) scaled by (z-0.5) while the sine
  // enters with coefficient 1, and no single logarithm carries both.
  const Vd l1 = lgamma_log_d(Vd::blendv(arg * sn, z, big));
  const Vd l2 = lgamma_log_d(Vd::blendv(one, sn, big & neg));

  // z >= 4: Stirling. Writing it as (z-0.5)*(log z - 1) + C is algebraically
  // identical and one operation shorter, but measures SLOWER (10.32 against
  // 9.99) because it puts log z - 1 on the serial path, where this form lets
  // the rest be computed while the log is still in flight.
  const Vd rz = one / z;
  const Vd w = rz * rz;
  Vd bs = Vd(-0x1.2a73847847132p-10);
  bs = fmadd(bs, w, Vd(0x1.a0ccbe2d203a7p-11));
  bs = fmadd(bs, w, Vd(-0x1.3757efe38828dp-11));
  bs = fmadd(bs, w, Vd(0x1.a01765cb92d0bp-11));
  bs = fmadd(bs, w, Vd(-0x1.6c16c08fffe2ep-9));
  bs = fmadd(bs, w, Vd(0x1.5555555553dbep-4));
  const Vd tail = fmadd(rz, bs, Vd(0x1.d67f1c864beb5p-1) - z);
  const Vd mainbig = fmadd(z - Vd(0.5), l1, tail);
  // z < 4: lgamma(y) = (y-1)(y-2)*P(y-1.5). y-1 and y-2 are exact in double
  // for every y this can produce, so the two zeros are exact and P only has
  // to be accurate RELATIVE to itself -- which is what turns an absolute
  // 1e-15 requirement into a reachable one.
  const Vd v = y - Vd(1.5);
  Vd cp = Vd(-0x1.b136081ab08c5p-14);
  cp = fmadd(cp, v, Vd(0x1.4ed034c21e33ap-13));
  cp = fmadd(cp, v, Vd(-0x1.3d27daa756979p-13));
  cp = fmadd(cp, v, Vd(0x1.0385448701d48p-12));
  cp = fmadd(cp, v, Vd(-0x1.cfc601fa02703p-12));
  cp = fmadd(cp, v, Vd(0x1.77c9a21f13946p-11));
  cp = fmadd(cp, v, Vd(-0x1.31174664112f9p-10));
  cp = fmadd(cp, v, Vd(0x1.f7ea878b16f65p-10));
  cp = fmadd(cp, v, Vd(-0x1.a504d3a7d6711p-9));
  cp = fmadd(cp, v, Vd(0x1.64f114fd1729fp-8));
  cp = fmadd(cp, v, Vd(-0x1.34dbd62d05759p-7));
  cp = fmadd(cp, v, Vd(0x1.133421f0f103bp-6));
  cp = fmadd(cp, v, Vd(-0x1.007aa84177e78p-5));
  cp = fmadd(cp, v, Vd(0x1.01af62a3a2749p-4));
  cp = fmadd(cp, v, Vd(-0x1.2aed059bd47eep-3));
  cp = fmadd(cp, v, Vd(0x1.eeb95b094bf6ep-2));
  const Vd core = (y - one) * (y - Vd(2.0)) * cp;

  // Reflection folds in as a sign flip on the positive-side value plus
  // log(pi), with the sine's log already inside l1 below the threshold and
  // subtracted as l2 above it.
  const Vd g = Vd::blendv(one, Vd(-1.0), neg);
  const Vd off = Vd::blendv(zero, Vd(0x1.250d048e7a1bdp+0), neg);
  const Vd vs = fmadd(g, fmadd(sigma, l1, core), off);
  const Vd vb = fmadd(g, mainbig, off) - l2;
  const Vd res = Vd::blendv(vs, vb, big);

  // Poles are forced at float width by the caller.
  return res;
}

// always_inline is load-bearing, not decoration: left to its own judgement
// the compiler keeps this out of line, and the caller's loop can then no
// longer overlap one element with the next. That is worth 11% (9.8 against
// 10.9) and costs nothing but code size at the single call site.
inline Vectorized<float> Vectorized<float>::lgamma() const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(VectorizedN<double, kR>::size() == Vf::size());

  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(*this);
  VectorizedN<double, kR> rd;

  rd[0] = lgamma_core_d(xd[0]);
  rd[1] = lgamma_core_d(xd[1]);

  // Poles: x <= 0 and integral, which also covers +-0 and -inf, plus +inf.
  // At float width and once, rather than at double width in a core that runs
  // twice: widening is exact, so x == trunc(x) is the same integrality test
  // the core's t == 0 was, and +inf narrows to +inf.
  const Vf x = *this;
  const Vf inf(std::numeric_limits<float>::infinity());
  const Vf pole = ((x <= Vf(0.0f)) & (x == x.trunc())) | (x.abs() == inf);
  // The narrowing is the only rounding the result sees, so it lands on the
  // float grid directly -- including overflow to +inf, which puts the
  // threshold at the right input instead of 1.2% early like SLEEF's.
  // NaN reaches no compare above and propagates through the polynomials.
  return Vf::blendv(convert<float, 1, double, kR>(rd), inf, pole);
}

inline Vectorized<double> i0_exp_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;

  // Round to nearest built from trunc, the idiom cos() above documents:
  // Vectorized<double>::round() is vrndiq_f64 and follows the DYNAMIC
  // rounding mode. Under round-toward-zero |r| below would reach ln2
  // instead of ln2/2, and the Taylor would lose 1.8 ULP. x >= 0 here, so
  // adding 0.5 and truncating is the whole of it.
  const Vd kf = fmadd(x, Vd(0x1.71547652b82fep+0), Vd(0.5)).trunc();

  // One-word ln2 is enough here, unlike lgamma's log: k only reaches 145,
  // so this leaves 145*ln2*2^-53 = 1.1e-14 of absolute error in r, which is
  // 1.1e-14 relative in exp -- 2e-7 ULP. A second word is pure cost.
  const Vd r = fnmadd(kf, Vd(0x1.62e42fefa39efp-1), x);

  // Taylor, not minimax: a degree-6 minimax is both more accurate (1.9e-9
  // against 5.2e-9) and one term shorter, yet measures SLOWER -- the
  // reciprocal-factorial constants evidently materialise more cheaply than
  // arbitrary ones.
  Vd p = Vd(1.0 / 5040.0);
  p = fmadd(p, r, Vd(1.0 / 720.0));
  p = fmadd(p, r, Vd(1.0 / 120.0));
  p = fmadd(p, r, Vd(1.0 / 24.0));
  p = fmadd(p, r, Vd(1.0 / 6.0));
  p = fmadd(p, r, Vd(0.5));
  p = fmadd(p, r, Vd(1.0));
  p = fmadd(p, r, Vd(1.0));

  // 2^k. Reading k back out of the mantissa instead (add 2^52, then shift
  // left 52 to drop the header) is one operation shorter and measures
  // slower.
  const Vl ki = cast<int64_t>(kf + Vd(0x1p52)) - Vl(0x4330000000000000LL);
  return p * cast<double>((ki + Vl(1023)) << Vl(52));
}

inline Vectorized<double> i0_core_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd ax = x.abs();

  // |x| <= 8: a plain polynomial in x^2, no exp anywhere. x is a widened
  // float, so x*x is EXACT here -- see the note above, that one fact is
  // worth 2.4 ULP. Degree 9 is the floor: degree 8 fits to only 2.3e-7,
  // which is 3.9 ULP on its own.
  const Vd s = ax * ax;
  Vd p = Vd(0x1.079b8dcec5ce0p-54);
  p = fmadd(p, s, Vd(0x1.a30f704b9a1ddp-48));
  p = fmadd(p, s, Vd(0x1.72ba9d7c79d4bp-39));
  p = fmadd(p, s, Vd(0x1.fc5c8858e467fp-32));
  p = fmadd(p, s, Vd(0x1.241440b2278cap-24));
  p = fmadd(p, s, Vd(0x1.c6f3b959f3d16p-18));
  p = fmadd(p, s, Vd(0x1.c720c04f7ab86p-12));
  p = fmadd(p, s, Vd(0x1.ffff9309ff5c0p-7));
  p = fmadd(p, s, Vd(0x1.000003c167677p-2));
  p = fmadd(p, s, Vd(0x1.ffffffb3be91ap-1));

  // |x| > 8: exp(x) * G(1/x) / sqrt(x), G(u) -> 1/sqrt(2pi) as u -> 0.
  //
  // The clamp costs nothing: i0 already exceeds FLT_MAX at 91.9 so it
  // cannot change a finite answer, and it keeps exp's 2^k inside the
  // exponent field. NaN survives it because minimum() propagates NaN.
  // ax - ax is +0 for finite ax, which leaves axc bit-identical, and NaN at
  // +-inf, which makes i0(+-inf) NaN to match calc_i0.
#if 1
  // On Pytorch, i0(+/-inf) yields NaN
  const Vd axc = minimum(ax, Vd(100.0)) + (ax - ax);
#else
  const Vd axc = minimum(ax, Vd(100.0));
#endif

  // One divide, not two: 1/sqrt(x) serves as both the series argument
  // (u = rs*rs) and the closing factor. Worth 0.5 ns -- the opposite of
  // what the same trick did to lgamma, because this kernel is light enough
  // that the divider, not the FMA pipes, is the constraint. An FRSQRTE
  // estimate plus Newton is slower still (3.89 against 3.68), so the plain
  // sqrt and divide are the fast choice here, not just the exact one.
  const Vd rs = Vd(1.0) / axc.sqrt();
  const Vd u = rs * rs;
  Vd g = Vd(0x1.8d4850611acecp-2);
  g = fmadd(g, u, Vd(-0x1.c5e69c126ab0ap-7));
  g = fmadd(g, u, Vd(0x1.14c4d0141ad16p-5));
  g = fmadd(g, u, Vd(0x1.c91808a07c92ap-6));
  g = fmadd(g, u, Vd(0x1.98881a2912fb8p-5));
  g = fmadd(g, u, Vd(0x1.9884530240f18p-2));
  const Vd big = i0_exp_d(axc) * g * rs;

  // NaN reaches neither arm's compare and propagates out of both.
  return Vd::blendv(big, p, ax <= Vd(8.0));
}

inline Vectorized<float> Vectorized<float>::i0() const {
  // Always 2, on every capability: it is sizeof(double)/sizeof(float), not
  // anything to do with the vector length.
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(kR == 2);

  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(*this);
  VectorizedN<double, kR> rd;
  // Written out rather than looped: a RUNTIME index into VectorizedN forces
  // its backing array out of registers and into memory, which measures 40%
  // on a kernel this size. The index has to be a compile-time constant.
  rd[0] = i0_core_d(xd[0]);
  rd[1] = i0_core_d(xd[1]);
  // The narrowing is the only rounding the result sees, which is what puts
  // the overflow threshold at the right input instead of 3.18 early.
  return convert<float, 1, double, kR>(rd);
}

inline Vectorized<double> i0e_core_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd ax = x.abs();

  // |x| <= 6: a polynomial in x itself. The argument is |x|, which is
  // exact, so unlike i0's small arm there is no argument rounding to be
  // amplified. P(0) = 0.99999999171, which narrows to exactly 1.0f, so
  // i0e(0) is exact -- the shipped code is one ULP low there.
  Vd p = Vd(-0x1.243b5c708b6fep-36);
  p = fmadd(p, ax, Vd(0x1.ed7111609d375p-31));
  p = fmadd(p, ax, Vd(-0x1.84514956b8b4ep-26));
  p = fmadd(p, ax, Vd(0x1.7c3f79d6746e4p-22));
  p = fmadd(p, ax, Vd(-0x1.0572b35324d21p-18));
  p = fmadd(p, ax, Vd(0x1.0ec9128f0a1fap-15));
  p = fmadd(p, ax, Vd(-0x1.bb781837c3b93p-13));
  p = fmadd(p, ax, Vd(0x1.29c1f569bf7dfp-10));
  p = fmadd(p, ax, Vd(-0x1.508e3370e81a4p-8));
  p = fmadd(p, ax, Vd(0x1.44d9f889c00b0p-6));
  p = fmadd(p, ax, Vd(-0x1.0c01ece11efe2p-4));
  p = fmadd(p, ax, Vd(0x1.751abed1efd21p-3));
  p = fmadd(p, ax, Vd(-0x1.aaa032e85adbfp-2));
  p = fmadd(p, ax, Vd(0x1.7ffefddc5eb03p-1));
  p = fmadd(p, ax, Vd(-0x1.ffffebad81409p-1));
  p = fmadd(p, ax, Vd(0x1.ffffffb8d118dp-1));

  // |x| > 6: G(1/x)/sqrt(x), G(u) -> 1/sqrt(2pi) as u -> 0. One divide and
  // one sqrt, not two divides: 1/sqrt(x) is both the series argument
  // (u = rs*rs) and the closing factor. ax = 0 makes rs infinite and the
  // series garbage, but that lane takes the small arm.
  const Vd rs = Vd(1.0) / ax.sqrt();
  const Vd u = rs * rs;
  Vd g = Vd(0x1.d56b37498e753p-1);
  g = fmadd(g, u, Vd(0x1.0682dedd3f568p-3));
  g = fmadd(g, u, Vd(0x1.116af990e4630p-7));
  g = fmadd(g, u, Vd(0x1.124f148a1b2d5p-5));
  g = fmadd(g, u, Vd(0x1.c88ea16e44458p-6));
  g = fmadd(g, u, Vd(0x1.988a09b40ee14p-5));
  g = fmadd(g, u, Vd(0x1.988452cf8a66cp-2));

  // NaN reaches neither arm's compare and propagates out of both.
  return Vd::blendv(g * rs, p, leAbs(x, Vd(6.0)));
}

inline Vectorized<float> Vectorized<float>::i0e() const {
  // Always 2, on every capability: it is sizeof(double)/sizeof(float), not
  // anything to do with the vector length.
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(kR == 2);

  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(*this);
  VectorizedN<double, kR> rd;
  // Written out rather than looped: a RUNTIME index into VectorizedN forces
  // its backing array out of registers and into memory, which measures 40%
  // on a kernel this size. The index has to be a compile-time constant.
  rd[0] = i0e_core_d(xd[0]);
  rd[1] = i0e_core_d(xd[1]);
  return convert<float, 1, double, kR>(rd);
}

inline Vectorized<double> dg_log_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;
  const Vl xi = cast<int64_t>(x);
  const Vl e = (xi - Vl(0x3FE6A09E667F3BCDLL)) >> Vl(52);
  const Vd m = cast<double>(xi - (e << Vl(52)));
  const Vd ef = cast<double>((e + Vl(1023)) | Vl(0x4330000000000000LL)) - Vd(0x1.00000000003FFp+52);
  const Vd r = (m - Vd(1.0)) / (m + Vd(1.0));
  const Vd s = r * r;
  Vd p = Vd(1.0 / 17.0);
  p = fmadd(p, s, Vd(1.0 / 15.0));
  p = fmadd(p, s, Vd(1.0 / 13.0));
  p = fmadd(p, s, Vd(1.0 / 11.0));
  p = fmadd(p, s, Vd(1.0 / 9.0));
  p = fmadd(p, s, Vd(1.0 / 7.0));
  p = fmadd(p, s, Vd(1.0 / 5.0));
  p = fmadd(p, s, Vd(1.0 / 3.0));
  p = fmadd(p, s, Vd(1.0));
  const Vd q = fmadd(Vd(2.0), r * p, ef * Vd(0x1.62e42fefa39efp-1));
  return fmadd(ef, Vd(0x1.abc9e3b39803fp-56), q);
}

// digamma runs the whole core in double and rounds once, which makes it
// correctly rounded for every float input but one neighbourhood -- and that
// is not slack to spend, it is what the kernel costs. The core is 36
// polynomial coefficients run TWICE per float vector, and with 32 vector
// registers that does not fit: written as one function it spills 49 times
// per iteration. So the core is split in two, and digamma() below runs each
// half for BOTH double vectors before moving on, which keeps the log's and
// the asymptotic's constants live across one phase instead of two. That and
// the unrolled recurrence are worth 21% for bit-identical output.
//
// Everything is reduced to z > 0; x < 0 reflects in dg_refl_d.
inline Vectorized<double> dg_main_d(Vectorized<double> x, Vectorized<double>* zo, Vectorized<double>* nego) {
  using Vd = Vectorized<double>;
  const Vd one(1.0), zero(0.0);

  const Vd neg = x < zero;
  const Vd z = Vd::blendv(x, one - x, neg);
  *nego = neg;
  *zo = z;

  // Push z up to >= 2 by psi(z) = psi(z+1) - 1/z, then use the asymptotic.
  // z > 0, so two increments always suffice and both masks come straight off
  // z -- writing this as a loop instead makes pp depend on the previous pp
  // and serialises six double ops for nothing.
  //
  // sum 1/(z+k) is computed as P'/P for P = prod(z+k), which needs ONE
  // divide instead of one per step and costs nothing in accuracy. The
  // inactive factors are 1, so P stays bounded and cannot overflow.
  const Vd m1 = z < Vd(2.0);
  const Vd m2 = z < one;
  const Vd a1 = Vd::blendv(one, z, m1);
  const Vd a2 = Vd::blendv(one, z + one, m2);
  const Vd acc = (Vd::blendv(zero, a2, m1) + Vd::blendv(zero, a1, m2)) / (a1 * a2);
  const Vd zz = z + (m1 & one) + (m2 & one);

  // psi(zz) = ln zz - 1/(2 zz) - w*P(w), w = 1/zz^2, zz >= 2.
  const Vd rz = one / zz;
  const Vd w = rz * rz;
  Vd p = Vd(0x1.03a0bf0d1d8acp+2);
  p = fmadd(p, w, Vd(-0x1.c9b0655292f64p+2));
  p = fmadd(p, w, Vd(0x1.71d4b8f497188p+2));
  p = fmadd(p, w, Vd(-0x1.706e5c33a9af9p+1));
  p = fmadd(p, w, Vd(0x1.046d267e15a61p+0));
  p = fmadd(p, w, Vd(-0x1.2522f850ab6e3p-2));
  p = fmadd(p, w, Vd(0x1.2d979f08c36c1p-4));
  p = fmadd(p, w, Vd(-0x1.525adeb130661p-6));
  p = fmadd(p, w, Vd(0x1.ef97877c493f6p-8));
  p = fmadd(p, w, Vd(-0x1.110cb84a78ca6p-8));
  p = fmadd(p, w, Vd(0x1.041035cb5c1a8p-8));
  p = fmadd(p, w, Vd(-0x1.1111110b4fdefp-7));
  p = fmadd(p, w, Vd(0x1.555555555535fp-4));
  return fnmadd(w, p, fnmadd(Vd(0.5), rz, dg_log_d(zz)) - acc);
}

inline Vectorized<double> dg_refl_d(Vectorized<double> x,
                                    Vectorized<double> res,
                                    Vectorized<double> z,
                                    Vectorized<double> neg) {
  using Vd = Vectorized<double>;
  const Vd one(1.0);
  const Vd inf(std::numeric_limits<double>::infinity());

  // Reflection: psi(x) = psi(1-x) - pi*cot(pi x), written as
  //   pi*cot(pi r) = (1 - 4r^2) * H(r^2) / r,   r = x - round(x).
  // BOTH the pole at r=0 and the zero at r=+-1/2 are factored out, so H
  // only needs relative accuracy. The natural 1/r + r*K(r^2) form cannot
  // work: its r*K term is O(5), so even a degree-12 K leaves 24 ULP once
  // the reflection cancels.
  //
  // Round to nearest built from trunc, the idiom cos() above documents --
  // Vectorized<double>::round() is vrndiq_f64 and follows the dynamic
  // rounding mode. Only x < 0 lanes are used, so trunc(x - 0.5) is the
  // copysign form for that sign.
  const Vd rr = x - (x - Vd(0.5)).trunc();
  const Vd qq = rr * rr;
  Vd h = Vd(0x1.2a28e51d5b8a5p+2);
  h = fmadd(h, qq, Vd(-0x1.c0afb3413487bp+1));
  h = fmadd(h, qq, Vd(0x1.70f200303876cp+1));
  h = fmadd(h, qq, Vd(-0x1.f1847e323aef2p-5));
  h = fmadd(h, qq, Vd(0x1.a6f0fa9366229p-1));
  h = fmadd(h, qq, Vd(0x1.48fe951e6274bp-1));
  h = fmadd(h, qq, Vd(0x1.56a346406f07fp-1));
  h = fmadd(h, qq, Vd(0x1.55415a8cbad89p-1));
  h = fmadd(h, qq, Vd(0x1.5567b9882d317p-1));
  h = fmadd(h, qq, Vd(0x1.559ac67e80a28p-1));
  h = fmadd(h, qq, Vd(0x1.5671eaeaf20afp-1));
  h = fmadd(h, qq, Vd(0x1.5a0d12fa8b4b8p-1));
  h = fmadd(h, qq, Vd(0x1.6b96676b3e812p-1));
  h = fmadd(h, qq, Vd(0x1.fffffffffffffp-1));
  const Vd cot = (fnmadd(Vd(4.0), qq, one) * h) / rr;
  res = Vd::blendv(res, res - cot, neg);

  // psi(+inf) = +inf. The log's exponent arithmetic returns ~709.78 there,
  // so it has to be forced; testing z catches -inf at the same time, which
  // the negative-integer arm below then turns into NaN.
  res = Vd::blendv(res, inf, z == inf);
  // A negative integer is a pole: NaN, per the C++ standard and SciPy.
  // Tempting to reuse rr == 0 for this test and drop the second trunc, but
  // rr is NaN at x = -inf, where x == x.trunc() is true -- that one input
  // comes out +inf instead of NaN.
  //
  // +-0 needs no case of its own: z = +-0 makes the recurrence's product
  // +-0, so acc = pp/pr is already -+inf and res = (...) - acc is -+inf --
  // which is copysign(inf, -x), exactly what psi(+-0) wants, including the
  // sign for -0 that copysign(.., x) would get wrong.
  return res | (neg & (x == x.trunc()));
}

inline Vectorized<float> Vectorized<float>::digamma() const {
  // Always 2, on every capability: it is sizeof(double)/sizeof(float), not
  // anything to do with the vector length.
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(kR == 2);
  using Vd = Vectorized<double>;

  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(*this);
  // Both halves through one phase before either enters the next: the two
  // phases' coefficient sets are 22 and 14 doubles, and interleaving them
  // does not fit the register file. Written out rather than looped, because
  // a RUNTIME index into VectorizedN forces its backing array out of
  // registers and into memory, which measures 40% on a kernel this size.
  Vd z0, z1, neg0, neg1;
  const Vd m0 = dg_main_d(xd[0], &z0, &neg0);
  const Vd m1 = dg_main_d(xd[1], &z1, &neg1);
  VectorizedN<double, kR> rd;
  rd[0] = dg_refl_d(xd[0], m0, z0, neg0);
  rd[1] = dg_refl_d(xd[1], m1, z1, neg1);
  return convert<float, 1, double, kR>(rd);
}

// Coefficient tables shared by the igamma and igammac ports. Both index
// these directly, so this block must precede them both.
//
// Sized for the ports as tuned, NOT for a full-precision double kernel:
// igamma reads kIgD[0..4][0..12] and igammac only kIgD[0..2][0..8]. The
// published table is 25x25; 5x13 is enough because inside the asymptotic
// band a > 20 and |eta| < 0.31, so the dropped terms are already below
// double. That trim is safe ONLY for an asymptotic series -- the other
// three tables below are minimax fits, where dropping a term leaves an
// unfitted polynomial rather than a small perturbation (truncating them
// measured 1.2e9, 657 and 28 ULP), so they must be used at full length.

// d_k(eta), DLMF 8.12.3 / 8.12.4
alignas(64) constexpr double kIgD[5][13] = {
    {-0.3333333333333333,
     0.08333333333333333,
     -0.014814814814814815,
     0.0011574074074074073,
     0.0003527336860670194,
     -0.0001787551440329218,
     3.919263178522438e-05,
     -2.1854485106799924e-06,
     -1.85406221071516e-06,
     8.296711340953087e-07,
     -1.7665952736826078e-07,
     6.707853543401498e-09,
     1.0261809784240309e-08},
    {-0.001851851851851852,
     -0.003472222222222222,
     0.0026455026455026454,
     -0.0009902263374485596,
     0.00020576131687242798,
     -4.018775720164609e-07,
     -1.8098550334489977e-05,
     7.64916091608111e-06,
     -1.6120900894563446e-06,
     4.647127802807434e-09,
     1.378633446915721e-07,
     -5.752545603517705e-08,
     1.1951628599778148e-08},
    {0.004133597883597883,
     -0.0026813271604938273,
     0.0007716049382716049,
     2.0093878600823047e-06,
     -0.00010736653226365161,
     5.2923448829120125e-05,
     -1.2760635188618728e-05,
     3.423578734096138e-08,
     1.3721957309062932e-06,
     -6.298992138380055e-07,
     1.4280614206064242e-07,
     -2.0477098421990866e-10,
     -1.409252991086752e-08},
    {0.0006494341563786008,
     0.00022947209362139917,
     -0.0004691894943952557,
     0.00026772063206283885,
     -7.561801671883977e-05,
     -2.396505113867297e-07,
     1.1082654115347302e-05,
     -5.6749528269915965e-06,
     1.4230900732435883e-06,
     -2.7861080291528143e-11,
     -1.6958404091930278e-07,
     8.099464905388083e-08,
     -1.9111168485973655e-08},
    {-0.0008618882909167117,
     0.0007840392217200666,
     -0.0002990724803031902,
     -1.4638452578843418e-06,
     6.641498215465122e-05,
     -3.968365047179435e-05,
     1.1375726970678419e-05,
     2.507497226237533e-10,
     -1.6954149536558305e-06,
     8.907507532205309e-07,
     -2.292934834000805e-07,
     2.956794137544049e-11,
     2.8865829742708783e-08},
};

// S(z) = erfc(z)*exp(z^2) on [0, 3.25], degree 18, 2.6e-11
alignas(64) constexpr double kIgErfcS[19] = {0x1.ffffffffc9287p-1,
                                             -0x1.20dd74ee2d0bep+0,
                                             0x1.fffffa056f349p-1,
                                             -0x1.8126f2b7ef776p-1,
                                             0x1.fffb0ea479cccp-2,
                                             -0x1.34085dcd4ca9ep-2,
                                             0x1.54c450b18bda9p-3,
                                             -0x1.5d992eb2e5c4dp-4,
                                             0x1.4cde28ae51c3ep-5,
                                             -0x1.237685a5d7949p-6,
                                             0x1.cc4a76fd0eb7ep-8,
                                             -0x1.3ed7bbece4082p-9,
                                             0x1.77496e6efefecp-11,
                                             -0x1.6a8d24678aedep-13,
                                             0x1.14af00280acfbp-15,
                                             -0x1.3e3bcd9455939p-18,
                                             0x1.01b458946ed08p-21,
                                             -0x1.04dec760cc89dp-25,
                                             0x1.ef3508c9adae6p-31};

// G(u) with S(z) = G(1/z^2)/(z*sqrt(pi)) for z >= 3, degree 12, 2.6e-14.
// G(0) = 1, which is why this arm stays valid out to any z.
alignas(64) constexpr double kIgG[13] = {0x1.fffffffffff19p-1,
                                         -0x1.fffffffe92659p-2,
                                         0x1.7ffffe80634a3p-1,
                                         -0x1.dfff60f6271ffp+0,
                                         0x1.a3eeb3c88430dp+2,
                                         -0x1.d75cda2726a78p+4,
                                         0x1.3e8c5d0622b83p+7,
                                         -0x1.df53e008e909dp+9,
                                         0x1.667130ca41a17p+12,
                                         -0x1.d201476605addp+14,
                                         0x1.caef6b14d2b07p+16,
                                         -0x1.22d9d166f8e7ep+18,
                                         0x1.5b58a7e95b028p+18};

// M2(s) = -2*(log1p(s) - s)/s^2, degree 16, 1.7e-14, used for
// eta = sigma*sqrt(M2(sigma)). Forming eta from log(x/a) - sigma instead
// loses everything once |sigma| < 1e-9.
alignas(64) constexpr double kIgM2[17] = {0x1.0000000000007p+0,
                                          -0x1.5555555553760p-1,
                                          0x1.fffffffff421bp-2,
                                          -0x1.999999a00224bp-2,
                                          0x1.55555561b1715p-2,
                                          -0x1.249245f38807bp-2,
                                          0x1.fffff654463d9p-3,
                                          -0x1.c71dde0af9981p-3,
                                          0x1.999b76140a420p-3,
                                          -0x1.74323c41668d8p-3,
                                          0x1.55229c2171d4ep-3,
                                          -0x1.3de9e22ef0987p-3,
                                          0x1.27b1420f31d0bp-3,
                                          -0x1.ec4524fa0b35fp-4,
                                          0x1.c7fba6c6c01ccp-4,
                                          -0x1.731bca0652038p-3,
                                          0x1.64d7d9bc1708cp-3};

inline Vectorized<double> igamma_exp_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  using Vl = Vectorized<int64_t>;
  const Vd kf = (x * Vd(0x1.71547652b82fep+0)).round();
  Vd r = fnmadd(kf, Vd(0x1.62e42fefa3800p-1), x);
  r = fnmadd(kf, Vd(0x1.ef35793c76730p-45), r);
  Vd p = Vd(1.0 / 479001600.0);
  p = fmadd(p, r, Vd(1.0 / 39916800.0));
  p = fmadd(p, r, Vd(1.0 / 3628800.0));
  p = fmadd(p, r, Vd(1.0 / 362880.0));
  p = fmadd(p, r, Vd(1.0 / 40320.0));
  p = fmadd(p, r, Vd(1.0 / 5040.0));
  p = fmadd(p, r, Vd(1.0 / 720.0));
  p = fmadd(p, r, Vd(1.0 / 120.0));
  p = fmadd(p, r, Vd(1.0 / 24.0));
  p = fmadd(p, r, Vd(1.0 / 6.0));
  p = fmadd(p, r, Vd(0.5));
  p = fmadd(p, r, Vd(1.0));
  p = fmadd(p, r, Vd(1.0));
  const Vl ki = cast<int64_t>(clamp(kf, Vd(-1100.0), Vd(1100.0)) + Vd(0x1.8p52)) - Vl(0x4338000000000000LL);
  const Vl kh = ki >> Vl(1);
  const Vd t1 = cast<double>((kh + Vl(1023)) << Vl(52));
  const Vd t2 = cast<double>(((ki - kh) + Vl(1023)) << Vl(52));
  return p * t1 * t2;
}

// True when any lane of `mask` is set. PRECONDITION: every lane is all-ones
// or all-zero, which holds because the masks are built only from compares
// and & / |. This sits in the series and continued-fraction loop conditions,
// so it runs up to kMaxIt times per core, twice per float vector.
//
// x86 reads the sign bits straight out. Neither AVX2 nor AVX512 has a
// horizontal add, so reduce_add is a shuffle ladder of dependent VADDPDs,
// where VMOVMSKPD / VPMOVQ2M is one instruction.
//
// aarch64 goes the other way and sums. reduce_max is no good -- an all-ones
// double is a NaN and MAXPD drops it rather than propagating -- but
// reduce_add does carry it, since any set lane makes the whole sum NaN, and
// an UNORDERED compare reads that off directly, so the mask does not have to
// be ANDed down to 1.0 to be counted first. Note the compare has to be !=,
// not >: NaN > 0.0 is false. FADDP is cheap there and the sign-bit route is
// not, because it ends in a vector-to-GPR move: UMAXV measured 0.7% SLOWER
// than this on NEON, against the 2-3% this is worth over the ANDed form.
inline bool igamma_any(Vectorized<double> mask) {
#if defined(CPU_CAPABILITY_AVX512)
  return _mm512_movepi64_mask(_mm512_castpd_si512(mask)) != 0;
#elif defined(CPU_CAPABILITY_AVX2)
  return _mm256_movemask_pd(mask) != 0;
#else
  return mask.reduce_add() != 0.0;
#endif
}

// lgamma for x > 0 only, which is all igamma ever needs.
//
// This is lgamma_core_d from the lgamma port with neg statically false, so
// the reflection collapses: the degree-7 sinc polynomial, the round-to-
// nearest reduction and the SECOND logarithm all disappear, and sigma is
// the only blend left. Worth 12% (42.5 -> 37.6 ns) over calling
// lgamma_core_d, which would compute the reflection branchlessly and then
// discard it. If both ports land together this and lgamma_core_d should
// share a common positive-side body rather than duplicating the Stirling
// and core coefficients.
inline Vectorized<double> igamma_lgamma_pos_d(Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd one(1.0), zero(0.0);
  const Vd inf(std::numeric_limits<double>::infinity());
  const Vd big = x >= Vd(4.0);

  const Vd n = x.floor() - one;
  const Vd y = x - n;
  Vd prod = Vd::blendv(one, x - one, Vd(1.0) <= n);
  prod = prod * Vd::blendv(one, x - Vd(2.0), Vd(2.0) <= n);
  // x < 1 is the one case that divides rather than multiplies: there
  // lgamma(x) = lgamma(x+1) - log(x), so the log enters with a minus sign.
  const Vd lo = n < zero;
  const Vd arg = Vd::blendv(prod, x, lo);
  const Vd sigma = Vd::blendv(one, Vd(-1.0), lo);
  const Vd l1 = lgamma_log_d(Vd::blendv(arg, x, big));

  const Vd rz = one / x;
  const Vd w = rz * rz;
  Vd bs = Vd(-0x1.2a73847847132p-10);
  bs = fmadd(bs, w, Vd(0x1.a0ccbe2d203a7p-11));
  bs = fmadd(bs, w, Vd(-0x1.3757efe38828dp-11));
  bs = fmadd(bs, w, Vd(0x1.a01765cb92d0bp-11));
  bs = fmadd(bs, w, Vd(-0x1.6c16c08fffe2ep-9));
  bs = fmadd(bs, w, Vd(0x1.5555555553dbep-4));
  const Vd tail = fmadd(rz, bs, Vd(0x1.d67f1c864beb5p-1) - x);
  const Vd mainbig = fmadd(x - Vd(0.5), l1, tail);

  const Vd v = y - Vd(1.5);
  Vd cp = Vd(-0x1.b136081ab08c5p-14);
  cp = fmadd(cp, v, Vd(0x1.4ed034c21e33ap-13));
  cp = fmadd(cp, v, Vd(-0x1.3d27daa756979p-13));
  cp = fmadd(cp, v, Vd(0x1.0385448701d48p-12));
  cp = fmadd(cp, v, Vd(-0x1.cfc601fa02703p-12));
  cp = fmadd(cp, v, Vd(0x1.77c9a21f13946p-11));
  cp = fmadd(cp, v, Vd(-0x1.31174664112f9p-10));
  cp = fmadd(cp, v, Vd(0x1.f7ea878b16f65p-10));
  cp = fmadd(cp, v, Vd(-0x1.a504d3a7d6711p-9));
  cp = fmadd(cp, v, Vd(0x1.64f114fd1729fp-8));
  cp = fmadd(cp, v, Vd(-0x1.34dbd62d05759p-7));
  cp = fmadd(cp, v, Vd(0x1.133421f0f103bp-6));
  cp = fmadd(cp, v, Vd(-0x1.007aa84177e78p-5));
  cp = fmadd(cp, v, Vd(0x1.01af62a3a2749p-4));
  cp = fmadd(cp, v, Vd(-0x1.2aed059bd47eep-3));
  cp = fmadd(cp, v, Vd(0x1.eeb95b094bf6ep-2));
  const Vd core = (y - one) * (y - Vd(2.0)) * cp;

  const Vd res = Vd::blendv(fmadd(sigma, l1, core), mainbig, big);
  // x <= 0 covers +-0 and the negative axis; igamma turns both into 0 or
  // NaN downstream, so +inf here is only ever a placeholder.
  return Vd::blendv(res, inf, (x <= zero) | (x == inf));
}

inline Vectorized<double> igamma_core_d(Vectorized<double> a, Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd one(1.0), zero(0.0);
  // 1e-8, not the 1e-11 a double result would want: the series and the CF
  // both truncate at a RELATIVE tol, and a float carries 6e-8, so anything
  // below ~1e-8 is refining bits that the narrowing throws away. Worth 18%
  // on the mixed case (44.6 -> 37.7 ns) for 0.50 -> 0.81 ULP. The cap is
  // tied to the band width below, not to Cephes' 2000: |sigma| >= 0.3
  // outside the band bounds the series at ~83 terms at this tol.
  constexpr int kMaxIt = 600;
  const Vd tol(1e-8);

  const Vd sigma = (x - a) / a;
  // Region split. This band is WIDER than Cephes', deliberately: Cephes uses
  // |sigma| < 4.5/sqrt(a) for a > 200, and just outside it the series needs
  // ~25/|sigma| terms -- 5600 at a = 1e6 -- so it never converges within any
  // sane cap. That gap measured 111 ULP. Taking |sigma| < 0.3 for every
  // a > 20 closes it and bounds the series at ~83 terms.
  const Vd asym = (a > Vd(20.0)) & (sigma.abs() < Vd(0.3));
  const Vd notasym = (a <= Vd(20.0)) | (sigma.abs() >= Vd(0.3));
  const Vd use_cf = notasym & (x > Vd(1.1)) & (x > a);

  // fac = x^a e^-x / Gamma(a). The Lanczos form avoids lgamma but measures
  // SLOWER (50.5 against 37.6 ns): its 26 coefficient blends and two extra
  // divides cost more than the lgamma kernel does.
  //
  // Guarded, because the asymptotic arm never reads it: this is an lgamma,
  // a log and an exp, and skipping it on an all-ridge vector is worth 30%
  // (55.6 -> 39.0 ns). The guard is vector-wide, so fac = 0 for a skipped
  // vector is safe only while every lane of such a vector is overwritten
  // later -- by the asymptotic blend, or by a boundary blend and the bad
  // mask. asym and notasym are NOT complements: an ordered compare is false
  // for NaN, so a lane whose sigma is NaN is in neither arm and it is the bad
  // mask that catches it. Widen that mask, not this guard, if another
  // not-in-any-arm case appears.
  Vd fac = zero;
  if (igamma_any(notasym)) {
    const Vd lax = fmadd(a, dg_log_d(x), (x + igamma_lgamma_pos_d(a)).neg());
    fac = Vd::blendv(igamma_exp_d(lax), zero, lax < Vd(-745.0));
  }

  // ---- P power series, DLMF 8.11.4. Every term is positive, so there is
  // no cancellation. Seeded with the region mask, NOT all-ones: an arm a
  // lane does not use never converges, and an unmasked loop then always
  // runs to the cap -- that alone measured 20-240x slower than the scalar.
  Vd r = a, c = one, ans = one;
  Vd live = notasym & ((x <= Vd(1.1)) | (x <= a));
  for (int i = 0; i < kMaxIt && igamma_any(live); i++) {
    r = r + one;
    const Vd cn = c * (x / r);
    c = Vd::blendv(c, cn, live);
    ans = Vd::blendv(ans, ans + cn, live);
    live = live & (c > tol * ans);
  }

  // ---- Q by modified Lentz continued fraction, then P = 1 - Q.
  const Vd tiny(1e-300);
  Vd bb = x - a + one, cc(1e300), dd = one / bb, hh = dd;
  live = use_cf;
  for (int i = 1; i < kMaxIt && igamma_any(live); i++) {
    const Vd an = Vd(-(double)i) * (Vd((double)i) - a);
    bb = bb + Vd(2.0);
    Vd d2 = fmadd(an, dd, bb);
    d2 = Vd::blendv(d2, tiny, d2.abs() < tiny);
    Vd c2 = bb + an / cc;
    c2 = Vd::blendv(c2, tiny, c2.abs() < tiny);
    d2 = one / d2;
    const Vd del = d2 * c2;
    dd = Vd::blendv(dd, d2, live);
    cc = Vd::blendv(cc, c2, live);
    hh = Vd::blendv(hh, hh * del, live);
    live = live & ((del - one).abs() > tol);
  }
  Vd res = Vd::blendv(ans * fac / a, one - fac * hh, use_cf);

  // ---- uniform asymptotic expansion, DLMF 8.12.3, for the a ~ x ridge.
  // Guarded: even trimmed this block is ~100 FMAs and the ridge is ~1% of a
  // log-uniform sample, so paying it unconditionally would dominate.
  //
  // 5 outer terms and degree 12 inner, not the 9 x 22 a full-double kernel
  // needs. This is an ASYMPTOTIC truncation in a^-k and eta^n, and the band
  // pins a > 20 and |eta| < 0.31, so the dropped terms are already below
  // double: ridge 87.0 -> 59.2 ns with the max ULP unmoved. kIgErfcS, kIgG
  // and kIgM2 do NOT trim the same way -- those are minimax fits, where
  // dropping a term leaves an unfitted polynomial rather than a small
  // perturbation (measured 1.2e9, 657 and 28 ULP), so they would need a
  // refit at lower degree, not a truncation.
  if (igamma_any(asym)) {
    // eta = sigma*sqrt(M2(sigma)). Forming eta from log(x/a) - sigma
    // instead loses everything once |sigma| < 1e-9, which is exactly the
    // large-a part of this band.
    Vd m2 = Vd(kIgM2[16]);
    for (int i = 15; i >= 0; i--)
      m2 = fmadd(m2, sigma, Vd(kIgM2[i]));
    const Vd eta = sigma * m2.sqrt();
    const Vd z = eta * (a * Vd(0.5)).sqrt();
    const Vd z2 = z * z;
    const Vd e2 = igamma_exp_d(z2.neg()); // = exp(-a eta^2 / 2)
    const Vd az = z.abs();
    // S(z) = erfc(z) e^{z^2}, in two branches. The small-z polynomial alone
    // is not enough once the band is widened: |z| reaches 0.3*sqrt(a/2), and
    // at z = 7.8 exp(-z^2) is still 3e-27, nowhere near underflowing, so an
    // out-of-range S there is simply wrong (that cost 6.2e14 ULP). For
    // z >= 3, S = G(1/z^2)/(z sqrt(pi)) stays valid out to any z, because
    // u = 1/z^2 -> 0 and G(0) = 1.
    Vd sp = Vd(kIgErfcS[18]);
    for (int i = 17; i >= 0; i--)
      sp = fmadd(sp, az, Vd(kIgErfcS[i]));
    const Vd uz = one / maximum(z2, Vd(9.0));
    Vd gp = Vd(kIgG[12]);
    for (int i = 11; i >= 0; i--)
      gp = fmadd(gp, uz, Vd(kIgG[i]));
    sp = Vd::blendv(sp, gp / (az * Vd(1.7724538509055160273)), az >= Vd(3.0));
    const Vd half = e2 * sp;
    const Vd erfcz = Vd::blendv(Vd(2.0) - half, half, z <= zero);
    Vd sum = zero, afac = one;
    const Vd ra = one / a;
    for (int k = 0; k < 5; k++) {
      Vd ck = Vd(kIgD[k][12]);
      for (int n = 11; n >= 0; n--)
        ck = fmadd(ck, eta, Vd(kIgD[k][n]));
      sum = fmadd(ck, afac, sum);
      afac = afac * ra;
    }
    const Vd corr = e2 * sum / (a * Vd(6.283185307179586477)).sqrt();
    res = Vd::blendv(res, Vd(0.5) * erfcz - corr, asym);
  }

  // Every boundary, and the NaN/domain-error set, is applied by
  // igamma_bounds_f after both halves come back. Lanes belonging to one of
  // those arms hold whatever the regions left here and are overwritten.
  return res;
}

// Every boundary igamma and igammac have, applied at float width once,
// rather than at double width inside a core that runs twice per float
// vector. float -> double widening is exact, so each compare answers here
// exactly what it answered on the widened values, and the limits blended in
// (0 and 1) narrow exactly, so it makes no difference which side of the
// narrowing they are written on.
//
// The two functions share every arm with the limits exchanged: `hi` is the
// limit where x dominates (x == inf, or a == 0 with x > 0), `lo` the one
// where a dominates (a == inf, or x == 0). igamma passes (1, 0), igammac
// (0, 1). The domain-error and NaN set is identical for both.
//
// That set is six compares and five ORs written plainly, folded here to four
// and three: minimum() propagates NaN, so m < 0 is both sign tests at once
// and m != m is both NaN tests; and the two ill-defined pairs (0, 0) and
// (inf, inf) share their a == x half, after which x carries whatever a does,
// so testing a alone settles both. Note a == x is true for (+0, -0), which
// is the pair the zero arm is there for.
inline Vectorized<float> igamma_bounds_f(Vf res, Vf a, Vf x, Vf hi, Vf lo) {
  const Vf zero(0.0f);
  const Vf inf(std::numeric_limits<float>::infinity());

#if 1
  // On Pytorch, a NaN input yields NaN only when no boundary arm claims it:
  // calc_igamma(inf, nan) = 0 and calc_igamma(nan, inf) = 1. So the NaN
  // inputs are ORed in before the boundary blends, which then override them.
  const Vf m = minimum(a, x);
  res = res | (m != m);
  res = Vf::blendv(res, hi, x == inf);
  res = Vf::blendv(res, lo, a == inf);
  res = Vf::blendv(res, lo, x == zero);
  res = Vf::blendv(res, hi, (a == zero) & (x > zero));

  const Vf bad = (m < zero) | ((a == x) & ((a == zero) | (a == inf)));
  return res | bad;
#else
  res = Vf::blendv(res, hi, x == inf);
  res = Vf::blendv(res, lo, a == inf);
  res = Vf::blendv(res, lo, x == zero);
  res = Vf::blendv(res, hi, (a == zero) & (x > zero));

  const Vf m = minimum(a, x);
  const Vf bad = (m < zero) | (m != m) | ((a == x) & ((a == zero) | (a == inf)));
  return res | bad;
#endif
}

inline Vectorized<float> Vectorized<float>::igamma(const Vectorized<float>& x) const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(kR == 2);

  const VectorizedN<double, kR> ad = convert<double, kR, float, 1>(*this);
  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(x);
  VectorizedN<double, kR> rd;
  rd[0] = igamma_core_d(ad[0], xd[0]);
  rd[1] = igamma_core_d(ad[1], xd[1]);
  return igamma_bounds_f(convert<float, 1, double, kR>(rd), *this, x, Vf(1.0f), Vf(0.0f));
}

inline Vectorized<double> lgamma_corep_d(Vectorized<double> v) {
  using Vd = Vectorized<double>;
  Vd cp = Vd(-0x1.b136081ab08c5p-14);
  cp = fmadd(cp, v, Vd(0x1.4ed034c21e33ap-13));
  cp = fmadd(cp, v, Vd(-0x1.3d27daa756979p-13));
  cp = fmadd(cp, v, Vd(0x1.0385448701d48p-12));
  cp = fmadd(cp, v, Vd(-0x1.cfc601fa02703p-12));
  cp = fmadd(cp, v, Vd(0x1.77c9a21f13946p-11));
  cp = fmadd(cp, v, Vd(-0x1.31174664112f9p-10));
  cp = fmadd(cp, v, Vd(0x1.f7ea878b16f65p-10));
  cp = fmadd(cp, v, Vd(-0x1.a504d3a7d6711p-9));
  cp = fmadd(cp, v, Vd(0x1.64f114fd1729fp-8));
  cp = fmadd(cp, v, Vd(-0x1.34dbd62d05759p-7));
  cp = fmadd(cp, v, Vd(0x1.133421f0f103bp-6));
  cp = fmadd(cp, v, Vd(-0x1.007aa84177e78p-5));
  cp = fmadd(cp, v, Vd(0x1.01af62a3a2749p-4));
  cp = fmadd(cp, v, Vd(-0x1.2aed059bd47eep-3));
  return fmadd(cp, v, Vd(0x1.eeb95b094bf6ep-2));
}

// lnGamma(1+a) for a > 0, RELATIVE-accurate as a -> 0.
//
// For a < 1 this is the lgamma core at y = 1+a, but written as
// a*(a-1)*P(a-0.5) so the (y-1) factor is EXACTLY a. Forming y = 1+a first
// is precisely the shipped bug: 1+a rounds to 1 below 2^-53 and the factor
// becomes 0. Nothing about the polynomial changes -- only the order of
// operations -- and that is the whole fix.
inline Vectorized<double> igammac_lg1p_d(Vectorized<double> a) {
  using Vd = Vectorized<double>;
  // clamped so the a >= 1 lanes cannot evaluate P far outside its fit range
  const Vd as = minimum(a, Vd(1.0));
  const Vd sm = as * (as - Vd(1.0)) * lgamma_corep_d(as - Vd(0.5));
  const Vd bg = igamma_lgamma_pos_d(Vd(1.0) + a);
  return Vd::blendv(bg, sm, a < Vd(1.0));
}

// expm1 over (-745, 0.2]. Only |u| < 0.25 needs the series; above that
// exp(u) - 1 has nothing to cancel. Degree 8, not the 14 a double result
// wants: the truncation is u^9/9! = 4.2e-11 relative at the cut, invisible
// here, and it measures 11-14% (38.6 -> 34.4 ns). Degree 6 would be
// 0.25^7/7! = 4.8e-8, right at float eps, and reads 2.60 ULP.
inline Vectorized<double> igammac_expm1_d(Vectorized<double> u) {
  using Vd = Vectorized<double>;
  const Vd one(1.0);
  Vd p = fmadd(Vd(1.0 / 8.0), u, one);
  p = fmadd(p * Vd(1.0 / 7.0), u, one);
  p = fmadd(p * Vd(1.0 / 6.0), u, one);
  p = fmadd(p * Vd(1.0 / 5.0), u, one);
  p = fmadd(p * Vd(0.25), u, one);
  p = fmadd(p * Vd(1.0 / 3.0), u, one);
  p = fmadd(p * Vd(0.5), u, one);
  return Vd::blendv(igamma_exp_d(u) - one, u * p, u.abs() < Vd(0.25));
}

inline Vectorized<double> igammac_core_d(Vectorized<double> a, Vectorized<double> x) {
  using Vd = Vectorized<double>;
  const Vd one(1.0), zero(0.0);
  constexpr int kMaxIt = 600;
  const Vd tol(3e-8);

  const Vd sigma = (x - a) / a;
  const Vd asym = (a > Vd(20.0)) & (sigma.abs() < Vd(0.3));
  const Vd notasym = (a <= Vd(20.0)) | (sigma.abs() >= Vd(0.3));
  const Vd big_x = x > Vd(1.5);
  const Vd small = notasym & (x <= Vd(1.5));
  const Vd use_cf = notasym & big_x & (x > a);
  const Vd use_ps = notasym & big_x & (x <= a);

  // Guarded: the asymptotic arm reads neither, and together they are a log
  // plus a full lgamma. Worth 25% on an all-ridge vector (49.3 -> 37.2 ns).
  Vd lx = zero, lg1 = zero;
  if (igamma_any(notasym)) {
    lx = dg_log_d(x);
    // ONE lgamma-class call, not two: lnGamma(a) = lnGamma(1+a) - ln(a),
    // so the CF and P-series arms reuse the lg1p the 8.7.3 arm needs.
    lg1 = igammac_lg1p_d(a);
  }

  // ---- DLMF 8.7.3, x <= 1.5.  u = a*ln x - lnGamma(1+a).
  // exp(u) is never formed separately: exp(u) = 1 + expm1(u) = 1 - t1.
  Vd qs = zero;
  if (igamma_any(small)) {
    const Vd u = fmadd(a, lx, lg1.neg());
    const Vd t1 = igammac_expm1_d(u).neg();
    Vd g = one, sum = zero;
    Vd live = small;
    for (int n = 1; n < kMaxIt && igamma_any(live); n++) {
      g = g * (x.neg() * Vd(1.0 / (double)n));
      const Vd t = g / (a + Vd((double)n));
      sum = Vd::blendv(sum, sum + t, live);
      live = live & (t.abs() > tol * sum.abs());
    }
    qs = t1 - (a * (one - t1)) * sum;
  }

  // ---- fac = x^a e^-x / Gamma(a); only the CF and P-series arms read it
  Vd fac = zero;
  if (igamma_any(use_cf | use_ps)) {
    const Vd lax = fmadd(a, lx, (x + lg1 - dg_log_d(a)).neg());
    fac = Vd::blendv(igamma_exp_d(lax), zero, lax < Vd(-745.0));
  }

  // ---- P power series, DLMF 8.11.4, for x > 1.5 && x <= a.  Q = 1 - P.
  Vd r = a, c = one, ans = one;
  Vd live = use_ps;
  for (int i = 0; i < kMaxIt && igamma_any(live); i++) {
    r = r + one;
    const Vd cn = c * (x / r);
    c = Vd::blendv(c, cn, live);
    ans = Vd::blendv(ans, ans + cn, live);
    live = live & (c > tol * ans);
  }

  // ---- Lentz CF, DLMF 8.9.2, for x > 1.5 && x > a.  Q = fac*hh, with no
  // subtraction at all -- this is the arm where Q gets small.
  const Vd tiny(1e-300);
  Vd bb = x - a + one, cc(1e300), dd = one / bb, hh = dd;
  live = use_cf;
  for (int i = 1; i < kMaxIt && igamma_any(live); i++) {
    const Vd an = Vd(-(double)i) * (Vd((double)i) - a);
    bb = bb + Vd(2.0);
    Vd d2 = fmadd(an, dd, bb);
    d2 = Vd::blendv(d2, tiny, d2.abs() < tiny);
    Vd c2 = bb + an / cc;
    c2 = Vd::blendv(c2, tiny, c2.abs() < tiny);
    d2 = one / d2;
    const Vd del = d2 * c2;
    dd = Vd::blendv(dd, d2, live);
    cc = Vd::blendv(cc, c2, live);
    hh = Vd::blendv(hh, hh * del, live);
    live = live & ((del - one).abs() > tol);
  }

  Vd res = Vd::blendv(one - ans * fac / a, fac * hh, use_cf);
  res = Vd::blendv(res, qs, small);

  // ---- uniform asymptotic expansion, DLMF 8.12.4. Identical to igamma's
  // 8.12.3 block except sgn = +1 instead of -1, which shows up as the two
  // erfc blend arms swapping and corr being ADDED rather than subtracted.
  //
  // 3 outer terms and degree 8 inner, trimmed harder than igamma's 5 and 12
  // because here the asymptotic arm is not what sets the maximum: the CF arm
  // reads 1.0961, so letting the ridge rise to 1.0141 costs nothing globally
  // and takes the ridge from 37.1 to 29.6 ns. Checked at the BOTTOM of the
  // band (a in 20..200), where the a^-k truncation is largest.
  if (igamma_any(asym)) {
    Vd m2 = Vd(kIgM2[16]);
    for (int i = 15; i >= 0; i--)
      m2 = fmadd(m2, sigma, Vd(kIgM2[i]));
    const Vd eta = sigma * m2.sqrt();
    const Vd z = eta * (a * Vd(0.5)).sqrt();
    const Vd z2 = z * z;
    const Vd e2 = igamma_exp_d(z2.neg());
    const Vd az = z.abs();
    Vd sp = Vd(kIgErfcS[18]);
    for (int i = 17; i >= 0; i--)
      sp = fmadd(sp, az, Vd(kIgErfcS[i]));
    const Vd uz = one / maximum(z2, Vd(9.0));
    Vd gp = Vd(kIgG[12]);
    for (int i = 11; i >= 0; i--)
      gp = fmadd(gp, uz, Vd(kIgG[i]));
    sp = Vd::blendv(sp, gp / (az * Vd(1.7724538509055160273)), az >= Vd(3.0));
    const Vd half = e2 * sp;
    // erfc(-z): the mirror of igamma's erfc(z), so the blend arms swap
    const Vd erfcmz = Vd::blendv(half, Vd(2.0) - half, z <= zero);
    Vd sum = zero, afac = one;
    const Vd ra = one / a;
    for (int k = 0; k < 3; k++) {
      Vd ck = Vd(kIgD[k][8]);
      for (int n = 7; n >= 0; n--)
        ck = fmadd(ck, eta, Vd(kIgD[k][n]));
      sum = fmadd(ck, afac, sum);
      afac = afac * ra;
    }
    const Vd corr = e2 * sum / (a * Vd(6.283185307179586477)).sqrt();
    res = Vd::blendv(res, fmadd(Vd(0.5), erfcmz, corr), asym);
  }

  // Every boundary, and the NaN/domain-error set, is applied by
  // igamma_bounds_f after both halves come back. Lanes belonging to one of
  // those arms hold whatever the regions left here and are overwritten.
  return res;
}

inline Vectorized<float> Vectorized<float>::igammac(const Vectorized<float>& x) const {
  constexpr int kR = Vf::size() / Vectorized<double>::size();
  static_assert(kR == 2);

  const VectorizedN<double, kR> ad = convert<double, kR, float, 1>(*this);
  const VectorizedN<double, kR> xd = convert<double, kR, float, 1>(x);
  VectorizedN<double, kR> rd;
  rd[0] = igammac_core_d(ad[0], xd[0]);
  rd[1] = igammac_core_d(ad[1], xd[1]);
  return igamma_bounds_f(convert<float, 1, double, kR>(rd), *this, x, Vf(0.0f), Vf(1.0f));
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
