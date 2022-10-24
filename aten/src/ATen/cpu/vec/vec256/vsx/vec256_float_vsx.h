#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <sleef.h>
namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]

inline namespace CPU_CAPABILITY {

template <>
class Vectorized<float> {
 private:
  union {
    struct {
      vfloat32 _vec0;
      vfloat32 _vec1;
    };
    struct {
      vbool32 _vecb0;
      vbool32 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = float;
  using vec_internal_type = vfloat32;
  using vec_internal_mask_type = vbool32;
  using size_type = int;

  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(vfloat32 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vfloat32 v1, vfloat32 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}
  C10_ALWAYS_INLINE Vectorized(float scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  C10_ALWAYS_INLINE Vectorized(
      float scalar1,
      float scalar2,
      float scalar3,
      float scalar4,
      float scalar5,
      float scalar6,
      float scalar7,
      float scalar8)
      : _vec0{vfloat32{scalar1, scalar2, scalar3, scalar4}},
        _vec1{vfloat32{scalar5, scalar6, scalar7, scalar8}} {}
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 0, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 1, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 2, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 3, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 4, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 5, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 6, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_2nd = VsxMask2(mask);
    // generated masks
    return {a._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 7, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_2nd = VsxMask2(mask);
    // generated masks
    return {b._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 8, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    const vbool32 mask_2nd = VsxMask2(mask);
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorized<float> C10_ALWAYS_INLINE blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    // the mask used here returned by comparision of vec256
    // assuming this we can use the same mask directly with vec_sel
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }

  template <typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }
  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      size_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }

    return b;
  }
  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    __at_align__ value_type tmp_values[size()];
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;

  Vectorized<float> map(float (*const f)(float)) const {
    Vectorized<float> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
    }
    for (int i = 0; i < size() / 2; i++) {
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  Vectorized<float> mapbi(float (*const f)(float, float), const Vectorized<float>& other)
      const {
    Vectorized<float> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i], other._vec0[i]);
    }
    for (int i = 0; i < size() / 2; i++) {
      ret._vec1[i] = f(_vec1[i], other._vec1[i]);
    }
    return ret;
  }

  Vectorized<float> _nor() const {
    return {vec_nor(_vec0, _vec0), vec_nor(_vec1, _vec1)};
  }

  Vectorized<float> isnan() const {
    auto x = *this;
    auto ret = (x == x);
    return ret._nor();
  }

  Vectorized<float> _isinf() const {
    auto x = *this;
    return (x == v_inf) | (x == v_minus_inf);
  }

  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    //__m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
    auto cmp = (*this == zero);
    // return _mm256_movemask_ps(cmp);
    // possible simulation  //mask= lvsl ( 0 ) vbpermq( vec, mask <<5)
    vuint64 result0 = vec_vbpermq((vuint8)cmp._vecb0, mask_zero_bits);
    vuint64 result1 = vec_vbpermq((vuint8)cmp._vecb1, mask_zero_bits);
    return (result0[1] >> 12 | (result1[1] >> 8));
  }

  Vectorized<float> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE acos() const {
     return {Sleef_acosf4_u10vsx(_vec0), Sleef_acosf4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE asin() const {
     return {Sleef_asinf4_u10vsx(_vec0), Sleef_asinf4_u10vsx(_vec1)};
  }
  Vectorized<float> atan() const {
     return {Sleef_atanf4_u10vsx(_vec0), Sleef_atanf4_u10vsx(_vec1)};
  }
  Vectorized<float> atan2(const Vectorized<float>& b) const {
     return {Sleef_atan2f4_u10vsx(_vec0, b._vec0), Sleef_atan2f4_u10vsx(_vec1, b._vec1)};
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return {Sleef_copysignf4_vsx(_vec0, sign._vec0), Sleef_copysignf4_vsx(_vec1, sign._vec1)};
  }
  Vectorized<float> lgamma() const {
     return {Sleef_lgammaf4_u10vsx(_vec0), Sleef_lgammaf4_u10vsx(_vec1)};
  }
  Vectorized<float> erf() const {
     return {Sleef_erff4_u10vsx(_vec0), Sleef_erff4_u10vsx(_vec1)};
  }

  Vectorized<float> erfc() const {
     return {Sleef_erfcf4_u15vsx(_vec0), Sleef_erfcf4_u15vsx(_vec1)};
  }

  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }

  Vectorized<float> angle() const {
    auto tmp = blendv(
      Vectorized<float>(0), Vectorized<float>(c10::pi<float>), *this < Vectorized<float>(0));
    return blendv(tmp, *this, isnan());
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return Vectorized<float>{0};
  }
  Vectorized<float> conj() const {
    return *this;
  }

  Vectorized<float> C10_ALWAYS_INLINE exp() const {
    // implementation logic from avx_mathfun with some modifications from sleef
    // Express e**x = e**g 2**n
    ///   = e**g e**( n loge(2) )
    ///   = e**( g + n loge(2) )
    //
    auto tmp_x = *this;
    auto fx = (tmp_x * log2e_inv).round();

    auto x = fx.madd(negln2f_hi, tmp_x);
    x = fx.madd(negln2f_lo, x);
    auto z = x * x;
    auto y = x.madd(exp_p0, exp_p1);
    y = y.madd(x, exp_p2);
    y = y.madd(x, exp_p3);
    y = y.madd(x, exp_p4);
    y = y.madd(x, exp_p5);
    y = y.madd(z, x) + one;

    // vm_pow2n 2^n
    vint32 imm0 = vec_signed(fx._vec0);
    vint32 imm1 = vec_signed(fx._vec1);
    // this pow2n logic is  from Sleef code
    vint32 imm00 = imm0 >> 1; //>>1
    vint32 imm01 = imm1 >> 1;
    vint32 imm10 = imm0 - imm00;
    vint32 imm11 = imm1 - imm01;
    imm00 = (imm00 + v0x7f) << vu_23;
    imm01 = (imm01 + v0x7f) << vu_23;
    imm10 = (imm10 + v0x7f) << vu_23;
    imm11 = (imm11 + v0x7f) << vu_23;
    // treat imm as float vector without conversion

    y._vec0 = (y._vec0 * (vfloat32)imm00) * (vfloat32)imm10;
    y._vec1 = (y._vec1 * (vfloat32)imm01) * (vfloat32)imm11;
    // boundary check
    auto tmp = blendv(y, v_inf, (Vectorized<float>(exp_hi) <= tmp_x));
    y = blendv(tmp, zero, (tmp_x < Vectorized<float>(exp_lo)));

    return y;
  }
  Vectorized<float> expm1() const {
    return exp() - one;
  }

  Vectorized<float> C10_ALWAYS_INLINE log() const {
    return {Sleef_logf4_u10vsx(_vec0), Sleef_logf4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE log10() const {
     return {Sleef_log10f4_u10vsx(_vec0), Sleef_log10f4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE log1p() const {
     return {Sleef_log1pf4_u10vsx(_vec0), Sleef_log1pf4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE log2() const {
     return {Sleef_log2f4_u10vsx(_vec0), Sleef_log2f4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE cos() const {
    // take the absolute value
    auto x = abs();
    // extract the sign bit (upper one)
    auto sign_bit = (*this) & sign_mask;
    // scale by 4/Pi
    auto y = x * _4div_pi;
    // store the integer part of y in mm0
    // j=(j+1) & (~1) (see the cephes sources)
    vint32 imm0 = (vec_signed(y._vec0) + vi_1) & vi_inv1;
    vint32 imm1 = (vec_signed(y._vec1) + vi_1) & vi_inv1;
    y._vec0 = vec_float(imm0);
    y._vec1 = vec_float(imm1);

    imm0 = imm0 - vi_2;
    imm1 = imm1 - vi_2;
    Vectorized<float> poly_mask;
    // get the swap sign flag
    vint32 tmp0 = vec_and(vec_nand(imm0, imm0), vi_4);
    vint32 tmp1 = vec_and(vec_nand(imm1, imm1), vi_4);
    sign_bit._vecb0 = (vbool32)vec_sl(tmp0, vu_29);
    sign_bit._vecb1 = (vbool32)vec_sl(tmp1, vu_29);
    // get the polynom selection mask
    // there is one polynom for 0 <= x <= Pi / 4
    // and another one for Pi / 4 < x <= Pi / 2
    // Both branches will be computed.

    poly_mask._vecb0 = (vbool32)vec_cmpeq((imm0 & vi_2), vi_0);
    poly_mask._vecb1 = (vbool32)vec_cmpeq((imm1 & vi_2), vi_0);

    // The magic pass: "Extended precision modular arithmetic"
    //  x = ((x - y * DP1) - y * DP2) - y * DP3;
    x = y.madd(minus_cephes_dp1, x);
    x = y.madd(minus_cephes_dp2, x);
    x = y.madd(minus_cephes_dp3, x);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    auto z = x * x;
    y = z.madd(coscof_p0, coscof_p1);
    y = y.madd(z, coscof_p2);
    y = y * z * z;
    y = y - z * half + one;

    // Evaluate the second polynom  (Pi/4 <= x <= 0)
    auto y_2 = z.madd(sincof_p0, sincof_p1);
    y_2 = y_2.madd(z, sincof_p2);
    y_2 = y_2 * z;
    y_2 = y_2.madd(x, x);

    // select the correct result from the two polynoms
    y = blendv(y, y_2, poly_mask);
    // update the sign
    y = y ^ sign_bit;

    return y;
  }
  Vectorized<float> C10_ALWAYS_INLINE cosh() const {
    // cosh = 1/2 * (e^x + e^-x)
    auto x = abs();
    auto e_x = x.exp();
    auto ret = (e_x + Vectorized<float>(one) / e_x) * half;
    // inf and nan checks
#if 0
                    ret = blendv(ret, v_inf, x >= vf_89);
                    ret = blendv(ret, v_inf, ret.isnan());
                    ret = blendv(ret, v_nan, this->isnan());
#endif
    return ret;
  }
  Vectorized<float> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE sin() const {
    // take the absolute value and xtract sign
    auto x = abs();
    auto sign_bit = (*this) & sign_mask;

    // scale by 4/Pi
    auto y = x * _4div_pi;
    // store the integer part of y in mm0

    // j=(j+1) & (~1) (see the cephes sources)
    vint32 imm0 = (vec_signed(y._vec0) + vi_1) & vi_inv1;
    vint32 imm1 = (vec_signed(y._vec1) + vi_1) & vi_inv1;
    y._vec0 = vec_float(imm0);
    y._vec1 = vec_float(imm1);
    // get the swap sign flag
    Vectorized<float> swap_sign_bit, poly_mask;
    swap_sign_bit._vecb0 = (vbool32)vec_sl(imm0 & vi_4, vu_29);
    swap_sign_bit._vecb1 = (vbool32)vec_sl(imm1 & vi_4, vu_29);
    // get the polynom selection mask
    // there is one polynom for 0 <= x <= Pi/4
    // and another one for Pi/4<x<=Pi/2
    // Both branches will be computed.

    poly_mask._vecb0 = vec_cmpeq((imm0 & vi_2), vi_0);
    poly_mask._vecb1 = vec_cmpeq((imm1 & vi_2), vi_0);
    sign_bit = sign_bit ^ swap_sign_bit; // xor operation

    // The magic pass: "Extended precision modular arithmetic"
    //  x = ((x - y * DP1) - y * DP2) - y * DP3;
    x = y.madd(minus_cephes_dp1, x);
    x = y.madd(minus_cephes_dp2, x);
    x = y.madd(minus_cephes_dp3, x);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    auto z = x * x;
    y = z.madd(coscof_p0, coscof_p1);
    y = y.madd(z, coscof_p2);
    y = y * z * z;
    y = y - z * half + one;

    // Evaluate the second polynom  (Pi/4 <= x <= 0)
    auto y2 = z.madd(sincof_p0, sincof_p1);
    y2 = y2.madd(z, sincof_p2);
    y2 = y2 * z;
    y2 = y2.madd(x, x);
    // select the correct result from the two polynoms
    y = blendv(y, y2, poly_mask);
    y = y ^ sign_bit;

    return y;
  }
  Vectorized<float> C10_ALWAYS_INLINE sinh() const {
    auto temp_abs = abs();
    // get exponent
    auto ret = temp_abs.exp();
    auto recp = Vectorized<float>(half) / ret;
    auto v = ret * half - recp;
    // extract the sign bit (upper one)
    auto sign_bit = (*this) & sign_mask;
    auto z = temp_abs * temp_abs;
    auto y = z.madd(p0, p1);
    y = y.madd(z, p2);
    y = (y * z).madd(temp_abs, temp_abs);
    // check and select
    auto result = blendv(y, v, temp_abs > one);
    return result | sign_bit;
  }
  Vectorized<float> C10_ALWAYS_INLINE tan() const {
     return {Sleef_tanf4_u10vsx(_vec0), Sleef_tanf4_u10vsx(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE tanh() const {
    auto x = *this;
    auto vabs = abs();
    // get exponent
    auto exp2x = (vabs + vabs).exp();
    auto vv = Vectorized<float>(one) - Vectorized<float>(two) / (exp2x + one);
    // extract the sign bit (upper one)
    auto sign_bit = (*this) & sign_mask;
    auto z = vabs * vabs;
    auto y = z.madd(tanh_p0, tanh_p1);
    auto tmp = y.madd(z, tanh_p2);
    y = z.madd(tmp, tanh_p3);
    tmp = y.madd(z, tanh_p4);
    y = tmp * z;
    tmp = y.madd(x, x);
    // add sign
    vv = vv | sign_bit;
    // check and select
    auto sel_mask = vabs >= tanh_0p625;
    auto max_mask = vabs > tanh_half_max;
    auto max_ret = sign_bit ^ one;
    return blendv(blendv(tmp, vv, sel_mask), max_ret, max_mask);
  }
  Vectorized<float> C10_ALWAYS_INLINE trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE frac() const {
    return *this - trunc();
  }

  Vectorized<float> C10_ALWAYS_INLINE sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<float> C10_ALWAYS_INLINE reciprocal() const {
    return Vectorized<float>(one) / (*this);
  }
  Vectorized<float> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }

  Vectorized<float> C10_ALWAYS_INLINE pow(const Vectorized<float>& exp) const {
    return {Sleef_powf4_u10vsx(_vec0, exp._vec0), Sleef_powf4_u10vsx(_vec1, exp._vec1)};
  }

  Vectorized<float> fmod(const Vectorized<float>& b) const {
     return {Sleef_fmodf4_vsx(_vec0, b._vec0),Sleef_fmodf4_vsx(_vec1, b._vec1)};
  }

  Vectorized<float> hypot(const Vectorized<float>& b) const {
     return {Sleef_hypotf4_u05vsx(_vec0, b._vec0), Sleef_hypotf4_u05vsx(_vec1, b._vec1)};
  }

  Vectorized<float> nextafter(const Vectorized<float>& b) const {
     return {Sleef_nextafterf4_vsx(_vec0, b._vec0), Sleef_nextafterf4_vsx(_vec1, b._vec1)};
  }

  Vectorized<float> igamma(const Vectorized<float>& x) const {
    return mapbi(calc_igamma, x);
  }

  Vectorized<float> igammac(const Vectorized<float>& x) const {
    return mapbi(calc_igammac, x);
  }

  Vectorized<float> i0() const {
    return map(calc_i0);
  }

  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }

  DEFINE_MEMBER_OP(operator==, float, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, float, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, float, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, float, vec_cmple)
  DEFINE_MEMBER_OP(operator>, float, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, float, vec_cmpge)
  DEFINE_MEMBER_OP_AND_ONE(eq, float, vec_cmpeq)
  DEFINE_MEMBER_OP_AND_ONE(ne, float, vec_cmpne)
  DEFINE_MEMBER_OP_AND_ONE(lt, float, vec_cmplt)
  DEFINE_MEMBER_OP_AND_ONE(le, float, vec_cmple)
  DEFINE_MEMBER_OP_AND_ONE(gt, float, vec_cmpgt)
  DEFINE_MEMBER_OP_AND_ONE(ge, float, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, float, vec_add)
  DEFINE_MEMBER_OP(operator-, float, vec_sub)
  DEFINE_MEMBER_OP(operator*, float, vec_mul)
  DEFINE_MEMBER_OP(operator/, float, vec_div)
  DEFINE_MEMBER_OP(maximum, float, vec_max_nan2)
  DEFINE_MEMBER_OP(minimum, float, vec_min_nan2)
  DEFINE_MEMBER_OP(operator&, float, vec_and)
  DEFINE_MEMBER_OP(operator|, float, vec_or)
  DEFINE_MEMBER_OP(operator^, float, vec_xor)
  DEFINE_MEMBER_TERNARY_OP(madd, float, vec_madd)
};

template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return a.maximum(b);
}

template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return a.minimum(b);
}

} // namespace
} // namespace vec
} // namespace at
