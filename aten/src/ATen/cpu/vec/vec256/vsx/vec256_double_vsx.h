#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <c10/util/irange.h>

#include <sleef.h>

namespace at {
namespace vec {

inline namespace CPU_CAPABILITY {


template <>
class Vectorized<double> {
 private:
  union {
    struct {
      vfloat64 _vec0;
      vfloat64 _vec1;
    };
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = double;
  using vec_internal_type = vfloat64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  C10_ALWAYS_INLINE Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}
  C10_ALWAYS_INLINE Vectorized(double scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  C10_ALWAYS_INLINE Vectorized(
      double scalar1,
      double scalar2,
      double scalar3,
      double scalar4)
      : _vec0{vfloat64{scalar1, scalar2}}, _vec1{vfloat64{scalar3, scalar4}} {}
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  int zero_mask() const {
    auto cmp = (*this == vd_zero);
    return (cmp._vecb0[0] & 1) | (cmp._vecb0[1] & 2) | (cmp._vecb1[0] & 4) |
        (cmp._vecb1[1] & 8);
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 0, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 1, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 2, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return { b._vec0, a._vec1 };
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 3, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return { a._vec0, b._vec1 };
  }


  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 4, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      return { (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1 };
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 5, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      return { (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1 };
  }


  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 6,
      Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_2nd = VsxDblMask2(mask);
      // generated masks
      return { a._vec0,
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 7,
      Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_2nd = VsxDblMask2(mask);
      // generated masks
      return { b._vec0,
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 8, Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      const vbool64 mask_2nd = VsxDblMask2(mask);
      return {
          (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st),
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }


  static Vectorized<double> C10_ALWAYS_INLINE blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    // the mask used here returned by comparision of vec256

    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }
  template <typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
  }

  static Vectorized<double> C10_ALWAYS_INLINE
  set(const Vectorized<double>& a, const Vectorized<double>& b, size_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
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

    __at_align__ value_type tmp_values[size()] = {};
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
  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;
  Vectorized<double> map(double (*const f)(double)) const {
    Vectorized<double> ret;
    for (const auto i : c10::irange(size()/2)) {
        ret._vec0[i] = f(_vec0[i]);
    }
    for (const auto i : c10::irange(size()/2)) {
        ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  Vectorized<double> mapbi(double (*const f)(double, double), const Vectorized<double>& other)
      const {
    Vectorized<double> ret;
    for (const auto i : c10::irange(size()/2)) {
        ret._vec0[i] = f(_vec0[i], other._vec0[i]);
    }
    for (const auto i : c10::irange(size()/2)) {
        ret._vec1[i] = f(_vec1[i], other._vec1[i]);
    }
    return ret;
  }
  Vectorized<double> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<double> C10_ALWAYS_INLINE acos() const {
     return {Sleef_acosd2_u10(_vec0), Sleef_acosd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE asin() const {
     return {Sleef_asind2_u10(_vec0), Sleef_asind2_u10(_vec1)};
  }
  Vectorized<double> atan() const {
     return {Sleef_atand2_u10(_vec0), Sleef_atand2_u10(_vec1)};
  }
  Vectorized<double> atanh() const {
     return {Sleef_atanhd2_u10(_vec0), Sleef_atanhd2_u10(_vec1)};
  }
  Vectorized<double> atan2(const Vectorized<double>& b) const {
     return {Sleef_atan2d2_u10(_vec0, b._vec0), Sleef_atan2d2_u10(_vec1, b._vec1)};
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
    return {Sleef_copysignd2(_vec0, sign._vec0), Sleef_copysignd2(_vec1, sign._vec1)};
  }
  Vectorized<double> erf() const {
     return {Sleef_erfd2_u10(_vec0), Sleef_erfd2_u10(_vec1)};
  }
  Vectorized<double> erfc() const {
     return {Sleef_erfcd2_u15(_vec0), Sleef_erfcd2_u15(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE exp() const {
     return {Sleef_expd2_u10(_vec0), Sleef_expd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE exp2() const {
    return {Sleef_exp2d2_u10(_vec0), Sleef_exp2d2_u10(_vec1)};
  }
  Vectorized<double> expm1() const {
     return {Sleef_expm1d2_u10(_vec0), Sleef_expm1d2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE exp_u20() const {
     return exp();
  }

  Vectorized<double> lgamma() const __ubsan_ignore_undefined__ {
     return {Sleef_lgammad2_u10(_vec0), Sleef_lgammad2_u10(_vec1)};
  }

  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }

  Vectorized<double> angle() const {
    auto tmp = blendv(
      Vectorized<double>(0), Vectorized<double>(c10::pi<double>), *this < Vectorized<double>(0));
    return blendv(tmp, *this, isnan());
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return Vectorized<double>{0};
  }
  Vectorized<double> conj() const {
    return *this;
  }

  Vectorized<double> C10_ALWAYS_INLINE log() const {
     return {Sleef_logd2_u10(_vec0), Sleef_logd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE log10() const {
     return {Sleef_log10d2_u10(_vec0), Sleef_log10d2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE log1p() const {
     return {Sleef_log1pd2_u10(_vec0), Sleef_log1pd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE log2() const {
     return {Sleef_log2d2_u10(_vec0), Sleef_log2d2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE cos() const {
     return {Sleef_cosd2_u10(_vec0), Sleef_cosd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE cosh() const {
     return {Sleef_coshd2_u10(_vec0), Sleef_coshd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE round() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE sin() const {
     return {Sleef_sind2_u10(_vec0), Sleef_sind2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE sinh() const {
     return {Sleef_sinhd2_u10(_vec0), Sleef_sinhd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE tan() const {
     return {Sleef_tand2_u10(_vec0), Sleef_tand2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE tanh() const {
     return {Sleef_tanhd2_u10(_vec0), Sleef_tanhd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<double> C10_ALWAYS_INLINE frac() const {
    return *this - trunc();
  }

  Vectorized<double> C10_ALWAYS_INLINE sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE reciprocal() const {
    return {
        vec_div(vd_one, _vec0), // vec_re(_vec0) is estimated one.
        vec_div(vd_one, _vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE rsqrt() const {
    return sqrt().reciprocal();
  }

  Vectorized<double> C10_ALWAYS_INLINE pow(const Vectorized<double>& b) const {
     return {Sleef_powd2_u10(_vec0, b._vec0), Sleef_powd2_u10(_vec1, b._vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE fmod(const Vectorized<double>& b) const {
     return {Sleef_fmodd2(_vec0, b._vec0),Sleef_fmodd2(_vec1, b._vec1)};
  }

  Vectorized<double> hypot(const Vectorized<double>& b) const {
     return {Sleef_hypotd2_u05(_vec0, b._vec0), Sleef_hypotd2_u05(_vec1, b._vec1)};
  }

  Vectorized<double> nextafter(const Vectorized<double>& b) const {
     return {Sleef_nextafterd2(_vec0, b._vec0), Sleef_nextafterd2(_vec1, b._vec1)};
  }

  Vectorized<double> igamma(const Vectorized<double>& x) const {
    return mapbi(calc_igamma, x);
  }

  Vectorized<double> igammac(const Vectorized<double>& x) const {
    return mapbi(calc_igammac, x);
  }


  Vectorized<double> i0() const {
    return map(calc_i0);
  }

  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }

  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }

  Vectorized<double> _nor() const {
    return {vec_nor(_vec0, _vec0), vec_nor(_vec1, _vec1)};
  }

  Vectorized<double> isnan() const {
    auto x = *this;
    auto ret = (x == x);
    return ret._nor();
  }
  bool has_inf_nan() const {
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec0[i]) || _isinf(_vec0[i])) {
        return true;
      }
    }
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec1[i]) || _isinf(_vec1[i])) {
        return true;
      }
    }
    return false;
  }

  DEFINE_MEMBER_OP(operator==, double, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, double, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, double, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, double, vec_cmple)
  DEFINE_MEMBER_OP(operator>, double, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, double, vec_cmpge)
  DEFINE_MEMBER_OP_AND_ONE(eq, double, vec_cmpeq)
  DEFINE_MEMBER_OP_AND_ONE(ne, double, vec_cmpne)
  DEFINE_MEMBER_OP_AND_ONE(lt, double, vec_cmplt)
  DEFINE_MEMBER_OP_AND_ONE(le, double, vec_cmple)
  DEFINE_MEMBER_OP_AND_ONE(gt, double, vec_cmpgt)
  DEFINE_MEMBER_OP_AND_ONE(ge, double, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, double, vec_add)
  DEFINE_MEMBER_OP(operator-, double, vec_sub)
  DEFINE_MEMBER_OP(operator*, double, vec_mul)
  DEFINE_MEMBER_OP(operator/, double, vec_div)
  DEFINE_MEMBER_OP(maximum, double, vec_max_nan2)
  DEFINE_MEMBER_OP(minimum, double, vec_min_nan2)
  DEFINE_MEMBER_OP(operator&, double, vec_and)
  DEFINE_MEMBER_OP(operator|, double, vec_or)
  DEFINE_MEMBER_OP(operator^, double, vec_xor)
  DEFINE_MEMBER_TERNARY_OP(madd, double, vec_madd)
};
template <>
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.maximum(b);
}

template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.minimum(b);
}
} // namespace
} // namespace vec
} // namespace at
