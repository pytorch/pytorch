#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/cpu/vec256/vsx/vsx_helpers.h>
#include <c10/util/qint32.h>
#include <array>

// This file defines Vec256<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vec256<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vec256<qint32> -> 1x Vec256<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vec256<T>::float_num_vecs
// iterations.

namespace at {
namespace vec256 {
namespace {

template <>
struct Vec256<c10::qint32> {
 private:
  union {
    struct {
      vint32 _vec0;
      vint32 _vec1;
    };
    struct {
      vbool32 _vecb0;
      vbool32 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  Vec256() {}

  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }

  static constexpr size_t float_num_vecs() {
    return 1;
  }
  static constexpr int int_num_vecs() {
    return 1;
  }
  using float_vec_return_type = std::array<Vec256<float>, 1>;
  using int_vec_return_type = std::array<Vec256<c10::qint32>, 1>;
  using value_type = c10::qint32::underlying;
  using vec_internal_type = vint32;
  using vec_internal_mask_type = vbool32;
  C10_ALWAYS_INLINE Vec256(vint32 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vec256(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vec256(vint32 v1, vint32 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vec256(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}

  Vec256(const c10::qint32& val)
      : _vec0(vec_splats(val.val_)), _vec1(vec_splats(val.val_)) {}

  static Vec256<c10::qint32> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    __at_align32__ value_type tmp_values[size()];
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      __at_align32__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    vfloat32 float_vals0 = vec_float(_vec0);
    vfloat32 float_vals1 = vec_float(_vec1);
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    vfloat32 scale_zp_premul0 = scale_zp_premul.vec0();
    vfloat32 scale_zp_premul1 = scale_zp_premul.vec1();
    return {Vec256<float>{
        vec_madd(scale_vec0, float_vals0, scale_zp_premul0),
        vec_madd(scale_vec1, float_vals1, scale_zp_premul1)}};
  }

  static Vec256<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    Vec256<c10::qint32> retval;

    const vint32 vmin = vec_splats(std::numeric_limits<value_type>::min());
    const vint32 vmax = vec_splats(std::numeric_limits<value_type>::max());
    vfloat32 inverse_scale_v = vec_splats(inverse_scale);
    vfloat32 vec_zero_point = vec_splats((float)(zero_point));
    Vec256<float> vf0 = rhs[0];

    vfloat32 vecf0 = vf0.vec0();
    vfloat32 vecf1 = vf0.vec1();
    vecf0 = vec_mul(vecf0, inverse_scale_v);
    vecf1 = vec_mul(vecf1, inverse_scale_v);
    vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
    vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
    vint32 veci0  = vec_signed(vecf0);
    vint32 veci1  = vec_signed(vecf1);

    veci0 = vec_max(veci0, vmin);
    veci1 = vec_max(veci1, vmin);
    veci0 = vec_min(veci0, vmax);
    veci1 = vec_min(veci1, vmax);

    return {veci0, veci1};
  }

  Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const {
    return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
  }

  Vec256<c10::qint32> relu6(
      Vec256<c10::qint32> zero_point,
      Vec256<c10::qint32> q_six) const {
    vint32 max0 = vec_max(_vec0, zero_point._vec0);
    vint32 max1 = vec_max(_vec1, zero_point._vec1);
    return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
  }

  int_vec_return_type widening_subtract(Vec256<c10::qint32> b) const {
    return {*this - b};
  }

  static Vec256<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    const vint32 vmin = vec_splats(std::numeric_limits<value_type>::min());
    const vint32 vmax = vec_splats(std::numeric_limits<value_type>::max());
    vfloat32 vec_mult = vec_splats(multiplier);
    vint32 vec_zero_point = vec_splats(zero_point);
    Vec256<c10::qint32> vi = inp[0];
    vfloat32 vecf0 = vec_float(vi.vec0());
    vfloat32 vecf1 = vec_float(vi.vec1());

    vecf0 = vec_mul(vecf0, vec_mult);
    vecf1 = vec_mul(vecf1, vec_mult);

    vecf0 = vec_rint(vecf0);
    vecf1 = vec_rint(vecf1);

    vint32 veci0  = vec_add(vec_signed(vecf0),vec_zero_point);
    vint32 veci1  = vec_add(vec_signed(vecf1),vec_zero_point);

    veci0 = vec_max(veci0, vmin);
    veci1 = vec_max(veci1, vmin);
    veci0 = vec_min(veci0, vmax);
    veci1 = vec_min(veci1, vmax);

    return {veci0, veci1};
  }

  void dump() const {
    std::cout << _vec0[0] << " ";
    std::cout << _vec0[1] << " ";
    std::cout << _vec0[2] << " ";
    std::cout << _vec0[3] << " ";
    std::cout << _vec1[0] << " ";
    std::cout << _vec1[1] << " ";
    std::cout << _vec1[2] << " ";
    std::cout << _vec1[3] << " ";
    std::cout << std::endl;
  }

  DEFINE_MEMBER_OP(operator==, c10::qint32, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, c10::qint32, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, c10::qint32, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, c10::qint32, vec_cmple)
  DEFINE_MEMBER_OP(operator>, c10::qint32, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, c10::qint32, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, c10::qint32, vec_add)
  DEFINE_MEMBER_OP(operator-, c10::qint32, vec_sub)
  DEFINE_MEMBER_OP(operator*, c10::qint32, vec_mul)
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, c10::qint32, /)
  DEFINE_MEMBER_OP(maximum, c10::qint32, vec_max)
  DEFINE_MEMBER_OP(minimum, c10::qint32, vec_min)
  DEFINE_MEMBER_OP(operator&, c10::qint32, vec_and)
  DEFINE_MEMBER_OP(operator|, c10::qint32, vec_or)
  DEFINE_MEMBER_OP(operator^, c10::qint32, vec_xor)
};

template <>
Vec256<c10::qint32> inline maximum(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vec256<c10::qint32> inline minimum(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  return a.minimum(b);
}
} // namespace
} // namespace vec256
} // namespace at
