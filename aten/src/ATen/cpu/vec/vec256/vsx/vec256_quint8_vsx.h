#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

#include <c10/util/irange.h>
#include <c10/util/quint8.h>
#include <array>

// This file defines Vectorized<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vectorized<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vectorized<quint8> -> 4x Vectorized<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vectorized<T>::float_num_vecs
// iterations.

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

const vint16 mask_unsigned = vec_splats((short int)0xFF);
template <>
struct Vectorized<c10::quint8> {
 private:
  union {
    struct {
      vuint8 _vec0;
      vuint8 _vec1;
    };
    struct {
      vbool8 _vecb0;
      vbool8 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  Vectorized() {}
  using size_type = int;
  static constexpr size_type size() {
    return 32;
  }

  static constexpr size_t float_num_vecs() {
    return 4;
  }
  static constexpr int int_num_vecs() {
    return 4;
  }
  using float_vec_return_type = std::array<Vectorized<float>, 4>;
  using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
  using value_type = typename c10::quint8::underlying;
  using vec_internal_type = vuint8;
  using vec_internal_mask_type = vbool8;
  // Broadcast constructor
  C10_ALWAYS_INLINE Vectorized(const c10::quint8& val)
      : _vec0(vec_splats(val.val_)), _vec1(vec_splats(val.val_)) {}

  C10_ALWAYS_INLINE Vectorized(const Vectorized<c10::quint8>& other)
      : _vec0{other._vec0}, _vec1(other._vec1) {}

  C10_ALWAYS_INLINE Vectorized(vuint8 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool8 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vuint8 v1, vuint8 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool8 v1, vbool8 v2) : _vecb0{v1}, _vecb1{v2} {}

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  static C10_ALWAYS_INLINE Vectorized<c10::quint8> loadu(
      const void* ptr,
      int count = size()) {
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

 public:
  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // unpacking unsigned as signed
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);

    vint16 vecshi2 = vec_unpackh((vint8)_vec1);
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);

    // signed ->  unsigned
    vecshi0 = vec_and(vecshi0, mask_unsigned);
    vecshi1 = vec_and(vecshi1, mask_unsigned);

    vecshi2 = vec_and(vecshi2, mask_unsigned);
    vecshi3 = vec_and(vecshi3, mask_unsigned);

    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 veci3 = vec_unpackl(vecshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 veci5 = vec_unpackl(vecshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    vfloat32 vecf0_0 = vec_float(veci0);
    vfloat32 vecf1_0 = vec_float(veci1);

    vfloat32 vecf0_1 = vec_float(veci2);
    vfloat32 vecf1_1 = vec_float(veci3);

    vfloat32 vecf0_2 = vec_float(veci4);
    vfloat32 vecf1_2 = vec_float(veci5);

    vfloat32 vecf0_3 = vec_float(veci6);
    vfloat32 vecf1_3 = vec_float(veci7);
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();

    vfloat32 zero_point_vec0 = zero_point.vec0();
    vfloat32 zero_point_vec1 = zero_point.vec1();

    vfloat32 vec_substract_src_zp0_0 =  vec_sub(vecf0_0, zero_point_vec0);
    vfloat32 vec_substract_src_zp1_0 =  vec_sub(vecf1_0, zero_point_vec1);
    Vectorized<float> vf0_zp = {vec_mul(scale_vec0, vec_substract_src_zp0_0), vec_mul(scale_vec1, vec_substract_src_zp1_0)};

    vfloat32 vec_substract_src_zp0_1 =  vec_sub(vecf0_1, zero_point_vec0);
    vfloat32 vec_substract_src_zp1_1 =  vec_sub(vecf1_1, zero_point_vec1);
    Vectorized<float> vf1_zp = {vec_mul(scale_vec0, vec_substract_src_zp0_1), vec_mul(scale_vec1, vec_substract_src_zp1_1)};

    vfloat32 vec_substract_src_zp0_2 = vec_sub(vecf0_2, zero_point_vec0);
    vfloat32 vec_substract_src_zp1_2 =  vec_sub(vecf1_2, zero_point_vec1);
    Vectorized<float> vf2_zp = {vec_mul(scale_vec0, vec_substract_src_zp0_2), vec_mul(scale_vec1, vec_substract_src_zp1_2)};

    vfloat32 vec_substract_src_zp0_3 = vec_sub(vecf0_3, zero_point_vec0);
    vfloat32 vec_substract_src_zp1_3 =  vec_sub(vecf1_3, zero_point_vec1);
    Vectorized<float> vf3_zp = {vec_mul(scale_vec0, vec_substract_src_zp0_3), vec_mul(scale_vec1, vec_substract_src_zp1_3)};

    return {vf0_zp, vf1_zp, vf2_zp, vf3_zp};
  }

  float_vec_return_type C10_ALWAYS_INLINE dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // unpacking unsigned as signed
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);

    vint16 vecshi2 = vec_unpackh((vint8)_vec1);
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);

    // signed ->  unsigned
    vecshi0 = vec_and(vecshi0, mask_unsigned);
    vecshi1 = vec_and(vecshi1, mask_unsigned);

    vecshi2 = vec_and(vecshi2, mask_unsigned);
    vecshi3 = vec_and(vecshi3, mask_unsigned);

    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 veci1 = vec_unpackl(vecshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 veci3 = vec_unpackl(vecshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 veci5 = vec_unpackl(vecshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 veci7 = vec_unpackl(vecshi3);

    vfloat32 vecf0_0 = vec_float(veci0);
    vfloat32 vecf1_0 = vec_float(veci1);

    vfloat32 vecf0_1 = vec_float(veci2);
    vfloat32 vecf1_1 = vec_float(veci3);

    vfloat32 vecf0_2 = vec_float(veci4);
    vfloat32 vecf1_2 = vec_float(veci5);

    vfloat32 vecf0_3 = vec_float(veci6);
    vfloat32 vecf1_3 = vec_float(veci7);
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();

    vfloat32 zero_point0 = zero_point.vec0();
    vfloat32 zero_point1 = zero_point.vec1();
    return {
    Vectorized<float>{
        (vecf0_0 - zero_point0) * scale_vec0,
        (vecf1_0 - zero_point1) * scale_vec1},
    Vectorized<float>{
        (vecf0_1 - zero_point0) * scale_vec0,
        (vecf1_1 - zero_point1) * scale_vec1},
    Vectorized<float>{
        (vecf0_2 - zero_point0) * scale_vec0,
        (vecf1_2 - zero_point1) * scale_vec1},
    Vectorized<float>{
        (vecf0_3 - zero_point0) * scale_vec0,
        (vecf1_3 - zero_point1) * scale_vec1}};
  }

  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    // constexpr int32_t min_val = std::numeric_limits<value_type>::min();
    // constexpr int32_t max_val = std::numeric_limits<value_type>::max();

    vfloat32 vec_inverse = vec_splats(inverse_scale);
    vfloat32 vec_zero_point = vec_splats((float)zero_point);
    // vuint32 vmin = vec_splats(min_val);
    // vuint32 vmax = vec_splats(max_val);
    Vectorized<float> vf0 = rhs[0];
    Vectorized<float> vf1 = rhs[1];
    Vectorized<float> vf2 = rhs[2];
    Vectorized<float> vf3 = rhs[3];
    vfloat32 vecf0 = vf0.vec0();
    vfloat32 vecf1 = vf0.vec1();
    vfloat32 vecf2 = vf1.vec0();
    vfloat32 vecf3 = vf1.vec1();

    vfloat32 vecf4 = vf2.vec0();
    vfloat32 vecf5 = vf2.vec1();
    vfloat32 vecf6 = vf3.vec0();
    vfloat32 vecf7 = vf3.vec1();

    vecf0 = vec_mul(vecf0, vec_inverse);
    vecf1 = vec_mul(vecf1, vec_inverse);
    vecf2 = vec_mul(vecf2, vec_inverse);
    vecf3 = vec_mul(vecf3, vec_inverse);

    vecf4 = vec_mul(vecf4, vec_inverse);
    vecf5 = vec_mul(vecf5, vec_inverse);
    vecf6 = vec_mul(vecf6, vec_inverse);
    vecf7 = vec_mul(vecf7, vec_inverse);

    vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
    vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
    vecf2 = vec_add(vec_rint(vecf2), vec_zero_point);
    vecf3 = vec_add(vec_rint(vecf3), vec_zero_point);

    vecf4 = vec_add(vec_rint(vecf4), vec_zero_point);
    vecf5 = vec_add(vec_rint(vecf5), vec_zero_point);
    vecf6 = vec_add(vec_rint(vecf6), vec_zero_point);
    vecf7 = vec_add(vec_rint(vecf7), vec_zero_point);

    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);

    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    vuint8 vec0 = vec_packsu(vecshi0, vecshi1);
    vuint8 vec1 = vec_packsu(vecshi2, vecshi3);

    return {vec0, vec1};
  }

  Vectorized<c10::quint8> C10_ALWAYS_INLINE relu(Vectorized<c10::quint8> zero_point) const {
    return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
  }

  Vectorized<c10::quint8> C10_ALWAYS_INLINE
  relu6(Vectorized<c10::quint8> zero_point, Vectorized<c10::quint8> q_six) const {
    vuint8 max0 = vec_max(_vec0, zero_point._vec0);
    vuint8 max1 = vec_max(_vec1, zero_point._vec1);
    return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
  }

  int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
    vint16 vecshi0 = vec_unpackh((vint8)_vec0);
    vint16 vecBshi0 = vec_unpackh((vint8)b._vec0);
    vint16 vecshi1 = vec_unpackl((vint8)_vec0);
    vint16 vecBshi1 = vec_unpackl((vint8)b._vec0);

    vint16 vecshi2 = vec_unpackh((vint8)_vec1);
    vint16 vecBshi2 = vec_unpackh((vint8)b._vec1);
    vint16 vecshi3 = vec_unpackl((vint8)_vec1);
    vint16 vecBshi3 = vec_unpackl((vint8)b._vec1);

    vecshi0 = vec_and(vecshi0, mask_unsigned);
    vecBshi0 = vec_and(vecBshi0, mask_unsigned);
    vecshi1 = vec_and(vecshi1, mask_unsigned);
    vecBshi1 = vec_and(vecBshi1, mask_unsigned);

    vecshi2 = vec_and(vecshi2, mask_unsigned);
    vecBshi2 = vec_and(vecBshi2, mask_unsigned);
    vecshi3 = vec_and(vecshi3, mask_unsigned);
    vecBshi3 = vec_and(vecBshi3, mask_unsigned);

    vint32 veci0 = vec_unpackh(vecshi0);
    vint32 vecBi0 = vec_unpackh(vecBshi0);
    vint32 veci1 = vec_unpackl(vecshi0);
    vint32 vecBi1 = vec_unpackl(vecBshi0);

    vint32 veci2 = vec_unpackh(vecshi1);
    vint32 vecBi2 = vec_unpackh(vecBshi1);
    vint32 veci3 = vec_unpackl(vecshi1);
    vint32 vecBi3 = vec_unpackl(vecBshi1);

    vint32 veci4 = vec_unpackh(vecshi2);
    vint32 vecBi4 = vec_unpackh(vecBshi2);
    vint32 veci5 = vec_unpackl(vecshi2);
    vint32 vecBi5 = vec_unpackl(vecBshi2);

    vint32 veci6 = vec_unpackh(vecshi3);
    vint32 vecBi6 = vec_unpackh(vecBshi3);
    vint32 veci7 = vec_unpackl(vecshi3);
    vint32 vecBi7 = vec_unpackl(vecBshi3);

    return {
        Vectorized<c10::qint32>(veci0 - vecBi0, veci1 - vecBi1),
        Vectorized<c10::qint32>(veci2 - vecBi2, veci3 - vecBi3),
        Vectorized<c10::qint32>(veci4 - vecBi4, veci5 - vecBi5),
        Vectorized<c10::qint32>(veci6 - vecBi6, veci7 - vecBi7)};
  }

  static Vectorized<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    vfloat32 vec_multiplier = vec_splats(multiplier);
    vint32 vec_zero_point = vec_splats(zero_point);

    Vectorized<c10::qint32> vi0 = inp[0];
    Vectorized<c10::qint32> vi1 = inp[1];
    Vectorized<c10::qint32> vi2 = inp[2];
    Vectorized<c10::qint32> vi3 = inp[3];

    vfloat32 vecf0 = vec_float(vi0.vec0());
    vfloat32 vecf1 = vec_float(vi0.vec1());
    vfloat32 vecf2 = vec_float(vi1.vec0());
    vfloat32 vecf3 = vec_float(vi1.vec1());

    vfloat32 vecf4 = vec_float(vi2.vec0());
    vfloat32 vecf5 = vec_float(vi2.vec1());
    vfloat32 vecf6 = vec_float(vi3.vec0());
    vfloat32 vecf7 = vec_float(vi3.vec1());

    vecf0 = vec_mul(vecf0, vec_multiplier);
    vecf1 = vec_mul(vecf1, vec_multiplier);
    vecf2 = vec_mul(vecf2, vec_multiplier);
    vecf3 = vec_mul(vecf3, vec_multiplier);

    vecf4 = vec_mul(vecf4, vec_multiplier);
    vecf5 = vec_mul(vecf5, vec_multiplier);
    vecf6 = vec_mul(vecf6, vec_multiplier);
    vecf7 = vec_mul(vecf7, vec_multiplier);

    vecf0 = vec_rint(vecf0);
    vecf1 = vec_rint(vecf1);
    vecf2 = vec_rint(vecf2);
    vecf3 = vec_rint(vecf3);

    vecf4 = vec_rint(vecf4);
    vecf5 = vec_rint(vecf5);
    vecf6 = vec_rint(vecf6);
    vecf7 = vec_rint(vecf7);

    vint32 veci0 = vec_signed(vecf0);
    vint32 veci1 = vec_signed(vecf1);
    vint32 veci2 = vec_signed(vecf2);
    vint32 veci3 = vec_signed(vecf3);

    vint32 veci4 = vec_signed(vecf4);
    vint32 veci5 = vec_signed(vecf5);
    vint32 veci6 = vec_signed(vecf6);
    vint32 veci7 = vec_signed(vecf7);

    veci0 = vec_add(veci0, vec_zero_point);
    veci1 = vec_add(veci1, vec_zero_point);
    veci2 = vec_add(veci2, vec_zero_point);
    veci3 = vec_add(veci3, vec_zero_point);

    veci4 = vec_add(veci4, vec_zero_point);
    veci5 = vec_add(veci5, vec_zero_point);
    veci6 = vec_add(veci6, vec_zero_point);
    veci7 = vec_add(veci7, vec_zero_point);

    vint16 vecshi0 = vec_packs(veci0, veci1);
    vint16 vecshi1 = vec_packs(veci2, veci3);
    vint16 vecshi2 = vec_packs(veci4, veci5);
    vint16 vecshi3 = vec_packs(veci6, veci7);

    vuint8 vec0 = vec_packsu(vecshi0, vecshi1);
    vuint8 vec1 = vec_packsu(vecshi2, vecshi3);

    return {vec0, vec1};
  }

  DEFINE_MEMBER_OP(operator==, c10::quint8, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, c10::quint8, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, c10::quint8, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, c10::quint8, vec_cmple)
  DEFINE_MEMBER_OP(operator>, c10::quint8, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, c10::quint8, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, c10::quint8, vec_add)
  DEFINE_MEMBER_OP(operator-, c10::quint8, vec_sub)
  DEFINE_MEMBER_OP(operator*, c10::quint8, vec_mul)
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, c10::quint8, /)
  DEFINE_MEMBER_OP(maximum, c10::quint8, vec_max)
  DEFINE_MEMBER_OP(minimum, c10::quint8, vec_min)
  DEFINE_MEMBER_OP(operator&, c10::quint8, vec_and)
  DEFINE_MEMBER_OP(operator|, c10::quint8, vec_or)
  DEFINE_MEMBER_OP(operator^, c10::quint8, vec_xor)
};

template <>
Vectorized<c10::quint8> inline maximum(
    const Vectorized<c10::quint8>& a,
    const Vectorized<c10::quint8>& b) {
  return a.maximum(b);
}

template <>
Vectorized<c10::quint8> inline minimum(
    const Vectorized<c10::quint8>& a,
    const Vectorized<c10::quint8>& b) {
  return a.minimum(b);
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator+(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator-(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator*(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_mul(a.vec0(), b.vec0()), vec_mul(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator/(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{a.vec0()/b.vec0(), a.vec1()/b.vec1()};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator&(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator|(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<c10::quint8> C10_ALWAYS_INLINE operator^(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return Vectorized<c10::quint8>{vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

} // namespace
} // namespace vec
} // namespace at
