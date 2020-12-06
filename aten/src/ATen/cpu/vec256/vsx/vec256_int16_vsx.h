#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/cpu/vec256/vsx/vsx_helpers.h>
namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

template <>
class Vec256<int16_t> {
 private:
  union {
    struct {
      vint16 _vec0;
      vint16 _vec1;
    };
    struct {
      vbool16 _vecb0;
      vbool16 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = int16_t;
  using vec_internal_type = vint16;
  using vec_internal_mask_type = vbool16;
  static constexpr int size() {
    return 16;
  }
  Vec256() {}
  C10_ALWAYS_INLINE Vec256(vint16 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vec256(vbool16 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vec256(vint16 v1, vint16 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vec256(vbool16 v1, vbool16 v2) : _vecb0{v1}, _vecb1{v2} {}
  C10_ALWAYS_INLINE Vec256(int16_t scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}

  C10_ALWAYS_INLINE Vec256(
      int16_t scalar1,
      int16_t scalar2,
      int16_t scalar3,
      int16_t scalar4,
      int16_t scalar5,
      int16_t scalar6,
      int16_t scalar7,
      int16_t scalar8,
      int16_t scalar9,
      int16_t scalar10,
      int16_t scalar11,
      int16_t scalar12,
      int16_t scalar13,
      int16_t scalar14,
      int16_t scalar15,
      int16_t scalar16)
      : _vec0{vint16{
            scalar1,
            scalar2,
            scalar3,
            scalar4,
            scalar5,
            scalar6,
            scalar7,
            scalar8}},
        _vec1{vint16{
            scalar9,
            scalar10,
            scalar11,
            scalar12,
            scalar13,
            scalar14,
            scalar15,
            scalar16}} {}
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vec256<int16_t>> C10_ALWAYS_INLINE
  blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    return a;
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask & 65535) == 65535, Vec256<int16_t>>
      C10_ALWAYS_INLINE blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    return b;
  }

  template <uint64_t mask>
  static std::enable_if_t<mask == 255, Vec256<int16_t>> C10_ALWAYS_INLINE
  blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    return {b._vec0, a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 255), Vec256<int16_t>>
      C10_ALWAYS_INLINE blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};

    return {(vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st), a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && (mask & 65535) != 65535 && ((mask & 255) == 255)),
      Vec256<int16_t>>
      C10_ALWAYS_INLINE blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // generated masks
    return {b._vec0, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) == 0)),
      Vec256<int16_t>>
      C10_ALWAYS_INLINE blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    constexpr int16_t mask2 = (mask & 65535) >> 16;
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // generated masks
    return {a, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) != 0) &&
       ((mask & 255) != 255)),
      Vec256<int16_t>>
      C10_ALWAYS_INLINE blend(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    constexpr int16_t mask2 = (mask & 65535) >> 16;
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // generated masks
    return {
        (vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st),
        (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  static Vec256<int16_t> C10_ALWAYS_INLINE blendv(
      const Vec256<int16_t>& a,
      const Vec256<int16_t>& b,
      const Vec256<int16_t>& mask) {
    // the mask used here returned by comparision of vec256
    // assuming this we can use the same mask directly with vec_sel
    // warning intel style mask will not work properly
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }

  static Vec256<int16_t> arange(int16_t base = 0, int16_t step = 1) {
    return Vec256<int16_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }
  static Vec256<int16_t> set(
      const Vec256<int16_t>& a,
      const Vec256<int16_t>& b,
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
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
    }
    return b;
  }
  static Vec256<value_type> C10_ALWAYS_INLINE
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
      std::memcpy(ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }
  const int16_t& operator[](int idx) const = delete;
  int16_t& operator[](int idx) = delete;

  Vec256<int16_t> angle() const {
    return Vec256<int16_t>{0};
  }
  Vec256<int16_t> real() const {
    return *this;
  }
  Vec256<int16_t> imag() const {
    return Vec256<int16_t>{0};
  }
  Vec256<int16_t> conj() const {
    return *this;
  }

  Vec256<int16_t> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vec256<int16_t> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  DEFINE_MEMBER_UNARY_OP(operator~, int16_t, vec_not)
  DEFINE_MEMBER_OP(operator==, int16_t, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, int16_t, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, int16_t, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, int16_t, vec_cmple)
  DEFINE_MEMBER_OP(operator>, int16_t, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, int16_t, vec_cmpge)
  DEFINE_MEMBER_OP_AND_ONE(eq, int16_t, vec_cmpeq)
  DEFINE_MEMBER_OP_AND_ONE(ne, int16_t, vec_cmpne)
  DEFINE_MEMBER_OP_AND_ONE(lt, int16_t, vec_cmplt)
  DEFINE_MEMBER_OP_AND_ONE(le, int16_t, vec_cmple)
  DEFINE_MEMBER_OP_AND_ONE(gt, int16_t, vec_cmpgt)
  DEFINE_MEMBER_OP_AND_ONE(ge, int16_t, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, int16_t, vec_add)
  DEFINE_MEMBER_OP(operator-, int16_t, vec_sub)
  DEFINE_MEMBER_OP(operator*, int16_t, vec_mul)
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, int16_t, /)
  DEFINE_MEMBER_OP(maximum, int16_t, vec_max)
  DEFINE_MEMBER_OP(minimum, int16_t, vec_min)
  DEFINE_MEMBER_OP(operator&, int16_t, vec_and)
  DEFINE_MEMBER_OP(operator|, int16_t, vec_or)
  DEFINE_MEMBER_OP(operator^, int16_t, vec_xor)
};

template <>
Vec256<int16_t> inline maximum(
    const Vec256<int16_t>& a,
    const Vec256<int16_t>& b) {
  return a.maximum(b);
}

template <>
Vec256<int16_t> inline minimum(
    const Vec256<int16_t>& a,
    const Vec256<int16_t>& b) {
  return a.minimum(b);
}


} // namespace
} // namespace vec256
} // namespace at
