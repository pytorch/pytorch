#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

template <>
class Vectorized<int64_t> {
 private:
  union {
    struct {
      vint64 _vec0;
      vint64 _vec1;
    };
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = int64_t;
  using vec_internal_type = vint64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  using ElementType = signed long long;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  C10_ALWAYS_INLINE Vectorized(vint64 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vint64 v1, vint64 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2)
      : _vecb0{v1}, _vecb1{v2} {}
  C10_ALWAYS_INLINE Vectorized(int64_t scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  C10_ALWAYS_INLINE Vectorized(
      int64_t scalar1,
      int64_t scalar2,
      int64_t scalar3,
      int64_t scalar4)
      : _vec0{vint64{scalar1, scalar2}}, _vec1{vint64{scalar3, scalar4}} {}

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return a;
  }

  template <uint64_t mask>
  static std::enable_if_t<mask == 3, Vectorized<int64_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return {b._vec0, a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask & 15) == 15, Vectorized<int64_t>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return b;
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 3), Vectorized<int64_t>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    constexpr uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
    constexpr uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
    const vbool64 mask_1st = (vbool64){g0, g1};
    return {(vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st), a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<(mask > 3) && (mask & 3) == 0, Vectorized<int64_t>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    constexpr uint64_t g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
    constexpr uint64_t g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;

    const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
    return {a._vec0, (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 3) && (mask & 3) != 0 && (mask & 15) != 15,
      Vectorized<int64_t>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    constexpr uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
    constexpr uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
    constexpr uint64_t g0_2 = ((mask & 4) >> 2) * 0xffffffffffffffff;
    constexpr uint64_t g1_2 = ((mask & 8) >> 3) * 0xffffffffffffffff;

    const vbool64 mask_1st = (vbool64){g0, g1};
    const vbool64 mask_2nd = (vbool64){g0_2, g1_2};
    return {
        (vint64)vec_sel(a._vec0, b._vec0, (vbool64)mask_1st),
        (vint64)vec_sel(a._vec1, b._vec1, (vbool64)mask_2nd)};
  }

  static Vectorized<int64_t> C10_ALWAYS_INLINE blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    // the mask used here returned by comparision of vec256

    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base, base + step, base + 2 * step, base + 3 * step);
  }

  static Vectorized<int64_t> C10_ALWAYS_INLINE
  set(const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
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
    }

    return b;
  }
  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      static_assert(sizeof(double) == sizeof(value_type));
      const double* dptr = reinterpret_cast<const double*>(ptr);
      return {// treat it as double load
              (vint64)vec_vsx_ld(offset0, dptr),
              (vint64)vec_vsx_ld(offset16, dptr)};
    }

    __at_align__ double tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {
        (vint64)vec_vsx_ld(offset0, tmp_values),
        (vint64)vec_vsx_ld(offset16, tmp_values)};
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      double* dptr = reinterpret_cast<double*>(ptr);
      vec_vsx_st((vfloat64)_vec0, offset0, dptr);
      vec_vsx_st((vfloat64)_vec1, offset16, dptr);
    } else if (count > 0) {
      __at_align__ double tmp_values[size()];
      vec_vsx_st((vfloat64)_vec0, offset0, tmp_values);
      vec_vsx_st((vfloat64)_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }
  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;

  Vectorized<int64_t> angle() const {
    return blendv(
        Vectorized<int64_t>(0),
        Vectorized<int64_t>(c10::pi<int64_t>),
        *this < Vectorized<int64_t>(0));
  }
  Vectorized<int64_t> real() const {
    return *this;
  }
  Vectorized<int64_t> imag() const {
    return Vectorized<int64_t>{0};
  }
  Vectorized<int64_t> conj() const {
    return *this;
  }

  Vectorized<int64_t> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<int64_t> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  DEFINE_MEMBER_UNARY_OP(operator~, int64_t, vec_not)
  DEFINE_MEMBER_OP(operator==, int64_t, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, int64_t, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, int64_t, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, int64_t, vec_cmple)
  DEFINE_MEMBER_OP(operator>, int64_t, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, int64_t, vec_cmpge)
  DEFINE_MEMBER_OP_AND_ONE(eq, int64_t, vec_cmpeq)
  DEFINE_MEMBER_OP_AND_ONE(ne, int64_t, vec_cmpne)
  DEFINE_MEMBER_OP_AND_ONE(lt, int64_t, vec_cmplt)
  DEFINE_MEMBER_OP_AND_ONE(le, int64_t, vec_cmple)
  DEFINE_MEMBER_OP_AND_ONE(gt, int64_t, vec_cmpgt)
  DEFINE_MEMBER_OP_AND_ONE(ge, int64_t, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, int64_t, vec_add)
  DEFINE_MEMBER_OP(operator-, int64_t, vec_sub)
  DEFINE_MEMBER_OP(operator*, int64_t, vec_mul)
  DEFINE_MEMBER_OP(operator/, int64_t, vec_div)
  DEFINE_MEMBER_OP(maximum, int64_t, vec_max)
  DEFINE_MEMBER_OP(minimum, int64_t, vec_min)
  DEFINE_MEMBER_OP(operator&, int64_t, vec_and)
  DEFINE_MEMBER_OP(operator|, int64_t, vec_or)
  DEFINE_MEMBER_OP(operator^, int64_t, vec_xor)
};

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  vuint64 shift_vec0 = reinterpret_cast<vuint64>(b.vec0());
  vuint64 shift_vec1 = reinterpret_cast<vuint64>(b.vec1());
  return Vectorized<int64_t>{
      vec_sl(a.vec0(), shift_vec0), vec_sl(a.vec1(), shift_vec1)};
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  vuint64 shift_vec0 = reinterpret_cast<vuint64>(b.vec0());
  vuint64 shift_vec1 = reinterpret_cast<vuint64>(b.vec1());
  return Vectorized<int64_t>{
      vec_sr(a.vec0(), shift_vec0), vec_sr(a.vec1(), shift_vec1)};
}

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return a.maximum(b);
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return a.minimum(b);
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_mul(a.vec0(), b.vec0()), vec_mul(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_div(a.vec0(), b.vec0()), vec_div(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator&(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator|(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
operator^(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return Vectorized<int64_t>{
      vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
