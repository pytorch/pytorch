#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>

namespace at::vec {
// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_SVE)

#define VEC_INT_SVE_TEMPLATE(vl, bit)                                         \
  template <>                                                                 \
  struct is_vec_specialized_for<int##bit##_t> : std::bool_constant<true> {};  \
                                                                              \
  template <>                                                                 \
  class Vectorized<int##bit##_t> {                                            \
   private:                                                                   \
    vls_int##bit##_t values;                                                  \
                                                                              \
   public:                                                                    \
    using value_type = int##bit##_t;                                          \
    using size_type = int;                                                    \
    static constexpr size_type size() {                                       \
      return vl;                                                              \
    }                                                                         \
    Vectorized() {                                                            \
      values = svdup_n_s##bit(0);                                             \
    }                                                                         \
    Vectorized(svint##bit##_t v) : values(v) {}                               \
    Vectorized(int##bit##_t val) {                                            \
      values = svdup_n_s##bit(val);                                           \
    }                                                                         \
    template <                                                                \
        typename... Args,                                                     \
        typename = std::enable_if_t<(sizeof...(Args) == size())>>             \
    Vectorized(Args... vals) {                                                \
      __at_align__ int##bit##_t buffer[size()] = {vals...};                   \
      values = svld1_s##bit(ptrue, buffer);                                   \
    }                                                                         \
    operator svint##bit##_t() const {                                         \
      return values;                                                          \
    }                                                                         \
    template <uint64_t mask>                                                  \
    static Vectorized<int##bit##_t> blend(                                    \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b) {                                  \
      __at_align__ int##bit##_t flag_arr[size()];                             \
      for (int i = 0; i < size(); ++i) {                                      \
        flag_arr[i] = (i < 64 && (mask & (1ULL << i))) ? 1 : 0;               \
      }                                                                       \
      svbool_t blend_mask = svcmpne_n_s##bit(                                 \
          svptrue_b##bit(), svld1_s##bit(svptrue_b##bit(), flag_arr), 0);     \
      return Vectorized<int##bit##_t>(                                        \
          svsel_s##bit(blend_mask, b.values, a.values));                      \
    }                                                                         \
    static Vectorized<int##bit##_t> blendv(                                   \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        const Vectorized<int##bit##_t>& mask_) {                              \
      svbool_t mask = svcmpeq_s##bit(ptrue, mask_, ALL_S##bit##_TRUE_MASK);   \
      return svsel_s##bit(mask, b, a);                                        \
    }                                                                         \
    /* step sometimes requires a higher precision type (e.g., T=int,          \
     * step_t=double) */                                                      \
    template <typename step_t>                                                \
    static Vectorized<int##bit##_t> arange(                                   \
        int##bit##_t base = 0,                                                \
        step_t step = static_cast<step_t>(1)) {                               \
      __at_align__ int##bit##_t buffer[size()];                               \
      for (int64_t i = 0; i < size(); i++) {                                  \
        buffer[i] = base + i * step;                                          \
      }                                                                       \
      return svld1_s##bit(ptrue, buffer);                                     \
    }                                                                         \
    static Vectorized<int##bit##_t> set(                                      \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        int##bit##_t count = size()) {                                        \
      if (count == 0) {                                                       \
        return a;                                                             \
      } else if (count < size()) {                                            \
        return svsel_s##bit(svwhilelt_b##bit(0ull, count), b, a);             \
      }                                                                       \
      return b;                                                               \
    }                                                                         \
    static Vectorized<int##bit##_t> loadu(                                    \
        const void* ptr,                                                      \
        int64_t count = size()) {                                             \
      if (count == size())                                                    \
        return svld1_s##bit(                                                  \
            ptrue, reinterpret_cast<const int##bit##_t*>(ptr));               \
      svbool_t pg = svwhilelt_b##bit(0ull, count);                            \
      return svld1_s##bit(pg, reinterpret_cast<const int##bit##_t*>(ptr));    \
    }                                                                         \
    void store(void* ptr, int64_t count = size()) const {                     \
      if (count == size()) {                                                  \
        svst1_s##bit(ptrue, reinterpret_cast<int##bit##_t*>(ptr), values);    \
      } else {                                                                \
        svbool_t pg = svwhilelt_b##bit(0ull, count);                          \
        svst1_s##bit(pg, reinterpret_cast<int##bit##_t*>(ptr), values);       \
      }                                                                       \
    }                                                                         \
    const int##bit##_t& operator[](int idx) const = delete;                   \
    int##bit##_t& operator[](int idx) = delete;                               \
    Vectorized<int##bit##_t> abs() const {                                    \
      return svabs_s##bit##_x(ptrue, values);                                 \
    }                                                                         \
    Vectorized<int##bit##_t> real() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> imag() const {                                   \
      return svdup_n_s##bit(0);                                               \
    }                                                                         \
    Vectorized<int##bit##_t> conj() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> frac() const;                                    \
    Vectorized<int##bit##_t> neg() const {                                    \
      return svneg_s##bit##_x(ptrue, values);                                 \
    }                                                                         \
    Vectorized<int##bit##_t> operator==(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmpeq_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> operator!=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmpne_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> operator<(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmplt_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> operator<=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmple_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> operator>(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmpgt_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> operator>=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      svbool_t mask = svcmpge_s##bit(ptrue, values, other);                   \
      return svsel_s##bit(                                                    \
          mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);             \
    }                                                                         \
    Vectorized<int##bit##_t> eq(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> ne(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> gt(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> ge(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> lt(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> le(const Vectorized<int##bit##_t>& other) const; \
  };                                                                          \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator+(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svadd_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator-(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svsub_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator*(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svmul_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline maximum(                                    \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svmax_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline minimum(                                    \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svmin_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline clamp(                                      \
      const Vectorized<int##bit##_t>& a,                                      \
      const Vectorized<int##bit##_t>& min,                                    \
      const Vectorized<int##bit##_t>& max) {                                  \
    return svmin_s##bit##_x(ptrue, max, svmax_s##bit##_x(ptrue, min, a));     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline clamp_max(                                  \
      const Vectorized<int##bit##_t>& a,                                      \
      const Vectorized<int##bit##_t>& max) {                                  \
    return svmin_s##bit##_x(ptrue, max, a);                                   \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline clamp_min(                                  \
      const Vectorized<int##bit##_t>& a,                                      \
      const Vectorized<int##bit##_t>& min) {                                  \
    return svmax_s##bit##_x(ptrue, min, a);                                   \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator&(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svand_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator|(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return svorr_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator^(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return sveor_s##bit##_x(ptrue, a, b);                                     \
  }                                                                           \
  template <>                                                                 \
  inline Vectorized<int##bit##_t> operator~(                                  \
      const Vectorized<int##bit##_t>& a) {                                    \
    return sveor_s##bit##_x(ptrue, a, svdup_n_s##bit(-1));                    \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::eq(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this == other) & Vectorized<int##bit##_t>(1);                    \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::ne(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this != other) & Vectorized<int##bit##_t>(1);                    \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::gt(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this > other) & Vectorized<int##bit##_t>(1);                     \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::ge(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this >= other) & Vectorized<int##bit##_t>(1);                    \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::lt(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this < other) & Vectorized<int##bit##_t>(1);                     \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::le(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return (*this <= other) & Vectorized<int##bit##_t>(1);                    \
  }

VEC_INT_SVE_TEMPLATE(VECTOR_WIDTH / sizeof(int64_t), 64)
VEC_INT_SVE_TEMPLATE(VECTOR_WIDTH / sizeof(int32_t), 32)
VEC_INT_SVE_TEMPLATE(VECTOR_WIDTH / sizeof(int16_t), 16)
VEC_INT_SVE_TEMPLATE(VECTOR_WIDTH / sizeof(int8_t), 8)

template <typename T>
Vectorized<T> inline intdiv_nosve(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  T values_a[Vectorized<T>::size()];
  T values_b[Vectorized<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    values_a[i] /= values_b[i];
  }
  return Vectorized<T>::loadu(values_a);
}

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svdiv_s64_x(ptrue, a, b);
}

template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svdiv_s32_x(ptrue, a, b);
}

template <>
Vectorized<int16_t> inline operator/(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return intdiv_nosve(a, b);
}

template <>
Vectorized<int8_t> inline operator/(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return intdiv_nosve(a, b);
}

template <>
inline void convert(const int32_t* src, int64_t* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<int64_t>::size();
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vectorized<int64_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<int64_t>::size())
    svst1_s64(pg_64, dst + i, svunpklo_s64(svldnt1_s32(pg_32, src + i)));
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<int64_t>::size()) {
    pg_32 = svwhilelt_b32(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svst1_s64(pg_64, dst + i, svunpklo_s64(svldnt1_s32(pg_32, src + i)));
  }
}

template <>
inline void convert(const int64_t* src, float* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<int64_t>::size();
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vectorized<int64_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<int64_t>::size()) {
    svint64_t src_vec_s64 = svldnt1_s64(pg_64, src + i);
    svfloat32_t src_vec_f32 =
        svuzp1_f32(svcvt_f32_s64_x(pg_64, src_vec_s64), ZERO_F32);
    svst1_f32(pg_32, dst + i, src_vec_f32);
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<int64_t>::size()) {
    pg_32 = svwhilelt_b32(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svint64_t src_vec_s64 = svldnt1_s64(pg_64, src + i);
    svfloat32_t src_vec_f32 =
        svuzp1_f32(svcvt_f32_s64_x(pg_64, src_vec_s64), ZERO_F32);
    svst1_f32(pg_32, dst + i, src_vec_f32);
  }
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<int32_t>::size();
  svbool_t pg = svwhilelt_b32(0ull, Vectorized<int32_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<int32_t>::size()) {
    svint32_t src_vec = svldnt1_s32(pg, src + i);
    svst1_f32(pg, dst + i, svcvt_f32_s32_x(pg, src_vec));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<int32_t>::size()) {
    pg = svwhilelt_b32(i, n);
    svint32_t src_vec = svldnt1_s32(pg, src + i);
    svst1_f32(pg, dst + i, svcvt_f32_s32_x(pg, src_vec));
  }
}

template <>
inline void convert(const bool* src, int64_t* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<int64_t>::size();
  svbool_t pg_8 = svwhilelt_b8(0ull, Vectorized<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vectorized<int64_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<int64_t>::size()) {
    svuint8_t src_vec_u8 =
        svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint64_t src_vec_u64 =
        svunpklo_u64(svunpklo_u32(svunpklo_u16(src_vec_u8)));
    svbool_t mask = svcmpne_u64(pg_64, src_vec_u64, ZERO_U64);
    svst1_s64(pg_64, dst + i, svsel_s64(mask, ONE_S64, ZERO_S64));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<int64_t>::size()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svuint8_t src_vec_u8 =
        svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint64_t src_vec_u64 =
        svunpklo_u64(svunpklo_u32(svunpklo_u16(src_vec_u8)));
    svbool_t mask = svcmpne_u64(pg_64, src_vec_u64, ZERO_U64);
    svst1_s64(pg_64, dst + i, svsel_s64(mask, ONE_S64, ZERO_S64));
  }
}

template <>
inline void convert(const bool* src, int32_t* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<int32_t>::size();
  svbool_t pg_8 = svwhilelt_b8(0ull, Vectorized<int32_t>::size());
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<int32_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<int32_t>::size()) {
    svuint8_t src_vec_u8 =
        svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_s32(pg_32, dst + i, svsel_s32(mask, ONE_S32, ZERO_S32));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<int32_t>::size()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svuint8_t src_vec_u8 =
        svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_s32(pg_32, dst + i, svsel_s32(mask, ONE_S32, ZERO_S32));
  }
}

template <>
inline void convert(const uint8_t* src, bool* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<uint8_t>::size();
  svbool_t pg = svwhilelt_b8(0ull, Vectorized<uint8_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<uint8_t>::size()) {
    svbool_t mask = svcmpne_u8(pg, svldnt1_u8(pg, src + i), ZERO_U8);
    svst1_u8(
        pg,
        reinterpret_cast<uint8_t*>(dst) + i,
        svsel_u8(mask, ALL_U8_TRUE_MASK, ALL_U8_FALSE_MASK));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<uint8_t>::size()) {
    pg = svwhilelt_b8(i, n);
    svbool_t mask = svcmpne_u8(pg, svldnt1_u8(pg, src + i), ZERO_U8);
    svst1_u8(
        pg,
        reinterpret_cast<uint8_t*>(dst) + i,
        svsel_u8(mask, ALL_U8_TRUE_MASK, ALL_U8_FALSE_MASK));
  }
}

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svlsl_s64_x(ptrue, a, svreinterpret_u64_s64(b));
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svlsl_s32_x(ptrue, a, svreinterpret_u32_s32(b));
}

template <>
Vectorized<int16_t> inline operator<<(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return svlsl_s16_x(ptrue, a, svreinterpret_u16_s16(b));
}

template <>
Vectorized<int8_t> inline operator<<(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return svlsl_s8_x(ptrue, a, svreinterpret_u8_s8(b));
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svasr_s64_x(ptrue, a, svreinterpret_u64_s64(b));
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svasr_s32_x(ptrue, a, svreinterpret_u32_s32(b));
}

template <>
Vectorized<int16_t> inline operator>>(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return svasr_s16_x(ptrue, a, svreinterpret_u16_s16(b));
}

template <>
Vectorized<int8_t> inline operator>>(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return svasr_s8_x(ptrue, a, svreinterpret_u8_s8(b));
}

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
