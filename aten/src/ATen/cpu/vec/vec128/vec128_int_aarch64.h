#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

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

#define VEC_INT_NEON_TEMPLATE(vl, bit)                                        \
  template <>                                                                 \
  struct is_vec_specialized_for<int##bit##_t> : std::bool_constant<true> {};  \
                                                                              \
  template <>                                                                 \
  class Vectorized<int##bit##_t> {                                            \
    using neon_type = int##bit##x##vl##_t;                                    \
                                                                              \
   private:                                                                   \
    neon_type values;                                                         \
                                                                              \
   public:                                                                    \
    using value_type = int##bit##_t;                                          \
    using size_type = int;                                                    \
    static constexpr size_type size() {                                       \
      return vl;                                                              \
    }                                                                         \
    Vectorized() {                                                            \
      values = vdupq_n_s##bit(0);                                             \
    }                                                                         \
    Vectorized(neon_type v) : values(v) {}                                    \
    Vectorized(int##bit##_t val);                                             \
    template <                                                                \
        typename... Args,                                                     \
        typename = std::enable_if_t<(sizeof...(Args) == size())>>             \
    Vectorized(Args... vals) {                                                \
      __at_align__ int##bit##_t buffer[size()] = {vals...};                   \
      values = vld1q_s##bit(buffer);                                          \
    }                                                                         \
    operator neon_type() const {                                              \
      return values;                                                          \
    }                                                                         \
    static Vectorized<int##bit##_t> loadu(                                    \
        const void* ptr,                                                      \
        int64_t count = size());                                              \
    void store(void* ptr, int64_t count = size()) const;                      \
    template <int64_t mask>                                                   \
    static Vectorized<int##bit##_t> blend(                                    \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b);                                   \
    static Vectorized<int##bit##_t> blendv(                                   \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        const Vectorized<int##bit##_t>& mask_) {                              \
      return vbslq_s##bit(vreinterpretq_u##bit##_s##bit(mask_.values), b, a); \
    }                                                                         \
    template <typename step_t>                                                \
    static Vectorized<int##bit##_t> arange(                                   \
        value_type base = 0,                                                  \
        step_t step = static_cast<step_t>(1));                                \
    static Vectorized<int##bit##_t> set(                                      \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        int64_t count = size());                                              \
    const int##bit##_t& operator[](int idx) const = delete;                   \
    int##bit##_t& operator[](int idx) = delete;                               \
    Vectorized<int##bit##_t> abs() const {                                    \
      return vabsq_s##bit(values);                                            \
    }                                                                         \
    Vectorized<int##bit##_t> real() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> imag() const {                                   \
      return vdupq_n_s##bit(0);                                               \
    }                                                                         \
    Vectorized<int##bit##_t> conj() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> neg() const {                                    \
      return vnegq_s##bit(values);                                            \
    }                                                                         \
    int##bit##_t reduce_add() const {                                         \
      return vaddvq_s##bit(values);                                           \
    }                                                                         \
    int##bit##_t reduce_max() const;                                          \
    Vectorized<int##bit##_t> operator==(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
          vreinterpretq_s##bit##_u##bit(vceqq_s##bit(values, other.values))); \
    }                                                                         \
    Vectorized<int##bit##_t> operator!=(                                      \
        const Vectorized<int##bit##_t>& other) const;                         \
    Vectorized<int##bit##_t> operator<(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
          vreinterpretq_s##bit##_u##bit(vcltq_s##bit(values, other.values))); \
    }                                                                         \
    Vectorized<int##bit##_t> operator<=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
          vreinterpretq_s##bit##_u##bit(vcleq_s##bit(values, other.values))); \
    }                                                                         \
    Vectorized<int##bit##_t> operator>(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
          vreinterpretq_s##bit##_u##bit(vcgtq_s##bit(values, other.values))); \
    }                                                                         \
    Vectorized<int##bit##_t> operator>=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
          vreinterpretq_s##bit##_u##bit(vcgeq_s##bit(values, other.values))); \
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
    return vaddq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator-(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vsubq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator&(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vandq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator|(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vorrq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator^(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return veorq_s##bit(a, b);                                                \
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

VEC_INT_NEON_TEMPLATE(2, 64)
VEC_INT_NEON_TEMPLATE(4, 32)
VEC_INT_NEON_TEMPLATE(8, 16)
VEC_INT_NEON_TEMPLATE(16, 8)

inline int32_t Vectorized<int32_t>::reduce_max() const {
  return vmaxvq_s32(values);
}

inline int16_t Vectorized<int16_t>::reduce_max() const {
  return vmaxvq_s16(values);
}

inline int8_t Vectorized<int8_t>::reduce_max() const {
  return vmaxvq_s8(values);
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vmulq_s32(a, b);
}

template <>
Vectorized<int16_t> inline operator*(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return vmulq_s16(a, b);
}

template <>
Vectorized<int8_t> inline operator*(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return vmulq_s8(a, b);
}

template <>
inline Vectorized<int64_t> operator~(const Vectorized<int64_t>& a) {
  int64x2_t val = a;
  return ~val;
}

template <>
inline Vectorized<int32_t> operator~(const Vectorized<int32_t>& a) {
  return vmvnq_s32(a);
}

template <>
inline Vectorized<int16_t> operator~(const Vectorized<int16_t>& a) {
  return vmvnq_s16(a);
}

template <>
inline Vectorized<int8_t> operator~(const Vectorized<int8_t>& a) {
  return vmvnq_s8(a);
}

inline Vectorized<int64_t> Vectorized<int64_t>::operator!=(
    const Vectorized<int64_t>& other) const {
  return ~(*this == other);
}

inline Vectorized<int32_t> Vectorized<int32_t>::operator!=(
    const Vectorized<int32_t>& other) const {
  return ~(*this == other);
}

inline Vectorized<int16_t> Vectorized<int16_t>::operator!=(
    const Vectorized<int16_t>& other) const {
  return ~(*this == other);
}

inline Vectorized<int8_t> Vectorized<int8_t>::operator!=(
    const Vectorized<int8_t>& other) const {
  return ~(*this == other);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vminq_s32(a, b);
}

template <>
Vectorized<int16_t> inline minimum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return vminq_s16(a, b);
}

template <>
Vectorized<int8_t> inline minimum(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return vminq_s8(a, b);
}

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vmaxq_s32(a, b);
}

template <>
Vectorized<int16_t> inline maximum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return vmaxq_s16(a, b);
}

template <>
Vectorized<int8_t> inline maximum(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return vmaxq_s8(a, b);
}

template <int64_t mask>
Vectorized<int64_t> Vectorized<int64_t>::blend(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // Build an array of flags: each bit of element is 1 if the corresponding bit
  // in 'mask' is set, 0 otherwise.
  uint64x2_t maskArray = {
      (mask & 1LL) ? 0xFFFFFFFFFFFFFFFF : 0,
      (mask & 2LL) ? 0xFFFFFFFFFFFFFFFF : 0};
  // Use BSL to select elements from b where the mask is 1, else from a
  return vbslq_s64(maskArray, b.values, a.values);
}

template <int64_t mask>
Vectorized<int32_t> Vectorized<int32_t>::blend(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  // Build an array of flags: each bit of element is 1 if the corresponding bit
  // in 'mask' is set, 0 otherwise.
  uint32x4_t maskArray = {
      (mask & 1LL) ? 0xFFFFFFFF : 0,
      (mask & 2LL) ? 0xFFFFFFFF : 0,
      (mask & 4LL) ? 0xFFFFFFFF : 0,
      (mask & 8LL) ? 0xFFFFFFFF : 0};
  // Use BSL to select elements from b where the mask is 1, else from a
  return vbslq_s32(maskArray, b.values, a.values);
}

template <int64_t mask>
Vectorized<int16_t> Vectorized<int16_t>::blend(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  // Build an array of flags: each bit of element is 1 if the corresponding bit
  // in 'mask' is set, 0 otherwise.
  uint16x8_t maskArray = {
      (mask & 1LL) ? 0xFFFF : 0,
      (mask & 2LL) ? 0xFFFF : 0,
      (mask & 4LL) ? 0xFFFF : 0,
      (mask & 8LL) ? 0xFFFF : 0,
      (mask & 16LL) ? 0xFFFF : 0,
      (mask & 32LL) ? 0xFFFF : 0,
      (mask & 64LL) ? 0xFFFF : 0,
      (mask & 128LL) ? 0xFFFF : 0};
  // Use BSL to select elements from b where the mask is 1, else from a
  return vbslq_s16(maskArray, b.values, a.values);
}

template <int64_t mask>
Vectorized<int8_t> Vectorized<int8_t>::blend(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  // Build an array of flags: each bit of element is 1 if the corresponding bit
  // in 'mask' is set, 0 otherwise.
  uint8x16_t maskArray = {
      (mask & 1LL) ? 0xFF : 0,
      (mask & 2LL) ? 0xFF : 0,
      (mask & 4LL) ? 0xFF : 0,
      (mask & 8LL) ? 0xFF : 0,
      (mask & 16LL) ? 0xFF : 0,
      (mask & 32LL) ? 0xFF : 0,
      (mask & 64LL) ? 0xFF : 0,
      (mask & 128LL) ? 0xFF : 0,
      (mask & 256LL) ? 0xFF : 0,
      (mask & 512LL) ? 0xFF : 0,
      (mask & 1024LL) ? 0xFF : 0,
      (mask & 2048LL) ? 0xFF : 0,
      (mask & 4096LL) ? 0xFF : 0,
      (mask & 8192LL) ? 0xFF : 0,
      (mask & 16384LL) ? 0xFF : 0,
      (mask & 32768LL) ? 0xFF : 0};
  // Use BSL to select elements from b where the mask is 1, else from a
  return vbslq_s8(maskArray, b.values, a.values);
}

#define VEC_INT_NEON_OPS(vl, bit)                                             \
  inline Vectorized<int##bit##_t>::Vectorized(int##bit##_t val) {             \
    values = vdupq_n_s##bit(val);                                             \
  }                                                                           \
  inline Vectorized<int##bit##_t> Vectorized<int##bit##_t>::loadu(            \
      const void* ptr, int64_t count) {                                       \
    if (count == size()) {                                                    \
      return vld1q_s##bit(reinterpret_cast<const int##bit##_t*>(ptr));        \
    } else {                                                                  \
      __at_align__ int##bit##_t tmp_values[size()];                           \
      for (const auto i : c10::irange(size())) {                              \
        tmp_values[i] = 0;                                                    \
      }                                                                       \
      std::memcpy(                                                            \
          tmp_values,                                                         \
          reinterpret_cast<const int##bit##_t*>(ptr),                         \
          count * sizeof(int##bit##_t));                                      \
      return vld1q_s##bit(reinterpret_cast<const int##bit##_t*>(tmp_values)); \
    }                                                                         \
  }                                                                           \
  inline void Vectorized<int##bit##_t>::store(void* ptr, int64_t count)       \
      const {                                                                 \
    if (count == size()) {                                                    \
      vst1q_s##bit(reinterpret_cast<int##bit##_t*>(ptr), values);             \
    } else {                                                                  \
      int##bit##_t tmp_values[size()];                                        \
      vst1q_s##bit(reinterpret_cast<int##bit##_t*>(tmp_values), values);      \
      std::memcpy(ptr, tmp_values, count * sizeof(int##bit##_t));             \
    }                                                                         \
  }

VEC_INT_NEON_OPS(2, 64)
VEC_INT_NEON_OPS(4, 32)
VEC_INT_NEON_OPS(8, 16)
VEC_INT_NEON_OPS(16, 8)

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t x = a;
  int64x2_t y = b;
  return x * y;
}

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t x = a;
  int64x2_t y = b;
  return x / y;
}

template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  int32x4_t x = a;
  int32x4_t y = b;
  return x / y;
}

inline int64_t Vectorized<int64_t>::reduce_max() const {
  return std::max(values[0], values[1]);
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t x = a;
  int64x2_t y = b;
  return {std::min(x[0], y[0]), std::min(x[1], y[1])};
}

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t x = a;
  int64x2_t y = b;
  return {std::max(x[0], y[0]), std::max(x[1], y[1])};
}

template <typename step_t>
inline Vectorized<int64_t> Vectorized<int64_t>::arange(
    int64_t base,
    step_t step) {
  const Vectorized<int64_t> base_vec(base);
  const Vectorized<int64_t> step_vec(step);
  const int64x2_t step_sizes = {0, 1};
  return base_vec.values + step_sizes * step_vec.values;
}

template <typename step_t>
inline Vectorized<int32_t> Vectorized<int32_t>::arange(
    int32_t base,
    step_t step) {
  const Vectorized<int32_t> base_vec(base);
  const Vectorized<int32_t> step_vec(step);
  const int32x4_t step_sizes = {0, 1, 2, 3};
  return vmlaq_s32(base_vec, step_sizes, step_vec);
}

template <typename step_t>
inline Vectorized<int16_t> Vectorized<int16_t>::arange(
    int16_t base,
    step_t step) {
  const Vectorized<int16_t> base_vec(base);
  const Vectorized<int16_t> step_vec(step);
  const int16x8_t step_sizes = {0, 1, 2, 3, 4, 5, 6, 7};
  return vmlaq_s16(base_vec, step_sizes, step_vec);
}

template <typename step_t>
inline Vectorized<int8_t> Vectorized<int8_t>::arange(int8_t base, step_t step) {
  const Vectorized<int8_t> base_vec(base);
  const Vectorized<int8_t> step_vec(step);
  const int8x16_t step_sizes = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return vmlaq_s8(base_vec, step_sizes, step_vec);
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t x = a;
  int64x2_t y = b;
  uint64x2_t u = vreinterpretq_u64_s64(y);
  uint64x2_t z = {std::min(u[0], (uint64_t)63), std::min(u[1], (uint64_t)63)};
  return x >> vreinterpretq_s64_u64(z);
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  int32x4_t x = a;
  int32x4_t y = b;
  uint32x4_t bound = vdupq_n_u32(31);
  uint32x4_t z = vminq_u32(vreinterpretq_u32_s32(y), bound);
  return x >> vreinterpretq_s32_u32(z);
}

template <>
Vectorized<int16_t> inline operator>>(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  int16x8_t x = a;
  int16x8_t y = b;
  uint16x8_t bound = vdupq_n_u16(15);
  uint16x8_t z = vminq_u16(vreinterpretq_u16_s16(y), bound);
  return x >> vreinterpretq_s16_u16(z);
}

template <>
Vectorized<int8_t> inline operator>>(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  int8x16_t x = a;
  int8x16_t y = b;
  uint8x16_t bound = vdupq_n_u8(7);
  int8x16_t z = vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(y), bound));
  return x >> z;
}

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t y = b;
  uint64x2_t u = vreinterpretq_u64_s64(y);
  uint64x2_t z = {std::min(u[0], (uint64_t)64), std::min(u[1], (uint64_t)64)};
  return vshlq_s64(a, vreinterpretq_s64_u64(z));
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  int32x4_t y = b;
  uint32x4_t bound = vdupq_n_u32(32);
  uint32x4_t z = vminq_u32(vreinterpretq_u32_s32(y), bound);
  return vshlq_s32(a, vreinterpretq_s32_u32(z));
}

template <>
Vectorized<int16_t> inline operator<<(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  int16x8_t y = b;
  uint16x8_t bound = vdupq_n_u16(16);
  uint16x8_t z = vminq_u16(vreinterpretq_u16_s16(y), bound);
  return vshlq_s16(a, vreinterpretq_s16_u16(z));
}

template <>
Vectorized<int8_t> inline operator<<(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  int8x16_t y = b;
  uint8x16_t bound = vdupq_n_u8(8);
  int8x16_t z = vreinterpretq_s8_u8(vminq_u8(vreinterpretq_u8_s8(y), bound));
  return vshlq_s8(a, z);
}

inline Vectorized<int64_t> Vectorized<int64_t>::set(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b,
    int64_t count) {
  if (count == 0) {
    return a;
  } else if (count >= 2) {
    return b;
  } else {
    int64x2_t c = {b.values[0], a.values[1]};
    return c;
  }
}

inline Vectorized<int32_t> Vectorized<int32_t>::set(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b,
    int64_t count) {
  if (count == 0) {
    return a;
  } else if (count >= 4) {
    return b;
  } else {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint32x4_t maskArray = {
        (count >= 1LL) ? 0xFFFFFFFF : 0,
        (count >= 2LL) ? 0xFFFFFFFF : 0,
        (count >= 3LL) ? 0xFFFFFFFF : 0,
        0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_s32(maskArray, b.values, a.values);
  }
}

inline Vectorized<int16_t> Vectorized<int16_t>::set(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b,
    int64_t count) {
  if (count == 0) {
    return a;
  } else if (count >= 8) {
    return b;
  } else {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint16x8_t maskArray = {
        static_cast<uint16_t>((count >= 1LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 2LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 3LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 4LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 5LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 6LL) ? 0xFFFF : 0),
        static_cast<uint16_t>((count >= 7LL) ? 0xFFFF : 0),
        0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_s16(maskArray, b.values, a.values);
  }
}

inline Vectorized<int8_t> Vectorized<int8_t>::set(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b,
    int64_t count) {
  if (count == 0) {
    return a;
  } else if (count >= 16) {
    return b;
  } else {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint8x16_t maskArray = {
        static_cast<uint8_t>((count >= 1LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 2LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 3LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 4LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 5LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 6LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 7LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 8LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 9LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 10LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 11LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 12LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 13LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 14LL) ? 0xFF : 0),
        static_cast<uint8_t>((count >= 15LL) ? 0xFF : 0),
        0};

    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_s8(maskArray, b.values, a.values);
  }
}

template <>
Vectorized<int16_t> inline operator/(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  Vectorized<int32_t> highBitsA = vmovl_high_s16(a);
  Vectorized<int32_t> highBitsB = vmovl_high_s16(b);
  Vectorized<int32_t> lowBitsA = vmovl_s16(vget_low_s16(a));
  Vectorized<int32_t> lowBitsB = vmovl_s16(vget_low_s16(b));
  int32x4_t highBitsResult = highBitsA / highBitsB;
  int32x4_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_s16(
      vreinterpretq_s16_s32(lowBitsResult),
      vreinterpretq_s16_s32(highBitsResult));
}

template <>
Vectorized<int8_t> inline operator/(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  Vectorized<int16_t> highBitsA = vmovl_high_s8(a);
  Vectorized<int16_t> highBitsB = vmovl_high_s8(b);
  Vectorized<int16_t> lowBitsA = vmovl_s8(vget_low_s8(a));
  Vectorized<int16_t> lowBitsB = vmovl_s8(vget_low_s8(b));
  int16x8_t highBitsResult = highBitsA / highBitsB;
  int16x8_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_s8(
      vreinterpretq_s8_s16(lowBitsResult),
      vreinterpretq_s8_s16(highBitsResult));
}

template <>
Vectorized<int64_t> inline clamp(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min,
    const Vectorized<int64_t>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<int32_t> inline clamp(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min,
    const Vectorized<int32_t>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<int16_t> inline clamp(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& min,
    const Vectorized<int16_t>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<int8_t> inline clamp(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& min,
    const Vectorized<int8_t>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<int64_t> inline clamp_max(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& max) {
  return minimum(max, a);
}

template <>
Vectorized<int32_t> inline clamp_max(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& max) {
  return minimum(max, a);
}

template <>
Vectorized<int16_t> inline clamp_max(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& max) {
  return minimum(max, a);
}

template <>
Vectorized<int8_t> inline clamp_max(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& max) {
  return minimum(max, a);
}

template <>
Vectorized<int64_t> inline clamp_min(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min) {
  return maximum(min, a);
}

template <>
Vectorized<int32_t> inline clamp_min(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min) {
  return maximum(min, a);
}

template <>
Vectorized<int16_t> inline clamp_min(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& min) {
  return maximum(min, a);
}

template <>
Vectorized<int8_t> inline clamp_min(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& min) {
  return maximum(min, a);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
