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

#define VEC_UINT_NEON_TEMPLATE(vl, bit)                                       \
  template <>                                                                 \
  struct is_vec_specialized_for<uint##bit##_t> : std::bool_constant<true> {}; \
                                                                              \
  template <>                                                                 \
  class Vectorized<uint##bit##_t> {                                           \
    using neon_type = uint##bit##x##vl##_t;                                   \
                                                                              \
   private:                                                                   \
    neon_type values;                                                         \
                                                                              \
   public:                                                                    \
    using value_type = uint##bit##_t;                                         \
    using size_type = int;                                                    \
    static constexpr size_type size() {                                       \
      return vl;                                                              \
    }                                                                         \
    Vectorized() {                                                            \
      values = vdupq_n_u##bit(0);                                             \
    }                                                                         \
    Vectorized(neon_type v) : values(v) {}                                    \
    Vectorized(uint##bit##_t val);                                            \
    template <                                                                \
        typename... Args,                                                     \
        typename = std::enable_if_t<(sizeof...(Args) == size())>>             \
    Vectorized(Args... vals) {                                                \
      __at_align__ uint##bit##_t buffer[size()] = {vals...};                  \
      values = vld1q_u##bit(buffer);                                          \
    }                                                                         \
    operator neon_type() const {                                              \
      return values;                                                          \
    }                                                                         \
    static Vectorized<uint##bit##_t> loadu(                                   \
        const void* ptr,                                                      \
        uint64_t count = size());                                             \
    void store(void* ptr, uint64_t count = size()) const;                     \
    template <uint64_t mask>                                                  \
    static Vectorized<uint##bit##_t> blend(                                   \
        const Vectorized<uint##bit##_t>& a,                                   \
        const Vectorized<uint##bit##_t>& b);                                  \
    static Vectorized<uint##bit##_t> blendv(                                  \
        const Vectorized<uint##bit##_t>& a,                                   \
        const Vectorized<uint##bit##_t>& b,                                   \
        const Vectorized<uint##bit##_t>& mask_) {                             \
      return vbslq_u##bit(mask_.values, b, a);                                \
    }                                                                         \
    template <typename step_t>                                                \
    static Vectorized<uint##bit##_t> arange(                                  \
        value_type base = 0,                                                  \
        step_t step = static_cast<step_t>(1));                                \
    static Vectorized<uint##bit##_t> set(                                     \
        const Vectorized<uint##bit##_t>& a,                                   \
        const Vectorized<uint##bit##_t>& b,                                   \
        uint64_t count = size());                                             \
    const uint##bit##_t& operator[](uint idx) const = delete;                 \
    uint##bit##_t& operator[](uint idx) = delete;                             \
    Vectorized<uint##bit##_t> abs() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> real() const {                                  \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> imag() const {                                  \
      return vdupq_n_u##bit(0);                                               \
    }                                                                         \
    Vectorized<uint##bit##_t> conj() const {                                  \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> neg() const {                                   \
      return vreinterpretq_u##bit##_s##bit(                                   \
          vnegq_s##bit(vreinterpretq_s##bit##_u##bit(values)));               \
    }                                                                         \
    uint##bit##_t reduce_add() const {                                        \
      return vaddvq_u##bit(values);                                           \
    }                                                                         \
    uint##bit##_t reduce_max() const;                                         \
    Vectorized<uint##bit##_t> operator==(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vceqq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator!=(                                     \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> operator<(                                      \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcltq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator<=(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcleq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator>(                                      \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcgtq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator>=(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcgeq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> eq(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> ne(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> gt(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> ge(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> lt(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> le(                                             \
        const Vectorized<uint##bit##_t>& other) const;                        \
  };                                                                          \
  template <>                                                                 \
  Vectorized<uint##bit##_t> inline operator+(                                 \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b) {                                   \
    return vaddq_u##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<uint##bit##_t> inline operator-(                                 \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b) {                                   \
    return vsubq_u##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<uint##bit##_t> inline operator&(                                 \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b) {                                   \
    return vandq_u##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<uint##bit##_t> inline operator|(                                 \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b) {                                   \
    return vorrq_u##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<uint##bit##_t> inline operator^(                                 \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b) {                                   \
    return veorq_u##bit(a, b);                                                \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::eq(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this == other) & Vectorized<uint##bit##_t>(1);                   \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::ne(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this != other) & Vectorized<uint##bit##_t>(1);                   \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::gt(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this > other) & Vectorized<uint##bit##_t>(1);                    \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::ge(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this >= other) & Vectorized<uint##bit##_t>(1);                   \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::lt(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this < other) & Vectorized<uint##bit##_t>(1);                    \
  }                                                                           \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::le(             \
      const Vectorized<uint##bit##_t>& other) const {                         \
    return (*this <= other) & Vectorized<uint##bit##_t>(1);                   \
  }

VEC_UINT_NEON_TEMPLATE(16, 8)

inline uint8_t Vectorized<uint8_t>::reduce_max() const {
  return vmaxvq_u8(values);
}

template <>
Vectorized<uint8_t> inline operator*(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return vmulq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator~(const Vectorized<uint8_t>& a) {
  return vmvnq_u8(a);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::operator!=(
    const Vectorized<uint8_t>& other) const {
  return ~(*this == other);
}

template <>
Vectorized<uint8_t> inline minimum(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return vminq_u8(a, b);
}

template <>
Vectorized<uint8_t> inline maximum(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return vmaxq_u8(a, b);
}

template <uint64_t mask>
Vectorized<uint8_t> Vectorized<uint8_t>::blend(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
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
  return vbslq_u8(maskArray, b.values, a.values);
}

#define VEC_UINT_NEON_OPS(vl, bit)                                             \
  inline Vectorized<uint##bit##_t>::Vectorized(uint##bit##_t val) {            \
    values = vdupq_n_u##bit(val);                                              \
  }                                                                            \
  inline Vectorized<uint##bit##_t> Vectorized<uint##bit##_t>::loadu(           \
      const void* ptr, uint64_t count) {                                       \
    if (count == size()) {                                                     \
      return vld1q_u##bit(reinterpret_cast<const uint##bit##_t*>(ptr));        \
    } else {                                                                   \
      __at_align__ uint##bit##_t tmp_values[size()];                           \
      for (const auto i : c10::irange(size())) {                               \
        tmp_values[i] = 0;                                                     \
      }                                                                        \
      std::memcpy(                                                             \
          tmp_values,                                                          \
          reinterpret_cast<const uint##bit##_t*>(ptr),                         \
          count * sizeof(uint##bit##_t));                                      \
      return vld1q_u##bit(reinterpret_cast<const uint##bit##_t*>(tmp_values)); \
    }                                                                          \
  }                                                                            \
  inline void Vectorized<uint##bit##_t>::store(void* ptr, uint64_t count)      \
      const {                                                                  \
    if (count == size()) {                                                     \
      vst1q_u##bit(reinterpret_cast<uint##bit##_t*>(ptr), values);             \
    } else {                                                                   \
      uint##bit##_t tmp_values[size()];                                        \
      vst1q_u##bit(reinterpret_cast<uint##bit##_t*>(tmp_values), values);      \
      std::memcpy(ptr, tmp_values, count * sizeof(uint##bit##_t));             \
    }                                                                          \
  }

VEC_UINT_NEON_OPS(16, 8)

template <typename step_t>
inline Vectorized<uint8_t> Vectorized<uint8_t>::arange(
    uint8_t base,
    step_t step) {
  const Vectorized<uint8_t> base_vec(base);
  const Vectorized<uint8_t> step_vec(step);
  const uint8x16_t step_sizes = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return vmlaq_u8(base_vec, step_sizes, step_vec);
}

template <>
Vectorized<uint8_t> inline operator>>(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  uint8x16_t x = a;
  uint8x16_t bound = vdupq_n_u8(8);
  uint8x16_t z = vminq_u8(b, bound);
  return x >> z;
}

template <>
Vectorized<uint8_t> inline operator<<(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  uint8x16_t bound = vdupq_n_u8(8);
  uint8x16_t z = vminq_u8(b, bound);
  return vshlq_u8(a, vreinterpretq_s8_u8(z));
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::set(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b,
    uint64_t count) {
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
    return vbslq_u8(maskArray, b.values, a.values);
  }
}

template <>
Vectorized<uint8_t> inline operator/(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  uint8x16_t x = a;
  uint8x16_t y = b;
  return x / y;
}

template <>
Vectorized<uint8_t> inline clamp(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min,
    const Vectorized<uint8_t>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<uint8_t> inline clamp_max(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& max) {
  return minimum(max, a);
}

template <>
Vectorized<uint8_t> inline clamp_min(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min) {
  return maximum(min, a);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
