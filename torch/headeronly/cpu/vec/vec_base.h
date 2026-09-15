#pragma once

#include <torch/headeronly/cpu/vec/intrinsics.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/NumericUtils.h>
#include <torch/headeronly/util/complex.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

// Capability-specific alignment and width definitions shared by the vec
// headers.
#ifdef CPU_CAPABILITY_AVX512
#ifndef __at_align__
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
#endif
#define VECTOR_WIDTH 64
#define int_vector __m512i
#elif defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
#ifndef __at_align__
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(16)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(16))
#else
#define __at_align__
#endif
#endif
// SVE code expects 256-bit vectors; leave that set for SVE256.
#define VECTOR_WIDTH 16
#else
#ifndef __at_align__
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(32))
#else
#define __at_align__
#endif
#endif
#define VECTOR_WIDTH 32
#define int_vector __m256i
#endif

// Primary templates required by the header-only NEON Vectorized<float>
// specialization. The generic Vectorized class remains in ATen.
namespace at::vec::inline CPU_CAPABILITY {

#if defined(__s390x__)
template <class T, class TEMP = void>
#else
template <typename T>
#endif
struct is_vec_specialized_for : std::bool_constant<false> {
};

template <typename T>
constexpr bool is_vec_specialized_for_v = is_vec_specialized_for<T>::value;

#if defined(__s390x__)
template <class T, class TEMP = void>
#else
template <class T>
#endif
struct Vectorized;

template <class T>
Vectorized<T> inline operator-(const Vectorized<T>& a) {
  return a.neg();
}

template <class T>
Vectorized<T> inline operator+(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator-(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator*(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator/(const Vectorized<T>& a, const Vectorized<T>& b)
    __ubsan_ignore_float_divide_by_zero__ {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec,
    const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_max(
    const Vectorized<T>& a,
    const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_min(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

struct Vectorizedi;

#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
namespace headeronly_detail {

template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  int_vector buffer;
#if defined(CPU_CAPABILITY_AVX2)
  int_vector a_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)b));
#elif defined(CPU_CAPABILITY_AVX512)
  int_vector a_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)b));
#endif
  buffer = op(a_buffer, b_buffer);
  __at_align__ std::array<T, Vectorized<T>::size()> results{};

#if defined(CPU_CAPABILITY_AVX2)
  _mm256_store_si256(reinterpret_cast<int_vector*>(results.data()), buffer);
#elif defined(CPU_CAPABILITY_AVX512)
  _mm512_store_si512(reinterpret_cast<int_vector*>(results.data()), buffer);
#endif
  return Vectorized<T>::loadu(results.data());
}

} // namespace headeronly_detail

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_and_si512 or _mm256_and_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_and_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_and_si512(a, b); });
#endif
}
template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_or_si512 or _mm256_or_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_or_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_or_si512(a, b); });
#endif
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_xor_si512 or _mm256_xor_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_xor_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_xor_si512(a, b); });
#endif
}

#else
namespace headeronly_detail {

template <typename T>
auto load(char const* data) -> T {
  T ret;
  std::memcpy(&ret, data, sizeof(ret));
  return ret;
}

template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  static constexpr uint32_t element_no = VECTOR_WIDTH / sizeof(intmax_t);
  __at_align__ std::array<intmax_t, element_no> buffer{};
  static_assert(
      VECTOR_WIDTH % sizeof(intmax_t) == 0,
      "VECTOR_WIDTH not a multiple of sizeof(intmax_t)");
  static_assert(
      sizeof(buffer) == sizeof(Vectorized<T>),
      "sizeof(buffer) must match sizeof(Vectorized<T>)");
  // We should be using memcpy in order to respect the strict aliasing rule
  // see: https://github.com/pytorch/pytorch/issues/66119
  // Using char* is defined in the C11 standard 6.5 Expression paragraph 7
  // (http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
  const auto* a_data = a.as_bytes();
  const auto* b_data = b.as_bytes();
  // load each intmax_t chunk and process; increase pointers by sizeof(intmax_t)
  for (auto& out : buffer) {
    out =
        op(headeronly_detail::load<intmax_t>(a_data),
           headeronly_detail::load<intmax_t>(b_data));
    a_data += sizeof(intmax_t);
    b_data += sizeof(intmax_t);
  }
  assert(a_data == a.as_bytes() + sizeof(a));
  assert(b_data == b.as_bytes() + sizeof(b));
  return Vectorized<T>::loadu(buffer.data());
}

} // namespace headeronly_detail

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_and<>());
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_or<>());
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_xor<>());
}

#endif // defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)

template <typename T>
inline Vectorized<T> fmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b + c;
}

template <typename T>
inline Vectorized<T> fnmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return -(a * b) + c;
}

template <typename T>
inline Vectorized<T> fmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b - c;
}

template <typename T>
inline Vectorized<T> fnmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return -(a * b) - c;
}

} // namespace at::vec::inline CPU_CAPABILITY

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, vec)

using at::vec::clamp;
using at::vec::clamp_max;
using at::vec::clamp_min;
using at::vec::fmadd;
using at::vec::fmsub;
using at::vec::fnmadd;
using at::vec::fnmsub;
using at::vec::is_vec_specialized_for;
using at::vec::is_vec_specialized_for_v;
using at::vec::maximum;
using at::vec::minimum;
using at::vec::Vectorized;
using at::vec::operator&;
using at::vec::operator*;
using at::vec::operator+;
using at::vec::operator-;
using at::vec::operator/;
using at::vec::operator^;
using at::vec::operator|;

HIDDEN_NAMESPACE_END(torch, headeronly, vec)
