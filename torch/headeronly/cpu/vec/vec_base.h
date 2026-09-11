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

// Primary templates required by the header-only NEON Vectorized<float>
// specialization. The generic Vectorized class remains in ATen.
namespace at::vec::inline CPU_CAPABILITY {

#if defined(__s390x__)
template <class T, class TEMP = void>
#else
template <typename T>
#endif
struct is_vec_specialized_for : std::bool_constant<false> {};

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
Vectorized<T> inline operator+(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator-(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator*(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator/(
    const Vectorized<T>& a,
    const Vectorized<T>& b) __ubsan_ignore_float_divide_by_zero__ {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (torch::headeronly::_isnan(a[i])) {
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (torch::headeronly::_isnan(a[i])) {
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> clamp(
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
Vectorized<T> clamp_max(
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
Vectorized<T> clamp_min(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

struct Vectorizedi;

namespace headeronly_detail {

template <class T, typename Op>
Vectorized<T> inline bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
#if defined(CPU_CAPABILITY_AVX512)
  using int_vector = __m512i;
  int_vector a_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)b));
  int_vector buffer = op(a_buffer, b_buffer);
  alignas(64) std::array<T, Vectorized<T>::size()> results{};
  _mm512_store_si512(reinterpret_cast<int_vector*>(results.data()), buffer);
  return Vectorized<T>::loadu(results.data());
#elif defined(CPU_CAPABILITY_AVX2)
  using int_vector = __m256i;
  int_vector a_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)b));
  int_vector buffer = op(a_buffer, b_buffer);
  alignas(32) std::array<T, Vectorized<T>::size()> results{};
  _mm256_store_si256(reinterpret_cast<int_vector*>(results.data()), buffer);
  return Vectorized<T>::loadu(results.data());
#else
  static_assert(sizeof(Vectorized<T>) % sizeof(intmax_t) == 0);
  constexpr auto chunk_count = sizeof(Vectorized<T>) / sizeof(intmax_t);
  std::array<intmax_t, chunk_count> buffer{};
  const auto* a_data = a.as_bytes();
  const auto* b_data = b.as_bytes();
  for (auto& out : buffer) {
    intmax_t a_value;
    intmax_t b_value;
    std::memcpy(&a_value, a_data, sizeof(a_value));
    std::memcpy(&b_value, b_data, sizeof(b_value));
    out = op(a_value, b_value);
    a_data += sizeof(intmax_t);
    b_data += sizeof(intmax_t);
  }
  assert(a_data == a.as_bytes() + sizeof(a));
  assert(b_data == b.as_bytes() + sizeof(b));
  return Vectorized<T>::loadu(buffer.data());
#endif
}

} // namespace headeronly_detail

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> inline operator&(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
#if defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m512i x, __m512i y) { return _mm512_and_si512(x, y); });
#elif defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m256i x, __m256i y) { return _mm256_and_si256(x, y); });
#else
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_and<>());
#endif
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> inline operator|(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
#if defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m512i x, __m512i y) { return _mm512_or_si512(x, y); });
#elif defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m256i x, __m256i y) { return _mm256_or_si256(x, y); });
#else
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_or<>());
#endif
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> inline operator^(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
#if defined(CPU_CAPABILITY_AVX512)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m512i x, __m512i y) { return _mm512_xor_si512(x, y); });
#elif defined(CPU_CAPABILITY_AVX2)
  return headeronly_detail::bitwise_binary_op(
      a, b, [](__m256i x, __m256i y) { return _mm256_xor_si256(x, y); });
#else
  return headeronly_detail::bitwise_binary_op(a, b, std::bit_xor<>());
#endif
}

template <typename T>
Vectorized<T> inline fmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b + c;
}

template <typename T>
Vectorized<T> inline fnmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return -(a * b) + c;
}

template <typename T>
Vectorized<T> inline fmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b - c;
}

template <typename T>
Vectorized<T> inline fnmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return -(a * b) - c;
}

} // namespace at::vec::inline CPU_CAPABILITY

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, vec)

using at::vec::Vectorized;
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
using at::vec::operator&;
using at::vec::operator*;
using at::vec::operator+;
using at::vec::operator-;
using at::vec::operator/;
using at::vec::operator^;
using at::vec::operator|;

HIDDEN_NAMESPACE_END(torch, headeronly, vec)
