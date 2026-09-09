#pragma once

#include <torch/headeronly/cpu/vec/intrinsics.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/complex.h>

#include <type_traits>

// Declarations required by the header-only NEON Vectorized<float>
// specialization. Generic Vectorized implementations remain in ATen.
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
Vectorized<T> operator+(const Vectorized<T>& a, const Vectorized<T>& b);

template <class T>
Vectorized<T> operator-(const Vectorized<T>& a, const Vectorized<T>& b);

template <class T>
Vectorized<T> operator*(const Vectorized<T>& a, const Vectorized<T>& b);

template <class T>
Vectorized<T> operator/(const Vectorized<T>& a, const Vectorized<T>& b);

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> maximum(const Vectorized<T>& a, const Vectorized<T>& b);

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> minimum(const Vectorized<T>& a, const Vectorized<T>& b);

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> clamp(
    const Vectorized<T>& a,
    const Vectorized<T>& min,
    const Vectorized<T>& max);

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> clamp_max(
    const Vectorized<T>& a,
    const Vectorized<T>& max);

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> clamp_min(
    const Vectorized<T>& a,
    const Vectorized<T>& min);

struct Vectorizedi;

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b);

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b);

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b);

template <typename T>
Vectorized<T> fmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c);

template <typename T>
Vectorized<T> fnmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c);

template <typename T>
Vectorized<T> fmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c);

template <typename T>
Vectorized<T> fnmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c);

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
