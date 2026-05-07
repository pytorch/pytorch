#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <c10/macros/Macros.h>
#include <tuple>
#include <utility>

namespace c10::guts {

#if defined(__HIP__)

// std::apply is not available in HIP device code because it lacks
// __host__ __device__ annotations in the standard library.
namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
C10_HOST_DEVICE constexpr auto apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>) {
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
} // namespace detail

template <class F, class Tuple>
C10_HOST_DEVICE constexpr auto apply(F&& f, Tuple&& t) {
  return detail::apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif

} // namespace c10::guts

#endif // C10_UTIL_CPP17_H_
