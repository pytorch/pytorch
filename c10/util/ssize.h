#pragma once

#include <c10/util/Exception.h>
#include <c10/util/TypeSafeSignMath.h>

#include <cstddef>
#include <type_traits>

namespace c10 {

// Implementations of std::ssize() from C++ 20.
//
// This is useful in particular for avoiding -Werror=sign-compare
// issues.
//
// Use this with argument-dependent lookup, e.g.:
// use c10::ssize;
// auto size = ssize(container);
//
// As with the standard library version, containers are permitted to
// specialize this with a free function defined in the same namespace.
//
// See https://en.cppreference.com/w/cpp/iterator/size for more
// information as well as the source of our implementations.
//
// We augment the implementation by adding an assert() if an overflow
// would occur.

template <typename C>
constexpr auto ssize(const C& c) -> std::
    common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>> {
  using R = std::
      common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>;
  // We expect this to be exceedingly rare to fire and don't wish to
  // pay a performance hit in release mode.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!greater_than_max<R>(c.size()));
  return static_cast<R>(c.size());
}

template <typename T, std::ptrdiff_t N>
// NOLINTNEXTLINE(*-c-arrays)
constexpr auto ssize(const T (&array)[N]) noexcept -> std::ptrdiff_t {
  return N;
}

} // namespace c10
