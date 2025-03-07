#pragma once

#include <cstring>
#include <type_traits>

#if __has_include(<bit>) && (defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L)
#include <bit>
#define C10_HAVE_STD_BIT_CAST 1
#else
#define C10_HAVE_STD_BIT_CAST 0
#endif // __has_include(<bit>) && (__cplusplus >= 202002L ||
       // (defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L))

namespace c10 {

#if C10_HAVE_STD_BIT_CAST
using std::bit_cast;
#else
// Implementations of std::bit_cast() from C++ 20.
//
// This is a less sketchy version of reinterpret_cast.
//
// See https://en.cppreference.com/w/cpp/numeric/bit_cast for more
// information as well as the source of our implementations.
template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}
#endif // C10_HAVE_STD_BIT_CAST
#undef C10_HAVE_STD_BIT_CAST

} // namespace c10
