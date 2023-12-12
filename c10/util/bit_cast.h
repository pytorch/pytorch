#pragma once

#include <cstring>
#include <type_traits>

namespace c10 {

// Implementations of std::bit_cast() from C++ 20.
//
// This is a less sketchy version of reinterpret_cast.
//
// See https://en.cppreference.com/w/cpp/numeric/bit_cast for more
// information as well as the source of our implementations.
template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
        std::is_trivially_copyable<To>::value,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

} // namespace c10
