#include <array>
#include <utility>

namespace c10 {

// This helper function creates a constexpr std::array
// From a compile time list of values, without requiring you to explicitly
// write out the length.
//
// See also https://stackoverflow.com/a/26351760/23845
template <typename V, typename... T>
inline constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  return {{std::forward<T>(t)...}};
}

} // namespace c10
