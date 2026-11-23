#pragma once

#include <utility>
#include <vector>

namespace c10 {

// Pack a parameter pack of std::vector<T> rvalues into a
// std::vector<std::vector<T>> without copying each inner vector.
// std::vector<X>{a, b} copies: the initializer_list backing array is const,
// so the vector's init-list ctor can only copy out of it.
template <typename T, typename... Vs>
std::vector<std::vector<T>> make_nested(std::vector<T>&& head, Vs&&... tail) {
  std::vector<std::vector<T>> out;
  out.reserve(1 + sizeof...(Vs));
  out.emplace_back(std::move(head));
  ((out.emplace_back(std::forward<Vs>(tail))), ...);
  return out;
}

} // namespace c10
