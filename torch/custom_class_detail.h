#pragma once

namespace torch {
namespace jit {

namespace detail {

template <class RetType, class...>
struct types {
  constexpr static bool hasRet = true;
  using type = types;
};
template <class... Args>
struct types<void, Args...> {
  constexpr static bool hasRet = false;
  using type = types;
};


}  // namespace detail
}}  // namespace torch::jit
