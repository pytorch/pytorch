#pragma once

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include <c10/util/ArrayRef.h>

namespace c10 {

namespace detail {

// Prefer consuming an owned element when the callable accepts an rvalue.
// This covers callables taking T, T&&, or const T&. Fall back to T& for
// callables that can only operate on an lvalue.
template <typename F, typename T>
inline decltype(auto) fmap_invoke_owned(T& input, const F& fn) {
  if constexpr (std::is_invocable_v<const F&, T&&>) {
    return fn(std::move(input));
  } else {
    static_assert(
        std::is_invocable_v<const F&, T&>,
        "fmap callable must accept the input element");
    return fn(input);
  }
}

} // namespace detail

// For non-consuming inputs, the passed function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference will result in an
// error like:
//
//    error: no type named 'type' in 'class std::invoke_result<foobar::__lambda, T>'
//
// No explicit template parameters are required.

// Overload for explicit function and ArrayRef
template<class F, class T>
inline auto fmap(const T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for(const auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

// Consuming overload for an owned vector. Prefer passing elements as rvalues
// when the callable supports it. If the map produces the same element type by
// value, transform in place so the result can reuse the vector's allocation.
template<class F, class T>
inline auto fmap(std::vector<T>&& inputs, const F& fn) {
  using raw_result_type = decltype(detail::fmap_invoke_owned(
      std::declval<T&>(), std::declval<const F&>()));
  using result_type = std::remove_cvref_t<raw_result_type>;

  static_assert(
      !std::is_void_v<result_type>,
      "fmap callable must return a value");

  // Do not reuse storage for reference results. In particular, a T&& result
  // may alias input, turning assignment back into the same slot into a
  // self-move.
  if constexpr (
      std::is_same_v<result_type, T> &&
      !std::is_reference_v<raw_result_type> &&
      std::is_assignable_v<T&, raw_result_type>) {
    for (auto&& input : inputs) {
      input = detail::fmap_invoke_owned(input, fn);
    }
    return std::move(inputs);
  } else {
    std::vector<result_type> r;
    r.reserve(inputs.size());
    for (auto&& input : inputs) {
      r.emplace_back(detail::fmap_invoke_owned(input, fn));
    }
    return r;
  }
}

// C++ forbids taking an address of a constructor, so here's a workaround...
// Overload for constructor (R) application
template<typename R, typename T>
inline std::vector<R> fmap(const T& inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(R(input));
  return r;
}

// Consuming overload for constructor application. Move from each element when
// R can be constructed from T&&; otherwise preserve the existing lvalue path.
template<typename R, typename T>
inline std::vector<R> fmap(std::vector<T>&& inputs) {
  if constexpr (std::is_same_v<R, T>) {
    return std::move(inputs);
  } else {
    std::vector<R> r;
    r.reserve(inputs.size());
    for (auto&& input : inputs) {
      if constexpr (std::is_constructible_v<R, T&&>) {
        r.emplace_back(std::move(input));
      } else {
        static_assert(
            std::is_constructible_v<R, T&>,
            "fmap result must be constructible from the input element");
        r.emplace_back(input);
      }
    }
    return r;
  }
}

template<typename F, typename T>
inline std::vector<T> filter(at::ArrayRef<T> inputs, const F& fn) {
  std::vector<T> r;
  r.reserve(inputs.size());
  for(auto & input : inputs) {
    if (fn(input)) {
      r.push_back(input);
    }
  }
  return r;
}

template<typename F, typename T>
inline std::vector<T> filter(const std::vector<T>& inputs, const F& fn) {
  return filter<F, T>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

// TODO: Constrain this overload to move-assignable T with concepts once all
// supported compilers handle C++20 constraints reliably.
template<typename F, typename T>
inline std::vector<T> filter(std::vector<T>&& inputs, const F& fn) {
  if constexpr (std::is_move_assignable_v<T>) {
    inputs.erase(
        std::remove_if(
            inputs.begin(),
            inputs.end(),
            [&](auto&& input) { return !fn(input); }),
        inputs.end());
    return std::move(inputs);
  } else {
    return filter<F, T>(static_cast<const std::vector<T>&>(inputs), fn);
  }
}

} // namespace c10
