#pragma once

#include <vector>
#include <ATen/core/ArrayRef.h>

namespace c10 {

// The passed in function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference
// will result in an error like:
//
//    error: no type named 'type' in 'class std::result_of<foobar::__lambda(T)>'
//
// No explicit template parameters are required.

// Overload for explicit function and ArrayRef
template<typename F, typename T>
inline auto fmap(const T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for(const auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

template<typename F, typename T>
inline auto fmap(T& inputs, const F& fn) -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
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

} // namespace c10
