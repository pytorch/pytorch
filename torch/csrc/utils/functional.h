#pragma once

#include <vector>
#include <ATen/ATen.h>

namespace torch {

// NB: Order matters.  If you put the forwarding definitions before
// the actual implementations, C++ will resolve the overload to
// itself, because there's an implicit conversion to vector in ArrayRef.

// The passed in function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference
// will result in an error like:
//
//    error: no type named 'type' in 'class std::result_of<foobar::__lambda(T)>'
//
// No explicit template parameters are required.

// Overload for explicit function and ArrayRef
template<typename F, typename T, typename R = typename std::result_of<F(T)>::type>
inline std::vector<R> fmap(at::ArrayRef<T> inputs, const F& fn) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

// Overload for explicit function and vector (this is required because
// template deduction will not apply an implicit conversion from std::vector
// to ArrayRef)
template<typename F, typename T, typename R = typename std::result_of<F(T)>::type>
inline std::vector<R> fmap(const std::vector<T>& inputs, const F& fn) {
  return fmap<F, T, R>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

// C++ forbids taking an address of a constructor, so here's a workaround...

// Overload for ArrayRef and constructor (R) application
template<typename R, typename T>
inline std::vector<R> fmap(at::ArrayRef<T> inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(R(input));
  return r;
}

// Overload for std::vector and constructor (R) application
template<typename R, typename T>
inline std::vector<R> fmap(const std::vector<T>& inputs) {
  return fmap<R, T>(static_cast<at::ArrayRef<T>>(inputs));
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

}
