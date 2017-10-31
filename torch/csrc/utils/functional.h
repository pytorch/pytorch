#pragma once

#include <vector>

namespace torch {

// NB: Order matters.  If you put the forwarding definitions before
// the actual implementations, C++ will resolve the overload to
// itself, because there's an implicit conversion to vector in ArrayRef.

template<typename F, typename T, typename R = typename std::result_of<F(T)>::type>
inline std::vector<R> fmap(at::ArrayRef<T> inputs, const F& fn) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

template<typename F, typename T, typename R = typename std::result_of<F(T)>::type>
inline std::vector<R> fmap(const std::vector<T>& inputs, const F& fn) {
  return fmap<F, T, R>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

template<typename R, typename T>
inline std::vector<R> fmap(at::ArrayRef<T> inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(R(input));
  return r;
}

// C++ forbids taking an address of a constructor, so here's a workaround...
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
