#pragma once

#include <vector>

namespace torch {

template<typename F, typename T, typename R = typename std::result_of<F(T)>::type>
static std::vector<R> fmap(const std::vector<T> & inputs, const F& fn) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

template<typename F, typename T>
static std::vector<T> filter(const std::vector<T> & inputs, const F& fn) {
  std::vector<T> r;
  r.reserve(inputs.size());
  for(auto & input : inputs) {
    if (fn(input)) {
      r.push_back(input);
    }
  }
  return r;
}

}
