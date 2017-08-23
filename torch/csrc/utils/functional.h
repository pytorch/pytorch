#pragma once

#include <vector>

namespace torch {

template<typename R, typename T>
static std::vector<R> fmap(const std::vector<T> & inputs, std::function<R(const T &)> fn) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for(auto & input : inputs)
    r.push_back(fn(input));
  return r;
}

}
