#pragma once

#include <vector>
#include <c10/util/ArrayRef.h>

namespace c10 {

// The passed in function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference
// will result in an error like:
//
//    error: no type named 'type' in 'class std::result_of<foobar::__lambda(T)>'
//
// No explicit template parameters are required.


// A traditional C++ approach with iterators
template<class F, class InputIt>
inline auto fmap(InputIt first, InputIt last, const F& fn)
            -> std::vector<decltype(fn(*first))> {
  std::vector<decltype(fn(*first))> r;
  while (first != last) {
    r.push_back(fn(*first));
  }
  return r;
}

// Overload for explicit function and ArrayRef
template<class F, class T>
inline auto fmap(const T& inputs, const F& fn)
            -> std::vector<decltype(fn(*inputs.begin()))> {
  return fmap(inputs.begin(), inputs.end(), fn);
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

template<class F, class InputIt, class OutputIt>
inline void fmap_to(InputIt first, InputIt last, OutputIt d_first, const F& fn) {
  while (first != last) {
    *d_first++ = *first++;
  }
}

template<class F, class From, class To>
inline void fmap_to(const From& inputs, To& outputs, const F& fn) {
  return fmap_to(inputs.begin(), inputs.end(), std::back_inserter(outputs), fn);
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
