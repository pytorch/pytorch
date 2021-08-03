#pragma once
#include <c10/macros/Macros.h>

// Utility to guaruntee complete unrolling of a loop where the bounds are known
// at compile time. Various pragmas achieve similar effects, but are not as
// portable across compilers.

// Example: c10::ForcedUnroll<4>{}(f); is equivalent to f(0); f(1); f(2); f(3);

namespace c10 {

template <int n>
struct ForcedUnroll {
  template <typename Func>
  C10_ALWAYS_INLINE void operator()(const Func& f) const {
    ForcedUnroll<n - 1>{}(f);
    f(n - 1);
  }
};

template <>
struct ForcedUnroll<1> {
  template <typename Func>
  C10_ALWAYS_INLINE void operator()(const Func& f) const {
    f(0);
  }
};

} // namespace c10
