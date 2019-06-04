#pragma once

namespace at { namespace native { namespace {

// n: number of function arguments (arity)
// traits: function_traits (see FunctionTraits.h)
// s: index of scalar argument or -1
template <int n, typename traits, int s=-1>
struct IsContiguous {
  static bool eval(const int64_t* strides) {
    using type = typename traits::template arg<n - 1>::type;
    return strides[n] == (s == n ? 0 : sizeof(type)) &&
        IsContiguous<n - 1, traits, s>::eval(strides);
  }
};

template <typename traits, int s>
struct IsContiguous<0, traits, s> {
  static bool eval(const int64_t* strides) {
    return strides[0] == sizeof(typename traits::result_type);
  }
};

// output and all inputs are contiguous
template <typename traits>
static inline bool is_contiguous(const int64_t* strides) {
  return IsContiguous<traits::arity, traits>::eval(strides);
}

// input at `s` is scalar (stride 0); output and other inputs are contiguous
// NB: output is typically at strides[0] so first input corresponds to s=1
template <typename traits, int s>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  return IsContiguous<traits::arity, traits, s>::eval(strides);
}

}}}
