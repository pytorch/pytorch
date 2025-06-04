#pragma once

namespace at::native { inline namespace CPU_CAPABILITY {

// n: number of function arguments (arity)
// traits: function_traits (see FunctionTraits.h)
// s: index of scalar argument or -1
template <int n, int stride_index, typename traits, int s=-1>
struct IsContiguous {
  static bool eval(const int64_t* strides) {
    using type = typename traits::template arg<n - 1>::type;
    return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
           IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
  }
};

// will be called when there is an output exists
template <typename traits, int s>
struct IsContiguous<0, 0, traits, s> {
  static bool eval(const int64_t* strides) {
    return strides[0] == sizeof(typename traits::result_type);
  }
};

// will be called when there is no output
template <typename traits, int s>
struct IsContiguous<0, -1, traits, s> {
  static bool eval(const int64_t* /*strides*/) {
    return true;
  }
};

// output and all inputs are contiguous
template <
    typename traits,
    std::enable_if_t<std::is_void_v<typename traits::result_type>>* =
        nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  return IsContiguous<traits::arity, traits::arity - 1, traits>::eval(strides);
}

template <typename traits,
    std::enable_if_t<!std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
}

// input at `s` is scalar (stride 0); output and other inputs are contiguous
// NB: output is typically at strides[0] so first input corresponds to s=1
template <typename traits, int s,
    std::enable_if_t<std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  return IsContiguous<traits::arity, traits::arity - 1, traits, s>::eval(strides);
}

template <typename traits, int s,
    std::enable_if_t<!std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  return IsContiguous<traits::arity, traits::arity, traits, s>::eval(strides);
}

}}
