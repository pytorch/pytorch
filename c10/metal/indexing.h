#pragma once
#include <c10/metal/utils.h>
#include <metal_stdlib>

namespace c10 {
namespace metal {
constant constexpr unsigned max_ndim = 16;

// Given coordinates and strides, calculates offset from the start of the
// tensors
template <typename T>
inline T offset_from_coord(
    thread T idx[max_ndim],
    constant long* strides,
    uint ndim) {
  T rc = 0;
  for (uint i = 0; i < ndim; ++i) {
    rc += idx[i] * T(strides[i]);
  }
  return rc;
}

// Given thread index calculates position in the ndim tensor
template <typename T>
inline void pos_from_thread_index(
    T idx,
    thread T pos[max_ndim],
    constant long* sizes,
    uint ndim) {
  for (uint i = 0; i < ndim; ++i) {
    pos[i] = idx % T(sizes[i]);
    idx /= T(sizes[i]);
  }
}

inline long offset_from_thread_index(
    long idx,
    constant long* sizes,
    constant long* strides,
    uint ndim) {
  long pos[max_ndim];
  pos_from_thread_index(idx, pos, sizes, ndim);
  return offset_from_coord(pos, strides, ndim);
}

template <typename T, typename F>
kernel void unary_dense(
    device result_of<F, T>* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  output[index] = f(input[index]);
}

#define REGISTER_UNARY_OP(NAME, DTYPE0, DTYPE1)                               \
  static_assert(                                                              \
      ::metal::                                                               \
          is_same_v<DTYPE1, ::c10::metal::result_of<NAME##_functor, DTYPE0>>, \
      "Output dtype mismatch for unary op " #NAME " and input " #DTYPE0);     \
  template [[host_name(#NAME "_dense_" #DTYPE1 "_" #DTYPE0)]] kernel void ::  \
      c10::metal::unary_dense<DTYPE0, NAME##_functor>(                        \
          device ::c10::metal::result_of<NAME##_functor, DTYPE0> * output,    \
          constant DTYPE0 * input,                                            \
          uint index)

#define DEFINE_UNARY_FLOATING_FUNCTOR(NAME)                                     \
  struct NAME##_functor {                                                       \
    template <typename T>                                                       \
    inline ::metal::enable_if_t<::metal::is_floating_point_v<T>, T> operator()( \
        const T x) {                                                            \
      return T(NAME(x));                                                        \
    }                                                                           \
    template <typename T>                                                       \
    inline ::metal::enable_if_t<::metal::is_integral_v<T>, float> operator()(   \
        const T x) {                                                            \
      return NAME(static_cast<float>(x));                                       \
    }                                                                           \
  }

} // namespace metal
} // namespace c10
