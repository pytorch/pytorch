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

template <typename T, typename F>
kernel void unary_strided(
    device result_of<F, T>* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* input_strides [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  output[output_offs] = f(input[input_offs]);
}

#define REGISTER_UNARY_OP(NAME, DTYPE0, DTYPE1)                                \
  static_assert(                                                               \
      ::metal::                                                                \
          is_same_v<DTYPE1, ::c10::metal::result_of<NAME##_functor, DTYPE0>>,  \
      "Output dtype mismatch for unary op " #NAME " and input " #DTYPE0);      \
  template [[host_name(#NAME "_dense_" #DTYPE1 "_" #DTYPE0)]] kernel void ::   \
      c10::metal::unary_dense<DTYPE0, NAME##_functor>(                         \
          device ::c10::metal::result_of<NAME##_functor, DTYPE0> * output,     \
          constant DTYPE0 * input,                                             \
          uint index);                                                         \
  template [[host_name(#NAME "_strided_" #DTYPE1 "_" #DTYPE0)]] kernel void :: \
      c10::metal::unary_strided<DTYPE0, NAME##_functor>(                       \
          device ::c10::metal::result_of<NAME##_functor, DTYPE0> * output,     \
          constant DTYPE0 * input,                                             \
          constant long* sizes,                                                \
          constant long* input_strides,                                        \
          constant long* output_strides,                                       \
          constant uint& ndim,                                                 \
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

template <typename T, typename F>
kernel void binary_strided(
    constant T* input [[buffer(0)]],
    constant T* other [[buffer(1)]],
    device result_of<F, T, T>* output [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* input_strides [[buffer(4)]],
    constant long* other_strides [[buffer(5)]],
    constant long* output_strides [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim) / sizeof(T);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim) / sizeof(T);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim) / sizeof(result_of<F, T, T>);
  output[output_offs] = f(input[input_offs], other[other_offs]);
}

template <typename T, typename F>
kernel void binary_dense(
    constant T* input [[buffer(0)]],
    constant T* other [[buffer(1)]],
    device result_of<F, T, T>* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid]);
}

#define REGISTER_BINARY_INDEXING_OP(NAME, DTYPE)                               \
  template [[host_name(#NAME "_strided_" #DTYPE)]] kernel void ::c10::metal::  \
      binary_strided<DTYPE, NAME##_functor>(                                   \
          constant DTYPE * input,                                              \
          constant DTYPE * other,                                              \
          device ::c10::metal::result_of<NAME##_functor, DTYPE, DTYPE> * out_, \
          constant long* sizes,                                                \
          constant long* input_strides,                                        \
          constant long* output_strides,                                       \
          constant long* other_strides,                                        \
          constant uint& ndim,                                                 \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPE)]] kernel void ::c10::metal::    \
      binary_dense<DTYPE, NAME##_functor>(                                     \
          constant DTYPE * input_,                                             \
          constant DTYPE * other_,                                             \
          device ::c10::metal::result_of<NAME##_functor, DTYPE, DTYPE> * out_, \
          uint tid)
} // namespace metal
} // namespace c10
