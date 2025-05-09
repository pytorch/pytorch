// Metal indexing primitives
#pragma once
#include <c10/metal/common.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

namespace c10 {
namespace metal {

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

template <typename T>
inline T val_at_offs(constant void* ptr, long offs) {
  return *reinterpret_cast<constant T*>(
      static_cast<constant char*>(ptr) + offs);
}

// Value at offset with dynamic cast from provided type
template <typename T>
inline T val_at_offs(constant void* ptr, long offs, ScalarType type) {
  switch (type) {
    case ScalarType::Bool:
      return cast_to<T>(val_at_offs<bool>(ptr, offs));
    case ScalarType::Byte:
      return cast_to<T>(val_at_offs<uchar>(ptr, offs));
    case ScalarType::Char:
      return cast_to<T>(val_at_offs<char>(ptr, offs));
    case ScalarType::Short:
      return cast_to<T>(val_at_offs<short>(ptr, offs));
    case ScalarType::Int:
      return cast_to<T>(val_at_offs<int>(ptr, offs));
    case ScalarType::Long:
      return cast_to<T>(val_at_offs<long>(ptr, offs));
    // Floats
    case ScalarType::Float:
      return cast_to<T>(val_at_offs<float>(ptr, offs));
    case ScalarType::Half:
      return cast_to<T>(val_at_offs<half>(ptr, offs));
#if __METAL_VERSION__ >= 310
    case ScalarType::BFloat16:
      return cast_to<T>(val_at_offs<bfloat>(ptr, offs));
#endif
      // Complex
    case ScalarType::ComplexHalf:
      return cast_to<T>(val_at_offs<half2>(ptr, offs));
    case ScalarType::ComplexFloat:
      return cast_to<T>(val_at_offs<float2>(ptr, offs));
  }
}

template <typename T>
inline device T& ref_at_offs(device void* ptr, long offs) {
  return *reinterpret_cast<device T*>(static_cast<device char*>(ptr) + offs);
}

template <typename T, typename F>
kernel void binary_strided(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* other_strides [[buffer(6)]],
    constant uint3& ndim [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim.x);
  const auto a = val_at_offs<T>(input, input_offs);
  const auto b = val_at_offs<T>(other, other_offs);
  ref_at_offs<result_of<F, T, T>>(output, output_offs) = f(a, b);
}

template <typename T, typename F>
kernel void alpha_binary_strided(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T* alpha [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* output_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant long* other_strides [[buffer(7)]],
    constant uint3& ndim [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim.x);
  const auto a = val_at_offs<T>(input, input_offs);
  const auto b = val_at_offs<T>(other, other_offs);
  ref_at_offs<result_of<F, T, T, T>>(output, output_offs) = f(a, b, *alpha);
}

template <typename T, typename F>
kernel void binary_strided_cast(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* other_strides [[buffer(6)]],
    constant uint3& ndim_types [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim_types.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim_types.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim_types.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim_types.x);
  const auto a =
      val_at_offs<T>(input, input_offs, static_cast<ScalarType>(ndim_types.y));
  const auto b =
      val_at_offs<T>(other, other_offs, static_cast<ScalarType>(ndim_types.z));
  ref_at_offs<result_of<F, T, T>>(output, output_offs) = f(a, b);
}

template <typename T, typename F>
kernel void alpha_binary_strided_cast(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T* alpha [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* output_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant long* other_strides [[buffer(7)]],
    constant uint3& ndim_types [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim_types.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim_types.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim_types.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim_types.x);
  const auto a =
      val_at_offs<T>(input, input_offs, static_cast<ScalarType>(ndim_types.y));
  const auto b =
      val_at_offs<T>(other, other_offs, static_cast<ScalarType>(ndim_types.z));
  ref_at_offs<result_of<F, T, T, T>>(output, output_offs) = f(a, b, *alpha);
}

template <typename T, typename F>
kernel void binary_dense(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid]);
}

template <typename T, typename F>
kernel void alpha_binary_dense(
    device result_of<F, T, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    constant T* alpha [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid], *alpha);
}

template <typename T, typename F>
kernel void binary_dense_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant uint4& sizes_types [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      other, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b);
}

template <typename T, typename F>
kernel void alpha_binary_dense_cast(
    device result_of<F, T, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T* alpha [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      other, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, *alpha);
}

#define REGISTER_BINARY_OP(NAME, DTYPEI, DTYPEO)                               \
  static_assert(                                                               \
      ::metal::is_same_v<                                                      \
          DTYPEO,                                                              \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI>>,            \
      "Output dtype mismatch for binary op " #NAME " and input " #DTYPEI);     \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI)]] kernel void :: \
      c10::metal::binary_strided<DTYPEI, NAME##_functor>(                      \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint3& ndim,                                                \
          uint tid);                                                           \
  template [[host_name(#NAME "_strided_cast_" #DTYPEI)]] kernel void ::c10::   \
      metal::binary_strided_cast<DTYPEI, NAME##_functor>(                      \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint3& ndim_types,                                          \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI)]] kernel void ::   \
      c10::metal::binary_dense<DTYPEI, NAME##_functor>(                        \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * other_,                                            \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_cast_" #DTYPEI)]] kernel void ::c10::     \
      metal::binary_dense_cast<DTYPEI, NAME##_functor>(                        \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant void* input,                                                \
          constant void* other,                                                \
          constant uint4& sizes_types,                                         \
          uint tid)

#define REGISTER_BINARY_ALPHA_OP(NAME, DTYPEI, DTYPEO)                         \
  static_assert(                                                               \
      ::metal::is_same_v<                                                      \
          DTYPEO,                                                              \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI>>,    \
      "Output dtype mismatch for binary op " #NAME " and input " #DTYPEI);     \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI)]] kernel void :: \
      c10::metal::alpha_binary_strided<DTYPEI, NAME##_functor>(                \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant DTYPEI* alpha,                                              \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint3& ndim,                                                \
          uint tid);                                                           \
  template [[host_name(#NAME "_strided_cast_" #DTYPEI)]] kernel void ::c10::   \
      metal::alpha_binary_strided_cast<DTYPEI, NAME##_functor>(                \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant DTYPEI* alpha,                                              \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint3& ndim_types,                                          \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI)]] kernel void ::   \
      c10::metal::alpha_binary_dense<DTYPEI, NAME##_functor>(                  \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI> *          \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * other_,                                            \
          constant DTYPEI * alpha,                                             \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_cast_" #DTYPEI)]] kernel void ::c10::     \
      metal::alpha_binary_dense_cast<DTYPEI, NAME##_functor>(                  \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI> *          \
              out_,                                                            \
          constant void* input,                                                \
          constant void* other,                                                \
          constant DTYPEI* alpha,                                              \
          constant uint4& sizes_types,                                         \
          uint tid)
} // namespace metal
} // namespace c10
