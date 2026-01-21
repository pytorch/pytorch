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

template <typename T, typename T2, typename F>
kernel void unary_alpha_dense(
    device result_of<F, T, T2>* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T2& alpha [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  output[index] = f(input[index], alpha);
}

template <typename T, typename T2, typename F>
kernel void unary_alpha_strided(
    device result_of<F, T, T2>* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* input_strides [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant T2& alpha [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  output[output_offs] = f(input[input_offs], alpha);
}

#define REGISTER_UNARY_ALPHA_OP(NAME, DTYPEI, DTYPEA, DTYPEO)              \
  static_assert(                                                           \
      ::metal::is_same_v<                                                  \
          DTYPEO,                                                          \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEA>>,        \
      "Output dtype mismatch for unary op " #NAME " and input " #DTYPEI);  \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI                 \
                             "_" #DTYPEA)]] kernel void ::c10::metal::     \
      unary_alpha_dense<DTYPEI, DTYPEA, NAME##_functor>(                   \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEA> * \
              output,                                                      \
          constant DTYPEI * input,                                         \
          constant DTYPEA & alpha,                                         \
          uint index);                                                     \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI               \
                             "_" #DTYPEA)]] kernel void ::c10::metal::     \
      unary_alpha_strided<DTYPEI, DTYPEA, NAME##_functor>(                 \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEA> * \
              output,                                                      \
          constant DTYPEI * input,                                         \
          constant long* sizes,                                            \
          constant long* input_strides,                                    \
          constant long* output_strides,                                   \
          constant uint& ndim,                                             \
          constant DTYPEA& alpha,                                          \
          uint index)

template <typename T>
inline T val_at_offs(constant void* ptr, long offs) {
  return *reinterpret_cast<constant T*>(
      static_cast<constant char*>(ptr) + offs);
}

// Value at offset with dynamic cast from provided type
template <typename T>
inline T val_at_offs(device void* ptr, long offs) {
  return *reinterpret_cast<device T*>(static_cast<device char*>(ptr) + offs);
}

template <typename T, typename P>
inline T val_at_offs(P ptr, long offs, ScalarType type) {
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
    case ScalarType::BFloat16:
      return cast_to<T>(val_at_offs<bfloat>(ptr, offs));
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

// Binary elementwise ops kernels
// Right now there are 4 flavors available:
// - binary_dense where both input, other and output are dense and share the
// same type
// - binary_strided when all inputs are of the same types, but some elements are
// strided
// - binary_dense_cast - inputs are dense, but of different dtypes
// - binary_strided_cast - inputs or output are strided and of different dtypes
// - binary_dense_broadcast - one input is dense, another one is broadcastable
// Note about accuracy (for more info see
// https://github.com/pytorch/pytorch/issues/152736) Sometimes when kernel is
// invoked to produce `half` output, but one of the arguments is float arguments
// should be upcast to float, rather than downcast to half At the moment this is
// expressed with `om_t` optional argument (which stands for opmath_type) which
// is identical to output type but could be something else

template <typename T, typename F, typename om_t = T>
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
  using res_t = result_of<F, T, T>;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim.x);
  const auto a = val_at_offs<T>(input, input_offs);
  const auto b = val_at_offs<T>(other, other_offs);
  ref_at_offs<res_t>(output, output_offs) =
      static_cast<res_t>(f(om_t(a), om_t(b)));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_strided(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
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
  ref_at_offs<result_of<F, T, T, T2>>(output, output_offs) = f(a, b, alpha);
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_strided_cast(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* other_strides [[buffer(6)]],
    constant uint4& ndim_types [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim_types.x);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim_types.x);
  const auto other_offs = offset_from_coord(pos, other_strides, ndim_types.x);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim_types.x);
  const auto a = val_at_offs<om_t>(
      input, input_offs, static_cast<ScalarType>(ndim_types.y));
  const auto b = val_at_offs<om_t>(
      other, other_offs, static_cast<ScalarType>(ndim_types.z));
  ref_at_offs<res_t>(output, output_offs) = static_cast<res_t>(f(a, b));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_strided_cast(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* output_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant long* other_strides [[buffer(7)]],
    constant uint4& ndim_types [[buffer(8)]],
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
  ref_at_offs<result_of<F, T, T, T2>>(output, output_offs) = f(a, b, alpha);
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_dense(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  out[tid] = static_cast<res_t>(f(om_t(input[tid]), om_t(other[tid])));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid], alpha);
}

template <typename T, typename F, typename om_t = T>
kernel void binary_dense_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant uint4& sizes_types [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  const auto a = val_at_offs<om_t>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<om_t>(
      other, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = static_cast<res_t>(f(a, b));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_cast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      other, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, alpha);
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_dense_broadcast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* broadcast [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  out[tid] = static_cast<res_t>(
      f(om_t(input[tid]), om_t(broadcast[tid % broadcast_numel])));
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_dense_broadcast_rhs(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant T* broadcast [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  out[tid] = static_cast<res_t>(
      f(om_t(broadcast[tid % broadcast_numel]), om_t(input[tid])));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_broadcast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* broadcast [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant T2& alpha [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], broadcast[tid % broadcast_numel], alpha);
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_broadcast_rhs(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant T* broadcast [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant T2& alpha [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(broadcast[tid % broadcast_numel], input[tid], alpha);
}

template <typename T, typename F, typename om_t = T>
kernel void binary_dense_broadcast_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* broadcast [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  const auto a = val_at_offs<om_t>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<om_t>(
      broadcast,
      (tid % broadcast_numel) * sizes_types.y,
      static_cast<ScalarType>(sizes_types.w));
  out[tid] = static_cast<res_t>(f(a, b));
}

template <typename T, typename F, typename om_t = T>
kernel void binary_dense_broadcast_rhs_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant void* broadcast [[buffer(1)]],
    constant void* input [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  const auto a = val_at_offs<om_t>(
      broadcast,
      (tid % broadcast_numel) * sizes_types.x,
      static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<om_t>(
      input, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = static_cast<res_t>(f(a, b));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_broadcast_cast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* broadcast [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant T2& alpha [[buffer(4)]],
    constant uint4& sizes_types [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      broadcast,
      (tid % broadcast_numel) * sizes_types.y,
      static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, alpha);
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_broadcast_rhs_cast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant void* broadcast [[buffer(1)]],
    constant void* input [[buffer(2)]],
    constant long& broadcast_numel [[buffer(3)]],
    constant T2& alpha [[buffer(4)]],
    constant uint4& sizes_types [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      broadcast,
      (tid % broadcast_numel) * sizes_types.x,
      static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      input, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, alpha);
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_dense_scalar(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    device T* scalar [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  out[tid] = static_cast<res_t>(f(om_t(input[tid]), om_t(scalar[0])));
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void binary_dense_scalar_lhs(
    device result_of<F, T, T>* out [[buffer(0)]],
    device T* scalar [[buffer(1)]],
    constant T* input [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  out[tid] = static_cast<res_t>(f(om_t(scalar[0]), om_t(input[tid])));
}

template <typename T, typename F, typename om_t = T>
kernel void binary_dense_scalar_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    device void* scalar [[buffer(2)]],
    constant uint4& sizes_types [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  const auto a = val_at_offs<om_t>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b =
      val_at_offs<om_t>(scalar, 0, static_cast<ScalarType>(sizes_types.w));
  out[tid] = static_cast<res_t>(f(a, b));
}

template <typename T, typename F, typename om_t = T>
kernel void binary_dense_scalar_lhs_cast(
    device result_of<F, T, T>* out [[buffer(0)]],
    device void* scalar [[buffer(1)]],
    constant void* input [[buffer(2)]],
    constant uint4& sizes_types [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T>;
  const auto a =
      val_at_offs<om_t>(scalar, 0, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<om_t>(
      input, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = static_cast<res_t>(f(a, b));
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_scalar(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    device T* scalar [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], scalar[0], alpha);
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_scalar_lhs(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    device T* scalar [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(scalar[0], input[tid], alpha);
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_scalar_cast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    device void* scalar [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a = val_at_offs<T>(
      input, tid * sizes_types.x, static_cast<ScalarType>(sizes_types.z));
  const auto b =
      val_at_offs<T>(scalar, 0, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, alpha);
}

template <typename T, typename T2, typename F>
kernel void binary_alpha_dense_scalar_lhs_cast(
    device result_of<F, T, T, T2>* out [[buffer(0)]],
    device void* scalar [[buffer(1)]],
    constant void* input [[buffer(2)]],
    constant T2& alpha [[buffer(3)]],
    constant uint4& sizes_types [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  const auto a =
      val_at_offs<T>(scalar, 0, static_cast<ScalarType>(sizes_types.z));
  const auto b = val_at_offs<T>(
      input, tid * sizes_types.y, static_cast<ScalarType>(sizes_types.w));
  out[tid] = f(a, b, alpha);
}

#define REGISTER_BINARY_OP_(NAME, DTYPEI, DTYPEO, OMT)                         \
  static_assert(                                                               \
      ::metal::is_same_v<                                                      \
          DTYPEO,                                                              \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI>>,            \
      "Output dtype mismatch for binary op " #NAME " and input " #DTYPEI);     \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI)]] kernel void :: \
      c10::metal::binary_strided<DTYPEI, NAME##_functor, OMT>(                 \
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
      metal::binary_strided_cast<DTYPEI, NAME##_functor, OMT>(                 \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint4& ndim_types,                                          \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI)]] kernel void ::   \
      c10::metal::binary_dense<DTYPEI, NAME##_functor, OMT>(                   \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * other_,                                            \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_cast_" #DTYPEI)]] kernel void ::c10::     \
      metal::binary_dense_cast<DTYPEI, NAME##_functor, OMT>(                   \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant void* input,                                                \
          constant void* other,                                                \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_" #DTYPEO "_" #DTYPEI)]]        \
  kernel void ::c10::metal::                                                   \
      binary_dense_broadcast<DTYPEI, NAME##_functor, OMT>(                     \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * broadcast_,                                        \
          constant long& broadcast_numel,                                      \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_rhs_" #DTYPEO "_" #DTYPEI)]]    \
  kernel void ::c10::metal::                                                   \
      binary_dense_broadcast_rhs<DTYPEI, NAME##_functor, OMT>(                 \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant DTYPEI * broadcast_,                                        \
          constant DTYPEI * input_,                                            \
          constant long& broadcast_numel,                                      \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_cast_" #DTYPEI)]]               \
  kernel void ::c10::metal::                                                   \
      binary_dense_broadcast_cast<DTYPEI, NAME##_functor, OMT>(                \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant void* input_,                                               \
          constant void* broadcast_,                                           \
          constant long& broadcast_numel,                                      \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_rhs_cast_" #DTYPEI)]]           \
  kernel void ::c10::metal::                                                   \
      binary_dense_broadcast_rhs_cast<DTYPEI, NAME##_functor, OMT>(            \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant void* broadcast_,                                           \
          constant void* input_,                                               \
          constant long& broadcast_numel,                                      \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_" #DTYPEO "_" #DTYPEI)]]           \
  kernel void ::c10::metal::binary_dense_scalar<DTYPEI, NAME##_functor, OMT>(  \
      device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> * out_,   \
      constant DTYPEI * input_,                                                \
      device DTYPEI * scalar_,                                                 \
      uint tid);                                                               \
  template [[host_name(#NAME "_dense_scalar_lhs_" #DTYPEO "_" #DTYPEI)]]       \
  kernel void ::c10::metal::                                                   \
      binary_dense_scalar_lhs<DTYPEI, NAME##_functor, OMT>(                    \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          device DTYPEI * scalar_,                                             \
          constant DTYPEI * input_,                                            \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_cast_" #DTYPEI)]]                  \
  kernel void ::c10::metal::                                                   \
      binary_dense_scalar_cast<DTYPEI, NAME##_functor, OMT>(                   \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          constant void* input_,                                               \
          device void* scalar_,                                                \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_lhs_cast_" #DTYPEI)]]              \
  kernel void ::c10::metal::                                                   \
      binary_dense_scalar_lhs_cast<DTYPEI, NAME##_functor, OMT>(               \
          device ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI> *     \
              out_,                                                            \
          device void* scalar_,                                                \
          constant void* input_,                                               \
          constant uint4& sizes_types,                                         \
          uint tid)

// OpMath Binary Op promotes inputs to higher precision type before Functor call
#define REGISTER_OPMATH_BINARY_OP(NAME, DTYPEI, DTYPEO) \
  REGISTER_BINARY_OP_(NAME, DTYPEI, DTYPEO, ::c10::metal::opmath_t<DTYPEI>)

#define REGISTER_BINARY_OP(NAME, DTYPEI, DTYPEO) \
  REGISTER_BINARY_OP_(NAME, DTYPEI, DTYPEO, DTYPEI)

#define REGISTER_BINARY_ALPHA_OP(NAME, DTYPEI, DTYPEA, DTYPEO)                 \
  static_assert(                                                               \
      ::metal::is_same_v<                                                      \
          DTYPEO,                                                              \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA>>,    \
      "Output dtype mismatch for binary op " #NAME " and input " #DTYPEI);     \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI                   \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_strided<DTYPEI, DTYPEA, NAME##_functor>(                    \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant DTYPEA& alpha,                                              \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint3& ndim,                                                \
          uint tid);                                                           \
  template [[host_name(#NAME "_strided_cast_" #DTYPEI                          \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_strided_cast<DTYPEI, DTYPEA, NAME##_functor>(               \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other,                                                \
          constant DTYPEA& alpha,                                              \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other_strides,                                        \
          constant uint4& ndim_types,                                          \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI                     \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense<DTYPEI, DTYPEA, NAME##_functor>(                      \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * other_,                                            \
          constant DTYPEA & alpha,                                             \
          uint tid);                                                           \
  template                                                                     \
      [[host_name(#NAME "_dense_cast_" #DTYPEI "_" #DTYPEA)]] kernel void ::   \
          c10::metal::binary_alpha_dense_cast<DTYPEI, DTYPEA, NAME##_functor>( \
              device ::c10::metal::                                            \
                      result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *      \
                  out_,                                                        \
              constant void* input,                                            \
              constant void* other,                                            \
              constant DTYPEA& alpha,                                          \
              constant uint4& sizes_types,                                     \
              uint tid);                                                       \
  template [[host_name(#NAME "_dense_broadcast_" #DTYPEO "_" #DTYPEI           \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_broadcast<DTYPEI, DTYPEA, NAME##_functor>(            \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * broadcast_,                                        \
          constant long& broadcast_numel,                                      \
          constant DTYPEA& alpha,                                              \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_rhs_" #DTYPEO "_" #DTYPEI       \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_broadcast_rhs<DTYPEI, DTYPEA, NAME##_functor>(        \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant DTYPEI * broadcast_,                                        \
          constant DTYPEI * input_,                                            \
          constant long& broadcast_numel,                                      \
          constant DTYPEA& alpha,                                              \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_cast_" #DTYPEI                  \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_broadcast_cast<DTYPEI, DTYPEA, NAME##_functor>(       \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant void* input_,                                               \
          constant void* broadcast_,                                           \
          constant long& broadcast_numel,                                      \
          constant DTYPEA& alpha,                                              \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_broadcast_rhs_cast_" #DTYPEI              \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_broadcast_rhs_cast<DTYPEI, DTYPEA, NAME##_functor>(   \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant void* broadcast_,                                           \
          constant void* input_,                                               \
          constant long& broadcast_numel,                                      \
          constant DTYPEA& alpha,                                              \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_" #DTYPEO "_" #DTYPEI              \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_scalar<DTYPEI, DTYPEA, NAME##_functor>(               \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          device DTYPEI * scalar_,                                             \
          constant DTYPEA & alpha,                                             \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_lhs_" #DTYPEO "_" #DTYPEI          \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_scalar_lhs<DTYPEI, DTYPEA, NAME##_functor>(           \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          device DTYPEI * scalar_,                                             \
          constant DTYPEI * input_,                                            \
          constant DTYPEA & alpha,                                             \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_cast_" #DTYPEI                     \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_scalar_cast<DTYPEI, DTYPEA, NAME##_functor>(          \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          constant void* input_,                                               \
          device void* scalar_,                                                \
          constant DTYPEA& alpha,                                              \
          constant uint4& sizes_types,                                         \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_scalar_lhs_cast_" #DTYPEI                 \
                             "_" #DTYPEA)]] kernel void ::c10::metal::         \
      binary_alpha_dense_scalar_lhs_cast<DTYPEI, DTYPEA, NAME##_functor>(      \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEA> *          \
              out_,                                                            \
          device void* scalar_,                                                \
          constant void* input_,                                               \
          constant DTYPEA& alpha,                                              \
          constant uint4& sizes_types,                                         \
          uint tid)

// Ternary elementwise ops kernels
// Right now there are 4 flavors available:
// - ternary_dense where both input, other1, other2, and output are dense and
// share the same type
// - ternary_strided when all inputs are of the same types, but some elements
// are strided
// - ternary_dense_cast - inputs are dense, but of different dtypes
// - ternary_strided_cast - inputs or output are strided and of different dtypes
// Note about accuracy (for more info see
// https://github.com/pytorch/pytorch/issues/152736) Sometimes when kernel is
// invoked to produce `half` output, but one of the arguments is float arguments
// should be upcast to float, rather than downcast to half At the moment this is
// expressed with `om_t` optional argument (which stands for opmath_type) which
// is identical to output type but could be something else

template <typename T, typename F, typename om_t = T>
kernel void ternary_strided(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other1 [[buffer(2)]],
    constant void* other2 [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* output_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant long* other1_strides [[buffer(7)]],
    constant long* other2_strides [[buffer(8)]],
    constant uint& ndim [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T, T>;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto other1_offs = offset_from_coord(pos, other1_strides, ndim);
  const auto other2_offs = offset_from_coord(pos, other2_strides, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  const auto a = val_at_offs<T>(input, input_offs);
  const auto b = val_at_offs<T>(other1, other1_offs);
  const auto c = val_at_offs<T>(other2, other2_offs);
  ref_at_offs<res_t>(output, output_offs) =
      static_cast<res_t>(f(om_t(a), om_t(b), om_t(c)));
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void ternary_strided_cast(
    device void* output [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other1 [[buffer(2)]],
    constant void* other2 [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* output_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant long* other1_strides [[buffer(7)]],
    constant long* other2_strides [[buffer(8)]],
    constant uint& ndim [[buffer(9)]],
    constant uint4& types [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T, T>;
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto other1_offs = offset_from_coord(pos, other1_strides, ndim);
  const auto other2_offs = offset_from_coord(pos, other2_strides, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  const auto a =
      val_at_offs<om_t>(input, input_offs, static_cast<ScalarType>(types.x));
  const auto b =
      val_at_offs<om_t>(other1, other1_offs, static_cast<ScalarType>(types.y));
  const auto c =
      val_at_offs<om_t>(other2, other2_offs, static_cast<ScalarType>(types.z));
  ref_at_offs<res_t>(output, output_offs) = static_cast<res_t>(f(a, b, c));
}

template <typename T, typename F, typename om_t = opmath_t<T>>
kernel void ternary_dense(
    device result_of<F, T, T, T>* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other1 [[buffer(2)]],
    constant T* other2 [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T, T>;
  out[tid] = static_cast<res_t>(
      f(om_t(input[tid]), om_t(other1[tid]), om_t(other2[tid])));
}

template <typename T, typename F, typename om_t = T>
kernel void ternary_dense_cast(
    device result_of<F, T, T, T>* out [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* other1 [[buffer(2)]],
    constant void* other2 [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    constant uint3& types [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  using res_t = result_of<F, T, T, T>;
  const auto a =
      val_at_offs<om_t>(input, tid * sizes.x, static_cast<ScalarType>(types.x));
  const auto b = val_at_offs<om_t>(
      other1, tid * sizes.y, static_cast<ScalarType>(types.y));
  const auto c = val_at_offs<om_t>(
      other2, tid * sizes.z, static_cast<ScalarType>(types.z));
  out[tid] = static_cast<res_t>(f(a, b, c));
}

#define REGISTER_TERNARY_OP_(NAME, DTYPEI, DTYPEO, OMT)                        \
  static_assert(                                                               \
      ::metal::is_same_v<                                                      \
          DTYPEO,                                                              \
          ::c10::metal::result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI>>,    \
      "Output dtype mismatch for ternary op " #NAME " and input " #DTYPEI);    \
  template [[host_name(#NAME "_strided_" #DTYPEO "_" #DTYPEI)]] kernel void :: \
      c10::metal::ternary_strided<DTYPEI, NAME##_functor, OMT>(                \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other1,                                               \
          constant void* other2,                                               \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other1_strides,                                       \
          constant long* other2_strides,                                       \
          constant uint& ndim,                                                 \
          uint tid);                                                           \
  template [[host_name(#NAME "_strided_cast_" #DTYPEI)]] kernel void ::c10::   \
      metal::ternary_strided_cast<DTYPEI, NAME##_functor, OMT>(                \
          device void* out,                                                    \
          constant void* input,                                                \
          constant void* other1,                                               \
          constant void* other2,                                               \
          constant long* sizes,                                                \
          constant long* output_strides,                                       \
          constant long* input_strides,                                        \
          constant long* other1_strides,                                       \
          constant long* other2_strides,                                       \
          constant uint& ndim,                                                 \
          constant uint4& types,                                               \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_" #DTYPEO "_" #DTYPEI)]] kernel void ::   \
      c10::metal::ternary_dense<DTYPEI, NAME##_functor, OMT>(                  \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI> *          \
              out_,                                                            \
          constant DTYPEI * input_,                                            \
          constant DTYPEI * other1_,                                           \
          constant DTYPEI * other2_,                                           \
          uint tid);                                                           \
  template [[host_name(#NAME "_dense_cast_" #DTYPEI)]] kernel void ::c10::     \
      metal::ternary_dense_cast<DTYPEI, NAME##_functor, OMT>(                  \
          device ::c10::metal::                                                \
                  result_of<NAME##_functor, DTYPEI, DTYPEI, DTYPEI> *          \
              out_,                                                            \
          constant void* input,                                                \
          constant void* other1,                                               \
          constant void* other2,                                               \
          constant uint3& sizes,                                               \
          constant uint3& types,                                               \
          uint tid)

// OpMath ternary Op promotes inputs to higher precision type before Functor
// call
#define REGISTER_OPMATH_TERNARY_OP(NAME, DTYPEI, DTYPEO) \
  REGISTER_TERNARY_OP_(NAME, DTYPEI, DTYPEO, ::c10::metal::opmath_t<DTYPEI>)

#define REGISTER_TERNARY_OP(NAME, DTYPEI, DTYPEO) \
  REGISTER_TERNARY_OP_(NAME, DTYPEI, DTYPEO, DTYPEI)

} // namespace metal
} // namespace c10
