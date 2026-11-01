#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// weight_norm backward per slice: dot = sum(grad_w * v), grad_g = dot / norm,
// grad_v = g / norm * (grad_w - v * dot / norm^2). A slice's elements sit at
// base + j * stride; a thread handles j in [first, len) in steps of `step`.
template <typename T>
inline opmath_t<T> slice_dot(
    device const T* grad_w,
    device const T* v,
    uint base,
    uint stride,
    uint first,
    uint step,
    uint len) {
  opmath_t<T> dot = 0;
  for (uint j = first; j < len; j += step) {
    dot += opmath_t<T>(grad_w[base + j * stride]) *
        opmath_t<T>(v[base + j * stride]);
  }
  return dot;
}

template <typename T>
inline void slice_apply(
    device const T* grad_w,
    device const T* v,
    device const T* g,
    device const T* norms,
    device T* grad_v,
    device T* grad_g,
    opmath_t<T> dot,
    uint slice,
    bool leader,
    uint base,
    uint stride,
    uint first,
    uint step,
    uint len) {
  const auto rnorm = 1 / opmath_t<T>(norms[slice]);
  const auto a = opmath_t<T>(g[slice]) * rnorm;
  const auto b = a * dot * rnorm * rnorm;
  if (leader) {
    grad_g[slice] = T(dot * rnorm);
  }
  for (uint j = first; j < len; j += step) {
    grad_v[base + j * stride] =
        T(a * opmath_t<T>(grad_w[base + j * stride]) -
          b * opmath_t<T>(v[base + j * stride]));
  }
}

// Sums one column of a tile stored row-major in threadgroup memory.
template <typename T>
inline T tile_column_sum(threadgroup T* tile, uint x, uint2 lsize) {
  T sum = 0;
  for (uint y = 0; y < lsize.y; ++y) {
    sum += tile[y * lsize.x + x];
  }
  return sum;
}

// dim == 0: one threadgroup per contiguous slice.
template <typename T>
kernel void weight_norm_backward_rows(
    device const T* grad_w [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device const T* g [[buffer(2)]],
    device const T* norms [[buffer(3)]],
    device T* grad_v [[buffer(4)]],
    device T* grad_g [[buffer(5)]],
    constant uint2& slices_len [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup opmath_t<T> shared[simdgroup_size];
  const auto len = slices_len.y;
  const auto dot = threadgroup_sum(
      shared, slice_dot(grad_w, v, row * len, 1, lid, lsize, len), lid, lsize);
  slice_apply(
      grad_w,
      v,
      g,
      norms,
      grad_v,
      grad_g,
      dot,
      row,
      lid == 0,
      row * len,
      1,
      lid,
      lsize,
      len);
}

// dim == last: a threadgroup covers up to simdgroup_size adjacent columns, so
// reads coalesce, and a block of rows, so few columns still fill the GPU. Pass
// 1 writes one partial dot per (row block, column).
template <typename T>
kernel void weight_norm_backward_cols_dot(
    device const T* grad_w [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device opmath_t<T>* partial [[buffer(2)]],
    constant uint4& cols_rows_block_nblocks [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup opmath_t<T> tile[max_threadgroup_size];
  const auto cols = cols_rows_block_nblocks.x, col = gid.x,
             block = cols_rows_block_nblocks.z;
  const auto first = tg.y * block + lid.y,
             last = uint(min(
                 ulong(cols_rows_block_nblocks.y), ulong(tg.y + 1) * block));
  tile[lid.y * lsize.x + lid.x] =
      col < cols ? slice_dot(grad_w, v, col, cols, first, lsize.y, last) : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && col < cols) {
    partial[tg.y * cols + col] = tile_column_sum(tile, lid.x, lsize);
  }
}

// Pass 2 reduces each column's partials into row 0 of the buffer.
kernel void weight_norm_backward_cols_sum(
    device float* partial [[buffer(0)]],
    constant uint4& cols_rows_block_nblocks [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]]) {
  threadgroup float tile[max_threadgroup_size];
  const auto cols = cols_rows_block_nblocks.x, col = gid.x;
  float dot = 0;
  for (uint b = lid.y; col < cols && b < cols_rows_block_nblocks.w;
       b += lsize.y) {
    dot += partial[b * cols + col];
  }
  tile[lid.y * lsize.x + lid.x] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && col < cols) {
    partial[col] = tile_column_sum(tile, lid.x, lsize);
  }
}

// Pass 3 scales the rows of each block with the reduced dot product.
template <typename T>
kernel void weight_norm_backward_cols(
    device const T* grad_w [[buffer(0)]],
    device const T* v [[buffer(1)]],
    device const T* g [[buffer(2)]],
    device const T* norms [[buffer(3)]],
    device T* grad_v [[buffer(4)]],
    device T* grad_g [[buffer(5)]],
    device const opmath_t<T>* partial [[buffer(6)]],
    constant uint4& cols_rows_block_nblocks [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const auto cols = cols_rows_block_nblocks.x, col = gid.x,
             block = cols_rows_block_nblocks.z;
  if (col >= cols) {
    return;
  }
  const auto first = tg.y * block + lid.y,
             last = uint(min(
                 ulong(cols_rows_block_nblocks.y), ulong(tg.y + 1) * block));
  slice_apply(
      grad_w,
      v,
      g,
      norms,
      grad_v,
      grad_g,
      partial[col],
      col,
      tg.y == 0 && lid.y == 0,
      col,
      cols,
      first,
      lsize.y,
      last);
}

#define INSTANTIATE_WEIGHT_NORM_BACKWARD(T)                               \
  template [[host_name("weight_norm_backward_rows_" #T)]] kernel void     \
  weight_norm_backward_rows<T>(                                           \
      device const T*,                                                    \
      device const T*,                                                    \
      device const T*,                                                    \
      device const T*,                                                    \
      device T*,                                                          \
      device T*,                                                          \
      constant uint2&,                                                    \
      uint,                                                               \
      uint,                                                               \
      uint);                                                              \
  template [[host_name("weight_norm_backward_cols_dot_" #T)]] kernel void \
  weight_norm_backward_cols_dot<T>(                                       \
      device const T*,                                                    \
      device const T*,                                                    \
      device opmath_t<T>*,                                                \
      constant uint4&,                                                    \
      uint2,                                                              \
      uint2,                                                              \
      uint2,                                                              \
      uint2);                                                             \
  template [[host_name("weight_norm_backward_cols_" #T)]] kernel void     \
  weight_norm_backward_cols<T>(                                           \
      device const T*,                                                    \
      device const T*,                                                    \
      device const T*,                                                    \
      device const T*,                                                    \
      device T*,                                                          \
      device T*,                                                          \
      device const opmath_t<T>*,                                          \
      constant uint4&,                                                    \
      uint2,                                                              \
      uint2,                                                              \
      uint2,                                                              \
      uint2);

INSTANTIATE_WEIGHT_NORM_BACKWARD(float)
INSTANTIATE_WEIGHT_NORM_BACKWARD(half)
INSTANTIATE_WEIGHT_NORM_BACKWARD(bfloat)
