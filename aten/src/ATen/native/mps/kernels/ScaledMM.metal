#include <ATen/native/mps/kernels/ScaledMM.h>
#include <c10/metal/float8.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;
using c10::metal::float8_e4m3fn;

// Kernel buffers plus the threadgroup staging tiles shared by both kernels.
template <typename T>
struct ScaledMMArgs {
  device const float8_e4m3fn* a;
  device const float8_e4m3fn* b;
  device T* out;
  device const float* scale_a;
  device const float* scale_b;
  device const uchar* bias;
  threadgroup half* a_tile;
  threadgroup half* b_tile;
};

template <typename T>
inline void scaled_mm_load_tiles(
    thread const ScaledMMArgs<T>& args,
    constant ScaledMMParams<>& p,
    uint row_start,
    uint col_start,
    uint k,
    uint tid) {
  constexpr auto tile = scaled_mm_tile;
  for (uint i = tid; i < tile * tile; i += scaled_mm_threads) {
    const uint row = i / tile;
    const uint col = i % tile;
    const auto a_idx =
        (row_start + row) * p.a_row_stride + (k + col) * p.a_col_stride;
    const auto b_idx =
        (k + row) * p.b_row_stride + (col_start + col) * p.b_col_stride;
    const auto av = row_start + row < p.m && k + col < p.k
        ? args.a[a_idx]
        : float8_e4m3fn(0.0f);
    const auto bv = k + row < p.k && col_start + col < p.n
        ? args.b[b_idx]
        : float8_e4m3fn(0.0f);
    args.a_tile[i] = half(float(av));
    args.b_tile[i] = half(float(bv));
  }
}

template <typename T>
inline void scaled_mm_store(
    float value,
    uint row,
    uint col,
    thread const ScaledMMArgs<T>& args,
    constant ScaledMMParams<>& p) {
  value = (value * args.scale_a[p.rowwise ? row : 0]) *
      args.scale_b[p.rowwise ? col : 0];
  if (p.has_bias) {
    value += p.bias_bfloat16
        ? float(reinterpret_cast<device const bfloat*>(args.bias)[col])
        : float(reinterpret_cast<device const half*>(args.bias)[col]);
  }
  args.out[row * p.out_row_stride + col * p.out_col_stride] = T(value);
}

template <typename T>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm(
    device const float8_e4m3fn* a [[buffer(0)]],
    device const float8_e4m3fn* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    device const uchar* bias [[buffer(5)]],
    constant ScaledMMParams<>& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr auto tile = scaled_mm_tile;
  // SIMD groups form a square grid over the tile; each owns a block built from
  // 8x8 simdgroup matrix fragments.
  constexpr uint frag = 8;
  constexpr uint sg_grid = 2;
  constexpr uint block = tile / sg_grid;
  constexpr uint frags = block / frag;
  static_assert(
      sg_grid * sg_grid == scaled_mm_simdgroups,
      "scaled_mm needs a square grid of SIMD groups");
  static_assert(
      frags * frag * sg_grid == tile,
      "scaled_mm_tile must split into 8x8 fragments per SIMD group");
  threadgroup half a_tile[tile * tile];
  threadgroup half b_tile[tile * tile];
  threadgroup float c_tile[tile * tile];
  const ScaledMMArgs<T> args{a, b, out, scale_a, scale_b, bias, a_tile, b_tile};
  const uint sg_row = sg / sg_grid * block;
  const uint sg_col = sg % sg_grid * block;
  simdgroup_float8x8 accum[frags][frags];
#pragma unroll
  for (uint i = 0; i < frags; ++i) {
#pragma unroll
    for (uint j = 0; j < frags; ++j) {
      accum[i][j] = simdgroup_float8x8(0.0f);
    }
  }
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  for (uint k = 0; k < p.k; k += tile) {
    scaled_mm_load_tiles(args, p, row_start, col_start, k, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint kk = 0; kk < tile; kk += frag) {
      simdgroup_half8x8 aa[frags], bb[frags];
#pragma unroll
      for (uint i = 0; i < frags; ++i) {
        simdgroup_load(aa[i], a_tile + (sg_row + i * frag) * tile + kk, tile);
        simdgroup_load(bb[i], b_tile + kk * tile + sg_col + i * frag, tile);
      }
#pragma unroll
      for (uint i = 0; i < frags; ++i) {
#pragma unroll
        for (uint j = 0; j < frags; ++j) {
          simdgroup_multiply_accumulate(accum[i][j], aa[i], bb[j], accum[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint i = 0; i < frags; ++i) {
#pragma unroll
    for (uint j = 0; j < frags; ++j) {
      simdgroup_store(
          accum[i][j],
          c_tile + (sg_row + i * frag) * tile + sg_col + j * frag,
          tile);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < tile * tile; i += scaled_mm_threads) {
    const uint row = row_start + i / tile;
    const uint col = col_start + i % tile;
    if (row < p.m && col < p.n) {
      scaled_mm_store(c_tile[i], row, col, args, p);
    }
  }
}

#define REGISTER_SCALED_MM_WITH_OUT_T(T)                            \
  template [[host_name("scaled_mm_" #T)]] kernel void scaled_mm<T>( \
      device const float8_e4m3fn*,                                  \
      device const float8_e4m3fn*,                                  \
      device T*,                                                    \
      device const float*,                                          \
      device const float*,                                          \
      device const uchar*,                                          \
      constant ScaledMMParams<>&,                                   \
      uint2,                                                        \
      uint,                                                         \
      uint);

REGISTER_SCALED_MM_WITH_OUT_T(float);
REGISTER_SCALED_MM_WITH_OUT_T(half);
REGISTER_SCALED_MM_WITH_OUT_T(bfloat);
REGISTER_SCALED_MM_WITH_OUT_T(float8_e4m3fn);

#if C10_METAL_HAS_MPP

template <typename T>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_mpp(
    device const float8_e4m3fn* a [[buffer(0)]],
    device const float8_e4m3fn* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    device const uchar* bias [[buffer(5)]],
    constant ScaledMMParams<>& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr auto tile = scaled_mm_tile;
  threadgroup half a_tile[tile * tile];
  threadgroup half b_tile[tile * tile];
  const ScaledMMArgs<T> args{a, b, out, scale_a, scale_b, bias, a_tile, b_tile};
  using tile_t =
      tensor<threadgroup half, extents<int, tile, tile>, tensor_inline>;
  tile_t aa(a_tile, extents<int, tile, tile>());
  tile_t bb(b_tile, extents<int, tile, tile>());
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      tile,
      tile,
      tile,
      false,
      false,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<scaled_mm_simdgroups>>
      op;
  auto accum =
      op.template get_destination_cooperative_tensor<tile_t, tile_t, float>();
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      accum[i] = 0.0f;
    }
  }
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  for (uint k = 0; k < p.k; k += tile) {
    scaled_mm_load_tiles(args, p, row_start, col_start, k, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(aa, bb, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      const auto coord = accum.get_multidimensional_index(i);
      const uint row = row_start + coord[1];
      const uint col = col_start + coord[0];
      if (row < p.m && col < p.n) {
        scaled_mm_store(accum[i], row, col, args, p);
      }
    }
  }
}

#define REGISTER_SCALED_MM_MPP_WITH_OUT_T(T)                                \
  template [[host_name("scaled_mm_mpp_" #T)]] kernel void scaled_mm_mpp<T>( \
      device const float8_e4m3fn*,                                          \
      device const float8_e4m3fn*,                                          \
      device T*,                                                            \
      device const float*,                                                  \
      device const float*,                                                  \
      device const uchar*,                                                  \
      constant ScaledMMParams<>&,                                           \
      uint2,                                                                \
      uint);

REGISTER_SCALED_MM_MPP_WITH_OUT_T(float);
REGISTER_SCALED_MM_MPP_WITH_OUT_T(half);
REGISTER_SCALED_MM_MPP_WITH_OUT_T(bfloat);
REGISTER_SCALED_MM_MPP_WITH_OUT_T(float8_e4m3fn);

#endif // C10_METAL_HAS_MPP
