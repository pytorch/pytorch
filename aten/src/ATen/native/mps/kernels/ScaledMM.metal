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
  if (p.has_tensor_scales) {
    value = (value * args.scale_a[p.rowwise ? row : 0]) *
        args.scale_b[p.rowwise ? col : 0];
  }
  if (p.has_bias) {
    value += p.bias_bfloat16
        ? float(reinterpret_cast<device const bfloat*>(args.bias)[col])
        : float(reinterpret_cast<device const half*>(args.bias)[col]);
  }
  args.out[row * p.out_row_stride + col * p.out_col_stride] = T(value);
}

// Accumulates the threadgroup's output tile over all of K into c_tile;
// load_tiles(k) stages the A and B tiles of the K step starting at k.
template <typename S, typename LoadTiles>
inline void scaled_mm_accumulate(
    threadgroup S* a_tile,
    threadgroup S* b_tile,
    threadgroup float* c_tile,
    uint k_size,
    uint sg,
    LoadTiles load_tiles) {
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
  for (uint k = 0; k < k_size; k += tile) {
    load_tiles(k);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint kk = 0; kk < tile; kk += frag) {
      simdgroup_matrix<S, frag, frag> aa[frags], bb[frags];
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
}

template <typename T>
inline void scaled_mm_store_tile(
    threadgroup const float* c_tile,
    uint row_start,
    uint col_start,
    uint tid,
    thread const ScaledMMArgs<T>& args,
    constant ScaledMMParams<>& p) {
  constexpr auto tile = scaled_mm_tile;
  for (uint i = tid; i < tile * tile; i += scaled_mm_threads) {
    const uint row = row_start + i / tile;
    const uint col = col_start + i % tile;
    if (row < p.m && col < p.n) {
      scaled_mm_store(c_tile[i], row, col, args, p);
    }
  }
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
  threadgroup half a_tile[tile * tile];
  threadgroup half b_tile[tile * tile];
  threadgroup float c_tile[tile * tile];
  const ScaledMMArgs<T> args{a, b, out, scale_a, scale_b, bias, a_tile, b_tile};
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  scaled_mm_accumulate(a_tile, b_tile, c_tile, p.k, sg, [&](uint k) {
    scaled_mm_load_tiles(args, p, row_start, col_start, k, tid);
  });
  scaled_mm_store_tile(c_tile, row_start, col_start, tid, args, p);
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

inline float4 scaled_mm_decode_scaled(uint bytes) {
  // mask the sign bit
  const uint magnitude = bytes & 0x7f7f7f7f;
  const uint nan_mask = 0x80808080;
  const uint nan_flags = (magnitude + 0x01010101) & nan_mask;
  // payload to detect nans here, all nans will be 11111111
  // this is done this way because 0 1111 111 and 1 1111 111 are both Nans in
  // fp8
  const uint payload = magnitude | nan_flags;
  // Reusing E4M3FN exponent bits avoids per-value bias adjustment. FP16's
  // bias is 15 instead of 7, so the decoded value is x * 2^(7 - 15) = x / 256.
  const uint low = ((payload & 0x00ff00ff) << 7) | ((bytes & 0x00800080) << 8);
  const uint high = ((payload & 0xff00ff00) >> 1) | (bytes & 0x80008000);
  const float2 lo = float2(as_type<half2>(low));
  const float2 hi = float2(as_type<half2>(high));
  return float4(lo.x, hi.x, lo.y, hi.y);
}

// Block-scaled operand formats: decode element k of a K-contiguous row or
// column, and its block's scale. stage_t is the threadgroup staging type.
struct nvfp4_format {
  // e2m1 times an e4m3 scale has at most 5 significant bits, so half is exact
  using stage_t = half;
  static constexpr constant uint block = 16;
  static float element(device const uchar* data, uint k) {
    // Two e2m1 values per byte, the first in the low nibble. Shifting the 3
    // magnitude bits into half's exponent and mantissa and rebiasing by 2^14
    // decodes them without a table, as MLX does.
    const uint code = data[k / 2] >> (k % 2 * 4);
    const float magnitude =
        float(as_type<half>(ushort((code & 7) << 9))) * 16384.0f;
    return code & 8 ? -magnitude : magnitude;
  }
  static float scale(uchar s) {
    return c10::metal::detail::fp8e4m3fn_to_fp32(s);
  }
  // The decode kernels read 32 values (two blocks, 16 bytes) per lane step and
  // decode them without the 2^14 rebias, which unbiased_scale applies once per
  // block. Masks put one nibble per 16-bit lane of a half2: magnitude into the
  // exponent and mantissa, sign into bit 15. That permutes the values within
  // each group of 8, which a dot product with an operand decoded the same way
  // does not see.
  static constexpr constant uint chunk_bytes = 16;
  static void decode_chunk(device const uchar* data, thread float* v) {
    const auto raw = *reinterpret_cast<device const uint4*>(data);
#pragma unroll
    for (uint w = 0; w < 4; ++w) {
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        const uint n = (raw[w] >> (4 * s)) & 0x000f000f;
        const auto x = float2(
            as_type<half2>(((n & 0x00070007) << 9) | ((n & 0x00080008) << 12)));
        v[8 * w + 2 * s] = x.x;
        v[8 * w + 2 * s + 1] = x.y;
      }
    }
  }
  static float unbiased_scale(uchar s) {
    return scale(s) * 16384.0f;
  }
};

struct mxfp8_format {
  // e8m0 scales span 2^-127..2^127, beyond half's range
  using stage_t = float;
  static constexpr constant uint block = 32;
  static float element(device const uchar* data, uint k) {
    return c10::metal::detail::fp8e4m3fn_to_fp32(data[k]);
  }
  static float scale(uchar s) {
    return c10::metal::detail::fp8e8m0fnu_to_fp32(s);
  }
  // 32 values (one block, 32 bytes) per lane step, decoded at 1/256 of their
  // value by scaled_mm_decode_scaled; unbiased_scale makes up for it.
  static constexpr constant uint chunk_bytes = 32;
  static void decode_chunk(device const uchar* data, thread float* v) {
    const auto raw = reinterpret_cast<device const uint4*>(data);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
#pragma unroll
      for (uint w = 0; w < 4; ++w) {
        const auto x = scaled_mm_decode_scaled(raw[h][w]);
#pragma unroll
        for (uint i = 0; i < 4; ++i) {
          v[16 * h + 4 * w + i] = x[i];
        }
      }
    }
  }
  static float unbiased_scale(uchar s) {
    return scale(s) * scaled_mm_decode_scale;
  }
};

// Kernel buffers of the block-scaled recipes. A is row-major and B
// column-major (both K-contiguous); scale_a and scale_b hold one scale per
// block of K for each row of A and column of B. The optional tensor-wise scales
// (two-level NVFP4) are applied by the store.
template <typename T>
struct ScaledMMBlockwiseArgs {
  device const uchar* a;
  device const uchar* b;
  device const uchar* scale_a;
  device const uchar* scale_b;
  ScaledMMArgs<T> store;
};

// Stages the decoded A and B tiles of the K step starting at k
template <typename F, typename T, typename S>
inline void scaled_mm_blockwise_load_tiles(
    thread const ScaledMMBlockwiseArgs<T>& args,
    constant ScaledMMParams<>& p,
    uint row_start,
    uint col_start,
    uint k,
    uint tid,
    threadgroup S* a_tile,
    threadgroup S* b_tile) {
  constexpr auto tile = scaled_mm_tile;
  const uint k_blocks = (p.k + F::block - 1) / F::block;
  const auto decode = [&](device const uchar* data,
                          device const uchar* scales,
                          int64_t stride,
                          uint outer,
                          uint outer_size,
                          uint k) {
    return outer < outer_size && k < p.k
        ? S(F::element(data + outer * stride, k) *
            F::scale(scales[outer * k_blocks + k / F::block]))
        : S(0);
  };
  // Consecutive threads walk K, the contiguous dimension of both operands
  for (uint i = tid; i < tile * tile; i += scaled_mm_threads) {
    const uint outer = i / tile;
    const uint kk = i % tile;
    a_tile[outer * tile + kk] = decode(
        args.a, args.scale_a, p.a_row_stride, row_start + outer, p.m, k + kk);
    b_tile[kk * tile + outer] = decode(
        args.b, args.scale_b, p.b_col_stride, col_start + outer, p.n, k + kk);
  }
}

#define SCALED_MM_BLOCKWISE_BUFFERS                                           \
  device const uchar *a [[buffer(0)]], device const uchar *b [[buffer(1)]],   \
      device T *out [[buffer(2)]], device const uchar *scale_a [[buffer(3)]], \
      device const uchar *scale_b [[buffer(4)]],                              \
      device const float *tensor_scale_a [[buffer(5)]],                       \
      device const float *tensor_scale_b [[buffer(6)]],                       \
      device const uchar *bias [[buffer(7)]],                                 \
      constant ScaledMMParams<>&p [[buffer(8)]]

#define SCALED_MM_BLOCKWISE_ARGS                                          \
  const ScaledMMBlockwiseArgs<T> args {                                   \
    a, b, scale_a, scale_b, {                                             \
      nullptr, nullptr, out, tensor_scale_a, tensor_scale_b, bias, {}, {} \
    }                                                                     \
  }

template <typename T, typename F>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_blockwise(
    SCALED_MM_BLOCKWISE_BUFFERS,
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr auto tile = scaled_mm_tile;
  using S = typename F::stage_t;
  threadgroup S a_tile[tile * tile];
  threadgroup S b_tile[tile * tile];
  threadgroup float c_tile[tile * tile];
  SCALED_MM_BLOCKWISE_ARGS;
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  scaled_mm_accumulate(a_tile, b_tile, c_tile, p.k, sg, [&](uint k) {
    scaled_mm_blockwise_load_tiles<F>(
        args, p, row_start, col_start, k, tid, a_tile, b_tile);
  });
  scaled_mm_store_tile(c_tile, row_start, col_start, tid, args.store, p);
}

#define REGISTER_SCALED_MM_BLOCKWISE(NAME, FORMAT, T)           \
  template [[host_name("scaled_mm_" #NAME "_" #T)]] kernel void \
  scaled_mm_blockwise<T, FORMAT>(                               \
      device const uchar*,                                      \
      device const uchar*,                                      \
      device T*,                                                \
      device const uchar*,                                      \
      device const uchar*,                                      \
      device const float*,                                      \
      device const float*,                                      \
      device const uchar*,                                      \
      constant ScaledMMParams<>&,                               \
      uint2,                                                    \
      uint,                                                     \
      uint);

#define REGISTER_SCALED_MM_BLOCKWISE_WITH_OUT_T(T)      \
  REGISTER_SCALED_MM_BLOCKWISE(nvfp4, nvfp4_format, T); \
  REGISTER_SCALED_MM_BLOCKWISE(mxfp8, mxfp8_format, T)

REGISTER_SCALED_MM_BLOCKWISE_WITH_OUT_T(float);
REGISTER_SCALED_MM_BLOCKWISE_WITH_OUT_T(half);
REGISTER_SCALED_MM_BLOCKWISE_WITH_OUT_T(bfloat);

// Every SIMD group computes one output column; its lanes take turns
// streaming scaled_mm_few_rows_lane_chunk elements of K.
template <typename T, uint32_t rows>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_few_rows(
    device const float8_e4m3fn* a [[buffer(0)]],
    device const float8_e4m3fn* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    device const uchar* bias [[buffer(5)]],
    constant ScaledMMParams<>& p [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using load_t = uint4;
  constexpr uint words = sizeof(load_t) / sizeof(uint);
  static_assert(
      sizeof(load_t) == scaled_mm_few_rows_load_bytes,
      "scaled_mm_few_rows_load_bytes must match the packed load type");
  static_assert(
      scaled_mm_alignment % sizeof(load_t) == 0,
      "loads must stay aligned and within K");
  static_assert(
      scaled_mm_alignment % scaled_mm_simdgroups == 0,
      "N must be a whole number of threadgroups");
  constexpr auto chunk = scaled_mm_few_rows_lane_chunk;
  const ScaledMMArgs<T> args{a, b, out, scale_a, scale_b, bias, {}, {}};
  const uint col = group * scaled_mm_simdgroups + sg;
  const auto b_col = b + col * p.b_col_stride;
  float4 accum[rows] = {};
  for (uint base = lane * chunk; base < p.k;
       base += c10::metal::simdgroup_size * chunk) {
    for (uint k = base; k < min(base + chunk, p.k); k += sizeof(load_t)) {
      const auto b_raw = *reinterpret_cast<device const load_t*>(b_col + k);
      float4 bv[words];
#pragma unroll
      for (uint i = 0; i < words; ++i) {
        bv[i] = scaled_mm_decode_scaled(b_raw[i]);
      }
#pragma unroll
      for (uint r = 0; r < rows; ++r) {
        const auto a_raw =
            *reinterpret_cast<device const load_t*>(a + r * p.a_row_stride + k);
#pragma unroll
        for (uint i = 0; i < words; ++i) {
          accum[r] = fma(scaled_mm_decode_scaled(a_raw[i]), bv[i], accum[r]);
        }
      }
    }
  }
#pragma unroll
  for (uint r = 0; r < rows; ++r) {
    const auto value = simd_sum(dot(accum[r], float4(1))) *
        (scaled_mm_decode_scale * scaled_mm_decode_scale);
    if (lane == 0) {
      scaled_mm_store(value, r, col, args, p);
    }
  }
}

#define REGISTER_SCALED_MM_FEW_ROWS(T, ROWS)                             \
  template [[host_name("scaled_mm_few_rows_" #ROWS "_" #T)]] kernel void \
  scaled_mm_few_rows<T, ROWS>(                                           \
      device const float8_e4m3fn*,                                       \
      device const float8_e4m3fn*,                                       \
      device T*,                                                         \
      device const float*,                                               \
      device const float*,                                               \
      device const uchar*,                                               \
      constant ScaledMMParams<>&,                                        \
      uint,                                                              \
      uint,                                                              \
      uint);

#define REGISTER_SCALED_MM_FEW_ROWS_WITH_OUT_T(T) \
  REGISTER_SCALED_MM_FEW_ROWS(T, 1);              \
  REGISTER_SCALED_MM_FEW_ROWS(T, 2);              \
  REGISTER_SCALED_MM_FEW_ROWS(T, 3);              \
  REGISTER_SCALED_MM_FEW_ROWS(T, 4);

static_assert(
    scaled_mm_few_rows_max == 4,
    "register scaled_mm_few_rows for every row count up to the maximum");

REGISTER_SCALED_MM_FEW_ROWS_WITH_OUT_T(float);
REGISTER_SCALED_MM_FEW_ROWS_WITH_OUT_T(half);
REGISTER_SCALED_MM_FEW_ROWS_WITH_OUT_T(bfloat);
REGISTER_SCALED_MM_FEW_ROWS_WITH_OUT_T(float8_e4m3fn);

// scaled_mm_few_rows for the block-scaled recipes. Every lane step covers
// scaled_mm_blockwise_chunk values of K, decoding B's chunk once for all rows
// of A and applying the scales once per block.
template <typename T, typename F, uint32_t rows>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_blockwise_few_rows(
    SCALED_MM_BLOCKWISE_BUFFERS,
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr auto chunk = scaled_mm_blockwise_chunk;
  constexpr uint blocks = chunk / F::block;
  static_assert(chunk % F::block == 0, "a chunk must hold whole blocks");
  SCALED_MM_BLOCKWISE_ARGS;
  const uint col = group * scaled_mm_simdgroups + sg;
  if (col >= p.n) {
    return;
  }
  const uint k_blocks = p.k / F::block;
  const auto b_col = b + col * p.b_col_stride;
  float accum[rows] = {};
  for (uint c = lane; c < p.k / chunk; c += c10::metal::simdgroup_size) {
    float bv[chunk];
    F::decode_chunk(b_col + c * F::chunk_bytes, bv);
    float b_scale[blocks];
#pragma unroll
    for (uint j = 0; j < blocks; ++j) {
      b_scale[j] = F::unbiased_scale(scale_b[col * k_blocks + c * blocks + j]);
    }
#pragma unroll
    for (uint r = 0; r < rows; ++r) {
      float av[chunk];
      F::decode_chunk(a + r * p.a_row_stride + c * F::chunk_bytes, av);
#pragma unroll
      for (uint j = 0; j < blocks; ++j) {
        float dot = 0;
#pragma unroll
        for (uint i = j * F::block; i < (j + 1) * F::block; ++i) {
          dot = fma(av[i], bv[i], dot);
        }
        const auto a_scale =
            F::unbiased_scale(scale_a[r * k_blocks + c * blocks + j]);
        accum[r] = fma(dot, a_scale * b_scale[j], accum[r]);
      }
    }
  }
#pragma unroll
  for (uint r = 0; r < rows; ++r) {
    const auto value = simd_sum(accum[r]);
    if (lane == 0) {
      scaled_mm_store(value, r, col, args.store, p);
    }
  }
}

#define REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS(NAME, FORMAT, T, ROWS)          \
  template                                                                    \
      [[host_name("scaled_mm_few_rows_" #NAME "_" #ROWS "_" #T)]] kernel void \
      scaled_mm_blockwise_few_rows<T, FORMAT, ROWS>(                          \
          device const uchar*,                                                \
          device const uchar*,                                                \
          device T*,                                                          \
          device const uchar*,                                                \
          device const uchar*,                                                \
          device const float*,                                                \
          device const float*,                                                \
          device const uchar*,                                                \
          constant ScaledMMParams<>&,                                         \
          uint,                                                               \
          uint,                                                               \
          uint);

#define REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_FORMAT(NAME, FORMAT, T) \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS(NAME, FORMAT, T, 1);          \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS(NAME, FORMAT, T, 2);          \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS(NAME, FORMAT, T, 3);          \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS(NAME, FORMAT, T, 4)

#define REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_WITH_OUT_T(T)             \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_FORMAT(nvfp4, nvfp4_format, T); \
  REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_FORMAT(mxfp8, mxfp8_format, T)

REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_WITH_OUT_T(float);
REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_WITH_OUT_T(half);
REGISTER_SCALED_MM_BLOCKWISE_FEW_ROWS_WITH_OUT_T(bfloat);

#if C10_METAL_HAS_MPP

template <typename T, typename Accum>
inline void scaled_mm_store_accum(
    thread const Accum& accum,
    uint row_start,
    uint col_start,
    thread const ScaledMMArgs<T>& args,
    constant ScaledMMParams<>& p) {
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
  scaled_mm_store_accum(accum, row_start, col_start, args, p);
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

// scaled_mm_blockwise with the MMA done by MPP on the decoded tiles
template <typename T, typename F>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_blockwise_mpp(
    SCALED_MM_BLOCKWISE_BUFFERS,
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr auto tile = scaled_mm_tile;
  using S = typename F::stage_t;
  threadgroup S a_tile[tile * tile];
  threadgroup S b_tile[tile * tile];
  SCALED_MM_BLOCKWISE_ARGS;
  using tile_t = tensor<threadgroup S, extents<int, tile, tile>, tensor_inline>;
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
    scaled_mm_blockwise_load_tiles<F>(
        args, p, row_start, col_start, k, tid, a_tile, b_tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(aa, bb, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  scaled_mm_store_accum(accum, row_start, col_start, args.store, p);
}

#define REGISTER_SCALED_MM_BLOCKWISE_MPP(NAME, FORMAT, T)           \
  template [[host_name("scaled_mm_mpp_" #NAME "_" #T)]] kernel void \
  scaled_mm_blockwise_mpp<T, FORMAT>(                               \
      device const uchar*,                                          \
      device const uchar*,                                          \
      device T*,                                                    \
      device const uchar*,                                          \
      device const uchar*,                                          \
      device const float*,                                          \
      device const float*,                                          \
      device const uchar*,                                          \
      constant ScaledMMParams<>&,                                   \
      uint2,                                                        \
      uint);

#define REGISTER_SCALED_MM_BLOCKWISE_MPP_WITH_OUT_T(T)      \
  REGISTER_SCALED_MM_BLOCKWISE_MPP(nvfp4, nvfp4_format, T); \
  REGISTER_SCALED_MM_BLOCKWISE_MPP(mxfp8, mxfp8_format, T)

REGISTER_SCALED_MM_BLOCKWISE_MPP_WITH_OUT_T(float);
REGISTER_SCALED_MM_BLOCKWISE_MPP_WITH_OUT_T(half);
REGISTER_SCALED_MM_BLOCKWISE_MPP_WITH_OUT_T(bfloat);

#if __METAL_VERSION__ >= 410

// Metal 4.1 MPP multiplies fp8 operands directly, without staging them as half
// in threadgroup memory. A must be row-major and B column-major (both
// K-contiguous), which is the layout cuBLASLt requires too.
template <typename T>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_native(
    device float8_e4m3fn* a [[buffer(0)]],
    device float8_e4m3fn* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    device const uchar* bias [[buffer(5)]],
    constant ScaledMMParams<>& p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr auto tile = scaled_mm_native_tile;
  // MPP rejects const-qualified element types, hence non-const a and b
  using fp8_tensor =
      tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>;
  const auto operand = [&](device float8_e4m3fn* data, uint rows, int64_t ld) {
    return fp8_tensor(
        reinterpret_cast<device uchar*>(data),
        dextents<int32_t, 2>(p.k, rows),
        array<int32_t, 2>{1, int32_t(ld)});
  };
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  auto a_tile = operand(a, p.m, p.a_row_stride).slice(0, row_start);
  auto b_tile = operand(b, p.n, p.b_col_stride).slice(0, col_start);
  // A single run over all of K, so the default multiply mode needs no zeroed
  // accumulator
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      tile, tile, static_cast<int>(dynamic_extent), false, true);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<scaled_mm_simdgroups>>
      op;
  auto accum = op.template get_destination_cooperative_tensor<
      decltype(a_tile),
      decltype(b_tile),
      float>();
  op.run(a_tile, b_tile, accum);
  const ScaledMMArgs<T> args{a, b, out, scale_a, scale_b, bias, {}, {}};
  scaled_mm_store_accum(accum, row_start, col_start, args, p);
}

#define REGISTER_SCALED_MM_NATIVE_WITH_OUT_T(T)              \
  template [[host_name("scaled_mm_native_" #T)]] kernel void \
  scaled_mm_native<T>(                                       \
      device float8_e4m3fn*,                                 \
      device float8_e4m3fn*,                                 \
      device T*,                                             \
      device const float*,                                   \
      device const float*,                                   \
      device const uchar*,                                   \
      constant ScaledMMParams<>&,                            \
      uint2);

REGISTER_SCALED_MM_NATIVE_WITH_OUT_T(float);
REGISTER_SCALED_MM_NATIVE_WITH_OUT_T(half);
REGISTER_SCALED_MM_NATIVE_WITH_OUT_T(bfloat);
REGISTER_SCALED_MM_NATIVE_WITH_OUT_T(float8_e4m3fn);

// MXFP8 on Metal 4.1: MPP applies the ue8m0 block scales itself, taking each
// operand's scales as a [rows, K / 32] plane alongside its fp8 data.
template <typename T>
[[max_total_threads_per_threadgroup(scaled_mm_threads)]]
kernel void scaled_mm_native_mxfp8(
    device uchar* a [[buffer(0)]],
    device uchar* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device uchar* scale_a [[buffer(3)]],
    device uchar* scale_b [[buffer(4)]],
    device const float* tensor_scale_a [[buffer(5)]],
    device const float* tensor_scale_b [[buffer(6)]],
    device const uchar* bias [[buffer(7)]],
    constant ScaledMMParams<>& p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr auto tile = scaled_mm_native_tile;
  using scales_t = tensor_blockwise<
      tensor_plane_scales,
      device metal_fp8_ue8m0_format,
      mxfp8_format::block,
      1>;
  using fp8_tensor = tensor<
      device metal_fp8_e4m3_format,
      dextents<int32_t, 2>,
      tensor_inline,
      scales_t>;
  const auto operand =
      [&](device uchar* data, device uchar* scales, uint rows, int64_t ld) {
        return fp8_tensor(
            data,
            dextents<int32_t, 2>(p.k, rows),
            array<int32_t, 2>{1, int32_t(ld)},
            scales_t(scales));
      };
  const uint row_start = group.y * tile;
  const uint col_start = group.x * tile;
  auto a_tile = operand(a, scale_a, p.m, p.a_row_stride).slice(0, row_start);
  auto b_tile = operand(b, scale_b, p.n, p.b_col_stride).slice(0, col_start);
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      tile, tile, static_cast<int>(dynamic_extent), false, true);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<scaled_mm_simdgroups>>
      op;
  auto accum = op.template get_destination_cooperative_tensor<
      decltype(a_tile),
      decltype(b_tile),
      float>();
  op.run(a_tile, b_tile, accum);
  const ScaledMMArgs<T> args{
      nullptr, nullptr, out, tensor_scale_a, tensor_scale_b, bias, {}, {}};
  scaled_mm_store_accum(accum, row_start, col_start, args, p);
}

#define REGISTER_SCALED_MM_NATIVE_MXFP8_WITH_OUT_T(T)              \
  template [[host_name("scaled_mm_native_mxfp8_" #T)]] kernel void \
  scaled_mm_native_mxfp8<T>(                                       \
      device uchar*,                                               \
      device uchar*,                                               \
      device T*,                                                   \
      device uchar*,                                               \
      device uchar*,                                               \
      device const float*,                                         \
      device const float*,                                         \
      device const uchar*,                                         \
      constant ScaledMMParams<>&,                                  \
      uint2);

REGISTER_SCALED_MM_NATIVE_MXFP8_WITH_OUT_T(float);
REGISTER_SCALED_MM_NATIVE_MXFP8_WITH_OUT_T(half);
REGISTER_SCALED_MM_NATIVE_MXFP8_WITH_OUT_T(bfloat);

#endif // __METAL_VERSION__ >= 410
#endif // C10_METAL_HAS_MPP
