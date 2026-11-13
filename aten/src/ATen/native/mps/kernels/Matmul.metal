#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/kernels/Matmul.h>
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
#include <ATen/native/mps/kernels/Gemv.h>

constant uint TILE_DIM = 16;

template <typename T, int BM, int BN, typename idx_t>
inline MatmulTile<T> matmul_tile(
    device const T* a,
    device const T* b,
    device T* out,
    device const T* bias,
    constant MatmulParams<idx_t>& p,
    uint3 tgid) {
  const idx_t row = idx_t(tgid.y) * BM, col = idx_t(tgid.x) * BN, z = tgid.z;
  return {
      a + z * p.a_batch + row * p.a_stride_m,
      b + z * p.b_batch + col * p.b_stride_n,
      out + z * p.out_batch + row * p.out_stride_m + col * p.out_stride_n,
      uint32_t(min(idx_t(BM), idx_t(p.m) - row)),
      uint32_t(min(idx_t(BN), idx_t(p.n) - col)),
      p.k,
      bias + z * p.bias_batch + row * p.bias_stride_m + col * p.bias_stride_n};
}

template <typename T, int BM, typename idx_t>
[[max_total_threads_per_threadgroup(
    matmul_simdgroups(BM, kMatmulTileN) * simdgroup_size)]]
kernel void matmul_simd(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    constant MatmulParams<idx_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  threadgroup T a_tile[BM * (kMatmulTileK + kMatmulTilePad)];
  threadgroup T b_tile[kMatmulTileK * (kMatmulTileN + kMatmulTilePad)];
  auto tile = matmul_tile<T, BM, kMatmulTileN>(a, b, out, bias, params, tgid);
  matmul_tile_gemm<T, BM>(tile, params, a_tile, b_tile, simd_group, simd_lane);
}

#define INSTANTIATE_MATMUL_SIMD(T, BM, IDX_T, SUFFIX)                         \
  template [[host_name("matmul_simd_" #T "_bm" #BM "_" #SUFFIX)]] kernel void \
  matmul_simd<T, BM, IDX_T>(                                                  \
      device const T*,                                                        \
      device const T*,                                                        \
      device T*,                                                              \
      device const T*,                                                        \
      constant MatmulParams<IDX_T>&,                                          \
      uint3,                                                                  \
      uint,                                                                   \
      uint)

#define INSTANTIATE_MATMUL_SIMD_BM(T, BM)        \
  INSTANTIATE_MATMUL_SIMD(T, BM, uint32_t, u32); \
  INSTANTIATE_MATMUL_SIMD(T, BM, uint64_t, u64)

#define INSTANTIATE_MATMUL_SIMD_DTYPE(T) \
  INSTANTIATE_MATMUL_SIMD_BM(T, 16);     \
  INSTANTIATE_MATMUL_SIMD_BM(T, 32);     \
  INSTANTIATE_MATMUL_SIMD_BM(T, 64)

INSTANTIATE_MATMUL_SIMD_DTYPE(float);
INSTANTIATE_MATMUL_SIMD_DTYPE(half);
INSTANTIATE_MATMUL_SIMD_DTYPE(bfloat);

#if C10_METAL_HAS_MPP
template <typename T, int BM, bool TA, bool TB>
[[max_total_threads_per_threadgroup(kMatmulBlockSimdgroups * simdgroup_size)]]
kernel void matmul_mpp(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    device const T* bias [[buffer(3)]],
    constant MatmulParams<uint64_t>& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr int BN = kMatmulBlock * kMatmulBlockSimdgroups;
  auto tile = matmul_tile<T, BM, BN>(a, b, out, bias, params, tgid);
  matmul_mpp_block_gemm<T, BM, TA, TB>(tile, params, simd_group);
}

#define INSTANTIATE_MATMUL_MPP(T, BM, TA, TB, LAYOUT)                        \
  template [[host_name("matmul_mpp_" #LAYOUT "_" #T "_bm" #BM)]] kernel void \
  matmul_mpp<T, BM, TA, TB>(                                                 \
      device const T*,                                                       \
      device const T*,                                                       \
      device T*,                                                             \
      device const T*,                                                       \
      constant MatmulParams<uint64_t>&,                                      \
      uint3,                                                                 \
      uint)

#define INSTANTIATE_MATMUL_MPP_BM(T, BM)           \
  INSTANTIATE_MATMUL_MPP(T, BM, false, false, nn); \
  INSTANTIATE_MATMUL_MPP(T, BM, false, true, nt);  \
  INSTANTIATE_MATMUL_MPP(T, BM, true, false, tn);  \
  INSTANTIATE_MATMUL_MPP(T, BM, true, true, tt)

// The tile rows metal_gemm picks (Matmul.mm): kMatmulBlock and half of it.
#define INSTANTIATE_MATMUL_MPP_DTYPE(T) \
  INSTANTIATE_MATMUL_MPP_BM(T, 16);     \
  INSTANTIATE_MATMUL_MPP_BM(T, 32)

INSTANTIATE_MATMUL_MPP_DTYPE(float);
INSTANTIATE_MATMUL_MPP_DTYPE(half);
INSTANTIATE_MATMUL_MPP_DTYPE(bfloat);

template <typename T, int RN, bool Tail = false>
[[max_total_threads_per_threadgroup(kMatmulBlockSimdgroups * simdgroup_size)]]
kernel void matmul_mpp_complex(
    device const vec<T, 2>* a [[buffer(0)]],
    device const vec<T, 2>* b [[buffer(1)]],
    device vec<T, 2>* out [[buffer(2)]],
    device const vec<T, 2>* bias [[buffer(3)]],
    constant MatmulComplexParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  using T2 = vec<T, 2>;
  using S = float;
  constexpr int BM = kMatmulComplexRows, BN = RN * kMatmulBlockSimdgroups;
  constexpr int KC = kMatmulComplexStep * kMatmulBlock / RN;
  constexpr int NA = BM * KC / simdgroup_size, NB = KC * RN / simdgroup_size;
  constexpr int NC = BM * RN / simdgroup_size;
  constexpr bool k_pass = RN == kMatmulBlock && is_same_v<T, float>;
  constant MatmulParams<uint64_t>& p = params.gemm;
  auto tile = matmul_tile<T2, BM, BN>(a, b, out, bias, p, tgid);
  const int col = simd_group * RN;
  const uint32_t k_start = k_pass ? params.k_start : 0;
  const uint32_t k = (k_pass ? params.k_end : p.k) - Tail;
  if (col >= int(tile.cols)) {
    return;
  }
  const int rows = tile.rows, cols = min(RN, int(tile.cols) - col);
  constexpr auto desc = matmul2d_descriptor(
      BM,
      RN,
      KC,
      false,
      false,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto a_re = op.template get_left_input_cooperative_tensor<S, S, float>();
  auto a_im = op.template get_left_input_cooperative_tensor<S, S, float>();
  auto b_re = op.template get_right_input_cooperative_tensor<S, S, float>();
  auto b_im = op.template get_right_input_cooperative_tensor<S, S, float>();
  using A = decltype(a_re);
  using B = decltype(b_re);
  auto acc_re = op.template get_destination_cooperative_tensor<A, B, float>();
  auto acc_im = op.template get_destination_cooperative_tensor<A, B, float>();
  // Where each element of this thread sits in a K step. Rows and columns past
  // the tile read its last ones: their products are not stored.
  uint32_t a_at[NA], b_at[NB];
  uint16_t a_k[NA], b_k[NB];
  for (ushort i = 0; i < NA; ++i) {
    const auto at = a_re.get_multidimensional_index(i);
    a_k[i] = at[0];
    a_at[i] = min(int(at[1]), rows - 1) * uint32_t(p.a_stride_m) +
        at[0] * uint32_t(p.a_stride_k);
  }
  for (ushort i = 0; i < NB; ++i) {
    const auto at = b_re.get_multidimensional_index(i);
    b_k[i] = at[1];
    b_at[i] = at[1] * uint32_t(p.b_stride_k) +
        min(int(at[0]), cols - 1) * uint32_t(p.b_stride_n);
  }
  for (ushort i = 0; i < NC; ++i) {
    const auto at = acc_re.get_multidimensional_index(i);
    const float2 acc = k_start && at[1] < rows && at[0] < cols
        ? float2(
              tile.out[at[1] * p.out_stride_m + (col + at[0]) * p.out_stride_n])
        : float2(0);
    acc_re[i] = acc.x;
    acc_im[i] = acc.y;
  }
  device const T2* b_tile = tile.b + col * p.b_stride_n;
  const T a_sign = T(params.a_sign), b_sign = T(params.b_sign);
  for (uint32_t kk = k_start; kk < k; kk += KC) {
    device const T2* a_step = tile.a + kk * p.a_stride_k;
    device const T2* b_step = b_tile + kk * p.b_stride_k;
    if (kk + KC <= k) {
      for (ushort i = 0; i < NA; ++i) {
        const T2 v = a_step[a_at[i]];
        a_re[i] = v.x;
        a_im[i] =
            is_same_v<T, S> ? a_sign * v.y : (params.a_sign < 0 ? -v.y : v.y);
      }
      for (ushort i = 0; i < NB; ++i) {
        const T2 v = b_step[b_at[i]];
        b_re[i] = v.x;
        b_im[i] =
            is_same_v<T, S> ? b_sign * v.y : (params.b_sign < 0 ? -v.y : v.y);
      }
    } else {
      for (ushort i = 0; i < NA; ++i) {
        const T2 v = kk + a_k[i] < k ? a_step[a_at[i]] : T2(0);
        a_re[i] = v.x;
        a_im[i] =
            is_same_v<T, S> ? a_sign * v.y : (params.a_sign < 0 ? -v.y : v.y);
      }
      for (ushort i = 0; i < NB; ++i) {
        const T2 v = kk + b_k[i] < k ? b_step[b_at[i]] : T2(0);
        b_re[i] = v.x;
        b_im[i] =
            is_same_v<T, S> ? b_sign * v.y : (params.b_sign < 0 ? -v.y : v.y);
      }
    }
    op.run(a_re, b_re, acc_re);
    op.run(a_re, b_im, acc_im);
    op.run(a_im, b_re, acc_im);
    for (ushort i = 0; i < NA; ++i) {
      a_im[i] = -a_im[i];
    }
    op.run(a_im, b_im, acc_re);
  }
  const float2 alpha = float2(p.alpha, params.alpha_im);
  const float2 beta = float2(p.beta, params.beta_im);
  const bool add = beta.x != 0 || beta.y != 0;
  for (ushort i = 0; i < NC; ++i) {
    const auto at = acc_re.get_multidimensional_index(i);
    if (at[1] >= rows || at[0] >= cols) {
      continue;
    }
    if (Tail) {
      const float2 x = float2(tile.a[at[1] * p.a_stride_m + k * p.a_stride_k]);
      const float2 y = float2(b_tile[k * p.b_stride_k + at[0] * p.b_stride_n]);
      const float xi = params.a_sign * x.y, yi = params.b_sign * y.y;
      acc_re[i] = fma(x.x, y.x, acc_re[i]);
      acc_im[i] = fma(x.x, yi, acc_im[i]);
      acc_im[i] = fma(xi, y.x, acc_im[i]);
      acc_re[i] = fma(-xi, yi, acc_re[i]);
    }
    const float2 acc = float2(acc_re[i], acc_im[i]);
    float2 v = acc;
    if (!k_pass || params.k_end == p.k) {
      v = alpha.x * acc + alpha.y * float2(-acc.y, acc.x);
      if (add) {
        float2 s = float2(
            tile.bias
                [at[1] * p.bias_stride_m + (col + at[0]) * p.bias_stride_n]);
        s.y *= params.bias_sign;
        v += beta.x * s + beta.y * float2(-s.y, s.x);
      }
    }
    tile.out[at[1] * p.out_stride_m + (col + at[0]) * p.out_stride_n] = T2(v);
  }
}

#define INSTANTIATE_MATMUL_MPP_COMPLEX(T, RN, TAIL, SUFFIX)                  \
  template[[host_name("matmul_mpp_complex_" #T "_" #RN SUFFIX)]] kernel void \
  matmul_mpp_complex<T, RN, TAIL>(                                           \
      device const vec<T, 2>*,                                               \
      device const vec<T, 2>*,                                               \
      device vec<T, 2>*,                                                     \
      device const vec<T, 2>*,                                               \
      constant MatmulComplexParams&,                                         \
      uint3,                                                                 \
      uint)

INSTANTIATE_MATMUL_MPP_COMPLEX(float, 16, false, "");
INSTANTIATE_MATMUL_MPP_COMPLEX(float, 32, false, "");
INSTANTIATE_MATMUL_MPP_COMPLEX(half, 16, false, "");
INSTANTIATE_MATMUL_MPP_COMPLEX(half, 32, false, "");
INSTANTIATE_MATMUL_MPP_COMPLEX(float, 16, true, "_tail");

template <typename T>
kernel void matmul_complex_reduce(
    device const vec<T, 2>* a [[buffer(0)]],
    device const vec<T, 2>* b [[buffer(1)]],
    device vec<T, 2>* out [[buffer(2)]],
    device const vec<T, 2>* bias [[buffer(3)]],
    constant MatmulComplexParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  using T2 = vec<T, 2>;
  constant MatmulParams<uint64_t>& p = params.gemm;
  const uint row = tid.y, col = tid.x / simdgroup_size;
  if (col >= p.n) {
    return;
  }
  device const T2* a_row = a + tid.z * p.a_batch + row * p.a_stride_m;
  device const T2* b_col = b + tid.z * p.b_batch + col * p.b_stride_n;
  float2 acc = 0;
  for (uint k = lane; k < p.k; k += simdgroup_size) {
    const float2 x = float2(a_row[k * p.a_stride_k]);
    const float2 y = float2(b_col[k * p.b_stride_k]);
    const float xi = params.a_sign * x.y, yi = params.b_sign * y.y;
    acc += float2(x.x * y.x - xi * yi, x.x * yi + xi * y.x);
  }
  acc = float2(simd_sum(acc.x), simd_sum(acc.y));
  if (lane == 0) {
    float2 v = p.alpha * acc + params.alpha_im * float2(-acc.y, acc.x);
    if (p.beta != 0 || params.beta_im != 0) {
      float2 s = float2(bias
                            [tid.z * p.bias_batch + row * p.bias_stride_m +
                             col * p.bias_stride_n]);
      s.y *= params.bias_sign;
      v += p.beta * s + params.beta_im * float2(-s.y, s.x);
    }
    out[tid.z * p.out_batch + row * p.out_stride_m + col * p.out_stride_n] =
        T2(v);
  }
}

#define INSTANTIATE_MATMUL_COMPLEX_REDUCE(T)                      \
  template [[host_name("matmul_complex_reduce_" #T)]] kernel void \
  matmul_complex_reduce<T>(                                       \
      device const vec<T, 2>*,                                    \
      device const vec<T, 2>*,                                    \
      device vec<T, 2>*,                                          \
      device const vec<T, 2>*,                                    \
      constant MatmulComplexParams&,                              \
      uint3,                                                      \
      uint)

INSTANTIATE_MATMUL_COMPLEX_REDUCE(float);
INSTANTIATE_MATMUL_COMPLEX_REDUCE(half);
#endif

template <typename T>
inline T matmul_conj(T value, bool conjugate) {
  return conjugate ? c10::metal::conj(value) : value;
}

template <typename T1, typename T2, typename TA>
inline TA matmul_inner(
    device const T1* mat1Data,
    device const T2* mat2Data,
    constant array<ulong2, 3>& strides,
    constant uint3& sizes,
    threadgroup TA A_tile[TILE_DIM][TILE_DIM],
    threadgroup TA B_tile[TILE_DIM][TILE_DIM],
    uint2 tid,
    uint2 thread_id,
    bool2 conjugate = bool2(false)) {
  TA sum = 0;

  const uint numTiles = c10::metal::ceil_div<ulong>(sizes.y, TILE_DIM);
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (thread_id.y < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = matmul_conj(
          static_cast<TA>(
              mat1Data[thread_id.y * strides[0].x + tiledCol * strides[0].y]),
          conjugate.x);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && thread_id.x < sizes.z) {
      B_tile[tid.y][tid.x] = matmul_conj(
          static_cast<TA>(
              mat2Data[tiledRow * strides[1].x + thread_id.x * strides[1].y]),
          conjugate.y);
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += c10::metal::mul(A_tile[tid.y][k], B_tile[k][tid.x]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  return sum;
}

template <typename T, uint N>
inline c10::metal::opmath_t<T> batched_matmul_inner(
    device const T* mat1Data,
    device const T* mat2Data,
    uint batch,
    constant array<ulong, N>& strides,
    constant uint4& sizes,
    threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM],
    threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM],
    uint3 tid,
    uint row,
    uint col,
    bool2 conjugate = bool2(false)) {
  using TA = c10::metal::opmath_t<T>;
  TA sum = 0;

  // Compute batch offsets
  ulong batch1Offset = batch * strides[2];
  ulong batch2Offset = batch * strides[5];

  const uint numTiles = c10::metal::ceil_div<ulong>(sizes.y, TILE_DIM);
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (row < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = matmul_conj(
          static_cast<TA>(
              mat1Data
                  [batch1Offset + row * strides[1] + tiledCol * strides[0]]),
          conjugate.x);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && col < sizes.z) {
      B_tile[tid.y][tid.x] = matmul_conj(
          static_cast<TA>(
              mat2Data
                  [batch2Offset + tiledRow * strides[4] + col * strides[3]]),
          conjugate.y);
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += c10::metal::mul(A_tile[tid.y][k], B_tile[k][tid.x]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  return sum;
}

template <typename T>
kernel void matmul(
    device const T* mat1Data [[buffer(0)]],
    device const T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    constant bool4& conjugate [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner(
      mat1Data,
      mat2Data,
      strides,
      sizes,
      A_tile,
      B_tile,
      tid,
      thread_id,
      conjugate.xy);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] =
        static_cast<T>(sum);
  }
}

template <typename T>
kernel void int_mm(
    device const T* mat1Data [[buffer(0)]],
    device const char* mat2Data [[buffer(1)]],
    device int* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup int A_tile[TILE_DIM][TILE_DIM];
  threadgroup int B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner(
      mat1Data, mat2Data, strides, sizes, A_tile, B_tile, tid, thread_id);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] = sum;
  }
}

template <typename T>
kernel void addmm(
    device const T* mat1Data [[buffer(0)]],
    device const T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    device const T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong2, 4>& strides [[buffer(5)]],
    constant uint3& sizes [[buffer(6)]],
    constant bool4& conjugate [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner<T>(
      mat1Data,
      mat2Data,
      reinterpret_cast<constant array<ulong2, 3>&>(strides),
      sizes,
      A_tile,
      B_tile,
      tid,
      thread_id,
      conjugate.xy);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = matmul_conj(
        static_cast<TA>(
            biasData[thread_id.y * strides[3].x + thread_id.x * strides[3].y]),
        conjugate.z);
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] =
        static_cast<T>(
            c10::metal::mul(alpha_beta[0], sum) +
            c10::metal::mul(alpha_beta[1], bias));
  }
}

template <typename T>
kernel void naive_bmm(
    device const T* mat1Data [[buffer(0)]],
    device const T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong, 9>& strides [[buffer(3)]],
    constant uint4& sizes [[buffer(4)]],
    constant bool4& conjugate [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 9>(
      mat1Data,
      mat2Data,
      batch,
      strides,
      sizes,
      A_tile,
      B_tile,
      tid,
      row,
      col,
      conjugate.xy);

  if (row < sizes.x && col < sizes.z) {
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(sum);
  }
}

template <typename T>
kernel void naive_baddbmm(
    device const T* mat1Data [[buffer(0)]],
    device const T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    device const T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong, 12>& strides [[buffer(5)]],
    constant uint4& sizes [[buffer(6)]],
    constant bool4& conjugate [[buffer(7)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 12>(
      mat1Data,
      mat2Data,
      batch,
      strides,
      sizes,
      A_tile,
      B_tile,
      tid,
      row,
      col,
      conjugate.xy);

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    ulong biasOffset = batch * strides[11];
    auto bias = matmul_conj(
        static_cast<TA>(
            biasData[biasOffset + row * strides[10] + col * strides[9]]),
        conjugate.z);
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(
            c10::metal::mul(alpha_beta[0], sum) +
            c10::metal::mul(alpha_beta[1], bias));
  }
}

template <typename T>
kernel void naive_addbmm(
    device const T* mat1Data [[buffer(0)]],
    device const T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    device const T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong, 12>& strides [[buffer(5)]],
    constant uint4& sizes [[buffer(6)]],
    constant bool4& conjugate [[buffer(7)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  c10::metal::opmath_t<T> sum = 0;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  // Iterate through all batches and accumulate
  for (uint batch = 0; batch < sizes.w; batch++) {
    sum += batched_matmul_inner<T, 12>(
        mat1Data,
        mat2Data,
        batch,
        strides,
        sizes,
        A_tile,
        B_tile,
        tid,
        row,
        col,
        conjugate.xy);
  }

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = matmul_conj(
        static_cast<TA>(biasData[row * strides[10] + col * strides[9]]),
        conjugate.z);
    outputData[row * strides[7] + col * strides[6]] = static_cast<T>(
        c10::metal::mul(alpha_beta[0], sum) +
        c10::metal::mul(alpha_beta[1], bias));
  }
}

#define INSTANTIATE_MM_OPS(DTYPE)                                           \
  template [[host_name("matmul_" #DTYPE)]] kernel void matmul<DTYPE>(       \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      constant array<ulong2, 3>& strides [[buffer(3)]],                     \
      constant uint3& sizes [[buffer(4)]],                                  \
      constant bool4& conjugate [[buffer(5)]],                              \
      uint2 tid [[thread_position_in_threadgroup]],                         \
      uint2 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_bmm_" #DTYPE)]] kernel void naive_bmm<DTYPE>( \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      constant array<ulong, 9>& strides [[buffer(3)]],                      \
      constant uint4& sizes [[buffer(4)]],                                  \
      constant bool4& conjugate [[buffer(5)]],                              \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("addmm_" #DTYPE)]] kernel void addmm<DTYPE>(         \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      device const DTYPE* biasData [[buffer(3)]],                           \
      constant array<c10::metal::opmath_t<DTYPE>, 2>& alpha_beta            \
      [[buffer(4)]],                                                        \
      constant array<ulong2, 4>& strides [[buffer(5)]],                     \
      constant uint3& sizes [[buffer(6)]],                                  \
      constant bool4& conjugate [[buffer(7)]],                              \
      uint2 tid [[thread_position_in_threadgroup]],                         \
      uint2 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_baddbmm_" #DTYPE)]]                           \
  kernel void naive_baddbmm<DTYPE>(                                         \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      device const DTYPE* biasData [[buffer(3)]],                           \
      constant array<c10::metal::opmath_t<DTYPE>, 2>& alpha_beta            \
      [[buffer(4)]],                                                        \
      constant array<ulong, 12>& strides [[buffer(5)]],                     \
      constant uint4& sizes [[buffer(6)]],                                  \
      constant bool4& conjugate [[buffer(7)]],                              \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_addbmm_" #DTYPE)]]                            \
  kernel void naive_addbmm<DTYPE>(                                          \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      device const DTYPE* biasData [[buffer(3)]],                           \
      constant array<c10::metal::opmath_t<DTYPE>, 2>& alpha_beta            \
      [[buffer(4)]],                                                        \
      constant array<ulong, 12>& strides [[buffer(5)]],                     \
      constant uint4& sizes [[buffer(6)]],                                  \
      constant bool4& conjugate [[buffer(7)]],                              \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]])

// Complex MM
INSTANTIATE_MM_OPS(float2);
INSTANTIATE_MM_OPS(half2);

// Integral MM
INSTANTIATE_MM_OPS(long);
INSTANTIATE_MM_OPS(int);
INSTANTIATE_MM_OPS(short);
INSTANTIATE_MM_OPS(char);
INSTANTIATE_MM_OPS(uchar);

#define INSTANTIATE_INT_MM_OP(DTYPE)                                  \
  template [[host_name("int_mm_" #DTYPE)]] kernel void int_mm<DTYPE>( \
      device const DTYPE* mat1Data [[buffer(0)]],                     \
      device const char* mat2Data [[buffer(1)]],                      \
      device int* outputData [[buffer(2)]],                           \
      constant array<ulong2, 3>& strides [[buffer(3)]],               \
      constant uint3& sizes [[buffer(4)]],                            \
      uint2 tid [[thread_position_in_threadgroup]],                   \
      uint2 thread_id [[thread_position_in_grid]])

INSTANTIATE_INT_MM_OP(char);
INSTANTIATE_INT_MM_OP(uchar);

#if C10_METAL_HAS_MPP

template <int BM, int BN, int NSG>
kernel void int_mm_mpp(
    device int8_t* mat1Data [[buffer(0)]],
    device int8_t* mat2Data [[buffer(1)]],
    device int* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const int m = int(sizes.x);
  const int k = int(sizes.y);
  const int n = int(sizes.z);
  const int row = int(tgid.y) * BM;
  const int col = int(tgid.x) * BN;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      false,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> mat1(
      mat1Data,
      dextents<int32_t, 2>(k, m),
      array<int32_t, 2>{int(strides[0].y), int(strides[0].x)});
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> mat2(
      mat2Data,
      dextents<int32_t, 2>(n, k),
      array<int32_t, 2>{int(strides[1].y), int(strides[1].x)});
  tensor<device int, dextents<int32_t, 2>, tensor_inline> output(
      outputData,
      dextents<int32_t, 2>(n, m),
      array<int32_t, 2>{int(strides[2].y), int(strides[2].x)});

  const bool inside = row + BM <= m && col + BN <= n;
  if (inside) {
    auto a = mat1.template slice<dynamic_extent, BM>(0, row);
    auto b = mat2.template slice<BN, dynamic_extent>(col, 0);
    auto c = output.template slice<BN, BM>(col, row);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(a),
        decltype(b),
        int>();
    op.run(a, b, result);
    result.store(c);
  } else {
    auto a = mat1.slice(0, row);
    auto b = mat2.slice(col, 0);
    auto c = output.slice(col, row);
    auto result = op.template get_destination_cooperative_tensor<
        decltype(a),
        decltype(b),
        int>();
    op.run(a, b, result);
    result.store(c);
  }
}

template [[host_name("int_mm_mpp_64_64_4")]]
kernel void int_mm_mpp<64, 64, 4>(
    device int8_t* mat1Data [[buffer(0)]],
    device int8_t* mat2Data [[buffer(1)]],
    device int* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]);

#endif // C10_METAL_HAS_MPP
