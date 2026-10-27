#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
#include <ATen/native/mps/kernels/Gemv.h>

constant uint TILE_DIM = 16;

template <typename T1, typename T2, typename TA>
inline TA matmul_inner(
    device const T1* mat1Data,
    device const T2* mat2Data,
    constant array<ulong2, 3>& strides,
    constant uint3& sizes,
    threadgroup TA A_tile[TILE_DIM][TILE_DIM],
    threadgroup TA B_tile[TILE_DIM][TILE_DIM],
    uint2 tid,
    uint2 thread_id) {
  TA sum = 0;

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (thread_id.y < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = static_cast<TA>(
          mat1Data[thread_id.y * strides[0].x + tiledCol * strides[0].y]);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && thread_id.x < sizes.z) {
      B_tile[tid.y][tid.x] = static_cast<TA>(
          mat2Data[tiledRow * strides[1].x + thread_id.x * strides[1].y]);
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
    uint col) {
  using TA = c10::metal::opmath_t<T>;
  TA sum = 0;

  // Compute batch offsets
  uint batch1Offset = batch * strides[2];
  uint batch2Offset = batch * strides[5];

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (row < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = static_cast<TA>(
          mat1Data[batch1Offset + row * strides[1] + tiledCol * strides[0]]);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && col < sizes.z) {
      B_tile[tid.y][tid.x] = static_cast<TA>(
          mat2Data[batch2Offset + tiledRow * strides[4] + col * strides[3]]);
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
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner(
      mat1Data, mat2Data, strides, sizes, A_tile, B_tile, tid, thread_id);
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
      thread_id);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = static_cast<TA>(
        biasData[thread_id.y * strides[3].x + thread_id.x * strides[3].y]);
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
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 9>(
      mat1Data, mat2Data, batch, strides, sizes, A_tile, B_tile, tid, row, col);

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
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 12>(
      mat1Data, mat2Data, batch, strides, sizes, A_tile, B_tile, tid, row, col);

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    uint biasOffset = batch * strides[11];
    auto bias = static_cast<TA>(
        biasData[biasOffset + row * strides[10] + col * strides[9]]);
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
        col);
  }

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = static_cast<TA>(biasData[row * strides[10] + col * strides[9]]);
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
      uint2 tid [[thread_position_in_threadgroup]],                         \
      uint2 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_bmm_" #DTYPE)]] kernel void naive_bmm<DTYPE>( \
      device const DTYPE* mat1Data [[buffer(0)]],                           \
      device const DTYPE* mat2Data [[buffer(1)]],                           \
      device DTYPE* outputData [[buffer(2)]],                               \
      constant array<ulong, 9>& strides [[buffer(3)]],                      \
      constant uint4& sizes [[buffer(4)]],                                  \
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
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]])

INSTANTIATE_MM_OPS(float);
INSTANTIATE_MM_OPS(half);
INSTANTIATE_MM_OPS(bfloat);

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
