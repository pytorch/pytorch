#include <c10/metal/utils.h>

#include <metal_stdlib>
using namespace metal;

kernel void weight_to_int4pack(constant int *W [[buffer(0)]],
                               device uchar *outputData [[buffer(1)]],
                               constant uint2 &sizes [[buffer(2)]],
                               uint2 thread_index [[thread_position_in_grid]]) {
  const uint K_int32 = sizes.y;
  const uint n = thread_index.x; // 0..N-1
  const uint k = thread_index.y; // 0..K_int32-1
  int32_t src_val = W[n * K_int32 + k];
  uint8_t src_val0 = (uint8_t)((src_val & 0xFF000000) >> 24);
  uint8_t src_val1 = (uint8_t)((src_val & 0x00FF0000) >> 16);
  uint8_t src_val2 = (uint8_t)((src_val & 0x0000FF00) >> 8);
  uint8_t src_val3 = (uint8_t)(src_val & 0x000000FF);
  outputData[n * K_int32 * 4 + k * 4] = ((src_val3 & 0xF) << 4) | (src_val3 >> 4);
  outputData[n * K_int32 * 4 + k * 4 + 1] = ((src_val2 & 0xF) << 4) | (src_val2 >> 4);
  outputData[n * K_int32 * 4 + k * 4 + 2] = ((src_val1 & 0xF) << 4) | (src_val1 >> 4);
  outputData[n * K_int32 * 4 + k * 4 + 3] = ((src_val0 & 0xF) << 4) | (src_val0 >> 4);
}

/*
   This code takes heavy inspiration from MLX qvm kernel here:
   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.metal#L381
   Specifically:
     - Multiplying activation by inverse scaling factor to reduce compute
   boundedness
     - Handling zero point by accumulating act in separate sum term. Needed with
   optimization done above. MLX MIT License:
   https://github.com/ml-explore/mlx/blob/main/LICENSE
*/

/*
   A matrix is [M x K] (right now this kernel does not support M > 1 but this is
   a very easy fix that will follow right after) B matrix is [N x K]. For 4 bit
   2 of the k values are packed in one byte so you can think of B as [N x K/2]
   matrix from layout perspective.

   Since this kernel is optimizing for gemv case, we split work, along reduction
   dim k, among the threads of same simdgroup. Ex: if K = 4096 and simdgroup
   size is 32 (current algorithm should work as long as simdgroup size is > 32).
   Then each thread will accumulate 4096/32 = 128 k values. However these 128
   values, handled by each thread are not laid out contiguously. Each thread
   handles 4 contiguous k values and then jumps 128 elements, k_jump =
   thread_per_channel (32) * ks_per_thread (4). Take a simpler example where
   simdgroup is of size 4. In this case threads_per_channel = 4. Assume K = 32
      k                thread
   [0, 1, 2, 3,          0
    4, 5, 6, 7,          1
    8, 9, 10, 11,        2
    12, 13, 14, 15,      3
    16, 17, 18, 19,      0
    20, 21, 22, 23,      1
    24, 25, 26, 27,      2
    28, 29, 30, 31]      3
   thread id in simd group that handle corresponding
   ks
   Thread 0 here is handling (0, 1, 2, 3) and then (16, 17, 18, 19). They are
   apart by k_jump = 4 * 4 = 16 This is done to improve memory access locality
   amonng threads that are working co-operatively. Once each thread has their
   partial sums accumulated, we use tree reduction (Metal offers simd_sum but
   not used so that we support simdgroup size = 64). In the
   example above we will have 4 partial sums.

   Each thread also handles 4 different output rows. Thus each simdgroup will be
   responsible for (1x4) tile of the output. We haven't evaluated whether a
   different tile size is better or not. We probably will do some auto-tuning
   once initial work is done.

*/

/*
   @brief This shader implements 4-bit matrix-vector multiplication where A
   matrix is fp16, bfloat or float and B matrix is a 4-bit groupwise-quantized weight
   matrix.
   @param [in] A is activation matrix of size M x K.
   @param [in] B is weight matrix of size M x K. Each byte contains 2 4-bit
   values, along K dim, packed together.
   @param [in] scales_and_zeros is scales and zero points corresponding each
   output channel x groups. These are packed as [num_groups = ceil(K / group_size), N, 2]. N = output
   @param [out] output_data is output matrix of size M x N.
   @param [in] sizes array contains values of M, N and K.
   @param [in] thread_index is global thread id.
   @param [in] tid_in_simdgruop is thread id in simdgroup. e.g. in simdgroup of size 32 it can be in [0-31].
*/
template <typename T, unsigned group_size>
kernel void int4pack_mm(constant T *A [[buffer(0)]],
                        constant uchar *B [[buffer(1)]],
                        constant T *scales_and_zeros [[buffer(2)]],
                        device T *output_data [[buffer(3)]],
                        constant uint3 &sizes [[buffer(4)]], // M, K, N
                        uint3 thread_index [[thread_position_in_grid]],
                        uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr uint threads_per_channel = 32;
  constexpr uint ks_per_thread = 4;
  constexpr uint k_pack_factor = 2;
  const uint K = sizes.y;
  const uint N = sizes.z;
  uint n = thread_index.x; // 0..N/4-1
  uint m = thread_index.z; // 0..M
  n = n / threads_per_channel;
  n = n * 4;
  // This is starting k for each thread. In the example above, for thread 1 this
  // value will be 4.
  uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
  constexpr int k_jump = threads_per_channel * ks_per_thread;

  using vecT = typename c10::metal::vec4type_t<T>;
  constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * K);
  constant uchar *B_ptr = B + ((n * K) / k_pack_factor);

  thread float4 result = float4(0.0);
  // We multiply group of 4 channels with these scales.
  // Because corresponding values from weight matrix are effectively left
  // shifted. This is to avoid doing right shift on those values which ends up
  // affecting performance. This is the trick applied in MLX kernels.
  float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};

  for (; k < K; k += k_jump) {
    // Find specific group to which channels handled by this thread
    // belong.
    uint k_block_index = k / group_size;
    // Since scales_and_zeros are packed as [num_groups, N, 2].
    // Finding a specific's group's scales and zero points requires jump by factor
    // of N*2
    uint scales_group_offset = (k_block_index * N + n) * 2;
    uint zeros_gruop_offset = scales_group_offset + 1;

    const T scale0 = scales_and_zeros[scales_group_offset];
    // Adding zero point results in 10% perf penalty.
    const T zero0 = scales_and_zeros[zeros_gruop_offset] - scale0 * T(8);

    const T scale1 = scales_and_zeros[scales_group_offset + 2];
    const T zero1 = scales_and_zeros[zeros_gruop_offset + 2] - scale1 * T(8);

    const T scale2 = scales_and_zeros[scales_group_offset + 4];
    const T zero2 = scales_and_zeros[zeros_gruop_offset + 4] - scale2 * T(8);

    const T scale3 = scales_and_zeros[scales_group_offset + 6];
    const T zero3 = scales_and_zeros[zeros_gruop_offset + 6] - scale3 * T(8);

    const float4 zeros = float4(zero0, zero1, zero2, zero3);

    float4 a_val = float4(A_ptr[k / 4]);
    // We are gonna skip right-shifts of the weights and hence divide by corresponding factor.
    float4 a_vec = a_val * act_div_scales;
    float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];

    float4x4 b_mat;
    ushort b_val0 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 0 * K) / k_pack_factor))[0];
    ushort b_val1 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 1 * K) / k_pack_factor))[0];
    ushort b_val2 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 2 * K) / k_pack_factor))[0];
    ushort b_val3 = (reinterpret_cast<constant ushort *>(
        B_ptr + (k + 3 * K) / k_pack_factor))[0];
    b_mat[0] = scale0 * float4(float(b_val0 & 0x000f), float(b_val0 & 0x00f0),
                               float(b_val0 & 0x0f00), float(b_val0 & 0xf000));
    b_mat[1] = scale1 * float4(float(b_val1 & 0x000f), float(b_val1 & 0x00f0),
                               float(b_val1 & 0x0f00), float(b_val1 & 0xf000));
    b_mat[2] = scale2 * float4(float(b_val2 & 0x000f), float(b_val2 & 0x00f0),
                               float(b_val2 & 0x0f00), float(b_val2 & 0xf000));
    b_mat[3] = scale3 * float4(float(b_val3 & 0x000f), float(b_val3 & 0x00f0),
                               float(b_val3 & 0x0f00), float(b_val3 & 0xf000));

    result += a_vec * b_mat;
    result += a_val_sum * zeros;
  }
  result += simd_shuffle_down(result, 1);
  result += simd_shuffle_down(result, 2);
  result += simd_shuffle_down(result, 4);
  result += simd_shuffle_down(result, 8);
  result += simd_shuffle_down(result, 16);
  if (tid_in_simdgroup % threads_per_channel == 0) {
    reinterpret_cast<device vecT *>(output_data + m * N)[n / 4] = vecT(result);
  }
}

#define INSTANTIATE_INT4MV(DTYPE, GSIZE)                                       \
  template [[host_name("int4pack_mm_" #GSIZE "_" #DTYPE)]] kernel void         \
  int4pack_mm<DTYPE, GSIZE>(                                                   \
      constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
      constant DTYPE * scales_and_zeros [[buffer(2)]],                         \
      device DTYPE * output_data [[buffer(3)]],                                \
      constant uint3 & sizes [[buffer(4)]],                                    \
      uint3 thread_index [[thread_position_in_grid]],                          \
      uint tid_in_simdgroup [[thread_index_in_simdgroup]])

INSTANTIATE_INT4MV(float, 32);
INSTANTIATE_INT4MV(half, 32);
INSTANTIATE_INT4MV(float, 64);
INSTANTIATE_INT4MV(half, 64);
INSTANTIATE_INT4MV(float, 128);
INSTANTIATE_INT4MV(half, 128);
INSTANTIATE_INT4MV(float, 256);
INSTANTIATE_INT4MV(half, 256);
INSTANTIATE_INT4MV(bfloat, 32);
INSTANTIATE_INT4MV(bfloat, 64);
INSTANTIATE_INT4MV(bfloat, 128);
INSTANTIATE_INT4MV(bfloat, 256);

// ------------------------------ int8 MM For M >= 12 ------------------------------------
/**
 * The following code is heavily inspired by llama.cpp (https://github.com/ggerganov/llama.cpp).
 * The original code is under MIT License: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
 *
 * Matrix Multiplication Algorithm:
 * 1. Load A and B blocks (32x32 and 64x32 respectively) into shared memory.
 * 2. In 4 simdgroups, calculate the outer product of the loaded blocks. Each simdgroup produces a 2x4 8x8 result.
 *      2.1 For how to use outer product to perform matrix multiplication, refer to
 *           https://web.archive.org/web/20230521063455/http://mlwiki.org/index.php/Matrix-Matrix_Multiplication#Sum_of_Outer_Products
 * 3. Repeat 1 & 2 along K axis, with K block size 32, accumulate the result in the 2x4 8x8 block.
 * 4. Dequantize the final result and store it in the output matrix.
 *
 * Variable names are changed to adapt to PyTorch convention such as M, N, K, etc.
 * Assuming row major order.
 * For more details please see inline comments.
 */
#include <metal_stdlib>
using namespace metal;
template <typename T> struct BlockType {};

template <> struct BlockType<float> {
  using simdgroup_type8x8 = simdgroup_float8x8;
  using type4 = float4;
};

template <> struct BlockType<half> {
  using simdgroup_type8x8 = simdgroup_half8x8;
  using type4 = half4;
};
template <> struct BlockType<bfloat> {
  using simdgroup_type8x8 = simdgroup_bfloat8x8;
  using type4 = bfloat4;
};

template<typename T>
float2 get_scale_zero_q8(constant T * scalesAndZeros, uint2 index) {
    T scale = scalesAndZeros[index[0]];
    return float2(scale, 0.0);
}

#define BLOCK_SIZE_M 32 // each block takes 32 rows in matrix A
#define BLOCK_SIZE_N 64 // each block takes 64 rows in matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 2 // in data loading stage, each thread load 2 simdgroup matrices from matrix A
#define THREAD_MAT_N 4 // in data loading stage, each thread load 4 simdgroup matrices from matrix B
#define THREAD_PER_ROW_A 4 // 4 thread for each row in matrix A to load numbers
#define THREAD_PER_ROW_B 2 // 2 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// T: input type, W: weight type
template<typename T, typename W, float2 (*get_scale_zero_func)(constant T *, uint2)>
kernel void kernel_mul_mm(
    constant T                 * A              [[buffer(0)]],
    constant char              * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device T                   * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup char           * shared_memory  [[threadgroup(0)]], // threadgroup buffer at index 0
    uint3                        tgpig          [[threadgroup_position_in_grid]], // 3d coordinates
    uint                         tiitg          [[thread_index_in_threadgroup]], // 128 per threadgroup
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) {

    using T4 = typename BlockType<T>::type4;
    using Tsimd8x8 = typename BlockType<T>::simdgroup_type8x8;
    // sizes: x = M, y = K, z = N
    // pytorch: M x K @ N x K -> M x N
    // ggml: K x N @ K x M -> N x M
    uint32_t M = sizes.x; // M
    uint32_t K = sizes.y; // K
    uint32_t N = sizes.z; // N
    uint32_t nbytes_B = sizeof(W); // number of bytes for one element in B
    uint32_t nbytes_B_row = nbytes_B * K; // number of bytes for one row in B
    uint32_t nbytes_A = sizeof(T); // number of bytes for one element in A
    uint32_t nbytes_A_row = nbytes_A * K; // number of bytes for one row in A

    // shared memory for A and B
    threadgroup T    * shared_memory_A = (threadgroup T    *)(shared_memory);
    // using half here to store int8, gives us about 8% perf gain comparing to bfloat but not sure why
    threadgroup half * shared_memory_B = (threadgroup half *)(shared_memory + 8192);

    const uint threadgroup_M = tgpig.x; // total number (M + 31)/32, the index of this threadgroup along M axis
    const uint threadgroup_N = tgpig.y; // total number (N + 63)/64, the index of this threadgroup along N axis

    // if this block is of 64x32 shape or smaller, bound the number of rows for A and B in this block.
    short n_rows_A = min(uint32_t(M - threadgroup_M * BLOCK_SIZE_M), uint32_t(BLOCK_SIZE_M));
    short n_rows_B = min(uint32_t(N - threadgroup_N * BLOCK_SIZE_N), uint32_t(BLOCK_SIZE_N));

    // a thread shouldn't load data outside of the matrix
    short thread_row_A = min(((short)tiitg/THREAD_PER_ROW_A), n_rows_A - 1);
    short thread_row_B = min(((short)tiitg/THREAD_PER_ROW_B), n_rows_B - 1);

    Tsimd8x8 simdgroup_A[2]; // input, each simdgroup load 128 values of input
    simdgroup_half8x8 simdgroup_B[4]; // weight, each simdgroup load 256 values of weight
    simdgroup_float8x8 simdgroup_C[8]; // outer product result, 2x4 8x8 blocks.
    for (short i = 0; i < 8; i++){
        simdgroup_C[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    constant T * a_ptr = (constant T *)((constant char *)A
        + nbytes_A_row * (threadgroup_M * BLOCK_SIZE_M + thread_row_A)
        + nbytes_A * (BLOCK_SIZE_K / THREAD_PER_ROW_A * (tiitg % THREAD_PER_ROW_A)));

    constant W * b_ptr = (constant W *)(B
        + nbytes_B_row * (threadgroup_N * BLOCK_SIZE_N + thread_row_B)
        + nbytes_B * (BLOCK_SIZE_K / THREAD_PER_ROW_B * (tiitg % THREAD_PER_ROW_B)));
/**
Load weight and input into shared memory:
8192: BLOCK_SIZE_M x BLOCK_SIZE_K x 4(max bytes per value) <----- numbers don't checkout, should be 4096. Changing it to 4096 gives wrong value.
4096: BLOCK_SIZE_N x BLOCK_SIZE_K x 2(storing int8 in half)

                          K
               ┌────────────────────────┐              8192(A)             4096(B)
               │                        │   ┌────────────────────────┬────────────┐
               │                        │   │++++++++++++++++++++++++│++++++++++++│
               │                        │   └────────────────────────┴────────────┘
               │                        │
               │32(BLOCK_SIZE_K)        │
               ├──┬──┬──────────────────┤                           K
               │++│  │                  │               ┌────────────────────────┐
             64│++│  │...               │               │                        │
 (BLOCK_SIZE_N)│++│  │                  │               │                        │
               ├──┴──┴──────────────────┤               │                        │
               │                        │               │                        │
               │      ───────────►      │               │32(BLOCK_SIZE_K)        │
               │       for loop         │               ├──┬──┬──────────────────┤
               │                        │             32│++│  │ ...              │
               │                        │ (BLOCK_SIZE_M)├──┴──┴──────────────────┤
               │                        │               │         ────────────►  │
               │                        │               │            for loop    │
               └────────────────────────┘               └────────────────────────┘
                           B                                        A

 */
    for (uint32_t loop_k = 0; loop_k < K; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            half weight = *(b_ptr + i);
            // for example, tiitg 32, i 12 -> 0 + 1 = 1, it needs to work on sg mat grid row 1
            short sg_mat_grid_row_index = (tiitg % THREAD_PER_ROW_B) * THREAD_PER_ROW_B + i / 8;
            // same example, sg mat grid col index: 32 / 2 / 8 = 2, so currently need to work with sg mat at (1, 2)
            short sg_mat_grid_col_index = tiitg / THREAD_PER_ROW_B / 8;
            // now inside sg mat, which index to write to? starting point is SG_MAT_SIZE * sg_mat_offset
            short row_offset = i % 8;
            short col_offset = (tiitg / THREAD_PER_ROW_B) % 8;
            // now calculates the overall offset for shared_memory_B
            short sb_offset = (sg_mat_grid_row_index * 8 + sg_mat_grid_col_index) * 64 + (row_offset * 8 + col_offset);
            *(shared_memory_B + sb_offset) = weight;
        }
        // read 8 values for input matrix

        #pragma unroll(2)
        for (short i = 0; i < 2; i++) {
            *((threadgroup T4 *)(shared_memory_A + (tiitg % THREAD_PER_ROW_A) * 8 * 32 + 8 * (tiitg / THREAD_PER_ROW_A)) + i) = *((constant T4 *)a_ptr + i);
        }

        a_ptr += BLOCK_SIZE_K;
        b_ptr += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        // pointing to the shared memory starting address for A, for current simdgroup.
        threadgroup T    * simdgroup_A_ptr = (shared_memory_A + THREAD_MAT_M * SG_MAT_SIZE * (sgitg / 2));
        // pointing to the shared memory starting address for B, for current simdgroup.
        threadgroup half * simdgroup_B_ptr = (shared_memory_B + THREAD_MAT_N * SG_MAT_SIZE * (sgitg % 2));

/**
Outer product:
              K
       ────────────►
     8    for loop              8   8
   ┌───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┬───┬───┬───┐
 8 │+++│   │   │   │      │  8│+++│+++│+++│+++│###│###│###│###│
   ├───┼───┼───┼───┤      │   ├───┼───┼───┼───┼───┼───┼───┼───┤
   │+++│   │   │   │      │   │   │   │   │   │   │   │   │   │
   ├───┼───┼───┼───┤      │ K ├───┼───┼───┼───┼───┼───┼───┼───┤
   │###│   │   │   │      │   │   │   │   │   │   │   │   │   │
   ├───┼───┼───┼───┤      │   ├───┼───┼───┼───┼───┼───┼───┼───┤
   │###│   │   │   │      │   │   │   │   │   │   │   │   │   │
   └───┴───┴───┴───┘      ▼   └───┴───┴───┴───┴───┴───┴───┴───┘
                       for loop
    + simdgroup 0,1                + simdgroup 0,2
    # simdgroup 2,3                # simdgroup 1,3
 */
        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(simdgroup_B[i], simdgroup_B_ptr + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(simdgroup_A[i], simdgroup_A_ptr + SG_MAT_SIZE * i);
            }

            simdgroup_A_ptr += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            simdgroup_B_ptr += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(simdgroup_C[i], simdgroup_A[i/4], simdgroup_B[i%4], simdgroup_C[i]);
            }
        }
    }

    /**
 * Each sgitg 0,1,2,3 handles 2x4 8x8.
    8   8
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
 8│ 0 │ 0 │ 0 │ 0 │ 1 │ 1 │ 1 │ 1 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │ 0 │ 0 │ 0 │ 0 │ 1 │ 1 │ 1 │ 1 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │ 2 │ 2 │ 2 │ 2 │ 3 │ 3 │ 3 │ 3 │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │ 2 │ 2 │ 2 │ 2 │ 3 │ 3 │ 3 │ 3 │
  └───┴───┴───┴───┴───┴───┴───┴───┘

   scale: 8 x BLOCK_SIZE_N, starting from shared_memory_A. Each sgitg handles 4 8x8 diagonal matrix.
    8   8
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
 8│   │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘
 */

    threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                  + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_N;
    for (int i = 0; i < 8; i++) {
        int block_start = 4 * 8 * (sgitg & 1) + (i % 4) * 8;
        threadgroup float * temp_scale = (threadgroup float *)shared_memory_B + block_start;
        threadgroup float * scale_iter = temp_scale;
        // dequantize
        for (int j = 0; j < 8; j++) {
            // clear next 8 values of scale_iter
            *((threadgroup float2x4 *)scale_iter) = float2x4(0.f);
            // find scale
            int scale_index = threadgroup_N * BLOCK_SIZE_N + block_start + j;
            float2 scale_zero = get_scale_zero_func(scalesAndZeros, uint2(scale_index, 0));
            // create diagonal matrix of scales
            *(scale_iter + j) = scale_zero[0];
            // go to next row
            scale_iter += BLOCK_SIZE_N;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 simd_scale;
        simdgroup_load(simd_scale, temp_scale, BLOCK_SIZE_N);
        simdgroup_multiply(simdgroup_C[i], simdgroup_C[i], simd_scale);
        simdgroup_store(simdgroup_C[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_N * (i/4), BLOCK_SIZE_N);
    }

    device T * C = outputData + (BLOCK_SIZE_N * threadgroup_N) + (BLOCK_SIZE_M * threadgroup_M) * N;
    if (sgitg == 0) {
        for (int i = 0; i < n_rows_B; i++) {
            for (int j = tiitg; j < n_rows_A; j += BLOCK_SIZE_M) {
                float temp = *(temp_str + i + j * BLOCK_SIZE_N);
                *(C + i + j * N) = (device T)(temp);
            }
        }
    }
}

#define INSTANTIATE_MM(DTYPE, WDTYPE, DEQUANT_FUNC)                      \
template                                                                 \
[[host_name("large_m_int8pack_mm_" #DTYPE)]]                             \
kernel void kernel_mul_mm<DTYPE, WDTYPE, DEQUANT_FUNC>(                  \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant char              * B              [[buffer(1)]],           \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],           \
    device   DTYPE             * outputData     [[buffer(3)]],           \
    constant uint3             & sizes          [[buffer(4)]],           \
    threadgroup char           * shared_memory  [[threadgroup(0)]],      \
    uint3                        tgpig          [[threadgroup_position_in_grid]], \
    uint                         tiitg          [[thread_index_in_threadgroup]],  \
    uint                         sgitg          [[simdgroup_index_in_threadgroup]])


INSTANTIATE_MM(float, char, get_scale_zero_q8);
INSTANTIATE_MM(half, char, get_scale_zero_q8);
INSTANTIATE_MM(bfloat, char, get_scale_zero_q8);
// ------------------------------ int8 MM For M < 12 ------------------------------------
/* Matrix vector multiplication, used for small M size for matrix multiplication as well.

                      for loop ->
                       1  1  1  1                                 1
  ┌──────────────────┬──┬──┬──┬──┬───────────┬─────┐             ┌──┐
  │      thread 0-> 8│  │  │  │  │           │     │            8│  │
  │                  ├──┼──┼──┼──┤           │     │             ├──┤
  │      thread 1-> 8│  │  │  │  │           │     │            8│  │
  │                  ├──┼──┼──┼──┤           │     │             ├──┤
  │      thread 2-> 8│  │  │  │  │           │     │            8│  │
  │                  ├──┼──┼──┼──┤           │     │             ├──┤
  │      thread 3-> 8│  │  │  │  │           │     │            8│  │
  │                  ├──┼──┼──┼──┤           │     │             ├──┤
  │                  │  │  │  │  │           │     │             │  │
  │    thread 4-7  32│  │  │  │  │           │     │           32│  │
  │                  │  │  │  │  │   SIMD    │     │             │  │
K │                  ├──┼──┼──┼──┤  Group 1  │     │             ├──┤
  │                  │  │  │  │  │           │     │             │  │
  │    thread 8-15 64│  │  │  │  │           │     │           64│  │
  │                  │  │  │  │  │           │     │             │  │
  │                  ├──┼──┼──┼──┤           │     │             ├──┤
  │                  │  │  │  │  │           │     │             │  │
  │  thread 16-31 128│  │  │  │  │           │     │          128│  │
  │                  │  │  │  │  │           │     │             │  │
  │                  ├──┼──┼──┼──┼───────────┤     │             ├──┤
  │                  │  │  │  │  │           │     │             │  │
  └──────────────────┴──┴──┴──┴──┴───────────┴─────┘             └──┘
                      SIMD Group 0                                input

                          N
  ┌──────────────────┬──┬──┬──┬──┬───────────┬─────┐
  │                  │  │  │  │  │           │     │
  └──────────────────┴──┴──┴──┴──┴───────────┴─────┘
                      scale

*/
// putting them in the kernel causes a significant performance penalty, could use function constant to optimize?
#define NB_Q8_0 8
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

template<typename T>
kernel void kernel_mul_mv(
    constant T                 * A              [[buffer(0)]],
    constant char              * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device T                   * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup char           * shared_memory  [[threadgroup(0)]],
    uint3                        tgpig          [[threadgroup_position_in_grid]],
    uint                         tiisg          [[thread_index_in_simdgroup]],
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) {

    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    // sizes: x = M, y = K, z = N, given mv, x = M = 1
    // pytorch: M x K @ N x K -> M x N
    // ggml: K x N @ K x M -> N x M
    uint32_t K = sizes.y; // K
    uint32_t N = sizes.z; // N

    const int nb = K/N_SIMDWIDTH; // number of blocks of 32 elements along K axis
    const int threadgroup_N = tgpig.x; // threadgroup index along N axis.
    const int threadgroup_M = tgpig.y; // threadgroup index along M axis. For matvec multiplication this will always be 0 but keep it for future usage.
    /*
     * Each SIMD group in a threadgroup handles N_DST = nr = 4 rows.
     *      - threadgroup_N is the x index of the threadgroup. threadgroup_N * nsg -> the overall offset of SIMD groups, for this threadgroup.
     *      - threadgroup_N * nsg + sgitg -> the overall index of SIMD group, in all SIMD groups.
     *      - (threadgroup_N * nsg + sgitg) * nr -> the starting index of the row that this SIMD group needs to handle.
     */
    const int first_row = (threadgroup_N * nsg + sgitg) * nr;

    const uint offset0 = first_row * K;

    // x: weight, y: input
    constant char * x = (constant char *) B + offset0;
    constant T    * y = (constant T    *) A + threadgroup_M*K;

    // Load data to shared memory
    threadgroup T * shared_scale = (threadgroup T *)(shared_memory); // length 8 * sizeof(float)
    // Load scale:
    if (tiisg < 4) {
        *(shared_scale + (sgitg % 2) * 4 + tiisg) = *(scalesAndZeros + (threadgroup_N * NB_Q8_0) + (sgitg % 2) * 4 + tiisg);
    }

    // Accumulate on float4
    float2x4 yl;
    float4x4 xl[2];
    float4 sumf = 0;

    // Group threads in SIMD group into 8x4 block, each thread handles 8 input values.
    const int ix = tiisg/4;
    const int il = tiisg%4;

    // N_SIMDWIDTH = 32 means we have 32 weights in 1 simdgroup.
    // Find the starting point of input that this thread need to work on, load yb into yl.
    constant T * yb = y + ix * N_SIMDWIDTH + NB_Q8_0*il;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (short ib = ix; ib < nb; ib += nw/4) {
        // Load y data
        for (short i = 0; i < 2; i++) {
            short offset = i * 4;
            yl[i] = {*(yb + offset), *(yb + offset + 1), *(yb + offset + 2), *(yb + offset + 3)};
        }

        for (short row = 0; row < nr; row++) {
            // Locate where x should be.
            // row offset: row * K
            // col offset: ib * N_SIMDWIDTH + il * NB_Q8_0
            // x index: row * K + ib * N_SIMDWIDTH + il * NB_Q8_0
            constant int8_t * qs = (constant int8_t *)(x + row * K + ib * N_SIMDWIDTH + il * NB_Q8_0);
            for (short batch = 0; batch < 2; batch++) {
                short offset = batch * 4;
                xl[batch][row] = {(float)qs[offset], (float)qs[offset+1], (float)qs[offset+2], (float)qs[offset+3]};
            }
        }
        sumf += yl[0] * xl[0];
        sumf += yl[1] * xl[1];
        yb += NB_Q8_0 * nw;
    }

    for (unsigned row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        float scale = *(shared_scale + (sgitg % 2) * 4 + row);
        if (tiisg == 0 && first_row + row < N) {
            outputData[threadgroup_M*N + first_row + row] = (device T)(tot * scale);
        }
    }
}


#define INSTANTIATE_MV(DTYPE)                                                   \
template                                                                        \
[[host_name("int8pack_mv_" #DTYPE)]]                                            \
kernel void kernel_mul_mv<DTYPE>(                                               \
    constant DTYPE             * A              [[buffer(0)]],                  \
    constant char              * B              [[buffer(1)]],                  \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],                  \
    device   DTYPE             * outputData     [[buffer(3)]],                  \
    constant uint3             & sizes          [[buffer(4)]],                  \
    threadgroup char           * shared_memory  [[threadgroup(0)]],             \
    uint3                        tgpig          [[threadgroup_position_in_grid]],   \
    uint                         tiisg          [[thread_index_in_simdgroup]],      \
    uint                         sgitg          [[simdgroup_index_in_threadgroup]])


INSTANTIATE_MV(float);
INSTANTIATE_MV(half);
INSTANTIATE_MV(bfloat);
