#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_int4pack_mm_native.h>
#include <ATen/ops/_weight_int8pack_mm_native.h>
#include <ATen/ops/empty.h>
#endif
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

// #define _CAPTURE_KERNEL 1

namespace at::native {

using namespace mps;

static at::native::mps::MetalShaderLibrary lib(R"METAL_QUANTIZED(
#include <metal_stdlib>
using namespace metal;

template <typename T> struct Vec4Type {};

template <> struct Vec4Type<float> {
  using type = float4;
};

template <> struct Vec4Type<half> {
  using type = half4;
};

#if __METAL_VERSION__ >= 310
template <> struct Vec4Type<bfloat> {
  using type = bfloat4;
};
#endif

template <typename T> struct Vec2Type {};

template <> struct Vec2Type<float> {
  using type = float2;
};

template <> struct Vec2Type<half> {
  using type = half2;
};

#if __METAL_VERSION__ >= 310
template <> struct Vec2Type<bfloat> {
  using type = bfloat2;
};
#endif

template<typename T, unsigned groupSize>
kernel void int4pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]], // M, K, N
    uint3 group_index [[threadgroup_position_in_grid]],
    uint3 threadgroup_index [[thread_position_in_threadgroup]]) {

    const uint K = sizes.y;
    const uint N = sizes.z;
    const uint nb = group_index.x; // 0..N/32-1
    const uint n2 = 16 * nb + threadgroup_index.x; // 0..N/2-1
    const uint m = group_index.z;
    const uint ldb = min(32U,  N - nb * 32);
    const uint32_t k_block = (K + groupSize - 1) / groupSize;

    using vec2T = typename Vec2Type<T>::type;
    using vec4T = typename Vec4Type<T>::type;

    constant vec4T *A_ptr = reinterpret_cast<constant vec4T *>(A + m * K);
    constant uchar *B_ptr = B + (nb * 16 * K);

    float2 rc = 0.0;
    uint k = threadgroup_index.y * 4;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      float2 scales, zeros;
      for (int i = 0; i < 2; ++i) {
        scales[i] = scalesAndZeros[(kb * N + 2*n2 + i) * 2 + 0];
        zeros[i] = scalesAndZeros[(kb * N + 2*n2 + i) * 2 + 1] - scales[i] * T(8);
      }

      for(uint idx = k % groupSize; idx < groupSize && k < K; idx += 16, k += 16) {
        threadgroup_barrier(mem_flags::mem_none);

        const auto a_vec = float4(A_ptr[k/4]);
        uchar4 b_byte;
        for (int i = 0; i < 4; i++) {
          b_byte[i] = B_ptr[((k + i) * ldb + (2*n2 % 32))/2];
        }

        float4x2 b_mat;

        for (int i = 0; i < 4; i++) {
          b_mat[i] = scales * float2(
            float(b_byte[i] & 0x0f),
            float(b_byte[i] >> 4)) + zeros;
        }

        rc += b_mat * a_vec;
      }
    }

    threadgroup float2 tgp_memory[16][4];
    tgp_memory[threadgroup_index.x][threadgroup_index.y] = rc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadgroup_index.y == 0) {
      for (unsigned i = 1; i < 4; i++) {
        rc += tgp_memory[threadgroup_index.x][i];
      }
      reinterpret_cast<device vec2T*>(outputData + m * N)[n2] = vec2T(rc);
    }
}

#define INSTANTIATE_INT4MM(DTYPE, GSIZE)                                 \
template                                                                 \
[[host_name("int4pack_mm_" #GSIZE "_" #DTYPE)]]                          \
kernel void int4pack_mm<DTYPE, GSIZE>(                                   \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant uchar             * B              [[buffer(1)]],           \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],           \
    device   DTYPE             * outputData     [[buffer(3)]],           \
    constant uint3             & sizes          [[buffer(4)]],           \
    uint3 group_index [[threadgroup_position_in_grid]], \
    uint3 threadgroup_index [[thread_position_in_threadgroup]])

INSTANTIATE_INT4MM(float, 32);
INSTANTIATE_INT4MM(half, 32);
INSTANTIATE_INT4MM(float, 64);
INSTANTIATE_INT4MM(half, 64);
INSTANTIATE_INT4MM(float, 128);
INSTANTIATE_INT4MM(half, 128);
INSTANTIATE_INT4MM(float, 256);
INSTANTIATE_INT4MM(half, 256);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT4MM(bfloat, 32);
INSTANTIATE_INT4MM(bfloat, 64);
INSTANTIATE_INT4MM(bfloat, 128);
INSTANTIATE_INT4MM(bfloat, 256);
#endif

// ------------------------------ int8 MM For M >= 12 ------------------------------------
/**
 * The following code is heavily inspired by llama.cpp (https://github.com/ggerganov/llama.cpp).
 * The original code is under MIT License: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
 *
 * Matrix Multiplication Algorithm:
 * 1. Load A and B blocks (32x32 and 64x32 respectively) into shared memory.
 * 2. In 4 simdgroups, calculate the outer product of the loaded blocks. Each simdgroup produces a 2x4 8x8 result.
 *      2.1 For how to use outer product to perform matrix multiplication, refer to
 *           http://mlwiki.org/index.php/Matrix-Matrix_Multiplication#Sum_of_Outer_Products
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
#if __METAL_VERSION__ >= 310
template <> struct BlockType<bfloat> {
  using simdgroup_type8x8 = simdgroup_bfloat8x8;
  using type4 = bfloat4;
};
#endif

template<typename T>
float2 get_scale_zero(constant T * scalesAndZeros, uint2 index) {
    return float2(1.0, 0.0);
}

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
#if __METAL_VERSION__ >= 310
INSTANTIATE_MM(bfloat, char, get_scale_zero_q8);
#endif
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

    using T4 = typename BlockType<T>::type4;

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

    for (int row = 0; row < nr; ++row) {
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
#if __METAL_VERSION__ >= 310
INSTANTIATE_MV(bfloat);
#endif

)METAL_QUANTIZED");

Tensor _weight_int4pack_mm_mps(const Tensor& A, const Tensor& B, int64_t qGroupSize, const Tensor& qScaleAndZeros) {
  constexpr int64_t kNTileSize = 8;

  auto M = A.size(0);
  auto N = B.size(0) * kNTileSize;
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
              __func__,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kInt, __func__, " : expect B to be int32 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.dim() == 4, __func__, " : expect B to 4d tensor.");

  TORCH_CHECK(qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 || qGroupSize == 256,
              __func__,
              ": expect qGroupSize to be 32, 64, 128 or 256, got ",
              qGroupSize);

  TORCH_CHECK(qScaleAndZeros.dim() == 3 && qScaleAndZeros.size(1) == N && qScaleAndZeros.size(2) == 2,
              __func__,
              ": expect qScaleAndZeros to be 3d tensor with sizes [:, ",
              N,
              ", 2]");

  auto C = at::empty({M, N}, A.options());
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture(fmt::format("int4pack_mm_{}x{}x{}", M, N, K), mpsStream);
      }
#endif
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = fmt::format("int4pack_mm_{}_{}", qGroupSize, scalarToMetalTypeString(A));
      id<MTLComputePipelineState> quantizedPSO = lib.getPipelineStateForFunc(kernel);
      const auto maxThreadsPerGroup = static_cast<decltype(M)>([quantizedPSO maxTotalThreadsPerThreadgroup]);
      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, qScaleAndZeros, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      [computeEncoder dispatchThreads:MTLSizeMake(N / 2, 4, M) threadsPerThreadgroup:MTLSizeMake(16, 4, 1)];
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });
  return C;
}

Tensor _weight_int8pack_mm_mps(const Tensor& A, const Tensor& B, const Tensor& scales) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
              __func__,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

  TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N, __func__, " : expect scales to be 1d tensor with size ", N);

  auto C = at::empty({M, N}, A.options());
  TORCH_CHECK(N % 32 == 0 && K % 32 == 0);
#if 1
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N), 0};
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture(fmt::format("int8pack_mm_{}x{}x{}", M, N, K), mpsStream);
      }
#endif
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      std::string kernel;
      // heuristic, to use mv kernel for mm with small M. M = 10 is the performance tipping point.
      if (M < 12) {
        kernel = fmt::format("int8pack_mv_{}", scalarToMetalTypeString(A));
      } else {
        kernel = fmt::format("large_m_int8pack_mm_{}", scalarToMetalTypeString(A));
      }
      id<MTLComputePipelineState> quantizedPSO = lib.getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, scales, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      if (M < 12) {
        [computeEncoder setThreadgroupMemoryLength:32 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((N + 7) / 8, M, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [computeEncoder setThreadgroupMemoryLength:12288 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((M + 31) / 32, (N + 63) / 64, 1)
                       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });
#else
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *ATensor = nil, *BTensor = nil, *scalesTensor = nil;
    MPSGraphTensor* outputTensor = nil;
  };
  @autoreleasepool {
    std::string key = __func__ + getTensorsStringKey({A, B, scales});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->ATensor = mpsGraphRankedPlaceHolder(mpsGraph, A);
      newCachedGraph->BTensor = mpsGraphRankedPlaceHolder(mpsGraph, B);
      newCachedGraph->scalesTensor = mpsGraphRankedPlaceHolder(mpsGraph, scales);
      auto castB = castMPSTensor(mpsGraph, newCachedGraph->BTensor, getMPSScalarType(A));
      auto transposedB = [mpsGraph transposeTensor:castB dimension:-1 withDimension:-2 name:nil];
      auto mmTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:newCachedGraph->ATensor
                                                      secondaryTensor:transposedB
                                                                 name:nil];
      newCachedGraph->outputTensor = [mpsGraph multiplicationWithPrimaryTensor:mmTensor
                                                               secondaryTensor:newCachedGraph->scalesTensor
                                                                          name:nil];
    });
    auto APlaceholder = Placeholder(cachedGraph->ATensor, A);
    auto BPlaceholder = Placeholder(cachedGraph->BTensor, B);
    auto scalesPlaceholder = Placeholder(cachedGraph->scalesTensor, scales);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, C);
    runMPSGraph(getCurrentMPSStream(),
                cachedGraph->graph(),
                dictionaryFromPlaceholders(APlaceholder, BPlaceholder, scalesPlaceholder),
                outputPlaceholder);
  }
#endif

  return C;
}

} // namespace at::native
