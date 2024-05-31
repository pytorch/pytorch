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

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// T: input type, W: weight type
template<typename T, typename W, float2 (*get_scale_zero_func)(constant T *, uint2)>
kernel void kernel_mul_mm(
    constant T                 * A              [[buffer(0)]],  // 2 x 4096
    constant char              * B              [[buffer(1)]],  // 1024 x 4096
    constant T                 * scalesAndZeros [[buffer(2)]],
    device T                   * outputData     [[buffer(3)]],  // 2 x 1024
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup uchar          * shared_memory  [[threadgroup(0)]], // threadgroup buffer at index 0
    uint3                        tgpig          [[threadgroup_position_in_grid]], // 3d coordinates
    uint                         tiitg          [[thread_index_in_threadgroup]], // 128 per threadgroup
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) {

    using T4 = typename BlockType<T>::type4;
    using Tsimd8x8 = typename BlockType<T>::simdgroup_type8x8;
    // sizes: x = M, y = K, z = N
    // pytorch: M x K @ N x K -> M x N
    // ggml: K x N @ K x M -> N x M
    uint32_t ne00 = sizes.y; // K
    uint32_t nb00 = sizeof(W);
    uint32_t nb01 = nb00 * ne00;
    uint32_t ne10 = sizes.y; // K
    uint32_t nb10 = sizeof(T);
    uint32_t nb11 = nb10 * ne10;
    uint32_t ne0 = sizes.z; // N
    uint32_t ne1 = sizes.x; // M
    constant char * src0 = (constant char *)B;
    constant char * src1 = (constant char *)A;

    // 8192 for sa, 4096 for sb
    threadgroup float * sa = (threadgroup float *)(shared_memory);
    threadgroup T     * sb = (threadgroup T     *)(shared_memory + 8192);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_float8x8 ma[4]; // dequantized weight
    Tsimd8x8 mb[2]; // input
    simdgroup_float8x8 c_res[8]; // outer product result
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    constant W * x = (constant W *)(src0
        + nb01 * (r0 * BLOCK_SIZE_M + thread_row)
        + nb00 * (BLOCK_SIZE_K / THREAD_PER_ROW * (tiitg % THREAD_PER_ROW)));
    constant T * y = (constant T *)(src1
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (uint32_t loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            float weight = *(x + i);
            // for example, tiitg 32, i 12 -> 0 + 1 = 1, it needs to work on sg mat grid row 1
            int sg_mat_grid_row_index = (tiitg % THREAD_PER_ROW) * THREAD_PER_ROW + i / 8;
            // same example, sg mat grid col index: 32 / 2 / 8 = 2, so currently need to work with sg mat at (1, 2)
            int sg_mat_grid_col_index = tiitg / THREAD_PER_ROW / 8;
            // now inside sg mat, which index to write to? starting point is SG_MAT_SIZE * sg_mat_offset
            int row_offset = i & 7;
            int col_offset = (tiitg / THREAD_PER_ROW) % 8;
            // now calculates the overall offset for sa
            int sa_offset = (sg_mat_grid_row_index * 8 + sg_mat_grid_col_index) * 64 + (row_offset * 8 + col_offset);
            *(sa + sa_offset) = weight;
        }
        // read 8 values for input matrix

        #pragma unroll(8)
        for (int i = 0; i < 8; i++) {
            *((threadgroup T *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) + i) = *(y + i);
        }

        x += BLOCK_SIZE_K;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup float * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup T     * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
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

   scale: 8 x BLOCK_SIZE_M, starting from sb. Each sgitg handles 4 8x8 diagonal matrix.
    8   8
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
 8│   │   │   │   │   │   │   │   │
  └───┴───┴───┴───┴───┴───┴───┴───┘
 */

    threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                  + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
    for (int i = 0; i < 8; i++) {
        int block_start = 4 * 8 * (sgitg & 1) + (i % 4) * 8;
        threadgroup float * temp_scale = (threadgroup float *)sb + block_start;
        threadgroup float * scale_itr = temp_scale;
        // dequantize
        for (int j = 0; j < 8; j++) {
            // clear next 8 values of scale_itr
            *((threadgroup float2x4 *)scale_itr) = float2x4(0.f);
            // find scale
            int scale_index = r0 * BLOCK_SIZE_M + block_start + j;
            float2 scale_zero = get_scale_zero_func(scalesAndZeros, uint2(scale_index, 0));
            // create diagonal matrix of scales
            *(scale_itr + j) = scale_zero[0];
            // go to next row
            scale_itr += BLOCK_SIZE_M;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 simd_scale;
        simdgroup_load(simd_scale, temp_scale, BLOCK_SIZE_M);
        simdgroup_float8x8 simd_zero = make_filled_simdgroup_matrix<float, 8>(0.f);
        simdgroup_multiply_accumulate(c_res[i], c_res[i], simd_scale, simd_zero);
        simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device T * C = outputData + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0;
    if (sgitg == 0) {
        for (int i = 0; i < n_rows; i++) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                float temp = *(temp_str + i + j * BLOCK_SIZE_M);
                *(C + i + j * ne0) = (device T)(temp);
            }
        }
    }
}

#define INSTANTIATE_MM(DTYPE, WDTYPE, DEQUANT_FUNC)                      \
template                                                                 \
[[host_name("int8pack_mm_" #DTYPE)]]                                     \
kernel void kernel_mul_mm<DTYPE, WDTYPE, DEQUANT_FUNC>(                  \
    constant DTYPE             * A              [[buffer(0)]],           \
    constant char              * B              [[buffer(1)]],           \
    constant DTYPE             * scalesAndZeros [[buffer(2)]],           \
    device   DTYPE             * outputData     [[buffer(3)]],           \
    constant uint3             & sizes          [[buffer(4)]],           \
    threadgroup uchar          * shared_memory  [[threadgroup(0)]],      \
    uint3                        tgpig          [[threadgroup_position_in_grid]], \
    uint                         tiitg          [[thread_index_in_threadgroup]],  \
    uint                         sgitg          [[simdgroup_index_in_threadgroup]])


INSTANTIATE_MM(float, char, get_scale_zero_q8);
INSTANTIATE_MM(half, char, get_scale_zero_q8);
#if __METAL_VERSION__ >= 310
INSTANTIATE_MM(bfloat, char, get_scale_zero_q8);
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
      const std::string kernel = fmt::format("int8pack_mm_{}", scalarToMetalTypeString(A));
      id<MTLComputePipelineState> quantizedPSO = lib.getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, scales, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:16 atIndex:4];
      [computeEncoder setThreadgroupMemoryLength:12288 atIndex:0];
      [computeEncoder dispatchThreadgroups:MTLSizeMake( (M + 31)/32, (N + 63)/64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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
