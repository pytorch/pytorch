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

// A is sizes.x x sizes.y
// B.T is sizes.z x sizes.y
// C is sizes.x x sizes.z

template<typename T, unsigned groupSize>
kernel void int4pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]],
    uint2                        thread_index   [[thread_position_in_grid]]) {
    const uint lda = sizes.y;
    const uint ldc = sizes.z;
    const uint m = thread_index.y; // 0..sizes.x-1
    const uint n = thread_index.x; // 0..sizes.z-1
    const uint nb = n / 32;
    const uint ldb = min(32U,  sizes.z - nb * 32);
    const uint32_t k_block = (sizes.y + groupSize - 1) / groupSize;
    constant T *A_ptr = A + m * lda;
    constant uchar *B_ptr = B + (nb * 16 * sizes.y);

    float rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < k_block ; kb ++) {
      const T scale = scalesAndZeros[(kb * ldc + n) * 2 + 0];
      const T zero = scalesAndZeros[(kb * ldc + n) * 2 + 1] - scale * T(8);
      for(uint idx = 0; idx < groupSize && k < sizes.y; idx++, k++) {
        const auto a_val = float(A_ptr[k]);
        uchar b_val = B_ptr[(k * ldb + (n % 32))/2];
        b_val = (n & 1) == 0 ? b_val & 0x0f : (b_val >> 4);
        rc += a_val * float(scale * T(b_val) + zero);
      }
    }
    outputData[m * sizes.z + n] = T(rc);
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
    uint2                        thread_index [[thread_position_in_grid]])

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

template<typename T>
struct Vec4Type {};

template<>
struct Vec4Type<float> {
  using type = float4;
};

template<>
struct Vec4Type<half> {
  using type = half4;
};

#if __METAL_VERSION__ >= 310
template<>
struct Vec4Type<bfloat> {
  using type = bfloat4;
};
#endif

template <typename T, unsigned blockSize=8>
kernel void
int8pack_mm(constant T *A [[buffer(0)]], constant char *B [[buffer(1)]],
            constant T *scales [[buffer(2)]],
            device T *outputData [[buffer(3)]],
            constant int3 &sizes [[buffer(4)]],
            uint2 group_index [[threadgroup_position_in_grid]],
            uint2 threadgroup_index [[thread_position_in_threadgroup]]) {
  using vecT = typename Vec4Type<T>::type;
  const uint lda = sizes.y;
  const uint ldc = sizes.z;
  int out_idx = (group_index.x * blockSize + threadgroup_index.x) * 4;
  int n = out_idx % sizes.z;
  int m = out_idx / sizes.z;
  // Offset pointers
  A += m * lda;
  B += n * lda;
  outputData += m *ldc;

  float4 rc = 0;
  for (unsigned k = threadgroup_index.y * 4; k < sizes.y; k += 4 * blockSize) {
    threadgroup_barrier(mem_flags::mem_none);
    auto a_val = float4(*reinterpret_cast<constant vecT *>(A  + k));
    float4x4 b_val;
    for (int i = 0; i < 4; ++i) {
      b_val[i] = float4(*reinterpret_cast<constant char4 *>(B + i * lda + k));
    }
    rc += transpose(b_val) * a_val;
  }

  // Accumulate results acorss SIMD group? (8 threads using vec4)
  threadgroup float4 tgp_memory[blockSize][blockSize];
  tgp_memory[threadgroup_index.x][threadgroup_index.y] = rc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadgroup_index.y == 0) {
    for (int i = 1; i < blockSize; i++) {
      rc += tgp_memory[threadgroup_index.x][i];
    }
    *reinterpret_cast<device vecT *>(outputData + n) =
        vecT(rc * float4(*reinterpret_cast<constant vecT *>(scales + n)));
  }
}

#define INSTANTIATE_INT8MM(DTYPE)                                              \
  template [[host_name("int8pack_mm_" #DTYPE)]] kernel void                    \
  int8pack_mm<DTYPE>(                                                          \
      constant DTYPE * A [[buffer(0)]], constant char *B [[buffer(1)]],        \
      constant DTYPE *scales [[buffer(2)]],                                    \
      device DTYPE *outputData [[buffer(3)]],                                  \
      constant int3 &sizes [[buffer(4)]],                                      \
      uint2 group_index [[threadgroup_position_in_grid]],                      \
      uint2 threadgroup_index [[thread_position_in_threadgroup]]);

INSTANTIATE_INT8MM(half);
INSTANTIATE_INT8MM(float);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT8MM(bfloat);
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
      [computeEncoder dispatchThreads:MTLSizeMake(N, M, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M), 1, 1)];
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
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      [computeEncoder dispatchThreads:MTLSizeMake(M * N / 4, 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
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
