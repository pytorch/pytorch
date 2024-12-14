#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_weight_to_int4pack_native.h>
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

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Quantized_metallib.h>
#endif

Tensor _convert_weight_to_int4pack_mps(const Tensor& in, int64_t innerKTiles) {
  TORCH_CHECK(in.dim() == 2, __func__, " : expect weight to be 2D tensor.");
  TORCH_CHECK(in.dtype() == at::kByte, __func__, " : expect weight to be kByte.");
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8,
              __func__,
              " : innerKTiles need to be 2, 4, or 8, got ",
              innerKTiles);

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto Kdiv2 = weight.size(1);
  auto K = Kdiv2 * 2;

  // Create fake shapes for cpu. The meta registration in dynamo requires
  // operator has the same output shape for each device. So creating a fake
  // shape {N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2}
  auto weight_packed = at::empty({N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2},
                                 at::TensorOptions().dtype(at::kInt).device(at::kMPS));
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(N), static_cast<uint32_t>(Kdiv2 / 4), 0, 0};
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture(fmt::format("weight_to_int4pack_{}x{}", N, Kdiv2), mpsStream);
      }
#endif
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = fmt::format("weight_to_int4pack");
      id<MTLComputePipelineState> quantizedPSO = lib.getPipelineStateForFunc(kernel);
      const auto maxThreadsPerGroup = [quantizedPSO maxTotalThreadsPerThreadgroup];
      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setArgs(computeEncoder, weight, weight_packed, sizes);
      [computeEncoder dispatchThreads:MTLSizeMake(N, Kdiv2 / 4, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });
  return weight_packed;
}

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
      mtl_setArgs(computeEncoder, A, B, qScaleAndZeros, C, sizes);
      [computeEncoder dispatchThreads:MTLSizeMake(N / 4 * 32, 1, M) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
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
      mtl_setArgs(computeEncoder, A, B, scales, C, sizes);
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
