#include <stdexcept>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_int4pack_mm_native.h>
#include <ATen/ops/_weight_int8pack_mm_native.h>
#include <ATen/ops/empty.h>
#endif
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#include <Metal/MTLCaptureManager.h>

namespace at::native {

using namespace mps;

static const char* METAL_QUANTIZED = R"BINARY_QUANTIZED(
#include <metal_stdlib>
using namespace metal;

template<typename T, unsigned groupSize>
kernel void int4pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant uchar             * B              [[buffer(1)]],
    constant T                 * scalesAndZeros [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]],
    uint                         thread_index   [[thread_position_in_grid]]) {
    const uint lda = sizes.y;
    const uint ldb = groupSize;
    const uint ldc = sizes.x;
    const uint m = thread_index / sizes.x;
    const uint n = thread_index % sizes.x;
    if (n >= sizes.x || m >= sizes.z) {
        return;
    }
    const uint32_t n_block = sizes.y / groupSize;
    float rc = 0.0;
    uint k = 0;
    for (uint32_t kb = 0; kb < n_block ; kb ++) {
      const T scale = scalesAndZeros[(kb * ldc + n) * 2 + 0];
      const T zero = scalesAndZeros[(kb * ldc + n) * 2 + 1] - scale * T(8);
      for(uint idx = 0; idx < groupSize; idx++, k++) {
        const auto a_val = float(A[m * lda + k]);
        uchar b_val = B[(k * ldb + n)/2];
        b_val = (n & 1) == 0 ? (b_val >> 4) & 0x0f : b_val & 0x0f;
        rc += a_val * float(scale * T(b_val) + zero);
      }
    }
    outputData[thread_index] = T(rc);
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
    uint                         thread_index [[thread_position_in_grid]])

INSTANTIATE_INT4MM(float, 32);
INSTANTIATE_INT4MM(half, 32);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT4MM(bfloat, 32);
#endif
)BINARY_QUANTIZED";

static id<MTLLibrary> compileQuantizedOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> binaryLibrary = nil;
  if (binaryLibrary) {
    return binaryLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS) ? MTLLanguageVersion3_1
                                                                                      : MTLLanguageVersion2_3];
  binaryLibrary = [device newLibraryWithSource:[NSString stringWithCString:METAL_QUANTIZED encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(binaryLibrary, "Failed to create metal quantized library, error: ", [[error description] UTF8String]);
  return binaryLibrary;
}

static id<MTLComputePipelineState> quantizedPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> binaryLib = compileQuantizedOpsLibrary(device);
  id<MTLFunction> binaryFunc = [binaryLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(binaryFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:binaryFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
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
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(M),
                                   static_cast<uint32_t>(K),
                                   static_cast<uint32_t>(N)};
  static bool firstCapture = true;
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      NSError *err = nil;
      MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
      if (firstCapture) {
        //mpsStream->synchronize(SyncType::COMMIT);
        MTLCaptureDescriptor* captureDescriptor = [MTLCaptureDescriptor new];
        captureDescriptor.captureObject = device;
        captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        captureDescriptor.outputURL = [NSURL fileURLWithPath:@"int4packmm.gputrace"];
        auto rc = [manager startCaptureWithDescriptor:captureDescriptor error:&err];
        TORCH_CHECK(rc, "Failed to start capture, error :", [[err description] UTF8String]);
      }
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel =  fmt::format("int4pack_mm_{}_{}",qGroupSize, scalarToMetalTypeString(A.scalar_type()));
      id<MTLComputePipelineState> quantizedPSO = quantizedPipelineState(device, kernel);
      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, qScaleAndZeros, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      mtl_dispatch1DJob(computeEncoder, quantizedPSO, C.numel());
      if (firstCapture) {
        mpsStream->synchronize(SyncType::COMMIT);
        [manager stopCapture];
        firstCapture = false;
      }
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

  return C;
}

} // namespace at::native
