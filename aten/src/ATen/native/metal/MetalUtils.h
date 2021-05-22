#include <ATen/Tensor.h>
#include <ATen/native/metal/mpscnn/MPSCNNContext.h>
#include <ATen/native/metal/MetalCommandBuffer.h>
#include <ATen/native/metal/MetalTensorImpl.h>
#include <ATen/native/metal/MetalTensorImplStorage.h>
#include <vector>

#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
typedef float16_t fp16_t;
#else
typedef uint16_t fp16_t;
#endif

namespace at {
namespace native {
namespace metal {

std::vector<fp16_t> Fp32ToFp16(const std::vector<float>& src);
std::vector<float> Fp16ToFp32(const std::vector<fp16_t>& src);

std::vector<float> NCHWToNC4(
    const float* src,
    const std::vector<int64_t>& sizes);
std::vector<float> NC4ToNCHW(
    const float* src,
    const std::vector<int64_t>& sizes);

// The MPSCNNConvolution class takes weights in the order
// [outputChannels][kernelHeight][kernelWidth][inputChannels/groups].
static inline std::vector<float> permuteWeights(
    const float* src,
    const std::vector<int64_t>& sizes) {
  const int64_t M = sizes[0];
  const int64_t Cf = sizes[1];
  const int64_t kH = sizes[2];
  const int64_t kW = sizes[3];
  std::vector<float> packedWeights(M * kH * kW * Cf);
  for (auto m = 0; m < M; ++m) {
    for (auto c = 0; c < Cf; ++c) {
      for (auto kh = 0; kh < kH; ++kh) {
        for (auto kw = 0; kw < kW; ++kw) {
          int64_t oc = m * kH * kW * Cf + kh * kW * Cf + kw * Cf + c;
          int64_t ic = m * Cf * kH * kW + c * kH * kW + kh * kW + kw;
          packedWeights[oc] = src[ic];
        }
      }
    }
  }
  return packedWeights;
}

// When copying the result back to a CPU tensor, the memory format becomes NCHW.
// Thus,we compute the strides based on contiguous memory format.
static inline std::vector<int64_t> compute_strides(
    const std::vector<int64_t>& sizes) {
  const auto dim = sizes.size();
  std::vector<int64_t> strides(dim, 0);
  if (dim > 0) {
    const auto last_idx = dim - 1;
    strides[last_idx] = 1;
    for (int i = last_idx - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

static inline MetalTensorImplStorage& getTensorImplStorage(
    const at::Tensor& tensor) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  TORCH_CHECK(tensor.is_metal());
  MetalTensorImpl* impl =
      static_cast<MetalTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

static inline at::Tensor makeTensor(
    MetalTensorImplStorage&& mt,
    const TensorOptions& options) {
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  auto sizes = mt.sizes(); // sizes is stored in TensorImpl
  auto strides = mt.strides(); // strides is stored in MetalTensorImpl
  return detail::make_tensor<MetalTensorImpl>(
      DispatchKeySet(DispatchKey::Metal),
      options.dtype(),
      at::Device(at::kMetal),
      std::move(mt),
      std::vector<int64_t>(sizes.begin(), sizes.end()),
      std::vector<int64_t>(strides.begin(), strides.end()));
}

static inline MetalCommandBuffer* getCommandBufferFromTensor(
    const Tensor& tensor) {
  TORCH_CHECK(tensor.is_metal());
  auto implStorage = getTensorImplStorage(tensor);
  MetalCommandBuffer* cmdBuffer = implStorage.texture()->commandBuffer();
  if (!cmdBuffer || !cmdBuffer.valid) {
    cmdBuffer = [MetalCommandBuffer currentBuffer];
  }
  return cmdBuffer;
}

template<typename T>
id<MTLBuffer>makeMTLBuffer(const std::vector<T>& src) {
    id<MTLBuffer> buffer = [[MPSCNNContext sharedInstance].device
          newBufferWithLength:src.size() * sizeof(T)
                      options:MTLResourceOptionCPUCacheModeWriteCombined];
    memcpy(buffer.contents, src.data(), src.size() * sizeof(T));
    return buffer;
}

} // namespace metal
} // namespace native
} // namespace at
