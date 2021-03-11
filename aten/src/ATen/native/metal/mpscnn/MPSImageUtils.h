#import <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at {
namespace native {
namespace metal {

MPSImage* createStaticImage(const std::vector<int64_t>& sizes);
MPSImage* createStaticImage(
    const uint16_t* src,
    const std::vector<int64_t>& sizes);
MPSImage* createStaticImage(
    const float* src,
    const std::vector<int64_t>& sizes);
MPSImage* createStaticImage(const at::Tensor& tensor);
MPSImage* createStaticImage(MPSImage* image);
MPSImage* createStaticImage(
    MPSTemporaryImage* image,
    MetalCommandBuffer* buffer,
    bool waitUntilCompleted);

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const std::vector<int64_t>& sizes);
MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const std::vector<int64_t>& sizes,
    const float* src);
MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    MPSImage* image);

void copyToHost(float* dst, MPSImage* image);

std::vector<uint16_t> staticImageToFp16Array(MPSImage* image);
at::Tensor staticImageToTensor(MPSImage* image);

static inline MPSImage* imageFromTensor(const Tensor& tensor) {
  TORCH_CHECK(tensor.is_metal());
  using MetalTensorImplStorage = at::native::metal::MetalTensorImplStorage;
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  MetalTensorImpl* impl = (MetalTensorImpl*)tensor.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  return implStorage.texture()->image();
}

} // namespace metal
} // namespace native
} // namespace at
