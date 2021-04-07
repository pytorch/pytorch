#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

Tensor& hardtanh_(Tensor& input, const Scalar& min_val, const Scalar& max_val) {
  TORCH_CHECK(input.is_metal());
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  MPSImage* Y = createTemporaryImage(commandBuffer, input.sizes().vec());
  float min = min_val.toFloat();
  float max = max_val.toFloat();
  MPSCNNClampOp* clampOp = [MPSCNNClampOp newWithTextures:@[ X, Y ]
                                                     Args:@[ @(min), @(max) ]];
  [clampOp encode:commandBuffer.buffer];
  using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->copyFromTexture(Y);
  return input;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("hardtanh_", TORCH_FN(hardtanh_));
};

}
}
}
