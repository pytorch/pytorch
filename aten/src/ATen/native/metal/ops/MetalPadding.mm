#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

API_AVAILABLE(ios(11.0), macos(10.13))
Tensor reflection_pad2d(const Tensor& input, IntArrayRef padding) {
  TORCH_CHECK(input.is_metal());

  const int pad_dim = padding.size();
  const IntArrayRef input_size = input.sizes();
  const int input_dim = input_size.size();

  TORCH_CHECK(pad_dim == 1 || pad_dim == 4, "Padding sizes must be a 1-tuple or 4-tuple!");
  TORCH_CHECK(input_dim >= 2, "Input tensor must have dim >= 2!");

  NSUInteger pad_left = padding[0];
  NSUInteger pad_right = padding[0];
  NSUInteger pad_top = padding[0];
  NSUInteger pad_bottom = padding[0];
  if (pad_dim == 4) {
    pad_right = padding[1];
    pad_top = padding[2];
    pad_bottom = padding[3];
  }

  std::vector<int64_t> output_size(input_dim);
  for (int d = 0; d < input_dim; ++d) {
    if (d == input_dim - 1) {
      output_size[d] = input_size[d] + pad_right + pad_left;
    }
    else if (d == input_dim - 2) {
      output_size[d] = input_size[d] + pad_top + pad_bottom;
    }
    else {
      output_size[d] = input_size[d];
    }
  }

  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  MetalTensorImplStorage mt{output_size};
  mt.texture()->allocateTemporaryStorage(output_size, commandBuffer);
  MPSImage* Y = mt.texture()->image();

  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:"reflection_pad2d"
                     Constants:@[
                       @(Y.height),
                       @(Y.width),
                       @(Y.featureChannels),
                       @(Y.numberOfImages),
                       @(X.height),
                       @(X.width),
                       @(X.featureChannels),
                       @(X.numberOfImages),
                       @(pad_left),
                       @(pad_right),
                       @(pad_top),
                       @(pad_bottom)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::reflection_pad2d"), TORCH_FN(reflection_pad2d));
};

}
}
}
