#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

Tensor upsample_nearest2d_vec(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_CHECK(input.is_metal());
  auto osize =
      upsample::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = upsample::get_scale_value(scale_factors, 0);
  auto scale_w = upsample::get_scale_value(scale_factors, 1);
  int64_t output_height = osize[0];
  int64_t output_width = osize[1];
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);
  std::vector<int64_t> outputSizes{
      nbatch, channels, output_height, output_width};
  MPSImage* X = imageFromTensor(input);
  MetalTensorImplStorage mt{outputSizes};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryStorage(outputSizes, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  if (@available(iOS 11.0, *)) {
    MPSCNNUpsamplingNearest* kernel = [[MPSCNNUpsamplingNearest alloc]
             initWithDevice:[MPSCNNContext sharedInstance].device
        integerScaleFactorX:(NSUInteger)scale_w.value()
        integerScaleFactorY:(NSUInteger)scale_h.value()];
    [kernel encodeToCommandBuffer:commandBuffer.buffer
                      sourceImage:X
                 destinationImage:Y];
  } else {
    NSUInteger sh = scale_h.value() * 10000;
    NSUInteger sw = scale_w.value() * 10000;
    id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
        specializedPipelineState:mpscnn::kernelFor(
                                     Y,
                                     "resize_nearest",
                                     "resize_nearest_nonarray")
                       Constants:@[
                         @(output_height),
                         @(output_width),
                         @(sh),
                         @(sw)
                       ]];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    const auto& launchParams =
        mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    [X markRead];
    [Y markRead];
  }
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("upsample_nearest2d.vec", TORCH_FN(upsample_nearest2d_vec));
};

}
}
}
