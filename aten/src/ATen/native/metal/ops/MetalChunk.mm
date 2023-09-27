#include <ATen/Tensor.h>
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

// Split the input tensor into two on channel dimension
// TODO: [T87567124] Fully implement chunk in Metal shader
static std::vector<Tensor> chunk(const Tensor& input, int64_t chunks, int64_t dim) {
  TORCH_CHECK(chunks == 2 && dim == 1);
  TORCH_CHECK(input.dim() == 4);
  TORCH_CHECK(input.size(0) == 1);
  int64_t dim_size = input.size(dim);
  int64_t split_size = (dim_size + chunks - 1) / chunks;
  int64_t num_splits = 1;
  if (split_size != 0) {
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  auto outputSize1 = {input.size(0), split_size, input.size(2), input.size(3)};
  auto outputSize2 = {input.size(0), last_split_size, input.size(2), input.size(3)};
  MetalTensorImplStorage mt1(outputSize1);
  MetalTensorImplStorage mt2(outputSize2);
  mt1.texture()->allocateTemporaryStorage(outputSize1, commandBuffer);
  mt2.texture()->allocateTemporaryStorage(outputSize2, commandBuffer);
  MPSImage* Y1 = mt1.texture()->image();
  MPSImage* Y2 = mt2.texture()->image();
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:"split_channels"
                     Constants:@[
                         @(X.featureChannels),
                         @(Y1.featureChannels),
                         @(Y2.featureChannels)]];
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y1 texture] atIndex:1];
  [encoder setTexture:[Y2 texture] atIndex:2];
  const auto& launchParams =
      mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  auto output1 = makeTensor(std::move(mt1), input.options());
  auto output2 = makeTensor(std::move(mt2), input.options());
  return {output1, output2};
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::chunk"), TORCH_FN(chunk));
};

}
}
}
