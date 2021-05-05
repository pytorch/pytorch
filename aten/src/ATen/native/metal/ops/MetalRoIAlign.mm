#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/library.h>
//#include <caffe2/fb/custom_ops/maskrcnn/roi_align.h>
namespace torch {
namespace fb {
namespace metal {

using namespace at::native::metal;

torch::Tensor RoIAlign(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    std::string order,
    double spatial_scale,
    int64_t aligned_height,
    int64_t aligned_width,
    int64_t sampling_ratio,
    bool aligned,
    c10::optional<std::vector<torch::Tensor>>) {
    
  TORCH_CHECK(features.is_metal());
  TORCH_CHECK(features.size(0) == 1);
  TORCH_CHECK(rois.is_cpu());
  TORCH_CHECK(rois.dim() == 2);
  TORCH_CHECK(rois.size(1) == 4 || rois.size(1) == 5);
  
  std::vector<int64_t> outputSize{rois.size(0), features.size(1), aligned_height, aligned_width};
  MetalTensorImplStorage mt{outputSize};

  const int64_t N = rois.size(0);
  if(N == 0){
    // if there is RoIs, simply return an empty metal tensor
    return makeTensor(std::move(mt), features.options());
  }
  const auto roiBytes = rois.size(0) * 4 * sizeof(fp16_t);
  id<MTLBuffer> roiBuffer = makeMTLBuffer(roiBytes);
  fp16_t* roiBufferPtr = (fp16_t*)roiBuffer.contents;
  auto Rdim = rois.size(1);
  auto off = Rdim == 5 ? 1 : 0;
  for (auto i = 0; i < rois.size(0); ++i) {
    // skip the batch index
    roiBufferPtr[i * 4 + 0] = rois.data_ptr<float>()[i * Rdim + off + 0];
    roiBufferPtr[i * 4 + 1] = rois.data_ptr<float>()[i * Rdim + off + 1];
    roiBufferPtr[i * 4 + 2] = rois.data_ptr<float>()[i * Rdim + off + 2];
    roiBufferPtr[i * 4 + 3] = rois.data_ptr<float>()[i * Rdim + off + 3];
  }
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(features);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  MPSImage* X = imageFromTensor(features);
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  NSUInteger scale = (NSUInteger)(spatial_scale * 10000);
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:"roi_align"
                     Constants:@[
                       @(scale),
                       @((NSUInteger)sampling_ratio),
                       @(X.featureChannels),
                       @(X.numberOfImages),
                       @(Y.numberOfImages),
                     ]];
  [encoder setComputePipelineState:state];
  [encoder setBuffer:roiBuffer offset:0 atIndex:0];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X markRead];
  [Y markRead];
  auto output = makeTensor(std::move(mt), features.options());
  return output;
}

} // metal
} // fb
} // torch

TORCH_LIBRARY_IMPL(_caffe2, Metal, m) {
  m.impl("_caffe2::RoIAlign", TORCH_FN(torch::fb::metal::RoIAlign));
}
