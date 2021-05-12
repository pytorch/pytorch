#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor view(const Tensor& input, IntArrayRef size) {
  TORCH_CHECK(input.is_metal());
  auto inferred_size = at::infer_size(size, input.numel());
  auto stride =
      at::detail::computeStride(input.sizes(), input.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  MetalTensorImplStorage mt{inferred_size, stride_value};
  mt.texture()->allocateTemporaryStorage(inferred_size, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:"reshape"
                     Constants:@[
                       @(Y.height),
                       @(Y.width),
                       @(Y.featureChannels),
                       @(Y.numberOfImages),
                       @(X.height),
                       @(X.width),
                       @(X.featureChannels),
                       @(X.numberOfImages),
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
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

Tensor reshape(const Tensor& input, IntArrayRef shape) {
  TORCH_CHECK(input.is_metal());
  return view(input, shape);
}

Tensor flatten_using_ints(
    const Tensor& input,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(input.is_metal());
  start_dim = maybe_wrap_dim(start_dim, input.dim());
  end_dim = maybe_wrap_dim(end_dim, input.dim());
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");
  std::vector<int64_t> shape;
  if (input.dim() == 0) {
    return input.reshape({1});
  }
  if (start_dim == end_dim) {
    return input;
  }
  const auto slice_numel = c10::multiply_integers(
      input.sizes().slice(start_dim, end_dim - start_dim + 1));
  shape.reserve(input.dim() - end_dim + start_dim);
  for (int64_t i = 0; i < start_dim; i++) {
    shape.push_back(input.size(i));
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < input.dim(); i++) {
    shape.push_back(input.size(i));
  }
  return input.reshape(shape);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("view", TORCH_FN(view));
  m.impl("reshape", TORCH_FN(reshape));
  m.impl("flatten.using_ints", TORCH_FN(flatten_using_ints));
};

}
}
}
