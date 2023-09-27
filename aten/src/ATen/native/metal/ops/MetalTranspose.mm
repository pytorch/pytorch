#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

// TODO: Move this function to MetalContext
template<typename T>
id<MTLBuffer> _makeMTLBuffer(const std::vector<T>& src) {
    id<MTLBuffer> buffer = [[MetalContext sharedInstance].device
          newBufferWithLength:src.size() * sizeof(T)
                      options:MTLResourceCPUCacheModeWriteCombined];
    memcpy(buffer.contents, src.data(), src.size() * sizeof(T));
    return buffer;
}

static Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(input.is_metal());
  auto ndims = input.dim();
  // Support maximum eight channels on mobile
  TORCH_CHECK(ndims <= 8);
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return input;
  }
  auto outputSizes = input.sizes().vec();
  std::swap(outputSizes[dim0], outputSizes[dim1]);
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  if (input.dim() == 2) {
    MetalTensorImplStorage mt{outputSizes};
    mt.texture()->allocateTemporaryStorage(outputSizes, commandBuffer);
    MPSImage* Y = mt.texture()->image();
    MPSImageTranspose* transpose = [[MPSImageTranspose alloc]
        initWithDevice:[MetalContext sharedInstance].device];
    [transpose encodeToCommandBuffer:commandBuffer.buffer
                         sourceImage:X
                    destinationImage:Y];
    auto output = makeTensor(std::move(mt), input.options());
    return output;
  } else {
    id<MTLBuffer> sizeBuf1 = _makeMTLBuffer<ushort>(
        std::vector<ushort>{input.sizes().begin(), input.sizes().end()});
    id<MTLBuffer> sizeBuf2 = _makeMTLBuffer<ushort>(
        std::vector<ushort>{outputSizes.begin(), outputSizes.end()});
    MetalTensorImplStorage mt{outputSizes};
    mt.texture()->allocateTemporaryStorage(outputSizes, commandBuffer);
    MPSImage* Y = mt.texture()->image();
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        [[MetalContext sharedInstance] specializedPipelineState:"transpose"
                                                       Constants:@[
                                                         @(dim0),
                                                         @(dim1),
                                                         @(input.dim()),
                                                         @(X.numberOfImages),
                                                         @(X.featureChannels),
                                                         @(Y.numberOfImages),
                                                         @(Y.featureChannels),
                                                       ]];

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    [encoder setBuffer:sizeBuf1 offset:0 atIndex:0];
    [encoder setBuffer:sizeBuf2 offset:0 atIndex:1];

    const auto& launchParams =
        mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    auto output = makeTensor(std::move(mt), input.options());
    return output;
  }
}

static Tensor t(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.dim() == 2);
  return metal::transpose(input, 0, input.dim() < 2 ? 0 : 1);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::t"), TORCH_FN(t));
  m.impl(TORCH_SELECTIVE_NAME("aten::transpose.int"), TORCH_FN(transpose));
};

}
}
}
