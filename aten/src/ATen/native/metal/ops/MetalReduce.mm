#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

API_AVAILABLE(ios(11.3), macos(10.13))
static inline MPSNNReduceUnary* kernelForReducedDim(int dim) {
  id<MTLDevice> device = [MetalContext sharedInstance].device;
  if (dim == 3) {
    return [[MPSNNReduceRowMean alloc] initWithDevice:device];
  } else if (dim == 2) {
    return [[MPSNNReduceColumnMean alloc] initWithDevice:device];
  } else if (dim == 1) {
    return [[MPSNNReduceFeatureChannelsMean alloc] initWithDevice:device];
  }
  return nil;
}

static Tensor wrapper_mean_dim(
    const Tensor& input,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  if (@available(iOS 11.3, *)) {
    MPSImage* X = imageFromTensor(input);
    auto imageSize = input.sizes().vec();
    TORCH_CHECK(imageSize.size() == 4);
    // TODO: [T87340633] Support reducing the batch dimension
    TORCH_CHECK(imageSize[0] == 1);
    auto mask = make_dim_mask(opt_dims, input.dim());
    MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
    MPSImage* Y = nil;
    if (opt_dims.has_value()) {
      auto dims = opt_dims.value();
      for (int dim : dims) {
        imageSize[dim] = 1;
        MPSNNReduceUnary* kernel = kernelForReducedDim(dim);
        if (kernel) {
          Y = createTemporaryImage(commandBuffer, imageSize);
          [kernel encodeToCommandBuffer:commandBuffer.buffer
                            sourceImage:X
                       destinationImage:Y];
          X = Y;
        }
      }
    }
    MetalTensorImplStorage mt{imageSize};
    mt.texture()->setCommandBuffer(commandBuffer);
    mt.texture()->setImage(Y);
    auto shape = DimVector(input.sizes());
    for (int dim = shape.size() - 1; dim >= 0; dim--) {
      if (mask[dim]) {
        if (keepdim) {
          shape[dim] = 1;
        } else {
          shape.erase(shape.begin() + dim);
        }
      }
    }
    auto output = makeTensor(std::move(mt), input.options()).view(shape);
    return output;
  } else {
    // TODO: [T87350528] Fallback to shader kernels for 10.0 users
    TORCH_CHECK(
        false, "MPSNNReduceUnary is only available on iOS 11.3 and above");
  }
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(wrapper_mean_dim));
};

}
}
}
