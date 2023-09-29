#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>

namespace at {
namespace native {
namespace metal {
namespace mpscnn {

static auto divRoundUp(uint x, uint y) -> uint {
  return (x + y - 1) / y;
}

LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    MPSImage* im) {
  return spatialPointwiseKernelLaunchParams(
      pipeline, im.numberOfImages, im.featureChannels, im.height, im.width);
};

LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    NSUInteger numberOfImages,
    NSUInteger featureChannels,
    NSUInteger height,
    NSUInteger width) {
  const auto threadsPerThreadgroup = MTLSizeMake(
      8 /* threadExecutionWidth */,
      4 /* maxThreadsPerThreadgroup / threadExecutionWidth */,
      1);
  const auto threadgroupsPerGrid = MTLSizeMake(
      divRoundUp(width, threadsPerThreadgroup.width),
      divRoundUp(height, threadsPerThreadgroup.height),
      numberOfImages * divRoundUp(featureChannels, 4));
  const auto threadsPerGrid = MTLSizeMake(
      width, height, numberOfImages * divRoundUp(featureChannels, 4));
  return {threadsPerThreadgroup, threadgroupsPerGrid, threadsPerGrid};
};

}
}
}
}
