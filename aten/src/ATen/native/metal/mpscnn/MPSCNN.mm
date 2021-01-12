#import <ATen/native/metal/mpscnn/MPSCNN.h>

namespace at {
namespace native {
namespace metal {
namespace mpscnn {

auto divRoundUp(uint x, uint y) -> uint {
  return (x + y - 1) / y;
}

int computeMPSAlignOffset(int kernel, int pad) {
  // To set the offset, we can just match the top-left pixel (in the input
  // image, with negative values for padding) that we look at. For 3x3s1p1, we
  // look at the (-1, -1) pixel in the original impl. For 3x3s1p0, we look at
  // (0, 0) pixel. For 3x3s1p2, look at (-2, -2) MPSCNN always looks at
  // (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2)) Thus, we just
  // need to match this up.

  // For 3x3s1p1, offset should be (0, 0)
  // For 3x3s1p0, offset should be (1, 1)
  // For 3x3s1p2, offset should be (-1, -1)
  const int mps_offset = kernel / 2;
  const int c2_offset = pad;
  return mps_offset - c2_offset;
}

NSString* kernelFor(
    MPSImage* X,
    NSString* arrayKernel,
    NSString* nonArrayKernel) {
  if (X.featureChannels > 4 || X.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    MPSImage* im) {
  const auto threadsPerThreadgroup = MTLSizeMake(
      8 /* threadExecutionWidth */,
      4 /* maxThreadsPerThreadgroup / threadExecutionWidth */,
      1);
  const auto threadgroupsPerGrid = MTLSizeMake(
      divRoundUp(im.width, threadsPerThreadgroup.width),
      divRoundUp(im.height, threadsPerThreadgroup.height),
      im.numberOfImages * divRoundUp(im.featureChannels, 4));
  const auto threadsPerGrid = MTLSizeMake(
      im.width,
      im.height,
      im.numberOfImages * divRoundUp(im.featureChannels, 4));
  return {threadsPerThreadgroup, threadgroupsPerGrid, threadsPerGrid};
};

}
}
}
}
