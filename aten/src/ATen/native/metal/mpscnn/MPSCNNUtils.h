#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at {
namespace native {
namespace metal {
namespace mpscnn {

struct LaunchParams {
  MTLSize threadsPerThreadgroup;
  MTLSize threadgroupsPerGrid;
  MTLSize threadsPerGrid; // iOS 11.0
};

API_AVAILABLE(ios(10.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    MPSImage* im);

API_AVAILABLE(ios(10.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    NSUInteger numberOfImages,
    NSUInteger featureChannels,
    NSUInteger height,
    NSUInteger width);

API_AVAILABLE(ios(10.0), macos(10.13))
static inline NSString* kernelFor(
    MPSImage* image,
    NSString* arrayKernel,
    NSString* nonArrayKernel) {
  if (image.featureChannels > 4 || image.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

static inline int computeMPSAlignOffset(int kernel, int pad) {
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
  const int pt_offset = pad;
  return mps_offset - pt_offset;
}

}
} // namespace metal
} // namespace native
} // namespace at
