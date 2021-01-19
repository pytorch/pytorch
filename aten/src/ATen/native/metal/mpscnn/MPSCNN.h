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
NSString* kernelFor(
    MPSImage* image,
    NSString* arrayKernel,
    NSString* nonArrayKernel);

int computeMPSAlignOffset(int kernel, int pad);

}
} // namespace metal
} // namespace native
} // namespace at
