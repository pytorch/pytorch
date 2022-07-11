#ifndef MPSImageWrapper_h
#define MPSImageWrapper_h

#import <ATen/native/metal/MetalCommandBuffer.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace metal {

class API_AVAILABLE(ios(11.0), macos(10.13)) MPSImageWrapper {
 public:
  MPSImageWrapper(IntArrayRef sizes);
  ~MPSImageWrapper();
  void copyDataFromHost(const float* inputData);
  void copyDataToHost(float* hostData);
  void allocateStorage(IntArrayRef sizes);
  void allocateTemporaryStorage(
      IntArrayRef sizes,
      MetalCommandBuffer* commandBuffer);
  void setCommandBuffer(MetalCommandBuffer* buffer);
  MetalCommandBuffer* commandBuffer() const;
  void setImage(MPSImage* image);
  MPSImage* image() const;
  id<MTLBuffer> buffer() const;
  void synchronize();
  void prepare();
  void release();

 private:
  std::vector<int64_t> _imageSizes;
  MPSImage* _image = nil;
  id<MTLBuffer> _buffer = nil;
  MetalCommandBuffer* _commandBuffer = nil;
  id<PTMetalCommandBuffer> _delegate = nil;
};

} // namespace metal
} // namespace native
} // namespace at

#endif /* MPSImageWrapper_h */
