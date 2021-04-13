#ifndef MPSImageWrapper_h
#define MPSImageWrapper_h

#import <ATen/native/metal/MetalCommandBuffer.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace metal {

class API_AVAILABLE(ios(10.0), macos(10.13)) MPSImageWrapper {
 public:
  MPSImageWrapper(IntArrayRef sizes);
  operator bool() const {
    return _image;
  }
  void copyDataFromHost(const float* inputData);
  void copyDataToHost(float* hostData);
  void allocateTextureStorage(IntArrayRef sizes);
  void allocateTemporaryTextureStorage(
      IntArrayRef sizes,
      MetalCommandBuffer* commandBuffer);
  void copyFromTexture(MPSImage* image);
  void setCommandBuffer(MetalCommandBuffer* buffer);
  MetalCommandBuffer* commandBuffer() const;
  IntArrayRef textureSizes() const;
  MPSImage* image() const;
  void recycleImage();
  void synchronize();

 private:
  std::vector<int64_t> _textureSizes;
  MPSImage* _image = nullptr;
  MetalCommandBuffer* _commandBuffer;
};

} // namespace metal
} // namespace native
} // namespace at

#endif /* MPSImageWrapper_h */
