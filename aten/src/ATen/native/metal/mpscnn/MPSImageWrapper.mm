#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

using namespace at::native::metal;
@interface MPSImageWrapperTrampoline : NSObject<PTMetalCommandBuffer>
+ (instancetype)newWithMPSImageWrapper:(MPSImageWrapper*)wrapper;
@end

@implementation MPSImageWrapperTrampoline {
  MPSImageWrapper* _imageWrapper;
}

+ (instancetype)newWithMPSImageWrapper:(MPSImageWrapper*)wrapper {
  MPSImageWrapperTrampoline* trampoline = [MPSImageWrapperTrampoline new];
  trampoline->_imageWrapper = wrapper;
  return trampoline;
}

- (void)dealloc {
  _imageWrapper = nullptr;
}

- (void)beginSynchronization {
  if (_imageWrapper) {
    _imageWrapper->prepare();
  }
}

- (void)endSynchronization:(NSError*)error {
  if (error) {
    if (_imageWrapper) {
      _imageWrapper->release();
    }
    // throw exceptions if we failed to flush the command buffer
    TORCH_CHECK(error);
  }
}

@end

namespace at {
namespace native {
namespace metal {

MPSImageWrapper::MPSImageWrapper(IntArrayRef sizes) {
  _imageSizes = computeImageSize(sizes);
  _delegate = [MPSImageWrapperTrampoline newWithMPSImageWrapper:this];
}

MPSImageWrapper::~MPSImageWrapper() {
  release();
}

void MPSImageWrapper::copyDataFromHost(const float* inputData) {
  TORCH_CHECK(inputData);
  _commandBuffer = [MetalCommandBuffer currentBuffer];
  [_commandBuffer addSubscriber:_delegate];
  _image = createTemporaryImage(_commandBuffer, _imageSizes, inputData);
}

void MPSImageWrapper::copyDataToHost(float* hostData) {
  TORCH_CHECK(_image);
  synchronize();
  TORCH_CHECK(_buffer);
  memcpy(hostData, _buffer.contents, _buffer.length);
}

MPSImage* MPSImageWrapper::image() const {
  return _image;
}

id<MTLBuffer> MPSImageWrapper::buffer() const {
  return _buffer;
}

void MPSImageWrapper::setCommandBuffer(MetalCommandBuffer* commandBuffer) {
  TORCH_CHECK(commandBuffer && commandBuffer.valid);
  _commandBuffer = commandBuffer;
  [_commandBuffer addSubscriber:_delegate];
}

MetalCommandBuffer* MPSImageWrapper::commandBuffer() const {
  return _commandBuffer;
}

void MPSImageWrapper::allocateStorage(IntArrayRef sizes) {
  _imageSizes = computeImageSize(sizes);
  _image = createStaticImage(_imageSizes);
}

void MPSImageWrapper::allocateTemporaryStorage(
    IntArrayRef sizes,
    MetalCommandBuffer* commandBuffer) {
  setCommandBuffer(commandBuffer);
  _imageSizes = computeImageSize(sizes);
  _image = createTemporaryImage(commandBuffer, _imageSizes);
}

void MPSImageWrapper::setImage(MPSImage* image) {
  TORCH_CHECK(image);
  if (image.isTemporaryImage) {
    TORCH_CHECK(_commandBuffer && _commandBuffer.valid);
  }
  _image = image;
}

void MPSImageWrapper::prepare() {
  if (!_buffer) {
    int64_t size_bytes = c10::multiply_integers([_image sizes]) * sizeof(float);
    _buffer = [[MPSCNNContext sharedInstance].device
        newBufferWithLength:size_bytes
                    options:MTLResourceCPUCacheModeWriteCombined];
  }
  copyToMetalBuffer(_commandBuffer, _buffer, _image);
}

void MPSImageWrapper::synchronize() {
  if (_commandBuffer && _commandBuffer.valid) {
    [_commandBuffer commit];
  }
}

void MPSImageWrapper::release() {
  if ([_image isTemporaryImage]) {
    [_image recycle];
    [_commandBuffer remove:(MPSTemporaryImage*)_image];
  }
  [_commandBuffer removeSubscriber:_delegate];
  _delegate = nil;
  _commandBuffer = nil;
  _image = nil;
  _buffer = nil;
}

}
}
}
