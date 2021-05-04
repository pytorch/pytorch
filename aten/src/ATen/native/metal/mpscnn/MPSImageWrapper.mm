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
  _textureSizes = computeTextureSize(sizes);
  _delegate = [MPSImageWrapperTrampoline newWithMPSImageWrapper:this];
}

MPSImageWrapper::~MPSImageWrapper() {
  release();
}

void MPSImageWrapper::copyDataFromHost(const float* inputData) {
  TORCH_CHECK(inputData);
  _commandBuffer = [MetalCommandBuffer currentBuffer];
  [_commandBuffer addSubscriber:_delegate];
  _image = createTemporaryImage(_commandBuffer, _textureSizes, inputData);
}

void MPSImageWrapper::copyDataToHost(float* hostData) {
  TORCH_CHECK(_image);
  synchronize();
  TORCH_CHECK(_image && !_image.isTemporaryImage);
  copyToHost(hostData, _image);
}

MPSImage* MPSImageWrapper::image() const {
  return _image;
}

void MPSImageWrapper::setCommandBuffer(MetalCommandBuffer* commandBuffer) {
  TORCH_CHECK(commandBuffer && commandBuffer.valid);
  _commandBuffer = commandBuffer;
  [_commandBuffer addSubscriber:_delegate];
}
MetalCommandBuffer* MPSImageWrapper::commandBuffer() const {
  return _commandBuffer;
}

IntArrayRef MPSImageWrapper::textureSizes() const {
  return _textureSizes;
}

void MPSImageWrapper::allocateTextureStorage(IntArrayRef sizes) {
  _textureSizes = computeTextureSize(sizes);
  _image = createStaticImage(_textureSizes);
}

void MPSImageWrapper::allocateTemporaryTextureStorage(
    IntArrayRef sizes,
    MetalCommandBuffer* commandBuffer) {
  setCommandBuffer(commandBuffer);
  _textureSizes = computeTextureSize(sizes);
  _image = createTemporaryImage(commandBuffer, _textureSizes);
}

void MPSImageWrapper::copyFromTexture(MPSImage* image) {
  if ([image isTemporaryImage]) {
    _image = createTemporaryImage(_commandBuffer, image);
  } else {
    _image = createStaticImage(image);
  }
}

void MPSImageWrapper::setTexture(MPSImage* image) {
  TORCH_CHECK(image);
  if(image.isTemporaryImage) {
    TORCH_CHECK(_commandBuffer && _commandBuffer.valid);
  }
  _image = image;
}

void MPSImageWrapper::prepare() {
  // If the temporary image is still alive in the current command buffer,
  // make it a static image.
  if (_image.isTemporaryImage && _image.readCount != 0) {
    _image =
        createStaticImage((MPSTemporaryImage*)_image, _commandBuffer, false);
  }
}

void MPSImageWrapper::synchronize() {
  if(_commandBuffer && _commandBuffer.valid) {
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
}

}
}
}
