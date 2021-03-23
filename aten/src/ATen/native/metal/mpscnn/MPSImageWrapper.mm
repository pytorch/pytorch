#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

using namespace at::native::metal;
@interface MPSImageWrapperTrampoline : NSObject<PTMetalCommandBufferDelegate>
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
  _imageWrapper = nil;
}

- (void)prepareForSynchronization {
  if (_imageWrapper) {
    _imageWrapper->prepareForSynchronization();
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
  _delegate = nil;
  _commandBuffer = nil;
  _image = nil;
}

void MPSImageWrapper::copyDataFromHost(const float* inputData) {
  TORCH_CHECK(inputData);
  _commandBuffer = [MetalCommandBuffer currentBuffer];
  [_commandBuffer addDelegate:_delegate];
  _image = createTemporaryImage(_commandBuffer, _textureSizes, inputData);
}

void MPSImageWrapper::copyDataToHost(float* hostData) {
  synchronize();
  TORCH_CHECK(_image && ![_image isTemporaryImage]);
  copyToHost(hostData, _image);
}

MPSImage* MPSImageWrapper::image() const {
  return _image;
}

void MPSImageWrapper::recycleImage() {
  if ([_image isTemporaryImage]) {
    [_image recycle];
    [_commandBuffer remove:(MPSTemporaryImage*)_image];
    [_commandBuffer removeDelegate:_delegate];
    _image = nil;
  }
}

void MPSImageWrapper::setCommandBuffer(MetalCommandBuffer* cb) {
  _commandBuffer = cb;
  [_commandBuffer addDelegate:_delegate];
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
  TORCH_CHECK(commandBuffer)
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

void MPSImageWrapper::prepareForSynchronization() {
  // If the temporary image is still alive in the current command buffer,
  // make it a static image.
  if ([_image isTemporaryImage] && _image.readCount != 0) {
#if DEBUG
    NSLog(
        @"[MPSImageWrapper] Found a temporary image: [%lld, %lld, %lld, %lld]",
        (int64_t)_image.numberOfImages,
        (int64_t)_image.featureChannels,
        (int64_t)_image.height,
        (int64_t)_image.width);
#endif
    _image =
        createStaticImage((MPSTemporaryImage*)_image, _commandBuffer, false);
  }
}

void MPSImageWrapper::synchronize() {
  TORCH_CHECK(commandBuffer());
  [commandBuffer() synchronize];
}

}
}
}
