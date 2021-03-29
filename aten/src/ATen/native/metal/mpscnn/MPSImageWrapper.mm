#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

namespace at {
namespace native {
namespace metal {

MPSImageWrapper::MPSImageWrapper(IntArrayRef sizes) {
  _textureSizes = computeTextureSize(sizes);
}

void MPSImageWrapper::copyDataFromHost(const float* inputData) {
  TORCH_CHECK(inputData);
  _commandBuffer = [MetalCommandBuffer currentBuffer];
  _image = createTemporaryImage(_commandBuffer, _textureSizes, inputData);
}

void MPSImageWrapper::copyDataToHost(float* hostData) {
  TORCH_CHECK(_image);
  synchronize();
  copyToHost(hostData, _image);
}

MPSImage* MPSImageWrapper::image() const {
  return _image;
}

void MPSImageWrapper::recycleImage() {
  if ([_image isTemporaryImage]) {
    [_image recycle];
    [_commandBuffer remove:(MPSTemporaryImage*)_image];
  }
}

void MPSImageWrapper::setCommandBuffer(MetalCommandBuffer* cb) {
  _commandBuffer = cb;
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
  _textureSizes = computeTextureSize(sizes);
  _commandBuffer = commandBuffer;
  _image = createTemporaryImage(commandBuffer, _textureSizes);
}

void MPSImageWrapper::copyFromTexture(MPSImage* image) {
  if ([image isTemporaryImage]) {
    _image = createTemporaryImage(_commandBuffer, image);
  } else {
    _image = createStaticImage(image);
  }
}

void MPSImageWrapper::synchronize() {
  if ([_image isTemporaryImage]) {
    _image =
        createStaticImage((MPSTemporaryImage*)_image, _commandBuffer, false);
  }
  [_commandBuffer synchronize];
  _commandBuffer = nil;
}

}
}
}
