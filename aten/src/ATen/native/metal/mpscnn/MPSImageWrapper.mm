#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNN.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

#include <numeric>

namespace at {
namespace native {
namespace metal {

std::vector<int64_t> textureSizeFromSizes(IntArrayRef sizes, TextureType type) {
  if (sizes.size() == 2) {
    if (type == TextureType::TextureType2DArray) {
      return {sizes[0], sizes[1], 1, 1};
    } else if (type == TextureType::TextureType2D) {
      return {1, 1, sizes[0], sizes[1]};
    } else {
      return {};
    }
  }
  return sizes.vec();
}
MPSImageWrapper::MPSImageWrapper(IntArrayRef sizes) {
  _textureSizes = textureSizeFromSizes(sizes, TextureType::TextureType2D);
}

void MPSImageWrapper::copyDataFromHost(const float* inputData) {
  TORCH_CHECK(inputData);
  TORCH_CHECK(_textureSizes.size() == 4);
  _commandBuffer = [MetalCommandBuffer currentBuffer];
  _image = [MPSImage temporaryImageFromHost:inputData
                                      Sizes:_textureSizes
                              CommandBuffer:_commandBuffer];
}

void MPSImageWrapper::copyDataToHost(float* hostData) {
  TORCH_CHECK(_image);
  synchronize();
  [MPSImage copyToHost:hostData FromImage:_image];
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

TextureType MPSImageWrapper::textureType() const {
  if (!_image) {
    return TextureType::TextureNone;
  }
  MTLTextureType textureType = _image.textureType;
  if (textureType == MTLTextureType2D) {
    return TextureType::TextureType2D;
  } else if (textureType == MTLTextureType2DArray) {
    return TextureType::TextureType2DArray;
  }
  return TextureType::TextureNone;
}

void MPSImageWrapper::allocateTextureStorage(IntArrayRef sizes) {
  _textureSizes = sizes.vec();
  _image = [MPSImage imageFromSize:_textureSizes];
}

void MPSImageWrapper::allocateTemporaryTextureStorage(
    IntArrayRef sizes,
    MetalCommandBuffer* commandBuffer) {
  TORCH_CHECK(commandBuffer)
  _textureSizes = sizes.vec();
  _commandBuffer = commandBuffer;
  _image = [MPSImage temporaryImageFromSize:_textureSizes
                              commandBuffer:commandBuffer];
}

void MPSImageWrapper::copyFromTexture(MPSImage* image) {
  if ([image isTemporaryImage]) {
    _image = [MPSImage temporaryImageFromImage:image
                                 CommandBuffer:_commandBuffer];
  } else {
    _image = [MPSImage imageFromImage:image];
  }
}

void MPSImageWrapper::synchronize() {
  if ([_image isTemporaryImage]) {
    _image = [MPSImage imageFromTemporaryImage:(MPSTemporaryImage*)_image
                                 CommandBuffer:_commandBuffer
                            waitUntilCompleted:NO];
  }
  [_commandBuffer synchronize];
  _commandBuffer = nil;
}

}
}
}
