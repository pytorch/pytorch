
#pragma once

#include "GLImageAllocator.h"

namespace caffe2 {

template <class T>
class ImageAllocator {
  GLImageAllocator<T>* glImageAllocator;

 public:
  ImageAllocator() : glImageAllocator(GLImageAllocator<T>::newGLImageAllocator()) {}

  virtual ~ImageAllocator() { delete glImageAllocator; }

  GLImageVector<T>* newImage(
      int num_images, int width, int height, int channels, bool is_output = false) {
    const int tile_x = 1, tile_y = 1;
    return glImageAllocator->newImage(
        num_images, width, height, channels, tile_x, tile_y, is_output);
  }

  GLImageVector<T>* newImage(int num_images,
                             int width,
                             int height,
                             int channels,
                             int tile_x,
                             int tile_y,
                             bool is_output = false) {
    return glImageAllocator->newImage(
        num_images, width, height, channels, tile_x, tile_y, is_output);
  }

  GLImageVector<T>* newImage(
      int num_images,
      int width,
      int height,
      int channels,
      int tile_x,
      int tile_y,
      std::function<const GLTexture*(const int width, const int height)> textureAllocator) {
    return glImageAllocator->newImage(
        num_images, width, height, channels, tile_x, tile_y, textureAllocator);
  }
};
} // namespace caffe2
