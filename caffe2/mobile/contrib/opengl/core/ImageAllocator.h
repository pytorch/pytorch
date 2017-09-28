/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


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
