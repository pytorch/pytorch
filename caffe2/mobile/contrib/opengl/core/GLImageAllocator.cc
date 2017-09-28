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


#include "GLImageAllocator.h"
#include "arm_neon_support.h"

template <class T>
GLImageVector<T>* GLImageAllocator<T>::newImage(
    int num_images, int width, int height, int channels, int tile_x, int tile_y, bool is_output) {
  GLImageVector<T>* images =
      new GLImageVector<T>(num_images, width, height, channels, tile_x, tile_y);
  for (int i = 0; i < num_images; i++) {
    images->push_back(
        new GLImage<T>(width, height, channels, tile_x, tile_y, [&](int slice) -> const GLTexture* {
          bool usePadding = is_output;
          return new GLPlainTexture(type, nullptr, width * tile_x, height * tile_y, usePadding);
        }));
  }
  return images;
}

template <class T>
GLImageVector<T>* GLImageAllocator<T>::newImage(
    int num_images,
    int width,
    int height,
    int channels,
    int tile_x,
    int tile_y,
    std::function<const GLTexture*(const int width, const int height)> textureAllocator) {
  GLImageVector<T>* images =
      new GLImageVector<T>(num_images, width, height, channels, tile_x, tile_y);
  for (int i = 0; i < num_images; i++) {
    images->push_back(
        new GLImage<T>(width, height, channels, tile_x, tile_y, [&](int slice) -> const GLTexture* {
          return textureAllocator(width, height);
        }));
  }
  return images;
}

template <class T>
GLImageVector<T>* GLImageAllocator<T>::ShareTexture(const GLuint textureID,
                                                    int num_images,
                                                    int width,
                                                    int height,
                                                    int channels,
                                                    int tile_x,
                                                    int tile_y) {
  GLImageVector<T>* images =
      new GLImageVector<T>(num_images, width, height, channels, tile_x, tile_y);
  for (int i = 0; i < num_images; i++) {
    images->push_back(
        new GLImage<T>(width, height, channels, tile_x, tile_y, [&](int slice) -> const GLTexture* {
          return new GLPlainTexture(
              GLImageAllocator<T>::type, textureID, width * tile_x, height * tile_y);
        }));
  }
  return images;
}

template <>
const GLTexture::Type& GLImageAllocator<float16_t>::type = GLTexture::FP16;
template <>
const GLTexture::Type& GLImageAllocator<uint8_t>::type = GLTexture::UI8;

template class GLImageAllocator<float16_t>;
template class GLImageAllocator<uint8_t>;
