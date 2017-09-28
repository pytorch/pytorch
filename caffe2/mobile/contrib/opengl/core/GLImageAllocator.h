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

#include "GLImage.h"
#include "GLPlainTexture.h"

template <class T>
class GLImageAllocator {
 public:
  static const GLTexture::Type& type;

  GLImageAllocator() { gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__); }

  virtual ~GLImageAllocator() { gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__); }

  virtual GLImageVector<T>* newImage(
      int num_images, int width, int height, int channels, int tile_x, int tile_y, bool is_output);

  virtual GLImageVector<T>* newImage(
      int num_images,
      int width,
      int height,
      int channels,
      int tile_x,
      int tile_y,
      std::function<const GLTexture*(const int width, const int height)> textureAllocator);

  virtual GLImageVector<T>* ShareTexture(const GLuint textureID,
                                         int num_images,
                                         int width,
                                         int height,
                                         int channels,
                                         int tile_x = 1,
                                         int tile_y = 1);

  static GLImageAllocator<T>* newGLImageAllocator();
};
