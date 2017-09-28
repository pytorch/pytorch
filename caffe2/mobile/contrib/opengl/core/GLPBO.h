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

#include "GLTexture.h"
#include <functional>

class GLPBO {
  GLuint pboId = 0;
  GLuint pboSize = 0;
  GLuint pboFrameBuffer = 0;

  ~GLPBO();

  static GLPBO* pboContext;

 public:
  void mapTextureData(GLuint _textureId,
                      GLsizei _width,
                      GLsizei _height,
                      GLsizei _stride,
                      GLsizei _channels,
                      const GLTexture::Type& type,
                      std::function<void(const void* buffer,
                                         size_t width,
                                         size_t height,
                                         size_t stride,
                                         size_t channels,
                                         const GLTexture::Type& type)> process);

  static GLPBO* getContext();
};
