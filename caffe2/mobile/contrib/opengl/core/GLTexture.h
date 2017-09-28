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
#include "GL.h"
#include "GLLogging.h"

class GLTexture {
 public:
  struct Type {
    const GLenum internalFormat;
    const GLenum format;
    const GLenum type;

    int dataSize() const {
      switch (type) {
      case GL_UNSIGNED_INT:
        return 4;
      case GL_HALF_FLOAT:
        return 2;
      case GL_UNSIGNED_BYTE:
        return 1;
      default:
        throw std::runtime_error("Unknown Texture Type");
      }
    }

    int channels() const {
      switch (format) {
      case GL_R8:
        return 1;
      case GL_RG8:
        return 2;
      // case GL_BGRA:
      case GL_RG_INTEGER:
      case GL_RGBA:
        return 4;
      default:
        throw std::runtime_error("Unknown Texture Format");
      }
    }
  };

  static const Type FP16;
  static const Type FP16_COMPAT;
  static const Type UI8;

 protected:
  const Type& _type;

  const GLsizei _width;
  const GLsizei _height;
  const GLsizei _stride;
  const GLsizei _channels;
  const bool _use_padding;

  GLint _filter;
  GLint _wrap;
  GLuint _textureId;

 public:
  GLTexture(const Type& type,
            int width,
            int height,
            int stride,
            bool use_padding,
            GLint filter,
            GLint wrap)
      : _type(type),
        _width(width),
        _height(height),
        _stride(stride),
        _channels(type.channels()),
        _use_padding(use_padding),
        _filter(filter),
        _wrap(wrap) {}

  GLTexture(const Type& type, int width, int height, bool use_padding, GLint filter, GLint wrap)
      : GLTexture(type,
                  width,
                  height,
                  use_padding ? (width + 7) / 8 * 8 : width,
                  use_padding,
                  filter,
                  wrap) {}

  virtual ~GLTexture() {}
  virtual GLuint name() const = 0;
  virtual GLenum target() const = 0;
  virtual bool flipped() const = 0;

  virtual void map_read(std::function<void(const void* buffer,
                                           size_t width,
                                           size_t height,
                                           size_t stride,
                                           size_t channels,
                                           const Type& type)> process) const;

  virtual void map_load(std::function<void(void* buffer,
                                           size_t width,
                                           size_t height,
                                           size_t stride,
                                           size_t channels,
                                           const Type& type)> process) const;

  void loadData(const void* pixels) const;
};
