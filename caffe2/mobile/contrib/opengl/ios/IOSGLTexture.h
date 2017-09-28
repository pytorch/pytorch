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

#include "../core/GLContext.h"
#include "../core/GLTexture.h"

#import <CoreVideo/CoreVideo.h>

class IOSGLTexture : public GLTexture {
  CVOpenGLESTextureRef textureRef;

  IOSGLTexture(const Type& type,
               CVOpenGLESTextureCacheRef textureCache,
               CVPixelBufferRef sourceImage,
               GLint _filter = GL_NEAREST,
               GLint _wrap = GL_CLAMP_TO_EDGE);

  friend class IOSGLContext;

 public:
  const CVPixelBufferRef sourceImage;

  ~IOSGLTexture() { CFRelease(textureRef); }

  void map_buffer(std::function<void(void* buffer,
                                     size_t width,
                                     size_t height,
                                     size_t stride,
                                     size_t channels,
                                     const Type& type)> process) const;

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

  GLuint name() const { return CVOpenGLESTextureGetName(textureRef); }
  GLenum target() const { return CVOpenGLESTextureGetTarget(textureRef); };
  bool flipped() const { return CVOpenGLESTextureIsFlipped(textureRef); };

  static CVPixelBufferRef createCVPixelBuffer(OSType pixelType, int32_t width, int32_t height);
};
