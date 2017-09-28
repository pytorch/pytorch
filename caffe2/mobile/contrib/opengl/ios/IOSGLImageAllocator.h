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

#include "../core/GLImageAllocator.h"

#import <CoreVideo/CoreVideo.h>

template <class T>
class IOSGLImageAllocator : public GLImageAllocator<T> {
  static const GLTexture::Type& type;

  std::vector<CVPixelBufferRef> pixelbuffers;

 public:
  static const FourCharCode pixelFormat;

  IOSGLImageAllocator() : GLImageAllocator<T>() { gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__); }

  ~IOSGLImageAllocator() {
    gl_log(GL_VERBOSE, "%s\n", __PRETTY_FUNCTION__);

    for (auto&& pixelbuffer : pixelbuffers) {
      CFRelease(pixelbuffer);
    }
  }

  GLImageVector<T>* newImage(int num_images,
                             int width,
                             int height,
                             int channels,
                             int tile_x,
                             int tile_y,
                             bool useCVPixelBuffer);
};
