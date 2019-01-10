
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
