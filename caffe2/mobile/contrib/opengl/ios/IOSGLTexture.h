
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
