
#pragma once

#include "../core/GLContext.h"
#include "../core/GLTexture.h"

#import <CoreVideo/CoreVideo.h>

class IOSGLContext : public GLContext {
  void* oglContext;
  void* oldContext;
  CVOpenGLESTextureCacheRef textureCache;

 public:
  IOSGLContext();
  ~IOSGLContext();

  const GLTexture* createNewTexture(CVPixelBufferRef pixelBuffer, const GLTexture::Type& type);
  void set_context();
  void reset_context();
  void flush_context();
};
