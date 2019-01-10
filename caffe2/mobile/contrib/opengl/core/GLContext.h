
#pragma once
#include "GLTexture.h"
#include "caffe2/core/common.h"
#include <functional>

class GLContext {
 private:
  static std::unique_ptr<GLContext> _glcontext;
  std::function<const GLTexture*(const int width, const int height)> foreignTextureAllocator =
      nullptr;

 protected:
  bool half_float_supported = true;

 public:
  virtual void set_context() = 0;
  virtual void reset_context() = 0;
  virtual void flush_context() = 0;
  virtual ~GLContext(){};

  static void initGLContext();
  static GLContext* getGLContext();
  static void deleteGLContext();

  static bool GL_EXT_texture_border_clamp_defined();

  inline bool halfFloatTextureSupported() { return half_float_supported; }

  void setTextureAllocator(
      std::function<const GLTexture*(const int width, const int height)> textureAllocator) {
    foreignTextureAllocator = textureAllocator;
  }

  std::function<const GLTexture*(const int width, const int height)> getTextureAllocator() {
    return foreignTextureAllocator;
  }
};

bool supportOpenGLES3(bool* hfs = nullptr);

bool isSupportedDevice();

#if CAFFE2_IOS
int iPhoneVersion();
#endif
