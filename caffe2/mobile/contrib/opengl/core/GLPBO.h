
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
