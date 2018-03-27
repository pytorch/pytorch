
#include "GLPlainTexture.h"
#include "GLPBO.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"

#define half_float_supported (GLContext::getGLContext()->halfFloatTextureSupported())

#define FIXED_TYPE(_t) (((_t).type != GL_HALF_FLOAT || half_float_supported) ? (_t) : GLTexture::FP16_COMPAT)

GLPlainTexture::GLPlainTexture(
    const Type& type, const void* input, GLsizei width, GLsizei height, bool use_padding, GLint filter, GLint wrap)
    : GLTexture(FIXED_TYPE(type), width, height, use_padding, filter, wrap) {
  //  caffe2::Timer timer;
  //  timer.Start();
  glGenTextures(1, &_textureId);
  glBindTexture(GL_TEXTURE_2D, _textureId);
  glTexImage2D(GL_TEXTURE_2D, 0, _type.internalFormat, _stride, _height, 0, _type.format, _type.type, input);

  gl_log(
      GL_VERBOSE,
      "GLPlainTexture() - allocated textureId %d, internalFormat: 0x%X, format: 0x%X, type: 0x%X\n",
      _textureId,
      _type.internalFormat,
      _type.format,
      _type.type);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, _filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, _filter);

#if GL_EXT_texture_border_clamp
  GLfloat borderColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR_EXT, borderColor);
  // Set the texture to use the border clamp wrapping mode.
  _wrap = GL_CLAMP_TO_BORDER_EXT;
#endif

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _wrap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _wrap);

  glBindTexture(GL_TEXTURE_2D, 0);
  //  LOG(INFO) << "glTexImage2D takes " << timer.MilliSeconds() << " ms";
}

GLPlainTexture::GLPlainTexture(
    const Type& type, const GLuint textureID, GLsizei width, GLsizei height, bool use_padding, GLint filter, GLint wrap)
    : GLTexture(FIXED_TYPE(type), width, height, use_padding, filter, wrap) {
  _textureId = textureID;
  isOwner = false;
  gl_log(
      GL_VERBOSE,
      "GLPlainTexture() - wrapped textureId %d, internalFormat: 0x%X, format: 0x%X, type: 0x%X\n",
      _textureId,
      _type.internalFormat,
      _type.format,
      _type.type);
}
