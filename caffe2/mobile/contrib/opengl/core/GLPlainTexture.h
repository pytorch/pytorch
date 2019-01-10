
#pragma once

#include "GLContext.h"
#include "GLTexture.h"

class GLPlainTexture : public GLTexture {
 private:
  bool isOwner = true;

 public:
  GLPlainTexture(const Type& type,
                 const void* input,
                 GLsizei width,
                 GLsizei height,
                 bool use_padding = false,
                 GLint filter = GL_NEAREST,
                 GLint wrap = GL_CLAMP_TO_EDGE);

  GLPlainTexture(const Type& type,
                 const GLuint textureID,
                 GLsizei width,
                 GLsizei height,
                 bool use_padding = false,
                 GLint filter = GL_NEAREST,
                 GLint wrap = GL_CLAMP_TO_EDGE);

  ~GLPlainTexture() {
    if (glIsTexture(_textureId)) {
      if (isOwner) {
        gl_log(GL_VERBOSE, "~GLPlainTexture() - deleting texture %d\n", _textureId);
        glDeleteTextures(1, &_textureId);
      }
    } else {
      gl_log(GL_ERR, "not deleting texture %d\n", _textureId);
    }
  }

  GLuint name() const { return _textureId; };

  GLenum target() const { return GL_TEXTURE_2D; };

  bool flipped() const { return false; };
};
