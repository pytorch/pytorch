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
