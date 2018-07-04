
#include "GLPBO.h"

#include "caffe2/core/logging.h"

GLPBO::~GLPBO() {
  if (pboId != 0) {
    gl_log(GL_LOG, "deleting PBO buffer %d\n", pboId);
    glDeleteBuffers(1, &pboId);
    pboId = 0;
  }
  if (pboFrameBuffer != 0) {
    gl_log(GL_LOG, "deleting PBO frame buffer %d\n", pboFrameBuffer);
    glDeleteFramebuffers(1, &pboFrameBuffer);
    pboFrameBuffer = 0;
  }
}

GLPBO* GLPBO::pboContext = NULL;

GLPBO* GLPBO::getContext() {
  if (pboContext == NULL) {
    pboContext = new GLPBO();
  }
  return pboContext;
}

void GLPBO::mapTextureData(GLuint _textureId,
                           GLsizei _width,
                           GLsizei _height,
                           GLsizei _stride,
                           GLsizei _channels,
                           const GLTexture::Type& _type,
                           std::function<void(const void* buffer,
                                              size_t width,
                                              size_t height,
                                              size_t stride,
                                              size_t channels,
                                              const GLTexture::Type& type)> process) {
  GLint defaultFramebuffer = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &defaultFramebuffer);

  if (pboFrameBuffer == 0) {
    glGenFramebuffers(1, &pboFrameBuffer);
    gl_log(GL_VERBOSE, "created PBO frame buffer %d\n", pboFrameBuffer);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, pboFrameBuffer);

  glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _textureId, 0);

  int fbs = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (fbs != GL_FRAMEBUFFER_COMPLETE) {
    std::stringstream errmsg;
    errmsg << ": Frame buffer incomplete: " << fbs;
    throw std::runtime_error(errmsg.str());
  }

  if (pboId == 0) {
    glGenBuffers(1, &pboId);
    gl_log(GL_VERBOSE, "created PBO buffer %d\n", pboId);
  }
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pboId);

  size_t buffer_size = _stride * _height * _channels * _type.dataSize();

  if (buffer_size > pboSize) {
    LOG(INFO) << "Allocating PBO of capacity " << buffer_size;

    glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, NULL, GL_DYNAMIC_READ);
    pboSize = buffer_size;
  }

  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, _stride, _height, _type.format, _type.type, 0);

  GLhalf* ptr = (GLhalf*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, buffer_size, GL_MAP_READ_BIT);

  if (ptr) {
    process(ptr, _width, _height, _stride, _channels, _type);
  } else {
    std::stringstream errmsg;
    errmsg << ": glMapBufferRange using PBO incomplete";
    throw std::runtime_error(errmsg.str());
  }

  // Unmap buffer
  glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  // Bind to the default FrameBuffer
  glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
}
