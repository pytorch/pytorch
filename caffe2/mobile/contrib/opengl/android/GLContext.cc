
#include "AndroidGLContext.h"

std::unique_ptr<GLContext> GLContext::_glcontext = nullptr;

void GLContext::initGLContext() {
  if (_glcontext == nullptr) {
    _glcontext.reset(new AndroidGLContext());
  }
}

GLContext* GLContext::getGLContext() {
  if (_glcontext == nullptr) {
    initGLContext();
  }
  return _glcontext.get();
}

void GLContext::deleteGLContext() { _glcontext.reset(nullptr); }
