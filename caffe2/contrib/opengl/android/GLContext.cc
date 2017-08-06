// Copyright 2004-present Facebook. All Rights Reserved.

#include "AndroidGLContext.h"

GLContext* GLContext::_glcontext = nullptr;

void GLContext::initGLContext() {
  if (_glcontext == nullptr) {
    _glcontext = new AndroidGLContext();
  }
}

GLContext* GLContext::getGLContext() {
  if (_glcontext == nullptr) {
    initGLContext();
  }
  return _glcontext;
}

void GLContext::deleteGLContext() {
  if (_glcontext != nullptr) {
    delete _glcontext;
  }
  _glcontext = nullptr;
}
