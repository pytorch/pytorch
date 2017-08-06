// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../core/GLContext.h"
#include "../core/GLTexture.h"
#include <unordered_map>

enum GL_Renderer { Adreno, Mali, PowerVR };

class AndroidGLContext : public GLContext {
 private:
  EGLContext _eglcontext;
  GL_Renderer _renderer;
  static std::unordered_map<std::string, GL_Renderer> _renderer_map;

  EGLContext create_opengl_thread_context();
  bool opengl_thread_context_exists();
  bool release_opengl_thread_context();

 public:
  AndroidGLContext();
  ~AndroidGLContext();
  void set_context();
  void reset_context();
  void flush_context();
  void init_gles3();
  void detect_platform();
  GL_Renderer get_platform();
};
