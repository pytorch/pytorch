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

#include "../core/GLContext.h"
#include "../core/GLTexture.h"
#include <unordered_map>

enum GL_Renderer { Adreno, Mali /*, PowerVR */ };

class AndroidGLContext : public GLContext {
 private:
  EGLContext _eglcontext;

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
  GL_Renderer get_platform();
};
