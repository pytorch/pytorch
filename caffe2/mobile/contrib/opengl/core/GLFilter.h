
#pragma once

#include "GLContext.h"
#include "GLTexture.h"
#include "arm_neon_support.h"

#include <functional>
#include <string>
#include <vector>

#define BINDING(variableName) (variableName = new binding{#variableName})
#define ATTRIBUTE(variableName, value) (variableName = new binding{#variableName, value})

class GLFilter {
 protected:
  const std::string kernel_name;
  GLuint program = 0;
  GLuint frameBuffer = 0;
  static constexpr int kMaxUniformBlocks = 12;
  GLuint uniformBlock[kMaxUniformBlocks] = {0};
  GLint blockSize[kMaxUniformBlocks]     = {0};
  bool frame_buffer_initialized = false;

  // glGetError() can be expensive, we should turn error checking off when we're done with debugging

  static constexpr bool check_opengl_errors = true;

public:
  typedef std::vector<std::pair<std::string, std::string>> replacements_t;

  struct binding {
    const std::string name;
    GLint location;
  };

  struct texture_attachment {
    const GLTexture* texture;
    const binding* uniform;
  };

  GLFilter(const std::string kernel_name,
           const std::string vertex_shader,
           const std::string fragment_shader,
           const std::vector<binding*> uniforms,
           const std::vector<binding*> uniform_blocks = {},
           const std::vector<binding*> attributes = {},
           const replacements_t& replacements = {});

  // TODO: The set and reset context need to be commented out for unit testing
  ~GLFilter() {
    releaseBuffers();
    deleteProgram();
    deleteBindings();
  }

  void throwRuntimeError(std::function<void(std::stringstream& errmsg)> error_formatter) const {
    std::stringstream errmsg;
    errmsg << kernel_name << ": ";
    error_formatter(errmsg);
    throw std::runtime_error(errmsg.str());
  }

  void checkGLError(std::function<void(std::stringstream& errmsg)> error_formatter) const {
    if (check_opengl_errors) {
      GLenum glError = glGetError();
      if (glError != GL_NO_ERROR) {
        throwRuntimeError([&](std::stringstream& errmsg) {
          error_formatter(errmsg);
          errmsg << ", " << glError;
        });
      }
    }
  }

  template <typename T>
  void attach_uniform_buffer(const binding* block,
                             GLuint bindingPoint, std::function<void(T*, size_t)> loader);

  void run(const std::vector<texture_attachment>& input,
           const std::vector<const GLTexture*>& output,
           std::function<void(void)> uniforms_initializer,
           int width,
           int height);

  void releaseBuffers();
  void deleteProgram();
  void deleteBindings();

  static const char* vertex_shader;

 private:
  const std::vector<binding*> uniforms_;
  const std::vector<binding*> uniform_blocks_;
  const std::vector<binding*> attributes_;

  std::string process_replacements(std::string source, const replacements_t& replacements) const;

  bool createProgram(const GLchar* vertSource, const GLchar* fragSource, GLuint* program) const;

  GLint compileShader(GLenum target, GLsizei count, const GLchar** sources, GLuint* shader) const;
  GLint linkProgram(GLuint program) const;
  GLint validateProgram(GLuint program) const;
};
