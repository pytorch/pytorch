
#include "GLFilter.h"
#include <sstream>

GLFilter::GLFilter(const std::string _kernel_name,
                   const std::string _vertex_shader,
                   const std::string _fragment_shader,
                   const std::vector<binding*> uniforms,
                   const std::vector<binding*> uniform_blocks,
                   const std::vector<binding*> attributes,
                   const replacements_t& _replacements)
    : kernel_name(_kernel_name),
      uniforms_(uniforms),
      uniform_blocks_(uniform_blocks),
      attributes_(attributes) {
  // shader program
  if (createProgram(_vertex_shader.c_str(),
                    process_replacements(_fragment_shader, _replacements).c_str(),
                    &program)) {
    gl_log(GL_VERBOSE, "created program %d\n", program);
  } else {
    releaseBuffers();

    throwRuntimeError(
        [&](std::stringstream& errmsg) { errmsg << "Problem initializing OpenGL program"; });
  }
}

const char* shader_utils = R"GLSL(
#define unpackHalf4x16(pd) vec4(unpackHalf2x16(pd.x), unpackHalf2x16(pd.y))
#define packHalf4x16(pd) uvec2(packHalf2x16(pd.xy), packHalf2x16(pd.zw))
)GLSL";

const char* half_float_texture_utils = R"GLSL(
precision mediump sampler2D;

#define TEXTURE_OUTPUT(_loc, _var) \
        layout(location = _loc) out mediump vec4 _var
#define TEXTURE_INPUT(_var) \
        uniform sampler2D _var
#define TEXTURE_LOAD(_input, _coord) \
        texelFetch((_input), (_coord), 0)
#define TEXTURE_STORE(_val) \
        (_val)
)GLSL";

const char* half_float_compat_texture_utils = R"GLSL(
precision highp usampler2D;

#define TEXTURE_OUTPUT(_loc, _var) \
        layout(location = _loc) out highp uvec2 _var
#define TEXTURE_INPUT(_var) \
        uniform usampler2D _var
#define TEXTURE_LOAD(_input, _coord) \
        unpackHalf4x16(texelFetch((_input), (_coord), 0).xy)
#define TEXTURE_STORE(_val) \
        (uvec2(packHalf4x16((_val))))
)GLSL";

std::string GLFilter::process_replacements(std::string shader,
                                           const replacements_t& replacements) const {
  for (auto&& replacement : replacements) {
    std::string tag = "$(" + replacement.first + ")";
    std::string value = replacement.second;

    size_t position = shader.find(tag);
    if (position != std::string::npos) {
      shader.replace(position, tag.size(), value);
    } else {
      throwRuntimeError(
          [&](std::stringstream& errmsg) { errmsg << "Couldn't find replacement tag: " << tag; });
    }
  }

  // Add some #defines for convenience
  std::string version_tag = "#version 300 es";
  if (GLContext::getGLContext()->halfFloatTextureSupported()) {
    shader.insert(shader.find(version_tag) + version_tag.size(), half_float_texture_utils);
  } else {
    shader.insert(shader.find(version_tag) + version_tag.size(), half_float_compat_texture_utils);
  }
  shader.insert(shader.find(version_tag) + version_tag.size(), shader_utils);
  return shader;
}

template <typename T>
void GLFilter::attach_uniform_buffer(const binding* block,
                                     GLuint bindingPoint,
                                     std::function<void(T*, size_t)> loader) {
  if (block->location >= 0) {
    if (bindingPoint < kMaxUniformBlocks) {
      if (uniformBlock[bindingPoint] == 0) {
        // Associate the uniform block index with a binding point
        glUniformBlockBinding(program, block->location, bindingPoint);

        // Get the size of block
        glGetActiveUniformBlockiv(program, block->location, GL_UNIFORM_BLOCK_DATA_SIZE, &blockSize[bindingPoint]);

        // Create and fill a buffer object
        glGenBuffers(1, &uniformBlock[bindingPoint]);

        gl_log(GL_VERBOSE, "created uniform buffer block %d\n", uniformBlock[bindingPoint]);
      }

      // Fill a buffer object
      glBindBuffer(GL_UNIFORM_BUFFER, uniformBlock[bindingPoint]);
      glBufferData(GL_UNIFORM_BUFFER, blockSize[bindingPoint], NULL, GL_DYNAMIC_DRAW);

      checkGLError([&](std::stringstream& errmsg) {
        errmsg << "Unable to bind uniform buffer " << block->name << ":" << block->location
               << " at binding point " << bindingPoint;
      });

      T* blockData = (T*)glMapBufferRange(
          GL_UNIFORM_BUFFER, 0, blockSize[bindingPoint], GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      if (blockData != NULL) {
        // Copy the data into the mapped buffer
        if (loader)
          loader(blockData, blockSize[bindingPoint]);

        // Unmap the buffer
        if (glUnmapBuffer(GL_UNIFORM_BUFFER) == GL_TRUE) {
          // Bind the buffer object to the uniform block binding point
          glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, uniformBlock[bindingPoint]);
        } else {
          throwRuntimeError([&](std::stringstream& errmsg) { errmsg << "Error unmapping element buffer object"; });
        }
      } else {
        throwRuntimeError([&](std::stringstream& errmsg) {
          errmsg << "Error mapping element buffer object, blockSize: " << blockSize;
        });
      }

      glBindBuffer(GL_UNIFORM_BUFFER, 0);
    } else {
      throwRuntimeError([&](std::stringstream& errmsg) {
        errmsg << "Uniform block binding point out of range: " << bindingPoint << ", should be < "
               << kMaxUniformBlocks;
      });
    }
  } else {
    throwRuntimeError([&](std::stringstream& errmsg) { errmsg << "unbound uniform block"; });
  }
}

template void GLFilter::attach_uniform_buffer<float16_t>(const binding* block,
                                                         GLuint bindingPoint,
                                                         std::function<void(float16_t*, size_t)> loader);

static const GLenum unused_capability[] = {GL_CULL_FACE,
                                           GL_BLEND,
                                           GL_DITHER,
                                           GL_STENCIL_TEST,
                                           GL_DEPTH_TEST,
                                           GL_SCISSOR_TEST,
                                           GL_POLYGON_OFFSET_FILL,
                                           GL_SAMPLE_ALPHA_TO_COVERAGE,
                                           GL_SAMPLE_COVERAGE};

void GLFilter::run(const std::vector<texture_attachment>& input,
                   const std::vector<const GLTexture*>& output,
                   std::function<void(void)> uniforms_initializer,
                   int width,
                   int height) {
  const int first_texture_id = GL_TEXTURE0;

  GLint defaultFramebuffer = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &defaultFramebuffer);

  gl_log(GL_VERBOSE,
         "GLFilter::run %s - inputs: %d, outputs: %d, width: %d, height: %d\n",
         kernel_name.c_str(),
         input.size(),
         output.size(),
         width,
         height);

  if (output.size() > 4) {
    throwRuntimeError([&](std::stringstream& errmsg) {
      errmsg << "Too many output textures: " << output.size() << ", should be <= 4";
    });
  }

  if (frameBuffer == 0) {
    // create the frame buffer
    glGenFramebuffers(1, &frameBuffer);
    gl_log(GL_VERBOSE, "created frame buffer %d\n", frameBuffer);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
  checkGLError([&](std::stringstream& errmsg) { errmsg << "glBindFramebuffer"; });

  // Set up the output textures
  for (int i = 0; i < output.size(); i++) {
    GLenum target = output[i]->target();
    GLuint texture = output[i]->name();

    glBindTexture(target, texture);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, target, texture, 0);

    checkGLError([&](std::stringstream& errmsg) {
      errmsg << "Unable to connect output texture " << texture << " at color attachment " << i;
    });

    gl_log(GL_VERBOSE, "connected output texture %d to color attachment %d\n", texture, i);
  }

  // Bind the output textures to the frame buffer attachments
  if (!frame_buffer_initialized) {
    const int attachments_number = output.size();
    const GLenum attachments[4] = {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};

    glDrawBuffers(attachments_number, attachments);

    int fbs = glCheckFramebufferStatus(GL_FRAMEBUFFER);

    if (fbs != GL_FRAMEBUFFER_COMPLETE) {
      throwRuntimeError(
          [&](std::stringstream& errmsg) { errmsg << "Frame buffer incomplete: " << fbs; });
    }

    frame_buffer_initialized = true;
  }

  glUseProgram(program);
  checkGLError([&](std::stringstream& errmsg) { errmsg << "glUseProgram"; });

  // Set up the input textures
  GLenum texture_idx = first_texture_id;
  for (int i = 0; i < input.size(); i++, texture_idx++) {
    if (input[i].uniform->location >= 0) {
      GLenum target = input[i].texture->target();
      GLuint texture = input[i].texture->name();

      glActiveTexture(texture_idx);
      glBindTexture(target, texture);
      glUniform1i(input[i].uniform->location, texture_idx - GL_TEXTURE0);

      checkGLError([&](std::stringstream& errmsg) {
        errmsg << ": Unable to attach input texture " << texture << " to uniform "
               << input[i].uniform->name << ":" << input[i].uniform->location << " at index "
               << texture_idx - GL_TEXTURE0;
      });

      gl_log(GL_VERBOSE,
             "connected input texture %d to texture unit %d\n",
             texture,
             texture_idx - GL_TEXTURE0);
    } else {
      gl_log(GL_VERBOSE, "something wrong happened when i = %d\n", i);
    }
  }

  // Caller supplied uniforms initializer
  if (uniforms_initializer) {
    uniforms_initializer();

    checkGLError([&](std::stringstream& errmsg) {
      errmsg << "errors in the uniforms initializer callback";
    });
  }

  // Validate program
  if (check_opengl_errors && !validateProgram(program)) {
    throwRuntimeError(
        [&](std::stringstream& errmsg) { errmsg << "Couldn't validate OpenGL program"; });
  }

  glViewport(0, 0, width, height);

  // Disable stuff we don't need and make sure that we have all the channels ebabled
  for (int i = 0; i < sizeof(unused_capability) / sizeof(GLenum); i++) {
    glDisable(unused_capability[i]);
  }
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  // glDrawElements should be more efficient, but on iOS glDrawArrays is faster.

  const bool useDrawArrays = true;

  if (useDrawArrays) {
    enum { ATTRIB_VERTEX, ATTRIB_TEXTUREPOSITON, NUM_ATTRIBUTES };

    static const GLfloat squareVertices[] = {
        -1.0f,
        -1.0f, // bottom left
        1.0f,
        -1.0f, // bottom right
        -1.0f,
        1.0f, // top left
        1.0f,
        1.0f, // top right
    };

    static const float textureVertices[] = {
        0.0f,
        0.0f, // bottom left
        1.0f,
        0.0f, // bottom right
        0.0f,
        1.0f, // top left
        1.0f,
        1.0f, // top right
    };

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, squareVertices);
    glEnableVertexAttribArray(ATTRIB_VERTEX);
    checkGLError(
        [&](std::stringstream& errmsg) { errmsg << "glEnableVertexAttribArray(ATTRIB_VERTEX)"; });

    glVertexAttribPointer(ATTRIB_TEXTUREPOSITON, 2, GL_FLOAT, 0, 0, textureVertices);
    glEnableVertexAttribArray(ATTRIB_TEXTUREPOSITON);
    checkGLError([&](std::stringstream& errmsg) {
      errmsg << "glEnableVertexAttribArray(ATTRIB_TEXTUREPOSITON)";
    });

    gl_log(GL_VERBOSE, "Calling glDrawArrays\n");
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    checkGLError([&](std::stringstream& errmsg) { errmsg << "glDrawArrays"; });
  } else {
    // Run the shaders on the output geometry
    static const GLfloat vVertices[] = {
        -1.0f, -1.0f, 0.0f, // Position 0
        0.0f,  0.0f, // TexCoord 0
        -1.0f, 1.0f,  0.0f, // Position 1
        0.0f,  1.0f, // TexCoord 1
        1.0f,  1.0f,  0.0f, // Position 2
        1.0f,  1.0f, // TexCoord 2
        1.0f,  -1.0f, 0.0f, // Position 3
        1.0f,  0.0f // TexCoord 3
    };
    static const GLushort indices[] = {0, 1, 2, 0, 2, 3};

    // Load the vertex position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), vVertices);
    // Load the texture coordinate
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), &vVertices[3]);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    gl_log(GL_VERBOSE, "Calling glDrawElements\n");
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);

    checkGLError([&](std::stringstream& errmsg) { errmsg << "glDrawElements"; });
  }

#if CAFFE2_ANDROID
  glFlush();
#endif

  // Unbind the current texture - Man, this is expensive!
  for (int i = texture_idx - 1; i >= first_texture_id; i--) {
    gl_log(GL_VERBOSE, "unbinding texture unit %d\n", i - GL_TEXTURE0);
    glActiveTexture(i);
    glBindTexture(GL_TEXTURE_2D, 0);

    checkGLError([&](std::stringstream& errmsg) {
      errmsg << "Error unbinding texture unit " << i - GL_TEXTURE0;
    });
  }

  glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebuffer);
}

void GLFilter::releaseBuffers() {
  for (int i = 0; i < kMaxUniformBlocks; i++) {
    if (uniformBlock[i]) {
      gl_log(GL_VERBOSE, "deleting uniform buffer block %d\n", uniformBlock[i]);
      glDeleteBuffers(1, &uniformBlock[i]);
      uniformBlock[i] = 0;
    }
  }
  if (frameBuffer) {
    gl_log(GL_VERBOSE, "deleting frame buffer %d\n", frameBuffer);
    glDeleteFramebuffers(1, &frameBuffer);
    frameBuffer = 0;
  }
}

void GLFilter::deleteProgram() {
  if (program) {
    gl_log(GL_VERBOSE, "deleting program %d\n", program);
    glDeleteProgram(program);
    program = 0;
  }
}

void GLFilter::deleteBindings() {
  for (binding* uniform : uniforms_) {
    delete uniform;
  }
  for (binding* uniform_block : uniform_blocks_) {
    delete uniform_block;
  }
  for (binding* attribute : attributes_) {
    delete attribute;
  }
}

// Simple vertex shader setting up the coordinates system
const char* GLFilter::vertex_shader = R"GLSL(#version 300 es

  layout(location = 0) in vec4 a_position;
  layout(location = 1) in vec2 a_texCoord;
  out vec2 v_texCoord;

  void main()
  {
     gl_Position = a_position;
     v_texCoord = a_texCoord;
  }
)GLSL";

bool GLFilter::createProgram(const GLchar* vertSource,
                             const GLchar* fragSource,
                             GLuint* program) const {
  GLuint vertShader = 0, fragShader = 0, prog = 0, status = 1;

  // Clear the error state. We check error state later in the function and
  // want to capture only errors in filter program initialization.
  glGetError();

  // Create shader program
  prog = glCreateProgram();

  // Create and compile vertex shader
  status *= compileShader(GL_VERTEX_SHADER, 1, &vertSource, &vertShader);

  // Create and compile fragment shader
  status *= compileShader(GL_FRAGMENT_SHADER, 1, &fragSource, &fragShader);

  // Attach vertex shader to program
  glAttachShader(prog, vertShader);

  // Attach fragment shader to program
  glAttachShader(prog, fragShader);

  // Bind attribute locations
  // This needs to be done prior to linking
  for (auto&& attribute : attributes_) {
    glBindAttribLocation(prog, attribute->location, attribute->name.c_str());

    checkGLError([&](std::stringstream& errmsg) {
      errmsg << "Couldn't bind attribute: " << attribute->name << " at location "
             << attribute->location;
    });
  }

  // Link program
  status *= linkProgram(prog);

  // Get locations of uniforms
  if (status) {
    for (auto&& uniform : uniforms_) {
      uniform->location = glGetUniformLocation(prog, uniform->name.c_str());

      checkGLError([&](std::stringstream& errmsg) {
        errmsg << "Couldn't resolve uniform: " << uniform->name;
      });
    }

    for (auto&& uniform_block : uniform_blocks_) {
      uniform_block->location = glGetUniformBlockIndex(prog, uniform_block->name.c_str());
      gl_log(GL_VERBOSE,
             "Getting location for uniform block: %s, location: %d\n",
             uniform_block->name.c_str(),
             uniform_block->location);

      checkGLError([&](std::stringstream& errmsg) {
        errmsg << "Couldn't resolve uniform block: " << uniform_block->name;
      });
    }

    *program = prog;
  }

  // Release vertex and fragment shaders
  if (vertShader) {
    glDetachShader(prog, vertShader);
    glDeleteShader(vertShader);
  }
  if (fragShader) {
    glDetachShader(prog, fragShader);
    glDeleteShader(fragShader);
  }

  return status == 1;
}

#include <stdlib.h>

/* Compile a shader from the provided source(s) */
GLint GLFilter::compileShader(GLenum target,
                              GLsizei count,
                              const GLchar** sources,
                              GLuint* shader) const {
  GLint status = 1;

  *shader = glCreateShader(target);
  glShaderSource(*shader, count, sources, NULL);
  glCompileShader(*shader);

  GLint logLength = 0;
  glGetShaderiv(*shader, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0) {
    std::vector<GLchar> log(logLength);
    glGetShaderInfoLog(*shader, logLength, &logLength, &log[0]);
    gl_log(GL_ERR, "Shader compile log:\n%s", &log[0]);
  }

  glGetShaderiv(*shader, GL_COMPILE_STATUS, &status);
  if (status == 0) {
    int i;

    gl_log(GL_ERR, "Failed to compile shader:\n");
    for (i = 0; i < count; i++)
      gl_log(GL_ERR, "%s", sources[i]);
  }

  return status;
}

/* Link a program with all currently attached shaders */
GLint GLFilter::linkProgram(GLuint program) const {
  GLint status = 1;

  glLinkProgram(program);

  GLint logLength = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0) {
    std::vector<GLchar> log(logLength);
    glGetProgramInfoLog(program, logLength, &logLength, &log[0]);
    gl_log(GL_ERR, "Program link log:\n%s", &log[0]);
  }

  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status == 0)
    gl_log(GL_ERR, "Failed to link program %d\n", program);

  return status;
}

/* Validate a program (for i.e. inconsistent samplers) */
GLint GLFilter::validateProgram(GLuint program) const {
  GLint status = 1;

  glValidateProgram(program);

  GLint logLength = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0) {
    std::vector<GLchar> log(logLength);
    glGetProgramInfoLog(program, logLength, &logLength, &log[0]);
    gl_log(GL_ERR, "Program validate log:\n%s", &log[0]);
  }

  glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
  if (status == 0)
    gl_log(GL_ERR, "Failed to validate program %d\n", program);

  return status;
}
