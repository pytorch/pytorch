#include <stdio.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include <c10/util/Exception.h>

#include <ATen/native/vulkan/gl/GLES.h>
#include <ATen/native/vulkan/glsl.h>

#define GL_CHECK_ERROR                                        \
  {                                                           \
    GLenum error = glGetError();                              \
    TORCH_CHECK(error == GL_NO_ERROR, "GLES error: ", error); \
  }

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace gl {

class GLContext {
 public:
  GLContext() {
    if (!(eglGetCurrentContext() != EGL_NO_CONTEXT)) {
      display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (display_ == EGL_NO_DISPLAY) {
        isCreateError_ = true;
      }
      int majorVersion;
      int minorVersion;
      eglInitialize(display_, &majorVersion, &minorVersion);
      EGLint numConfigs;
      static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                             EGL_PBUFFER_BIT,
                                             EGL_RENDERABLE_TYPE,
                                             EGL_OPENGL_ES2_BIT,
                                             EGL_RED_SIZE,
                                             8,
                                             EGL_GREEN_SIZE,
                                             8,
                                             EGL_BLUE_SIZE,
                                             8,
                                             EGL_ALPHA_SIZE,
                                             8,
                                             EGL_NONE};

      EGLConfig surfaceConfig;
      if (!eglChooseConfig(
              display_, configAttribs, &surfaceConfig, 1, &numConfigs)) {
        eglMakeCurrent(
            display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(display_);
        display_ = EGL_NO_DISPLAY;
        isCreateError_ = true;
      }

      static const EGLint contextAttribs[] = {
          EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
      context_ =
          eglCreateContext(display_, surfaceConfig, NULL, contextAttribs);
      static const EGLint surfaceAttribs[] = {
          EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
      surface_ =
          eglCreatePbufferSurface(display_, surfaceConfig, surfaceAttribs);
      eglMakeCurrent(display_, surface_, surface_, context_);
      eglBindAPI(EGL_OPENGL_ES_API);
      int major;
      glGetIntegerv(GL_MAJOR_VERSION, &major);
      int minor;
      glGetIntegerv(GL_MINOR_VERSION, &minor);

      int maxShaderStorageBlockSize;
      glGetIntegerv(
          GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxShaderStorageBlockSize);

      GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);

      GLint maxCompGroupCountX, maxCompGroupCountY, maxCompGroupCountZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxCompGroupCountX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxCompGroupCountY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxCompGroupCountZ);

      GLint maxCompGroupInvocations;
      glGetIntegerv(
          GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxCompGroupInvocations);

      GLint maxCompUniformBlocks;
      glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &maxCompUniformBlocks);

      GLint maxCompSharedMemorySize;
      glGetIntegerv(
          GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxCompSharedMemorySize);

      int extNum;
      glGetIntegerv(GL_NUM_EXTENSIONS, &extNum);
      if (major < 3) {
        isCreateError_ = true;
      }
    } else {
      context_ = EGL_NO_CONTEXT;
      isCreateError_ = true;
    }
  }

  ~GLContext() {
    if (display_ != EGL_NO_DISPLAY) {
      if (context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(display_, context_);
        context_ = EGL_NO_CONTEXT;
      }
      if (surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(display_, surface_);
        surface_ = EGL_NO_SURFACE;
      }
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglTerminate(display_);
      display_ = EGL_NO_DISPLAY;
    }
    eglReleaseThread();
  }

  bool isCreateError() const {
    return isCreateError_;
  }

 private:
  EGLContext context_;
  EGLDisplay display_;
  EGLSurface surface_;
  bool isCreateError_{false};
};

using buffer_size_t = GLsizeiptr;

class GLBuffer {
 public:
  GLBuffer(buffer_size_t size, GLenum type = GL_SHADER_STORAGE_BUFFER) {
    type_ = type;
    assert(size > 0);
    glGenBuffers(1, &id_);
    GL_CHECK_ERROR;
    glBindBuffer(type_, id_);
    GL_CHECK_ERROR;
    assert(id_ > 0);
    glBufferData(type_, size, NULL, GL_DYNAMIC_DRAW);
    GL_CHECK_ERROR;
    size_ = size;
  }

  ~GLBuffer() {
    glDeleteBuffers(1, &id_);
    GL_CHECK_ERROR;
  }

  void* map(GLbitfield bufMask) {
    glBindBuffer(type_, id_);
    GL_CHECK_ERROR;
    auto p = glMapBufferRange(type_, 0, size_, bufMask);
    GL_CHECK_ERROR;
    return p;
  }

  void unmap() {
    glBindBuffer(type_, id_);
    glUnmapBuffer(type_);
    GL_CHECK_ERROR;
  }

  buffer_size_t size() const {
    return size_;
  }

  void bindInProgram(int binding) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id_);
    GL_CHECK_ERROR;
  }

  std::unique_ptr<GLBuffer> static from(
      const float* data,
      GLsizeiptr size,
      size_t sizeCopy) {
    auto buffer = std::make_unique<GLBuffer>(size);
    float* bufferDataPtr =
        (float*)(buffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if (!bufferDataPtr) {
      assert(false);
    }
    memset(bufferDataPtr, 0, size);
    memcpy(bufferDataPtr, data, sizeCopy);
    buffer->unmap();
    return buffer;
  }

  std::unique_ptr<GLBuffer> static from(const float* data, buffer_size_t size) {
    return from(data, size, size);
  }

  auto copyToHostVec() {
    int64_t n = size_ / sizeof(float);
    std::vector<float> ret(n);
    float* retDataPtr = ret.data();
    const float* bufferDataPtr = (const float*)map(GL_MAP_READ_BIT);
    if (!bufferDataPtr) {
      assert(false);
    }
    memset(retDataPtr, 0, n);
    memcpy(retDataPtr, bufferDataPtr, size_);
    unmap();
    return ret;
  }

  void copyToHost(float* outputDataPtr, size_t sizeCopy) {
    const float* bufferDataPtr = (const float*)map(GL_MAP_READ_BIT);
    if (!bufferDataPtr) {
      assert(false);
    }
    memcpy(outputDataPtr, bufferDataPtr, sizeCopy);
    unmap();
  }

 private:
  GLuint id_ = 0;
  buffer_size_t size_;
  GLenum type_;
};

GLTexture::GLTexture(int w, int h, int d, GLenum texFormat, GLenum target) {
  texFormat_ = texFormat;
  if (target == GL_TEXTURE_3D) {
    assert(w > 0 && h > 0 && d > 0);
    target_ = target;
    glGenTextures(1, &id_);
    GL_CHECK_ERROR;
    glBindTexture(target_, id_);
    glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    GL_CHECK_ERROR;
    glTexStorage3D(target_, 1 /* level */, texFormat_, w, h, d);
    GL_CHECK_ERROR;
  } else if (target == GL_TEXTURE_2D) {
    assert(w > 0 && h > 0);
    target_ = target;
    glGenTextures(1, &id_);
    GL_CHECK_ERROR;
    glBindTexture(target_, id_);
    GL_CHECK_ERROR;
    glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    GL_CHECK_ERROR;
    glTexStorage2D(target_, 1 /* level */, texFormat_, w, h);
    GL_CHECK_ERROR;
  }
}

GLTexture::~GLTexture() {
  glDeleteTextures(1, &id_);
  GL_CHECK_ERROR;
}

unsigned int GLTexture::id() const {
  return id_;
}

void GLTexture::read(GLuint unit) {
  glBindImageTexture(
      unit,
      id_,
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_READ_ONLY,
      texFormat_);
  GL_CHECK_ERROR;
}

void GLTexture::write(GLuint unit) {
  glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_WRITE_ONLY, texFormat_);
  GL_CHECK_ERROR;
}

class GLShader {
 public:
  GLShader(const std::string& shaderCode) {
    shaderId_ = glCreateShader(GL_COMPUTE_SHADER);
    GL_CHECK_ERROR;

    const char* shaderCodeArr[1];
    shaderCodeArr[0] = shaderCode.c_str();
    glShaderSource(shaderId_, 1, shaderCodeArr, NULL);
    GL_CHECK_ERROR;

    compileShader(shaderId_);

    programId_ = glCreateProgram();
    GL_CHECK_ERROR;
    glAttachShader(programId_, shaderId_);
    GL_CHECK_ERROR;
    glLinkProgram(programId_);
    GL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(programId_, GL_LINK_STATUS, &linked);
    if (!linked) {
      GLsizei len;
      glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &len);
      if (len <= 0) {
        GLsizei infoLogLen;
        glGetProgramInfoLog(programId_, 0, &infoLogLen, NULL);
        if (infoLogLen > 0) {
          char* buffer = new char[infoLogLen + 1];
          buffer[len] = '\0';
          glGetProgramInfoLog(programId_, infoLogLen, NULL, buffer);
          TORCH_CHECK(false, "Shader linking error:", buffer);
          delete[] buffer;
        } else {
          TORCH_CHECK(false, "Shader linking error");
        }
      }
    }
  }

  ~GLShader() {
    glDeleteShader(shaderId_);
    glDeleteProgram(programId_);
    GL_CHECK_ERROR;
  }

  unsigned int getProgramId() const {
    return programId_;
  }

  void useProgram() {
    glUseProgram(programId_);
    GL_CHECK_ERROR;
  }

  int getAttribLocation(const char* name) const {
    assert(NULL != name && 0 != programId_);
    return glGetAttribLocation(programId_, name);
  }

  int getUniformLocation(const char* name) const {
    assert(NULL != name && 0 != programId_);
    return glGetUniformLocation(programId_, name);
  }

 private:
  bool compileShader(GLuint s) {
    GLint status;
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &status);
    if (!status) {
      int len;
      glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
      if (0 >= len) {
        glGetShaderInfoLog(s, 0, &len, NULL);
      }
      char* buffer = new char[len + 1];
      glGetShaderInfoLog(s, len, NULL, buffer);
      buffer[len] = 0;
      TORCH_CHECK(false, "Shader compilation error:", buffer);
      delete[] buffer;
      return false;
    }
    return true;
  }

  unsigned int shaderId_ = 0;
  unsigned int programId_ = 0;
};

GLenum getTexFormat() {
  return GL_RGBA32F;
}

void bindTexInProgram(int texId, int programTexId, int binding) {
  glActiveTexture(GL_TEXTURE0 + programTexId);
  GL_CHECK_ERROR;
  glUniform1i(binding, programTexId);
  GL_CHECK_ERROR;
  glBindTexture(GL_TEXTURE_3D, texId);
  GL_CHECK_ERROR;
}

void bindImageTexInProgram(int texId, GLuint unit) {
  glBindImageTexture(
      unit,
      texId,
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());
  GL_CHECK_ERROR;
}

void wait() {
  glFinish();
  glFlush();
}

void compute(GLuint dim0, GLuint dim1, GLuint dim2) {
  glDispatchCompute(dim0, dim1, dim2);
}

inline auto atime_now() {
  return std::chrono::high_resolution_clock::now();
}

inline double atime_duration(
    std::chrono::high_resolution_clock::time_point tp0,
    std::chrono::high_resolution_clock::time_point tp1) {
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(tp1 - tp0);
  return time_span.count();
}

inline double atime_duration_to_now(
    std::chrono::high_resolution_clock::time_point tp0) {
  auto tp1 = std::chrono::high_resolution_clock::now();
  return atime_duration(tp0, tp1);
}
double compute(
    GLuint dim0,
    GLuint dim1,
    GLuint dim2,
    const char* log,
    int compGroupSize0,
    int compGroupSize1,
    int compGroupSize2) {
  glFlush();
  glFinish();

  auto tp = atime_now();
  compute(dim0, dim1, dim2);
  glFinish();
  return atime_duration_to_now(tp);
}

static std::unique_ptr<GLContext> glContext;

bool initGLContextOnce() {
  static const int once = []() {
    glContext = std::make_unique<GLContext>();
    TORCH_WARN(
        glContext && !glContext->isCreateError(), "Failed to create GLContext");
    return 0;
  }();
  ((void)once);
  return static_cast<bool>(glContext);
}

std::unique_ptr<GLShader> createShader(
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  std::ostringstream tc;
  for (auto& s : prefix) {
    tc << s << "\n";
  }
  tc << content;
  return std::make_unique<GLShader>(tc.str());
}

std::shared_ptr<GLShader> getShader(
    const std::string& key,
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  std::shared_ptr<GLShader> shader{createShader(content, prefix)};
  return shader;
}

void addCompGroupSizeDefines(
    std::vector<std::string>& header,
    int* compGroupSize,
    int compGroupSizeX,
    int compGroupSizeY,
    int compGroupSizeZ) {
  static GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ,
      maxCompGroupInvocations;
  static const int once = []() {
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);
    glGetIntegerv(
        GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxCompGroupInvocations);
    return 0;
  }();
  ((void)once);

  compGroupSize[0] =
      compGroupSizeX < maxCompGroupSizeX ? compGroupSizeX : maxCompGroupSizeX;
  compGroupSize[1] =
      compGroupSizeY < maxCompGroupSizeY ? compGroupSizeY : maxCompGroupSizeY;
  compGroupSize[2] =
      compGroupSizeZ < maxCompGroupSizeZ ? compGroupSizeZ : maxCompGroupSizeZ;

  const int compGroupInvocations =
      compGroupSize[0] * compGroupSize[1] * compGroupSize[2];
  if (compGroupInvocations > maxCompGroupInvocations) {
    int oldCompGroupSizeZ = compGroupSize[2];
    compGroupSize[2] =
        maxCompGroupInvocations / (compGroupSize[0] * compGroupSize[1]);
  }

  header.push_back(
      std::string{"#define WORKGROUP_X "} + std::to_string(compGroupSize[0]));
  header.push_back(
      std::string{"#define WORKGROUP_Y "} + std::to_string(compGroupSize[1]));
  header.push_back(
      std::string{"#define WORKGROUP_Z "} + std::to_string(compGroupSize[2]));
}

void hostCHW_to_deviceTex(
    GLuint texId,
    const float* inputData,
    const int C,
    const int H,
    const int W) {
  const int C_4 = UP_DIV(C, 4);
  GLsizeiptr size = ROUND_UP(C, 4) * W * H * sizeof(float);
  auto buffer = GLBuffer::from(inputData, size, C * H * W * sizeof(float));

  auto shader = getShader(
      "nchw_buf_to_tex_glsl", at::native::vulkan::nchw_buf_to_tex_glsl);
  shader->useProgram();

  bindImageTexInProgram(texId, 0 /* unit */);
  GL_CHECK_ERROR;

  buffer->bindInProgram(1);
  glUniform1i(2, W);
  glUniform1i(3, H);
  GL_CHECK_ERROR;

  compute(UP_DIV(W, 8), UP_DIV(H, 8), C_4, "hCHW2dTex", 8, 8, 1);
  GL_CHECK_ERROR;
}

double deviceTex2hostCHW(
    GLuint texId,
    float* outputData,
    int d0,
    int d1,
    int d2) {
  auto d2_4 = UP_DIV(d2, 4);
  auto size = d2_4 * 4 * d0 * d1 * sizeof(float);
  auto buffer = std::make_unique<GLBuffer>(size);
  auto program = getShader(
      "tex_to_nchw_buf_glsl", at::native::vulkan::tex_to_nchw_buf_glsl);
  program->useProgram();

  bindImageTexInProgram(texId, 0 /* unit */);
  GL_CHECK_ERROR;
  buffer->bindInProgram(1);

  glUniform1i(2, d0);
  glUniform1i(3, d1);
  GL_CHECK_ERROR;

  double shaderTime =
      compute(UP_DIV(d0, 8), UP_DIV(d1, 8), d2_4, "dTex2hCHW", 8, 8, 1);
  GL_CHECK_ERROR;

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  GL_CHECK_ERROR;

  auto dOutputData = buffer->map(GL_MAP_READ_BIT);
  if (dOutputData) {
    ::memcpy(outputData, dOutputData, d0 * d1 * d2 * sizeof(float));
  }
  buffer->unmap();
  return shaderTime;
}

GLTensor::GLTensor(std::vector<int64_t> sizes) : sizes_(sizes) {
  TORCH_CHECK(initGLContextOnce(), "Failed to create GLContext");
  assert(sizes_.size() == 4);
}

void GLTensor::setDataFromHost(const float* data) {
  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  auto tex =
      std::make_unique<GLTexture>(W, H, C_4, getTexFormat(), GL_TEXTURE_3D);
  hostCHW_to_deviceTex(tex->id(), data, C, H, W);
  tex_ = std::move(tex);
}

void GLTensor::copyDataToHost(float* output) {
  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  deviceTex2hostCHW(tex_->id(), output, W, H, C);
}

void GLTensor::allocateStorage() {
  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  auto tex =
      std::make_unique<GLTexture>(W, H, C_4, getTexFormat(), GL_TEXTURE_3D);
  tex_ = std::move(tex);
}

int GLTensor::texId() const {
  if (!tex_) {
    assert(false);
  }

  return tex_->id();
}

void upsample_nearest2d(
    GLTensor& output,
    const GLTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t _N,
    int64_t _C,
    float scaleH,
    float scaleW) {
  int64_t C = _N * _C;
  int64_t C_4 = UP_DIV(C, 4);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 8, 8, 1);
  auto shaderKey = "upsampleNearest2d_glsl";
  auto program =
      getShader(shaderKey, at::native::vulkan::upsampleNearest2d_glsl, header);

  program->useProgram();
  bindImageTexInProgram(output.texId(), 0 /* unit */);
  bindTexInProgram(input.texId(), 0, 1 /* binding */);

  glUniform3i(2, IW, IH, C_4);
  glUniform3i(3, OW, OH, C_4);

  glUniform1f(4, scaleW);
  glUniform1f(5, scaleH);
  GL_CHECK_ERROR;

  compute(
      UP_DIV(OW, compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  GL_CHECK_ERROR;
}

void add(
    GLTensor& output,
    const GLTensor& input0,
    const GLTensor& input1,
    float alpha) {
  auto sizes = output.sizes();
  auto N = sizes[0];
  auto C = sizes[1];
  auto H = sizes[2];
  auto W = sizes[3];
  auto C_4 = UP_DIV(C, 4);

  int compGroupSize[3];
  std::vector<std::string> prefix;
  addCompGroupSizeDefines(prefix, compGroupSize, 8, 8, 1);

  auto shaderKey = "add_glsl";
  auto addProgram = getShader(shaderKey, at::native::vulkan::add_glsl, prefix);
  addProgram->useProgram();

  bindImageTexInProgram(output.texId(), 0 /* unit */);
  bindTexInProgram(input0.texId(), 0, 1 /* binding */);
  bindTexInProgram(input1.texId(), 1, 2 /* binding */);

  glUniform4i(3, W, H, C_4, 1);
  glUniform1f(4, alpha);
  GL_CHECK_ERROR;

  compute(
      UP_DIV(W, compGroupSize[0]),
      UP_DIV(H, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  GL_CHECK_ERROR;
}

auto kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW) {
  const uint32_t kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto kernelBuf = std::make_unique<GLBuffer>(sizeof(float) * kBufSizeNumel);
  const int oc_4SizeNumel = UP_DIV(C, 4) * KW * KH * 16;
  float* kernelPtr =
      (float*)(kernelBuf->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
  if (kernelPtr) {
    memset(kernelPtr, 0, sizeof(float) * kBufSizeNumel);
    const float* src = weights;
    float* dst = kernelPtr;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = dst + oc_4 * oc_4SizeNumel;
      for (int ic = 0; ic < C; ++ic) {
        int ic_4 = ic / 4;
        int ic_4_i = ic % 4;
        float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
        for (int ky = 0; ky < KH; ++ky) {
          float* dst_ky = dst_ic + ky * KW * 16;
          for (int kx = 0; kx < KW; ++kx) {
            float* dst_kx = dst_ky + kx * 16;
            dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
          }
        }
      }
    }
  }
  kernelBuf->unmap();
  return kernelBuf;
}

std::unique_ptr<GLTexture> conv2d_kernel_tex_from_hostCHW(
    const float* data,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto kernelBuf = kernelNCHW_OCHW_repack_O4C4HWi4o4(data, OC, C, KH, KW);

  auto OC_4 = UP_DIV(OC, 4);
  auto C_4 = UP_DIV(C, 4);

  auto kernelOutTex = std::make_unique<GLTexture>(
      C_4 * 4, OC_4, KH * KW, getTexFormat(), GL_TEXTURE_3D);

  auto p =
      getShader("KO4C4HW_to_tex_glsl", at::native::vulkan::KO4C4HW_to_tex_glsl);
  p->useProgram();
  bindImageTexInProgram(kernelOutTex->id(), 0 /* unit */);
  kernelBuf->bindInProgram(2);
  glUniform1i(3, KW * KH);
  glUniform1i(4, C_4);
  GL_CHECK_ERROR;

  compute(C_4, OC_4, KH * KW, "hK2dTex", 1, 1, 1);
  GL_CHECK_ERROR;
  return kernelOutTex;
}

void conv2d(
    GLTensor& output,
    const GLTensor& input,
    const GLTensor& weight,
    const float* bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  assert(false);
  // weights are repacked on gpu to texture in special format that does not
  // match VTensor format, do not want to mix in VTensor at the moment -
  // processing through CPU as a hack IKTODO: how to handle/store kernel in
  // special format?
}

void conv2d(
    GLTensor& output,
    const GLTensor& input,
    const float* weight,
    int64_t KH,
    int64_t KW,
    const float* bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  auto osizes = output.sizes();
  auto isizes = input.sizes();

  int64_t OC = osizes[1];
  int64_t C = isizes[1];
  int64_t H = isizes[2];
  int64_t W = isizes[3];
  const int64_t OC_4 = UP_DIV(OC, 4);
  const int64_t C_4 = UP_DIV(C, 4);

  const int64_t KWE = (KW - 1) * DX + 1;
  const int64_t KHE = (KH - 1) * DY + 1;
  const int64_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const int64_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  assert(osizes[2] == OH);
  assert(osizes[3] == OW);

  auto biasBuf =
      GLBuffer::from(bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  auto kernelTex = conv2d_kernel_tex_from_hostCHW(weight, OC, C, KH, KW);

  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 1, 1, OC_4);

  auto shaderKey = "conv_tex_IKnc4hw_glsl";
  auto convProgram =
      getShader(shaderKey, at::native::vulkan::conv_tex_IKnc4hw_glsl, header);

  convProgram->useProgram();
  GL_CHECK_ERROR;
  bindImageTexInProgram(output.texId(), 0 /* unit */);
  bindTexInProgram(input.texId(), 0, 1 /* binding */);
  bindTexInProgram(kernelTex->id(), 1, 2 /* binding */);
  biasBuf->bindInProgram(3);
  GL_CHECK_ERROR;

  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform3i(8, OW, OH, OC_4);
  glUniform3i(9, W, H, C_4);
  GL_CHECK_ERROR;

  compute(
      UP_DIV(OW, 4 * compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(OC_4, compGroupSize[2]),
      "conv_tex",
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  GL_CHECK_ERROR;
}

void clamp(
    GLTensor& output,
    const GLTensor& input,
    float min,
    float max) {
  TORCH_INTERNAL_ASSERT(false, "clamp not implemented for GLES");
}

void addmm(
    GLTensor& output,
    const GLTensor& t,
    const GLTensor& m1,
    const GLTensor& m2,
    float beta,
    float alpha) {
  TORCH_INTERNAL_ASSERT(false, "addmm not implemented for GLES");
}

void mean(GLTensor& output, const GLTensor& input) {
  TORCH_INTERNAL_ASSERT(false, "mean not implemented for GLES");
}

bool is_available() {
  return initGLContextOnce();
}

} // namespace gl
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
