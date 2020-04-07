#ifdef USE_VULKANGL

#include <stdio.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include <ATen/native/vulkan/VulkanDebugUtils.h>
#include <ATen/native/vulkan/VulkanGL.h>
#include <ATen/native/vulkan/glsl.h>

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace gl {

class AGLContext {
 public:
  AGLContext() {
    COUT_FLF;
    if (!(eglGetCurrentContext() != EGL_NO_CONTEXT)) {
      display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (display_ == EGL_NO_DISPLAY) {
        APRINT("eglGetDisplay error");
        isCreateError_ = true;
      }
      int majorVersion;
      int minorVersion;
      eglInitialize(display_, &majorVersion, &minorVersion);
      APRINT("GLContext version major:%d minor:%d", majorVersion, minorVersion);
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
        APRINT("eglChooseConfig error !!!");
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
      APRINT(
          "GLContext: GL_MAJOR_VERSION:%d GL_MINOR_VERSION:%d", major, minor);
      APRINT(
          "GLContext: GL_SHADING_LANGUAGE_VERSION:%s",
          (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

      std::string vendor{(const char*)glGetString(GL_VENDOR)};
      std::string renderer{(const char*)glGetString(GL_RENDERER)};
      APRINT("GLContext: GL_VENDOR:%s", vendor.c_str());
      APRINT("GLContext: GL_RENDERER:%s", renderer.c_str());

      std::string s;
      s.append(vendor);
      s.append(" ");
      s.append(renderer);
      APRINT("GLContext gGLInfo: %s", s.c_str());

      int maxShaderStorageBlockSize;
      glGetIntegerv(
          GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxShaderStorageBlockSize);
      APRINT(
          "GLContext: GL_MAX_SHADER_STORAGE_BLOCK_SIZE:%d",
          maxShaderStorageBlockSize);

      GLint maxCompGroupSizeX, maxCompGroupSizeY, maxCompGroupSizeZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxCompGroupSizeX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxCompGroupSizeY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxCompGroupSizeZ);
      APRINT(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_SIZE: %d,%d,%d",
          maxCompGroupSizeX,
          maxCompGroupSizeY,
          maxCompGroupSizeZ);

      GLint maxCompGroupCountX, maxCompGroupCountY, maxCompGroupCountZ;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxCompGroupCountX);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxCompGroupCountY);
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxCompGroupCountZ);
      APRINT(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_COUNT: %d,%d,%d",
          maxCompGroupCountX,
          maxCompGroupCountY,
          maxCompGroupCountZ);

      GLint maxCompGroupInvocations;
      glGetIntegerv(
          GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxCompGroupInvocations);
      APRINT(
          "GLContext: GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: %d",
          maxCompGroupInvocations);

      GLint maxCompUniformBlocks;
      glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &maxCompUniformBlocks);
      APRINT(
          "GLContext: GL_MAX_COMPUTE_UNIFORM_BLOCKS: %d", maxCompUniformBlocks);

      GLint maxCompSharedMemorySize;
      glGetIntegerv(
          GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxCompSharedMemorySize);
      APRINT(
          "GLContext: GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: %d",
          maxCompSharedMemorySize);

      int extNum;
      glGetIntegerv(GL_NUM_EXTENSIONS, &extNum);
      for (int i = 0; i < extNum; i++) {
        const GLubyte* extName = glGetStringi(GL_EXTENSIONS, i);
        APRINT("GLContext ext %3d: %s", i, extName);
      }

      if (major < 3) {
        isCreateError_ = true;
      }
    } else {
      context_ = EGL_NO_CONTEXT;
      APRINT("eglGetCurrentContext() != EGL_NO_CONTEXT");
      isCreateError_ = true;
    }
  }

  ~AGLContext() {
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

class AGLSSBuffer {
 public:
  AGLSSBuffer(buffer_size_t size, GLenum type = GL_SHADER_STORAGE_BUFFER) {
    type_ = type;
    assert(size > 0);
    glGenBuffers(1, &id_);
    AGL_CHECK_ERROR;
    glBindBuffer(type_, id_);
    AGL_CHECK_ERROR;
    assert(id_ > 0);
    glBufferData(type_, size, NULL, GL_DYNAMIC_DRAW);
    AGL_CHECK_ERROR;
    size_ = size;
  }

  ~AGLSSBuffer() {
    glDeleteBuffers(1, &id_);
    AGL_CHECK_ERROR;
  }

  void* map(GLbitfield bufMask) {
    glBindBuffer(type_, id_);
    AGL_CHECK_ERROR;
    auto p = glMapBufferRange(type_, 0, size_, bufMask);
    AGL_CHECK_ERROR;
    return p;
  }

  void unmap() {
    glBindBuffer(type_, id_);
    glUnmapBuffer(type_);
    AGL_CHECK_ERROR;
  }

  buffer_size_t size() const {
    return size_;
  }

  void bindInProgram(int binding) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id_);
    AGL_CHECK_ERROR;
  }

  std::unique_ptr<AGLSSBuffer> static from(
      const float* data,
      GLsizeiptr size,
      size_t sizeCopy) {
    auto buffer = std::make_unique<AGLSSBuffer>(size);
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

  std::unique_ptr<AGLSSBuffer> static from(
      const float* data,
      buffer_size_t size) {
    return from(data, size, size);
  }

  auto copyToHostVec() {
    int64_t n = size_ / sizeof(float);
    std::vector<float> ret(n);
    float* retDataPtr = ret.data();
    std::cout << "copyToHostVec size:" << size_ << " n:" << n << std::endl;
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
}; // class AGLSSBuffer

AGLTexture::AGLTexture(
    int w,
    int h,
    int d,
    GLenum texFormat,
    GLenum target,
    bool HWC4) {
  texFormat_ = texFormat;
  if (target == GL_TEXTURE_3D) {
    assert(w > 0 && h > 0 && d > 0);
    target_ = target;
    glGenTextures(1, &id_);
    AGL_CHECK_ERROR;
    glBindTexture(target_, id_);
    glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    AGL_CHECK_ERROR;
    int realW = w;
    int realH = h;
    int realD = d;
    if (HWC4) {
      realH = h;
      realW = w;
      realD = UP_DIV(d, 4);
    }
    glTexStorage3D(target_, 1 /* level */, texFormat_, realW, realH, realD);
    AGL_CHECK_ERROR;
  } else if (target == GL_TEXTURE_2D) {
    assert(w > 0 && h > 0);
    target_ = target;
    glGenTextures(1, &id_);
    AGL_CHECK_ERROR;
    glBindTexture(target_, id_);
    AGL_CHECK_ERROR;
    glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    AGL_CHECK_ERROR;
    glTexStorage2D(target_, 1 /* level */, texFormat_, w, h);
    AGL_CHECK_ERROR;
  }
}

AGLTexture::~AGLTexture() {
  glDeleteTextures(1, &id_);
  AGL_CHECK_ERROR;
}

unsigned int AGLTexture::id() const {
  return id_;
}

void AGLTexture::read(GLuint unit) {
  glBindImageTexture(
      unit,
      id_,
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_READ_ONLY,
      texFormat_);
  AGL_CHECK_ERROR;
}

void AGLTexture::write(GLuint unit) {
  glBindImageTexture(unit, id_, 0, GL_TRUE, 0, GL_WRITE_ONLY, texFormat_);
  AGL_CHECK_ERROR;
}

void AGLTexture::sample(GLuint unit, GLuint texId) {
  glActiveTexture(GL_TEXTURE0 + texId);
  glUniform1i(unit, texId);
  glBindTexture(target_, id_);
  AGL_CHECK_ERROR;
}

enum AGLPrecision { highp = 0, mediump = 1, lowp = 2, count = 3 };
static AGLPrecision gPrecision = highp;
std::string getPrecision() {
  static const char* precisionStr[AGLPrecision::count] = {
      "highp", "mediump", "lowp"};
  return precisionStr[gPrecision];
}

class AGLShader {
 public:
  AGLShader(const std::string& shaderCode) {
    shaderId_ = glCreateShader(GL_COMPUTE_SHADER);
    AGL_CHECK_ERROR;

    const char* shaderCodeArr[1];
    shaderCodeArr[0] = shaderCode.c_str();
    glShaderSource(shaderId_, 1, shaderCodeArr, NULL);
    AGL_CHECK_ERROR;

    bool res = compileShader(shaderId_);
    assert(res);

    programId_ = glCreateProgram();
    AGL_CHECK_ERROR;
    glAttachShader(programId_, shaderId_);
    AGL_CHECK_ERROR;
    glLinkProgram(programId_);
    AGL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(programId_, GL_LINK_STATUS, &linked);
    if (!linked) {
      GLsizei len;
      glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &len);
      if (len <= 0) {
        glGetProgramInfoLog(programId_, 0, &len, NULL);
      }
      if (len > 0) {
        char* buffer = new char[len + 1];
        buffer[len] = '\0';
        glGetProgramInfoLog(programId_, len, NULL, buffer);
        APRINT("shaderCompile ERROR:%s", buffer);
        delete[] buffer;
      }
    }
  }

  ~AGLShader() {
    glDeleteShader(shaderId_);
    glDeleteProgram(programId_);
    AGL_CHECK_ERROR;
  }

  unsigned int getProgramId() const {
    return programId_;
  }

  static std::string getHead(std::string imageFormat, std::string precision) {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION " << precision << "\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    return headOs.str();
  }

  void useProgram() {
    glUseProgram(programId_);
    AGL_CHECK_ERROR;
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
      APRINT("shaderCompile ERROR:%s", buffer);
      delete[] buffer;
      return false;
    }
    return true;
  }

  unsigned int shaderId_ = 0;
  unsigned int programId_ = 0;
}; // class AGLShader

GLenum getTexFormat() {
  return GL_RGBA32F;
}

std::string getImageFormat() {
  return "rgba32f";
}

void bindTexInProgram(int texId, int programTexId, int binding) {
  COUT_FLF;
  glActiveTexture(GL_TEXTURE0 + programTexId);
  AGL_CHECK_ERROR;
  glUniform1i(binding, programTexId);
  AGL_CHECK_ERROR;
  glBindTexture(GL_TEXTURE_3D, texId);
  AGL_CHECK_ERROR;
}

void bindImageTexInProgram(int texId, GLuint unit) {
  COUT_FLF;
  glBindImageTexture(
      unit,
      texId,
      0 /* level */,
      GL_TRUE /* layered */,
      0 /* layer */,
      GL_WRITE_ONLY,
      getTexFormat());
  AGL_CHECK_ERROR;
}

void wait() {
  COUT_FLF;
  glFinish();
  glFlush();
}

void compute(GLuint dim0, GLuint dim1, GLuint dim2) {
  COUT_FLF;
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
  COUT_FLF;
  glFlush();
  glFinish();

  auto tp = atime_now();
  compute(dim0, dim1, dim2);
  glFinish();
  auto ret = atime_duration_to_now(tp);

  APRINT(
      "compute %16s(%3d,%3d,%3d)xCG(%3d,%3d,%3d) cpuTime:%.6fs",
      log,
      dim0,
      dim1,
      dim2,
      compGroupSize0,
      compGroupSize1,
      compGroupSize2,
      ret);
  return ret;
}

static std::unique_ptr<AGLContext> glContext;

void initGLContextOnce() {
  COUT_FLF;

  static const int once = []() {
    APRINT("Creating GLContext...");
    glContext = std::make_unique<AGLContext>();
    if (!glContext || glContext->isCreateError()) {
      APRINT("ERROR Failed to create GLContext");
      assert(false);
    }
    APRINT("GLContext created ok");
    return 0;
  }();
  ((void)once);
}

void printShaderCode(const std::string& s) {
  std::string token;
  std::istringstream tokenStream(s);
  int i = 0;
  while (std::getline(tokenStream, token, '\n')) {
    std::printf("%03d %s", i++, token.c_str());
  }
}

std::unique_ptr<AGLShader> createShader(
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  COUT_FLF;
  std::ostringstream tc;
  tc << AGLShader::getHead(getImageFormat(), getPrecision());
  for (auto& s : prefix) {
    tc << s << "\n";
  }
  tc << content;

  auto shaderCode = tc.str();
  printShaderCode(shaderCode);
  return std::make_unique<AGLShader>(tc.str());
}

std::shared_ptr<AGLShader> getShader(
    const std::string& key,
    const char* content,
    const std::vector<std::string>& prefix = {}) {
  COUT_FLF;
  initGLContextOnce();
  std::shared_ptr<AGLShader> shader{createShader(content, prefix)};
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
    APRINT(
        "compGroupSize(%3d, %3d, %3d) compGroupInvocations:%4d > GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS:%4d => changeZto (%3d, %3d, %3d)",
        compGroupSize[0],
        compGroupSize[1],
        oldCompGroupSizeZ,
        compGroupInvocations,
        maxCompGroupInvocations,
        compGroupSize[0],
        compGroupSize[1],
        compGroupSize[2]);
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
  COUT_FLF;
  const int C_4 = UP_DIV(C, 4);
  GLsizeiptr size = ROUND_UP(C, 4) * W * H * sizeof(float);
  std::cout << "hostCHW_to_deviceTex(C:" << C << " H:" << H << " W:" << W << ")"
            << std::endl;

  std::cout << "size:" << size << std::endl;
  auto buffer = AGLSSBuffer::from(inputData, size, C * H * W * sizeof(float));

  auto shader = getShader(
      "nchw_buf_to_tex_glsl", at::native::vulkan::nchw_buf_to_tex_glsl);
  shader->useProgram();

  bindImageTexInProgram(texId, 0 /* unit */);
  AGL_CHECK_ERROR;

  buffer->bindInProgram(1);
  glUniform1i(2, W);
  glUniform1i(3, H);
  AGL_CHECK_ERROR;

  compute(UP_DIV(W, 8), UP_DIV(H, 8), C_4, "hCHW2dTex", 8, 8, 1);
  AGL_CHECK_ERROR;
}

double deviceTex2hostCHW(
    GLuint texId,
    float* outputData,
    int d0,
    int d1,
    int d2) {
  auto d2_4 = UP_DIV(d2, 4);
  auto size = d2_4 * 4 * d0 * d1 * sizeof(float);
  auto buffer = std::make_unique<AGLSSBuffer>(size);
  auto program = getShader(
      "tex_to_nchw_buf_glsl", at::native::vulkan::tex_to_nchw_buf_glsl);
  program->useProgram();

  bindImageTexInProgram(texId, 0 /* unit */);
  AGL_CHECK_ERROR;
  buffer->bindInProgram(1);

  glUniform1i(2, d0);
  glUniform1i(3, d1);
  AGL_CHECK_ERROR;

  double shaderTime =
      compute(UP_DIV(d0, 8), UP_DIV(d1, 8), d2_4, "dTex2hCHW", 8, 8, 1);
  AGL_CHECK_ERROR;

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  AGL_CHECK_ERROR;

  auto dOutputData = buffer->map(GL_MAP_READ_BIT);
  if (dOutputData) {
    ::memcpy(outputData, dOutputData, d0 * d1 * d2 * sizeof(float));
  }
  buffer->unmap();
  return shaderTime;
}

VulkanGLTensor::VulkanGLTensor(std::vector<int64_t> sizes) : sizes_(sizes) {
  COUT_FLF;
  assert(sizes_.size() == 4);
  initGLContextOnce();
}

void VulkanGLTensor::setDataFromHost(const float* data) {
  COUT_FLF;
  initGLContextOnce();

  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  auto tex = std::make_unique<AGLTexture>(
      W, H, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  hostCHW_to_deviceTex(tex->id(), data, C, H, W);
  tex_ = std::move(tex);
}

void VulkanGLTensor::copyDataToHost(float* output) {
  COUT_FLF;
  initGLContextOnce();

  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  deviceTex2hostCHW(tex_->id(), output, W, H, C);

  at::native::vulkan::debug::vk_print4d(
      "dense_to_phvulkan", output, N, C, H, W);
}

void VulkanGLTensor::allocateStorage() {
  COUT_FLF;
  at::native::vulkan::gl::initGLContextOnce();
  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  auto tex = std::make_unique<AGLTexture>(
      W, H, C_4, getTexFormat(), GL_TEXTURE_3D, false);
  tex_ = std::move(tex);
}

int VulkanGLTensor::texId() const {
  if (!tex_) {
    assert(false);
  }

  return tex_->id();
}

void upsample_nearest2d(
    VulkanGLTensor& output,
    const VulkanGLTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t _N,
    int64_t _C,
    float scaleH,
    float scaleW) {
  COUT_FLF;

  int64_t C = _N * _C;
  int64_t C_4 = UP_DIV(C, 4);

  COUT_FLF;
  int compGroupSize[3];
  std::vector<std::string> header;
  addCompGroupSizeDefines(header, compGroupSize, 8, 8, 1);
  auto shaderKey = "upsampleNearest2d_glsl";
  auto program = getShader(
      shaderKey, at::native::vulkan::upsampleNearest2d_glsl, header);

  program->useProgram();
  // { binding
  bindImageTexInProgram(output.texId(), 0 /* unit */);
  bindTexInProgram(input.texId(), 0, 1 /* binding */);

  glUniform3i(2, IW, IH, C_4);
  glUniform3i(3, OW, OH, C_4);

  glUniform1f(4, scaleW);
  glUniform1f(5, scaleH);
  AGL_CHECK_ERROR;
  // } binding

  compute(
      UP_DIV(OW, compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;
}

void add(
    VulkanGLTensor& output,
    const VulkanGLTensor& input0,
    const VulkanGLTensor& input1,
    float alpha) {
  COUT_FLF;

  auto sizes = output.sizes();
  auto N = sizes[0];
  auto C = sizes[1];
  auto H = sizes[2];
  auto W = sizes[3];
  auto C_4 = UP_DIV(C, 4);

  int compGroupSize[3];
  std::vector<std::string> prefix;
  addCompGroupSizeDefines(prefix, compGroupSize, 8, 8, 1);

  auto shaderKey = "binary_add_glsl";
  auto binAddProgram =
      getShader(shaderKey, at::native::vulkan::binary_add_glsl, prefix);
  binAddProgram->useProgram();

  bindImageTexInProgram(output.texId(), 0 /* unit */);
  bindTexInProgram(input0.texId(), 0, 1 /* binding */);
  bindTexInProgram(input1.texId(), 1, 2 /* binding */);

  glUniform4i(3, W, H, C_4, 1);
  glUniform1f(4, alpha);
  AGL_CHECK_ERROR;

  compute(
      UP_DIV(W, compGroupSize[0]),
      UP_DIV(H, compGroupSize[1]),
      UP_DIV(C_4, compGroupSize[2]),
      shaderKey,
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;
}

auto kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW) {
  const uint32_t kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
  auto kernelBuf = std::make_unique<AGLSSBuffer>(sizeof(float) * kBufSizeNumel);
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

std::unique_ptr<AGLTexture> conv2d_kernel_tex_from_hostCHW(
    const float* data,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto kernelBuf = kernelNCHW_OCHW_repack_O4C4HWi4o4(data, OC, C, KH, KW);

  auto OC_4 = UP_DIV(OC, 4);
  auto C_4 = UP_DIV(C, 4);

  auto kernelOutTex = std::make_unique<AGLTexture>(
      C_4 * 4, OC_4, KH * KW, getTexFormat(), GL_TEXTURE_3D, false);

  auto p = getShader(
      "KO4C4HW_to_tex_glsl", at::native::vulkan::gl::KO4C4HW_to_tex_glsl);
  p->useProgram();
  bindImageTexInProgram(kernelOutTex->id(), 0 /* unit */);
  kernelBuf->bindInProgram(2);
  glUniform1i(3, KW * KH);
  glUniform1i(4, C_4);
  AGL_CHECK_ERROR;

  compute(C_4, OC_4, KH * KW, "hK2dTex", 1, 1, 1);
  AGL_CHECK_ERROR;
  return kernelOutTex;
}

void conv2d(
    VulkanGLTensor& output,
    const VulkanGLTensor& input,
    const VulkanGLTensor& weight,
    const float* bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G) {
  COUT_FLF;
  assert(false);
  // weights are repacked on gpu to texture in special format that does not
  // match VTensor format, do not want to mix in VTensor at the moment -
  // processing through CPU as a hack IKTODO: how to handle/store kernel in
  // special format?
}

void conv2d(
    VulkanGLTensor& output,
    const VulkanGLTensor& input,
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
  COUT_FLF;

  std::cout << "KH:" << KH << " KW:" << KW << " SY:" << SY << " SX:" << SX
            << " PY:" << PY << " PX:" << PX << " DY:" << DY << " DX:" << DX
            << " G:" << G << std::endl;

  auto osizes = output.sizes();
  auto isizes = input.sizes();

  COUT_FLF;
  int64_t OC = osizes[1];
  int64_t C = isizes[1];
  std::cout << "OC:" << OC << " C:" << C << std::endl;

  int64_t H = isizes[2];
  int64_t W = isizes[3];
  std::cout << "H:" << H << " W:" << W << std::endl;

  const int64_t OC_4 = UP_DIV(OC, 4);
  const int64_t C_4 = UP_DIV(C, 4);

  const int64_t KWE = (KW - 1) * DX + 1;
  const int64_t KHE = (KH - 1) * DY + 1;
  const int64_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const int64_t OH = ((H - KHE + 2 * PY) / SY) + 1;
  std::cout << "OH:" << OH << " OW:" << OW << std::endl;
  COUT_FLF;
  assert(osizes[2] == OH);
  assert(osizes[3] == OW);

  COUT_FLF;
  auto biasBuf = AGLSSBuffer::from(
      bias, sizeof(float) * ALIGN_UP4(OC), sizeof(float) * OC);

  COUT_FLF;
  auto kernelTex = conv2d_kernel_tex_from_hostCHW(weight, OC, C, KH, KW);

  int compGroupSize[3];
  std::vector<std::string> header;
  COUT_FLF;
  addCompGroupSizeDefines(header, compGroupSize, 1, 1, OC_4);

  auto shaderKey = "conv_tex_IKnc4hw_glsl";
  COUT_FLF;
  auto convProgram = getShader(
      shaderKey, at::native::vulkan::conv_tex_IKnc4hw_glsl, header);

  COUT_FLF;
  convProgram->useProgram();
  AGL_CHECK_ERROR;
  COUT_FLF;
  bindImageTexInProgram(output.texId(), 0 /* unit */);
  AGL_CHECK_ERROR;
  COUT_FLF;
  bindTexInProgram(input.texId(), 0, 1 /* binding */);
  AGL_CHECK_ERROR;
  COUT_FLF;
  bindTexInProgram(kernelTex->id(), 1, 2 /* binding */);
  AGL_CHECK_ERROR;
  COUT_FLF;
  biasBuf->bindInProgram(3);
  AGL_CHECK_ERROR;

  COUT_FLF;
  glUniform2i(4, PX, PY);
  glUniform2i(5, KW, KH);
  glUniform2i(6, SX, SY);
  glUniform2i(7, DX, DY);
  glUniform1i(8, 4);
  AGL_CHECK_ERROR;

  COUT_FLF;
  glUniform3i(10, OW, OH, OC_4);
  glUniform3i(11, W, H, C_4);
  AGL_CHECK_ERROR;

  COUT_FLF;
  compute(
      UP_DIV(OW, 4 * compGroupSize[0]),
      UP_DIV(OH, compGroupSize[1]),
      UP_DIV(OC_4, compGroupSize[2]),
      "conv_tex",
      compGroupSize[0],
      compGroupSize[1],
      compGroupSize[2]);
  AGL_CHECK_ERROR;
}

} // namespace gl
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
