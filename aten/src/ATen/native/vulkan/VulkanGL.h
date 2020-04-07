#pragma once
#ifdef USE_VULKANGL

#include <memory>
#include <vector>

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#ifdef __ANDROID__
#include <android/log.h>
//  __android_log_print(ANDROID_LOG_ERROR, "AGPU", format, ##__VA_ARGS__)
#define AGPU_ERROR(format, ...) printf(format, ##__VA_ARGS__)
//  __android_log_print(ANDROID_LOG_INFO, "AGPU", format, ##__VA_ARGS__)
#define APRINT(format, ...) printf(format, ##__VA_ARGS__)

#define FUNC_PRINT(x) APRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) \
  APRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AGL_CHECK_ERROR                                                   \
  {                                                                       \
    GLenum error = glGetError();                                          \
    if (GL_NO_ERROR != error) {                                           \
      APRINT("File:%s Line:%d F:%s\n", __FILE__, __LINE__, __FUNCTION__); \
      FUNC_PRINT_ALL(error, 0x);                                          \
    }                                                                     \
    assert(GL_NO_ERROR == error);                                         \
  }
#else

// if not __ANDROID__
#define APRINT(format, ...) printf(format, ##__VA_ARGS__)
#define AGPU_ERROR(format, ...) printf(format, ##__VA_ARGS__)

#endif

namespace at {
namespace native {
namespace vulkan {

namespace details {
namespace gl {

class AGLTexture {
 public:
  ~AGLTexture();
  // IKTODO: Disable copy
  AGLTexture(
      int w,
      int h,
      int d,
      GLenum texFormat,
      GLenum target = GL_TEXTURE_3D,
      bool HWC4 = true);

  unsigned int id() const;

  void read(GLuint unit);
  void write(GLuint unit);
  void sample(GLuint unit, GLuint texId);

 private:
  unsigned int id_;
  GLenum target_;
  GLenum texFormat_{GL_RGBA32F};
}; // class AGLTexture

class VulkanGLTensor {
 public:
  VulkanGLTensor(std::vector<int64_t> sizes);

  ~VulkanGLTensor() = default;

  VulkanGLTensor(VulkanGLTensor&&) = default;
  VulkanGLTensor& operator=(VulkanGLTensor&&) = default;

  VulkanGLTensor(const VulkanGLTensor&) = delete;
  VulkanGLTensor& operator=(const VulkanGLTensor&) = delete;

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  void setDataFromHost(const float* data);
  void copyDataToHost(float* data);

  bool hasStorage() {
    return static_cast<bool>(tex_);
  }
  void allocateStorage();

  // IKTODO: avoid exposing these internals?
  int texId() const;

 private:
  std::vector<int64_t> sizes_;
  std::unique_ptr<AGLTexture> tex_;
};

void upsample_nearest2d(
    VulkanGLTensor& output,
    const VulkanGLTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t N,
    int64_t C,
    float scaleH,
    float scaleW);

void add(
    VulkanGLTensor& output,
    const VulkanGLTensor& input0,
    const VulkanGLTensor& input1,
    float alpha);

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
    int64_t G);

} // namespace gl
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at

#endif
