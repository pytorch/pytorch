#pragma once
#ifdef USE_GLES

#include <memory>
#include <vector>

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#include <c10/util/intrusive_ptr.h>

namespace at {
namespace native {
namespace vulkan {

namespace details {
namespace gl {

class AGLTexture {
 public:
  ~AGLTexture();
  AGLTexture(
      int w,
      int h,
      int d,
      GLenum texFormat,
      GLenum target = GL_TEXTURE_3D);
  AGLTexture(const AGLTexture&) = delete;
  AGLTexture& operator=(const AGLTexture&) = delete;

  unsigned int id() const;

  void read(GLuint unit);
  void write(GLuint unit);
  void sample(GLuint unit, GLuint texId);

 private:
  unsigned int id_;
  GLenum target_;
  GLenum texFormat_{GL_RGBA32F};
}; // class AGLTexture

class GLTensor : public c10::intrusive_ptr_target {
 public:
  GLTensor(std::vector<int64_t> sizes);

  ~GLTensor() = default;

  GLTensor(GLTensor&&) = default;
  GLTensor& operator=(GLTensor&&) = default;

  GLTensor(const GLTensor&) = delete;
  GLTensor& operator=(const GLTensor&) = delete;

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
    GLTensor& output,
    const GLTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t N,
    int64_t C,
    float scaleH,
    float scaleW);

void add(
    GLTensor& output,
    const GLTensor& input0,
    const GLTensor& input1,
    float alpha);

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
    int64_t G);

} // namespace gl
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at

#endif
