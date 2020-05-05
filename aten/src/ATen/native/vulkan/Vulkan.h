#pragma once
#ifdef USE_VULKAN

#include <memory>
#include <vector>

#ifdef USE_VULKAN_WRAPPER
#include "vulkan_wrapper.h"
#else
#include <vulkan/vulkan.h>
#endif

#include <c10/util/intrusive_ptr.h>

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

bool is_available();

class VulkanTensor final : public c10::intrusive_ptr_target {
  class Impl;

 public:
  VulkanTensor(std::vector<int64_t> sizes);
  ~VulkanTensor() = default;

  VulkanTensor(VulkanTensor&&) = default;
  VulkanTensor& operator=(VulkanTensor&&) = default;

  VulkanTensor(const VulkanTensor&) = default;
  VulkanTensor& operator=(const VulkanTensor&) = default;

  inline std::shared_ptr<Impl> impl() {
    return pImpl;
  }
  inline std::shared_ptr<Impl> impl() const {
    return pImpl;
  }
  std::vector<int64_t> sizes() const;
  void setDataFromHost(const float* data);
  void copyDataToHost(float* data);
  bool hasStorage() const;
  void allocateStorage();

 private:
  std::shared_ptr<Impl> pImpl;
};

void upsample_nearest2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t N,
    int64_t C,
    float scaleH,
    float scaleW);

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
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
} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
