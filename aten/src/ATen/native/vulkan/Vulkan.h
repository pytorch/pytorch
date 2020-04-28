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

void initVulkanContextOnce();

class VImage {
 public:
  VImage(int64_t W, int64_t H, int64_t C);
  ~VImage();

  VkImageViewCreateInfo imageViewCreateInfo();
  VkSamplerCreateInfo samplerCreateInfo();

  VkImage image_;
  VkImageViewType viewType_;
  VkFormat format_;
  VkDeviceMemory imageMemory_;
  VkImageLayout initialLayout_;
  VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_GENERAL;
  VkFilter filter_;
  VkSamplerAddressMode samplerAddressMode_;

  VkImageView imageView_;
  VkSampler sampler_;
};

class VulkanTensor : public c10::intrusive_ptr_target {
 public:
  VulkanTensor(std::vector<int64_t> sizes);
  ~VulkanTensor() = default;

  VulkanTensor(VulkanTensor&&) = default;
  VulkanTensor& operator=(VulkanTensor&&) = default;

  VulkanTensor(const VulkanTensor&) = delete;
  VulkanTensor& operator=(const VulkanTensor&) = delete;

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  void setDataFromHost(const float* data);
  void copyDataToHost(float* data);

  bool hasStorage() {
    return static_cast<bool>(tensorImage_);
  }
  void allocateStorage();

 private:
  std::vector<int64_t> sizes_;
  std::unique_ptr<VImage> tensorImage_;
};

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
