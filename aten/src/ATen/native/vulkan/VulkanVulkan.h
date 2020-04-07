#pragma once
#ifdef USE_VULKAN

#include <memory>

#include "vulkan_wrapper.h"

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

void initVulkanContextOnce();

class AVKImage {
 public:
  AVKImage(int64_t W, int64_t H, int64_t C);
  ~AVKImage();

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

class VulkanVulkanTensor {
 public:
  VulkanVulkanTensor(std::vector<int64_t> sizes);
  ~VulkanVulkanTensor() = default;

  VulkanVulkanTensor(VulkanVulkanTensor&&) = default;
  VulkanVulkanTensor& operator=(VulkanVulkanTensor&&) = default;

  VulkanVulkanTensor(const VulkanVulkanTensor&) = delete;
  VulkanVulkanTensor& operator=(const VulkanVulkanTensor&) = delete;

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  void setDataFromHost(const float* data);
  void copyDataToHost(float* data);

  bool hasStorage() {
    return static_cast<bool>(tensorImage_);
  }
  void allocateStorage() {
    assert(false);
  }

 private:
  std::vector<int64_t> sizes_;
  std::unique_ptr<AVKImage> tensorImage_;
};

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
