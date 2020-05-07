#pragma once
#ifdef USE_VULKAN

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#ifdef USE_VULKAN_WRAPPER
#include "vulkan_wrapper.h"
#else
#include <vulkan/vulkan.h>
#endif

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#define GLSL_SPV(name) name##_glsl
#else
#include <ATen/native/vulkan/spv.h>
#define GLSL_SPV(name) name##_spv, name##_spv_len
#endif

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#include <c10/util/intrusive_ptr.h>

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

static constexpr bool kEnableValidationLayers = true;
bool is_available();

class VContext;
const VContext& context();

// VulkanTensor is a handle that holds shared pointer to VulkanTensor:Impl,
// that owns Tensor representation on GPU.
//  VulkanTensor is copyable and moveable (copying and moving pointer to Impl).
//
// VulkanTensor::Impl is moveable only, owns Vulkan device memory for Tensor
// data. Tensor can be represented in several formats.
//
// 0. VBuffer - (wrapper on  vulkan VkBuffer), supports all tensor dimensions,
// data is in  Contiguous format (NCHW), in plan to preserve at::Tensor memory
// format (3d or 4d tensors can be in NHWC ChannelsLast format). It is located
// in host visible memory that can be memory mapped to CPU memory.
//
// 1. VImage(TexC4) - (wrapper on vulkan VkImage), optional representation of
// tensors with dimension <= 4 as VkImage, sed in shaders as texture or storage
// image. It is 3-dimensional image (x, y, z) with 4 component * 16 bit for each
// triple (x, y, z).
// For NCHW, NHWC: 
// For dim==4: image.x - W sizes[3]; image.y -  H sizes[2]; image.z - (N sizes[0] * C sizes[1]) / 4
// For dim==3: image.x - W sizes[2]; image.y - H sizes[1]; image.z - (C sizes[0]) / 4
// For dim==2: image.x - W sizes[1]; image.y - H sizes[0]; image.z : 1
// For dim==1: image.x - W sizes[0]; image.y : 1; image.z : 1
//
//
// 2. VImage (other format) - Currently not added, but for some operations
// another texture
//  packing format can be beneficial for performance.
//
// Contract about synchronization between representations:
// 1.VImage(TexC4) representation is allocated lazily with calling image(),
// fails for dimensions > 4.
//
// Tensor data can be in 0.VBuffer and/or 1.VImage(TexC4),
// If Tensor can be represented as image - VulkanTensor::Impl::canBeImage()
// returns true. Image representation created lazily by call
// VulkanTensor::Impl::image(), if it is called on Tensor with !canBeImage() -
// it fails.
//
// If image allocated - image data has priority.
// VulkanTensor::copyDataToHost checks if image allocated -
// copyFromImageToBuffer first.
class VBuffer;
class VImage;
class VulkanTensor final : public c10::intrusive_ptr_target {
  class Impl;

 public:
  VulkanTensor(std::vector<int64_t> sizes);
  ~VulkanTensor() = default;

  VulkanTensor(VulkanTensor&&) = default;
  VulkanTensor& operator=(VulkanTensor&&) = default;

  VulkanTensor(const VulkanTensor&) = default;
  VulkanTensor& operator=(const VulkanTensor&) = default;

  inline std::vector<int64_t> sizes() const;
  inline int64_t dim() const;
  inline int64_t numel() const;

  inline bool hasStorage() const;
  inline void allocateStorage();
  inline void setDataFromHost(const float* inputData);
  inline void copyDataToHost(float* outputData);

  inline bool hasBuffer();
  inline VBuffer& buffer();
  inline bool canBeImage();
  inline bool hasImage();
  inline VImage& image() const;
  inline VImage& image();

 private:
  inline std::shared_ptr<Impl> impl();
  inline std::shared_ptr<Impl> impl() const;
  std::shared_ptr<Impl> pImpl;
};

class VContext final {
 public:
  VContext(bool enableValidationLayers);
  ~VContext();
  VContext(const VContext&) = delete;
  VContext& operator=(const VContext&) = delete;
  VContext(VContext&&) = default;
  VContext& operator=(VContext&&) = default;

  inline VkDevice device() const {
    return device_;
  }
  inline VkPhysicalDevice physicalDevice() const {
    return physicalDevice_;
  }
  inline VkPhysicalDeviceLimits limits() const {
    return physicalDeviceLimits_;
  }
  inline VkCommandPool commandPool() const {
    return commandPool_;
  }
  inline VkQueue queue() const {
    return queue_;
  }

 private:
  void createInstance();
  void findPhysicalDevice();
  void createDevice();
  uint32_t getComputeQueueFamilyIndex();

  VkInstance instance_;
  VkDebugReportCallbackEXT debugReportCallback_;
  VkDevice device_;
  VkPhysicalDevice physicalDevice_;
  VkPhysicalDeviceLimits physicalDeviceLimits_;
  std::vector<const char*> enabledValidationLayers_;
  VkQueue queue_;
  uint32_t queueFamilyIndex_;
  bool enableValidationLayers_;
  VkCommandPool commandPool_;
}; // class VContext

class VBuffer final {
 public:
  class MapMemory final {
   public:
    MapMemory(
        VkDevice device,
        VkDeviceMemory deviceMemory,
        VkDeviceSize offset,
        VkDeviceSize size)
        : device_(device), deviceMemory_(deviceMemory) {
      vkMapMemory(device_, deviceMemory_, 0, size, 0, &mappedMemory_);
    }
    ~MapMemory() {
      vkUnmapMemory(device_, deviceMemory_);
    }
    MapMemory(const MapMemory&) = delete;
    MapMemory& operator=(const MapMemory&) = delete;
    MapMemory(MapMemory&&) = default;
    MapMemory& operator=(MapMemory&&) = default;
    inline void* ptr() {
      return mappedMemory_;
    }

   private:
    VkDevice device_;
    VkDeviceMemory deviceMemory_;
    void* mappedMemory_;
  };

  explicit VBuffer(
      VkDeviceSize bufferSizeBytes,
      VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

  ~VBuffer() noexcept;

  VBuffer(const VBuffer&) = delete;
  VBuffer& operator=(const VBuffer&) = delete;
  VBuffer(VBuffer&&) = default;
  VBuffer& operator=(VBuffer&&) = default;

  static inline VBuffer makeUniformBuffer(VkDeviceSize bufferSize) {
    return VBuffer{bufferSize,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  }

  MapMemory map() {
    return MapMemory{context().device(), bufferMemory_, 0, bufferSizeBytes_};
  }

  void copyFromDeviceToHost(void* outputData, int64_t size);
  void copyFromHostToDevice(void* data, int64_t size);

  VkDescriptorBufferInfo makeDescriptorBufferInfo();
  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      const VkDescriptorBufferInfo* bufferInfo);

  void bind(VkDescriptorSet descriptorSet, uint32_t binding);

  inline VkDeviceSize sizeBytes() const {
    return bufferSizeBytes_;
  }

  void addBufferMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkDeviceSize offset,
      VkDeviceSize size);

 private:
  VkDeviceSize bufferSizeBytes_;
  VkDescriptorType descriptorType_;
  VkBuffer buffer_;
  VkDeviceMemory bufferMemory_;
}; // class VBuffer

VBuffer makeUniformConstBuffer(void* ptr, VkDeviceSize size);

class VImage final {
 public:
  explicit VImage(uint32_t W, uint32_t H, uint32_t C);

  ~VImage() noexcept;
  VImage(const VImage&) = delete;
  VImage& operator=(const VImage&) = delete;
  VImage(VImage&&) = default;
  VImage& operator=(VImage&&) = default;

  inline auto W() {
    return W_;
  }
  inline auto H() {
    return H_;
  }
  inline auto C() {
    return C_;
  }
  inline auto D() {
    return D_;
  }

  VkImageViewCreateInfo makeImageViewCreateInfo();
  VkSamplerCreateInfo makeSamplerCreateInfo();
  VkDescriptorImageInfo makeDescriptorImageInfo(VkImageLayout imageLayout);
  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      const VkDescriptorImageInfo* imageInfo);
  void bind(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      VkImageLayout imageLayout);
  inline VkDeviceSize sizeBytes() const {
    return sizeof(float) * W_ * H_ * C_;
  }

  inline VkDeviceSize capacityBytes() const {
    return sizeof(float) * W_ * H_ * D_ * 4;
  }

  void addImageMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkImageLayout oldLayout,
      VkImageLayout newLayout);

 private:
  uint32_t W_;
  uint32_t H_;
  uint32_t C_;
  uint32_t D_;

  VkImage image_;
  VkDeviceMemory imageMemory_;
  VkImageView imageView_;
  VkSampler sampler_;

  // configuration
  VkImageType imageType_ = VK_IMAGE_TYPE_3D;
  VkFilter filter_ = VK_FILTER_NEAREST;
  VkFormat format_ = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkImageLayout imageLayout_ = VK_IMAGE_LAYOUT_GENERAL;
  VkImageLayout imageLayoutInitial_ = VK_IMAGE_LAYOUT_UNDEFINED;
  VkSamplerAddressMode samplerAddressMode_ =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  VkImageViewType viewType_ = VK_IMAGE_VIEW_TYPE_3D;
}; // class VImage

void copyFromBufferToImage(VBuffer& buffer, VImage& image);

void copyFromImageToBuffer(VImage& image, VBuffer& buffer);


namespace vkutil {
VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    uint32_t binding,
    VkDescriptorType descriptorType);

void createDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t bindingCount,
    VkDescriptorSetLayout* setLayout);

void createDescriptorPool(
    VkDevice device,
    const VkDescriptorPoolSize* poolSizes,
    uint32_t poolSizeCount,
    uint32_t maxSets,
    VkDescriptorPool* descriptorPool);

void allocateDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    const VkDescriptorSetLayout* descriptorSetLayout,
    VkDescriptorSet* descriptorSet);
} // namespace vkutil

struct WorkGroupSize {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

class ComputeUnit final {
 public:
#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
  ComputeUnit(
      const char* glslSrc,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize) {
    createComputePipelineCompile(
        std::string{glslSrc, std::strlen(glslSrc)},
        descrSetLayout,
        workGroupSize);
  }
#else
  ComputeUnit(
      const unsigned char* spvCode,
      const unsigned int spvCodeSize,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize) {
    const uint32_t* code = reinterpret_cast<const uint32_t*>(spvCode);
    const auto codeSize = spvCodeSize;
    createComputePipeline(code, codeSize, descrSetLayout, workGroupSize);
  }
#endif

  ~ComputeUnit();
  ComputeUnit(const ComputeUnit&) = delete;
  ComputeUnit& operator=(const ComputeUnit&) = delete;
  ComputeUnit(ComputeUnit&&) = default;
  ComputeUnit& operator=(ComputeUnit&&) = default;

  void createComputePipeline(
      const uint32_t* code,
      const uint32_t codeSize,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize);

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
  void createComputePipelineCompile(
      std::string glslSrc,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize);
#endif

  void createCommandBuffer(VkDescriptorSet& descriptorSet);
  void dispatchCommandBuffer(
      uint32_t groupCountX,
      uint32_t groupCountY,
      uint32_t groupCountZ);
  void runCommandBuffer();
  inline VkCommandBuffer commandBuffer() {
    return commandBuffer_;
  }

 private:
  VkCommandBuffer commandBuffer_;
  VkPipeline pipeline_;
  VkPipelineLayout pipelineLayout_;
  VkShaderModule computeShaderModule_;
}; // class ComputeUnit
} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
