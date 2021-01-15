#pragma once

#include <c10/util/Optional.h>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#ifdef USE_VULKAN_WRAPPER
#include <vulkan_wrapper.h>
#else
#include <vulkan/vulkan.h>
#endif

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#define GLSL_SPV(name) #name, name##_glsl
#else
#include <ATen/native/vulkan/spv.h>
#define GLSL_SPV(name) #name, name##_spv, name##_spv_len
#endif
#include <ATen/native/vulkan/VulkanCommon.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {

#ifdef DEBUG
static constexpr bool kEnableValidationLayers = true;
#else
static constexpr bool kEnableValidationLayers = false;
#endif

bool is_available();

class VContext;
const VContext& context();

// VulkanTensor is a handle that holds shared pointer to VulkanTensor:Impl,
// that owns Tensor representation on GPU.
// VulkanTensor is copyable and moveable (copying and moving pointer to Impl).
//
// VulkanTensor::Impl is moveable only, owns Vulkan device memory for Tensor
// data. Tensor can be represented in several formats.
//
// 0. VBuffer - (wrapper on  vulkan VkBuffer), supports all tensor dimensions,
// data is in Contiguous format (NCHW), in plan to preserve at::Tensor memory
// format (3d or 4d tensors can be in NHWC ChannelsLast format). It is located
// in host visible memory that can be memory mapped to CPU memory.
//
// 1. VImage(TexC4) - (wrapper on vulkan VkImage), optional representation of
// tensors with dimension <= 4 as VkImage, used in shaders as texture or storage
// image. It is 3-dimensional image (x, y, z) with 4 component * 16 bit for each
// triple (x, y, z).
// For NCHW, NHWC:
//
// For dim==4: image.x - W sizes[3]; image.y -  H sizes[2]; image.z - (N
// sizes[0] * C sizes[1]) / 4;
//
// For dim==3: image.x - W sizes[2]; image.y - H sizes[1]; image.z - (C
// sizes[0]) / 4
//
// For dim==2: image.x - W sizes[1]; image.y - H sizes[0]; image.z : 1
//
// For dim==1: image.x - W sizes[0]; image.y : 1; image.z : 1
//
//
// 2. VImage (other format) - Currently not added, but for some operations
// another texture packing format can be beneficial for performance.
//
// Contract about synchronization between representations:
// 1.VImage(TexC4) representation is allocated lazily with calling image(),
// fails for dimensions > 4.
//
// Tensor data can be in 0.VBuffer and/or 1.VImage(TexC4),
// If Tensor can be represented as image - VulkanTensor::Impl::can_be_image()
// returns true. Image representation created lazily by call
// VulkanTensor::Impl::image(), if it is called on Tensor with !can_be_image() -
// it fails.
//
// If image allocated - image data has priority.
// VulkanTensor::copy_data_to_host checks if image allocated -
// copy_image_to_buffer first.
class VBuffer;
class VImage;

using ImageSize = std::array<int32_t, 3>;
struct ImageSizes {
  ImageSize imageSize;
  ImageSize dataSize;
};

class VulkanTensor final {
  class Impl;

 public:
  VulkanTensor() = default;
  explicit VulkanTensor(std::vector<int64_t> sizes);
  ~VulkanTensor() = default;

  VulkanTensor(VulkanTensor&&) = default;
  VulkanTensor& operator=(VulkanTensor&&) = default;

  VulkanTensor(const VulkanTensor&) = default;
  VulkanTensor& operator=(const VulkanTensor&) = default;

  bool defined() const {
    return static_cast<bool>(impl_);
  }

  std::vector<int64_t> sizes() const;
  std::vector<int64_t> strides() const;
  int64_t dim() const;
  int64_t numel() const;

  bool has_storage() const;
  void allocate_storage();
  void set_data_from_host(const float* inputData);
  void copy_data_to_host(float* outputData) const;

  bool has_buffer() const;
  VBuffer* buffer();
  const VBuffer* buffer() const;

  bool can_be_image() const;
  bool has_image() const;

  void sync_image_to_buffer() const;

  // if imageSizes argument is not specified:
  // Allocates VImage of sizes{W,H,NC4} and fills it from tensor VBuffer if it
  // exists, see comment for VulkanTensor.
  //
  // if imageSizes argument is specified:
  // Only allocates VImage of specified sizes, that will be returned on
  // subsequent image() calls. Can be used when user wants to store tensor image
  // not in default{W, H, NC4} format (For performance or other reasons).
  VImage* image(c10::optional<ImageSizes> imageSizes = c10::nullopt);
  const VImage* image(
      c10::optional<ImageSizes> imageSizes = c10::nullopt) const;

 private:
  std::shared_ptr<Impl> impl();
  std::shared_ptr<const Impl> impl() const;
  std::shared_ptr<Impl> impl_;
};

class ComputeUnitFactory;
class VContext final {
 public:
  explicit VContext(bool enableValidationLayers);
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
  ComputeUnitFactory& computeUnitFactory() const {
    return *(computeUnitFactory_.get());
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
  std::unique_ptr<ComputeUnitFactory> computeUnitFactory_;
};

class VBuffer final {
 public:
  class MapMemory final {
   public:
    MapMemory(
        const VkDevice device,
        const VkDeviceMemory deviceMemory,
        const VkDeviceSize offset,
        const VkDeviceSize size)
        : device_(device),
          deviceMemory_(deviceMemory),
          offset_(offset),
          size_(size) {
      vkMapMemory(device_, deviceMemory_, 0, size, 0, &mappedMemory_);
    }
    ~MapMemory() {
      vkUnmapMemory(device_, deviceMemory_);
    }
    MapMemory(const MapMemory&) = delete;
    MapMemory& operator=(const MapMemory&) = delete;
    MapMemory(MapMemory&&) = default;
    MapMemory& operator=(MapMemory&&) = default;
    inline const void* ptr() const {
      return mappedMemory_;
    }
    inline void* ptr() {
      return mappedMemory_;
    }
    void flushWriteToHost();
    void flushWriteToDevice();

   private:
    VkDevice device_;
    VkDeviceMemory deviceMemory_;
    VkDeviceSize offset_;
    VkDeviceSize size_;
    void* mappedMemory_;
  };

  explicit VBuffer(
      VkDeviceSize bufferSizeBytes,
      VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

  ~VBuffer();

  VBuffer(const VBuffer&) = delete;
  VBuffer& operator=(const VBuffer&) = delete;
  VBuffer(VBuffer&&) = default;
  VBuffer& operator=(VBuffer&&) = default;

  static inline VBuffer makeUniformBuffer(const VkDeviceSize bufferSize) {
    return VBuffer{bufferSize,
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  }

  MapMemory map() const {
    return MapMemory{context().device(), bufferMemory_, 0, bufferSizeBytes_};
  }

  void copy_from_device_to_host(void* outputData, int64_t size) const;
  void copy_from_host_to_device(const void* data, int64_t size);
  void set_zeros();

  VkDescriptorBufferInfo makeDescriptorBufferInfo() const;
  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      const VkDescriptorBufferInfo* bufferInfo) const;

  void bind(VkDescriptorSet descriptorSet, uint32_t binding) const;

  inline VkDeviceSize sizeBytes() const {
    return bufferSizeBytes_;
  }

  void addBufferMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkDeviceSize offset,
      VkDeviceSize size) const;

  inline VkBuffer vkbuffer() const {
    return buffer_;
  }

 private:
  VkDeviceSize bufferSizeBytes_;
  VkDescriptorType descriptorType_;
  VkBuffer buffer_;
  VkDeviceMemory bufferMemory_;
};

VBuffer makeUniformConstBuffer(const void* ptr, VkDeviceSize size);

class VImage final {
 public:
  static constexpr VkImageType kImageType = VK_IMAGE_TYPE_3D;
  static constexpr VkFilter kFilter = VK_FILTER_NEAREST;
  static constexpr VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  static constexpr VkSamplerAddressMode kSamplerAddressMode =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  static constexpr VkImageViewType kImageViewType = VK_IMAGE_VIEW_TYPE_3D;

  explicit VImage(ImageSize imageSize, ImageSize dataSize);
  explicit VImage(ImageSizes imageSizes)
      : VImage(imageSizes.imageSize, imageSizes.dataSize) {}
  ~VImage();
  VImage(const VImage&) = delete;
  VImage& operator=(const VImage&) = delete;
  VImage(VImage&&) = default;
  VImage& operator=(VImage&&) = default;

  inline auto w() const {
    return imageSize_[0];
  }
  inline auto h() const {
    return imageSize_[1];
  }
  inline auto d() const {
    return imageSize_[2];
  }

  VkImageViewCreateInfo makeImageViewCreateInfo() const;
  VkSamplerCreateInfo makeSamplerCreateInfo() const;
  VkDescriptorImageInfo makeDescriptorImageInfo(
      VkImageLayout imageLayout) const;
  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      const VkDescriptorImageInfo* imageInfo) const;
  void bind(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      VkImageLayout imageLayout) const;
  void bindShaderRead(VkDescriptorSet descriptorSet, uint32_t binding) const;
  void bindStorageImage(VkDescriptorSet descriptorSet, uint32_t binding) const;
  inline VkDeviceSize sizeBytes() const {
    return sizeof(float) * dataSize_[0] * dataSize_[1] * dataSize_[2];
  }

  inline VkDeviceSize capacityBytes() const {
    // Every VImage pixel(texel) contains 4 float elements
    return sizeof(float) * 4 * imageSize_[0] * imageSize_[1] * imageSize_[2];
  }

  ImageSize sizes() const {
    return imageSize_;
  }

  void addImageMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkImageLayout newLayout) const;
  void addImageMemoryBarrierToGeneral(VkCommandBuffer commandBuffer) const;
  void addImageMemoryBarrierToShaderRead(VkCommandBuffer commandBuffer) const;

 private:
  ImageSize imageSize_;
  ImageSize dataSize_;
  VkImage image_;
  VkDeviceMemory imageMemory_;
  VkImageView imageView_;
  VkSampler sampler_;
  // Holds current image layout that will be used in
  // addImageMemoryBarrier as the previous layout. Need to be mutable to
  // use addImageMemoryBarrier() for const VImage.
  mutable VkImageLayout imageLayout_;
};

void copy_buffer_to_image(const VBuffer& buffer, VImage& image);

void copy_image_to_buffer(
    const VImage& image,
    VBuffer& buffer,
    bool addBufferMemoryBarrierForHost = false);

void copy_buffer_to_buffer(
    const VBuffer& srcBuffer,
    VBuffer& dstBuffer,
    VkDeviceSize size,
    VkDeviceSize srcOffset = 0,
    VkDeviceSize dstOffset = 0);

VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    uint32_t binding,
    VkDescriptorType descriptorType);

void createDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t bindingCount,
    VkDescriptorSetLayout* setLayout);

void allocateDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    const VkDescriptorSetLayout* descriptorSetLayout,
    VkDescriptorSet* descriptorSet);

void createDescriptorSetLayoutSinglePool(
    VkDevice device,
    const std::vector<VkDescriptorType>& descrTypes,
    VkDescriptorSetLayout* descrSetLayout,
    VkDescriptorPool* descrPool,
    VkDescriptorSet* descrSet);

void allocateCommandBuffer(VkDevice device, VkCommandBuffer* commandBuffer);
void beginCommandBuffer(VkCommandBuffer commandBuffer);
void endCommandBuffer(VkCommandBuffer commandBuffer);
void submitAndWaitCommandBuffer(VkDevice device, VkCommandBuffer commandBuffer);

struct WorkGroupSize {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

class ComputeUnit final {
 public:
  static constexpr uint64_t kFenceTimeoutNanos = 100000000000;
#ifdef USE_VULKAN_SHADERC_RUNTIME
  ComputeUnit(
      const char* const glslSrc,
      const VkPipelineCache pipelineCache,
      const VkDescriptorSetLayout descrSetLayout,
      const WorkGroupSize workGroupSize) {
    createComputePipelineCompile(
        glslSrc, pipelineCache, descrSetLayout, workGroupSize);
  }
#else
  ComputeUnit(
      const uint32_t* const spvCode,
      const unsigned int spvCodeSize,
      const VkPipelineCache pipelineCache,
      const VkDescriptorSetLayout& descrSetLayout,
      const WorkGroupSize workGroupSize) {
    const auto codeSize = spvCodeSize;
    createComputePipeline(
        spvCode, codeSize, pipelineCache, descrSetLayout, workGroupSize);
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
      VkPipelineCache pipelineCache,
      VkDescriptorSetLayout descrSetLayout,
      WorkGroupSize workGroupSize);

#ifdef USE_VULKAN_SHADERC_RUNTIME
  void createComputePipelineCompile(
      const std::string& glslSrc,
      const VkPipelineCache pipelineCache,
      const VkDescriptorSetLayout descrSetLayout,
      const WorkGroupSize workGroupSize);
#endif

  void createCommandBuffer(VkDescriptorSet& descriptorSet);
  void addMemoryBarrier(
      VkPipelineStageFlags srcStageMask,
      VkAccessFlags srcAccessMask,
      VkPipelineStageFlags dstStageMask,
      VkAccessFlags dstAccessMask);
  void dispatchCommandBuffer(
      uint32_t groupCountX,
      uint32_t groupCountY,
      uint32_t groupCountZ);
  void dispatchCommandBuffer(
      uint32_t gridX,
      uint32_t gridY,
      uint32_t gridZ,
      WorkGroupSize workGroupSize);
  void submitAndWaitCommandBuffer();
  void endCommandBuffer();
  inline VkCommandBuffer commandBuffer() {
    return commandBuffer_;
  }

 private:
  VkCommandBuffer commandBuffer_;
  VkPipeline pipeline_;
  VkPipelineLayout pipelineLayout_;
  VkShaderModule computeShaderModule_;
};

class ComputeUnitFactory {
 public:
  explicit ComputeUnitFactory(const VkDevice device);
  ~ComputeUnitFactory();
  ComputeUnitFactory(const ComputeUnitFactory&) = default;
  ComputeUnitFactory& operator=(const ComputeUnitFactory&) = default;
  ComputeUnitFactory(ComputeUnitFactory&&) = default;
  ComputeUnitFactory& operator=(ComputeUnitFactory&&) = default;

#ifdef USE_VULKAN_SHADERC_RUNTIME
  ComputeUnit& get(
      const char* key,
      const char* glslSrc,
      VkDescriptorSetLayout descrSetLayout,
      WorkGroupSize workGroupSize);
#else
  ComputeUnit& get(
      const char* key,
      const uint32_t* code,
      const uint32_t codeSize,
      VkDescriptorSetLayout descrSetLayout,
      WorkGroupSize workGroupSize);
#endif
 private:
  std::string getCacheKey(const char* key, WorkGroupSize workGroupSize);
  ComputeUnit& get(
      const std::string& cacheKey,
      std::function<std::shared_ptr<ComputeUnit>()> factoryFn);

  VkDevice device_;
  VkPipelineCache pipelineCache_;
  std::unordered_map<std::string, std::shared_ptr<ComputeUnit>> computeUnits_;
};

std::ostream& operator<<(std::ostream& s, const WorkGroupSize& workGroupSize);
std::ostream& operator<<(std::ostream& s, const ImageSize& imageSize);
std::ostream& operator<<(std::ostream& s, const ImageSizes& imageSizes);

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
