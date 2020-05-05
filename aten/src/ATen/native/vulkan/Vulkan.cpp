#ifdef USE_VULKAN

#include <stdio.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <iostream>

#include <c10/util/Exception.h>

#ifdef USE_VULKAN_WRAPPER
#include "vulkan_wrapper.h"
#else
#include <vulkan/vulkan.h>
#endif

#include <ATen/native/vulkan/Vulkan.h>

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#include "shaderc/shaderc.hpp"
#define GLSL_SPV(name) name##_glsl
#else
#include <ATen/native/vulkan/spv.h>
#define GLSL_SPV(name) name##_spv, name##_spv_len
#endif

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#define VK_CHECK(f)                                                \
  {                                                                \
    VkResult res = (f);                                            \
    TORCH_CHECK(res == VK_SUCCESS, "Vulkan error VkResult:", res); \
  }

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

class VContext final {
 public:
  VContext(bool enableValidationLayers)
      : enableValidationLayers_(enableValidationLayers) {
    createInstance();
    findPhysicalDevice();
    createDevice();
  }
  ~VContext() {
    cleanup();
  }

  inline VkDevice device() {
    return device_;
  }

  inline VkPhysicalDevice physicalDevice() {
    return physicalDevice_;
  }

  inline VkPhysicalDeviceLimits limits() {
    return physicalDeviceLimits_;
  }

  inline VkCommandPool commandPool() {
    return commandPool_;
  }

  inline VkQueue queue() {
    return queue_;
  }

 private:
  void createInstance() {
    std::vector<const char*> enabledExtensions;
    if (enableValidationLayers_) {
      uint32_t layerPresentCount;
      vkEnumerateInstanceLayerProperties(&layerPresentCount, nullptr);
      std::vector<VkLayerProperties> layerProps(layerPresentCount);
      vkEnumerateInstanceLayerProperties(&layerPresentCount, layerProps.data());
      const char* instanceLayers[] = {
          "VK_LAYER_GOOGLE_unique_objects",
          "VK_LAYER_GOOGLE_threading",
          "VK_LAYER_LUNARG_object_tracker",
          "VK_LAYER_LUNARG_core_validation",
          "VK_LAYER_LUNARG_parameter_validation",
          "VK_LAYER_KHRONOS_validation",
      };

      uint32_t instanceLayersRequestCount =
          sizeof(instanceLayers) / sizeof(instanceLayers[0]);
      for (uint32_t i = 0; i < instanceLayersRequestCount; i++) {
        bool found = false;
        for (uint32_t j = 0; j < layerPresentCount; j++) {
          if (strcmp(instanceLayers[i], layerProps[j].layerName) == 0) {
            found = true;
          }
        }
        if (found) {
          enabledValidationLayers_.push_back(instanceLayers[i]);
        }
      }

      uint32_t extCount;
      vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
      std::vector<VkExtensionProperties> extProps(extCount);
      vkEnumerateInstanceExtensionProperties(
          nullptr, &extCount, extProps.data());
      bool foundExt = false;
      for (VkExtensionProperties p : extProps) {
        if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, p.extensionName) == 0) {
          foundExt = true;
          break;
        }
      }
      if (foundExt) {
        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
      }
    }

    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "pytorch";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "compute";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledLayerCount = enabledValidationLayers_.size();
    createInfo.ppEnabledLayerNames = enabledValidationLayers_.data();
    createInfo.enabledExtensionCount = enabledExtensions.size();
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    vkCreateInstance(&createInfo, nullptr, &instance_);

    if (enableValidationLayers_) {
      VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo = {};
      debugReportCallbackCreateInfo.sType =
          VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
      debugReportCallbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
          VK_DEBUG_REPORT_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
      debugReportCallbackCreateInfo.pfnCallback = &debugReportCallbackFn;

      auto vkCreateDebugReportCallbackEXT =
          (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
              instance_, "vkCreateDebugReportCallbackEXT");
      TORCH_CHECK(
          vkCreateDebugReportCallbackEXT,
          "Could not load vkCreateDebugReportCallbackEXT");
      VK_CHECK(vkCreateDebugReportCallbackEXT(
          instance_,
          &debugReportCallbackCreateInfo,
          nullptr,
          &debugReportCallback_));
    }
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
      VkDebugReportFlagsEXT msgFlags,
      VkDebugReportObjectTypeEXT objectType,
      uint64_t object,
      size_t location,
      int32_t msgCode,
      const char* pLayerPrefix,
      const char* pMsg,
      void* pUserData) {
    std::stringstream s;
    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
      s << "ERROR:";
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
      s << "WARN:";
    } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
      s << "PERF_WARNING:";
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
      s << "INFO:";
    }
    s << pLayerPrefix << " " << msgCode << " " << pMsg << std::endl;
    std::cout << s.str();
    // TODO Where to log if VLOG,LOG disabled?
    return VK_FALSE;
  }

  void findPhysicalDevice() {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    TORCH_CHECK(
        deviceCount > 0, "Vulkan: Could not find a device with vulkan support");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());
    int i = 0;
    bool found = false;
    for (VkPhysicalDevice device : devices) {
      if (!found) {
        physicalDevice_ = device;
        found = true;
      }
      i++;
    }
  }

  uint32_t getComputeQueueFamilyIndex() {
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice_, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice_, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;

    bool queueFound = false;
    for (; i < queueFamilies.size(); ++i) {
      VkQueueFamilyProperties props = queueFamilies[i];
      if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        queueFound = true;
        break;
      }
    }

    TORCH_CHECK(
        queueFound,
        "Vulkan: Could not find a queue family that supports operations");
    return i;
  }

  void createDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueFamilyIndex_ = getComputeQueueFamilyIndex();
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
    queueCreateInfo.queueCount = 1;
    float queuePriorities = 1.0;
    queueCreateInfo.pQueuePriorities = &queuePriorities;
    VkDeviceCreateInfo deviceCreateInfo = {};
    VkPhysicalDeviceFeatures deviceFeatures = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = enabledValidationLayers_.size();
    deviceCreateInfo.ppEnabledLayerNames = enabledValidationLayers_.data();
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VK_CHECK(
        vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_));
    queue_ = {};
    vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
    VK_CHECK(vkCreateCommandPool(
        device_, &commandPoolCreateInfo, nullptr, &commandPool_));
    physicalDeviceLimits_ = physicalDeviceProperties.limits;
  }

  void cleanup() {
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (enableValidationLayers_) {
      auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
          instance_, "vkDestroyDebugReportCallbackEXT");
      TORCH_CHECK(func, "Could not load vkDestroyDebugReportCallbackEXT");
      func(instance_, debugReportCallback_, nullptr);
    }

    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
  }

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
static std::unique_ptr<VContext> vkContext;
static constexpr bool kEnableValidationLayers = true;

bool initVulkanContextOnce() {
  static const int once = []() {
#ifdef USE_VULKAN_WRAPPER
    if (!InitVulkan()) {
      TORCH_WARN("Vulkan Wrapper Failed to InitVulkan");
      return 1;
    }
#endif
    vkContext = std::make_unique<VContext>(kEnableValidationLayers);
    if (!vkContext) {
      TORCH_WARN("Vulkan Failed to create Vulkan Context");
      return 2;
    }
    return 0;
  }();
  ((void)once);
  return static_cast<bool>(vkContext);
}

uint32_t findMemoryType(
    VkPhysicalDevice& physicalDevice,
    uint32_t memoryTypeBits,
    VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if ((memoryTypeBits & (1 << i)) &&
        ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
         properties))
      return i;
  }
  return -1;
}

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
      VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      : bufferSizeBytes_(bufferSizeBytes), descriptorType_(descriptorType) {
    auto device = vkContext->device();
    auto physicalDevice = vkContext->physicalDevice();
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSizeBytes_;
    bufferCreateInfo.usage = bufferUsageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer_));
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer_, &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(
        physicalDevice,
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory_));
    VK_CHECK(vkBindBufferMemory(device, buffer_, bufferMemory_, 0));
  }

  ~VBuffer() noexcept {
    vkFreeMemory(vkContext->device(), bufferMemory_, nullptr);
    vkDestroyBuffer(vkContext->device(), buffer_, nullptr);
  }
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
    return MapMemory{vkContext->device(), bufferMemory_, 0, bufferSizeBytes_};
  }

  void setZeros() {
    void* mappedMemory = nullptr;
    vkMapMemory(
        vkContext->device(),
        bufferMemory_,
        0,
        bufferSizeBytes_,
        0,
        &mappedMemory);
    ::memset(mappedMemory, 0, bufferSizeBytes_);
    vkUnmapMemory(vkContext->device(), bufferMemory_);
  }

  void copyFromDeviceToHost(void* outputData, int64_t size) {
    auto mm = map();
    TORCH_INTERNAL_ASSERT(
        mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
    ::memcpy(outputData, mm.ptr(), size);
  }

  void copyFromHostToDevice(void* data, int64_t size) {
    auto mm = map();
    TORCH_INTERNAL_ASSERT(
        mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
    ::memcpy(mm.ptr(), data, size);
  }

  VkDescriptorBufferInfo makeDescriptorBufferInfo() {
    VkDescriptorBufferInfo info = {};
    info.buffer = buffer_;
    info.offset = 0;
    info.range = bufferSizeBytes_;
    return info;
  }

  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      const VkDescriptorBufferInfo* bufferInfo) {
    VkWriteDescriptorSet writeSet{};
    writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.pNext = nullptr;
    writeSet.dstSet = descriptorSet;
    writeSet.dstBinding = binding;
    writeSet.dstArrayElement = 0;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType = descriptorType_;
    writeSet.pImageInfo = nullptr;
    writeSet.pBufferInfo = bufferInfo;
    writeSet.pTexelBufferView = nullptr;
    return writeSet;
  }

  void bind(VkDescriptorSet descriptorSet, uint32_t binding) {
    auto descrBufferInfo = makeDescriptorBufferInfo();
    auto writeDescrSet =
        makeWriteDescriptorSet(descriptorSet, binding, &descrBufferInfo);
    vkUpdateDescriptorSets(vkContext->device(), 1, &writeDescrSet, 0, nullptr);
  }

  inline VkDeviceSize sizeBytes() const {
    return bufferSizeBytes_;
  }

  void addBufferMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkDeviceSize offset,
      VkDeviceSize size) {
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer = buffer_;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.offset = offset;
    barrier.pNext = nullptr;
    barrier.size = size;
    barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &barrier,
        0,
        nullptr);
  }

 private:
  VkDeviceSize bufferSizeBytes_;
  VkDescriptorType descriptorType_;
  VkBuffer buffer_;
  VkDeviceMemory bufferMemory_;
}; // class VBuffer

class VImage final {
 public:
  explicit VImage(uint32_t W, uint32_t H, uint32_t C)
      : W_(W), H_(H), C_(C), D_(UP_DIV(C, 4)) {
    auto device = vkContext->device();
    auto physicalDevice = vkContext->physicalDevice();

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = imageType_;
    imageInfo.extent.width = W_;
    imageInfo.extent.height = H_;
    imageInfo.extent.depth = D_;

    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format_;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = imageLayoutInitial_;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.pNext = nullptr;
    imageInfo.flags = 0;

    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image_));

    VkMemoryRequirements memReqs = {};
    vkGetImageMemoryRequirements(device, image_, &memReqs);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        physicalDevice,
        memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory_));
    VK_CHECK(vkBindImageMemory(device, image_, imageMemory_, 0));

    VkImageViewCreateInfo imageViewCreateInfo = makeImageViewCreateInfo();
    VK_CHECK(
        vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView_));

    VkSamplerCreateInfo samplerCreateInfo = makeSamplerCreateInfo();
    VK_CHECK(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler_));
  }
  ~VImage() noexcept {
    vkFreeMemory(vkContext->device(), imageMemory_, nullptr);
    vkDestroySampler(vkContext->device(), sampler_, nullptr);
    vkDestroyImageView(vkContext->device(), imageView_, nullptr);
    vkDestroyImage(vkContext->device(), image_, nullptr);
  }

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

  VkImageViewCreateInfo makeImageViewCreateInfo() {
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image_;
    info.viewType = viewType_;
    info.format = format_;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;
    return info;
  }

  VkSamplerCreateInfo makeSamplerCreateInfo() {
    VkSamplerCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter_;
    info.minFilter = filter_;
    info.addressModeU = samplerAddressMode_;
    info.addressModeV = samplerAddressMode_;
    info.addressModeW = samplerAddressMode_;
    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy = 1.0f;
    info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.compareEnable = VK_FALSE;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.mipLodBias = 0.0f;
    info.minLod = 0.0f;
    info.maxLod = 0.0f;
    return info;
  }

  VkDescriptorImageInfo makeDescriptorImageInfo(VkImageLayout imageLayout) {
    VkDescriptorImageInfo info = {};
    info.sampler = sampler_;
    info.imageView = imageView_;
    info.imageLayout = imageLayout;
    return info;
  }

  VkWriteDescriptorSet makeWriteDescriptorSet(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      const VkDescriptorImageInfo* imageInfo) {
    VkWriteDescriptorSet writeSet{};
    writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.pNext = nullptr;
    writeSet.dstSet = descriptorSet;
    writeSet.dstBinding = binding;
    writeSet.dstArrayElement = 0;
    writeSet.descriptorCount = 1;
    writeSet.descriptorType = descriptorType, writeSet.pImageInfo = imageInfo;
    writeSet.pBufferInfo = nullptr;
    writeSet.pTexelBufferView = nullptr;
    return writeSet;
  }

  void bind(
      VkDescriptorSet descriptorSet,
      uint32_t binding,
      VkDescriptorType descriptorType,
      VkImageLayout imageLayout) {
    auto descrImageInfo = makeDescriptorImageInfo(imageLayout);
    auto writeDescrSet = makeWriteDescriptorSet(
        descriptorSet, binding, descriptorType, &descrImageInfo);
    vkUpdateDescriptorSets(vkContext->device(), 1, &writeDescrSet, 0, nullptr);
  }

  inline VkDeviceSize sizeBytes() const {
    return sizeof(float) * W_ * H_ * C_;
  }

  inline VkDeviceSize capacityBytes() const {
    return sizeof(float) * W_ * H_ * D_ * 4;
  }

  void addImageMemoryBarrier(
      VkCommandBuffer commandBuffer,
      VkImageLayout oldLayout,
      VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_;
    barrier.newLayout = newLayout;
    barrier.oldLayout = oldLayout;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    } else if (
        oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
        newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (
        oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
        newLayout == VK_IMAGE_LAYOUT_GENERAL) {
      barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Vulkan: Unsupported Vulkan Image Layout transition");
    }
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);
  }

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

VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    uint32_t binding,
    VkDescriptorType descriptorType) {
  return {binding, descriptorType, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
}

void createDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t bindingCount,
    VkDescriptorSetLayout* setLayout) {
  VkDescriptorSetLayoutCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.bindingCount = bindingCount;
  createInfo.pBindings = bindings;
  VK_CHECK(
      vkCreateDescriptorSetLayout(device, &createInfo, nullptr, setLayout));
}

void createDescriptorPool(
    VkDevice device,
    const VkDescriptorPoolSize* poolSizes,
    uint32_t poolSizeCount,
    uint32_t maxSets,
    VkDescriptorPool* descriptorPool) {
  VkDescriptorPoolCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.maxSets = maxSets;
  createInfo.poolSizeCount = poolSizeCount;
  createInfo.pPoolSizes = poolSizes;
  VK_CHECK(
      vkCreateDescriptorPool(device, &createInfo, nullptr, descriptorPool));
}

void allocateDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    const VkDescriptorSetLayout* descriptorSetLayout,
    VkDescriptorSet* descriptorSet) {
  VkDescriptorSetAllocateInfo allocateInfo = {};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.pNext = nullptr;
  allocateInfo.descriptorPool = descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = descriptorSetLayout;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSet));
}

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

  ~ComputeUnit() {
    vkDestroyShaderModule(vkContext->device(), computeShaderModule_, nullptr);
    vkDestroyPipelineLayout(vkContext->device(), pipelineLayout_, nullptr);
    vkDestroyPipeline(vkContext->device(), pipeline_, nullptr);
  }

  ComputeUnit(const ComputeUnit&) = delete;
  ComputeUnit& operator=(const ComputeUnit&) = delete;
  ComputeUnit(ComputeUnit&&) = default;
  ComputeUnit& operator=(ComputeUnit&&) = default;

  void createComputePipeline(
      const uint32_t* code,
      const uint32_t codeSize,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize) {
    auto device = vkContext->device();
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = codeSize;

    VK_CHECK(vkCreateShaderModule(
        device, &createInfo, nullptr, &computeShaderModule_));

    VkSpecializationMapEntry spMapEntries[3];
    spMapEntries[0].constantID = 1;
    spMapEntries[0].offset = 0;
    spMapEntries[0].size = sizeof(WorkGroupSize::x);
    spMapEntries[1].constantID = 2;
    spMapEntries[1].offset = 4;
    spMapEntries[1].size = sizeof(WorkGroupSize::y);
    spMapEntries[2].constantID = 3;
    spMapEntries[2].offset = 8;
    spMapEntries[2].size = sizeof(WorkGroupSize::z);

    VkSpecializationInfo spInfo;
    spInfo.mapEntryCount = 3;
    spInfo.pMapEntries = spMapEntries;
    spInfo.dataSize = sizeof(workGroupSize);
    spInfo.pData = &workGroupSize;

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeShaderModule_;
    shaderStageCreateInfo.pName = "main";
    shaderStageCreateInfo.pSpecializationInfo = &spInfo;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descrSetLayout;

    VK_CHECK(vkCreatePipelineLayout(
        device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout_;

    VK_CHECK(vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline_));
  }

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
  void createComputePipelineCompile(
      std::string glslSrc,
      const VkDescriptorSetLayout& descrSetLayout,
      WorkGroupSize& workGroupSize) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetGenerateDebugInfo();
    options.SetTargetEnvironment(
        shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
    options.SetForcedVersionProfile(450, shaderc_profile_core);
    shaderc::SpvCompilationResult compilationResult = compiler.CompileGlslToSpv(
        glslSrc.c_str(),
        glslSrc.size(),
        shaderc_compute_shader,
        "vulkan_shader.comp",
        "main",
        options);
    auto compilationStatus = compilationResult.GetCompilationStatus();
    TORCH_INTERNAL_ASSERT(
        compilationStatus == shaderc_compilation_status_success,
        "Shader compilation error: status:",
        compilationStatus,
        compilationResult.GetErrorMessage());
    std::vector<uint32_t> shaderSpvCode(
        compilationResult.cbegin(), compilationResult.cend());
    const auto codeSizeBytes = 4 * shaderSpvCode.size();
    createComputePipeline(
        shaderSpvCode.data(), codeSizeBytes, descrSetLayout, workGroupSize);
  }
#endif

  void createCommandBuffer(VkDescriptorSet& descriptorSet) {
    auto device = vkContext->device();
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = vkContext->commandPool();
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(
        device, &commandBufferAllocateInfo, &commandBuffer_));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

    vkCmdBindPipeline(
        commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout_,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
  }

  void dispatchCommandBuffer(
      uint32_t groupCountX,
      uint32_t groupCountY,
      uint32_t groupCountZ) {
    vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
    VK_CHECK(vkEndCommandBuffer(commandBuffer_));
  }

  void runCommandBuffer() {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK(vkCreateFence(vkContext->device(), &fenceCreateInfo, NULL, &fence))

    VK_CHECK(vkQueueSubmit(vkContext->queue(), 1, &submitInfo, fence));
    vkWaitForFences(vkContext->device(), 1, &fence, VK_TRUE, 100000000000);

    vkDestroyFence(vkContext->device(), fence, NULL);
  }

  inline VkCommandBuffer commandBuffer() {
    return commandBuffer_;
  }

 private:
  VkCommandBuffer commandBuffer_;
  VkPipeline pipeline_;
  VkPipelineLayout pipelineLayout_;
  VkShaderModule computeShaderModule_;
}; // class ComputeUnit

VBuffer makeUniformConstBuffer(void* ptr, VkDeviceSize size) {
  auto sizeAligned =
      ROUND_UP(size, vkContext->limits().minUniformBufferOffsetAlignment);
  VBuffer constBuffer = VBuffer::makeUniformBuffer(sizeAligned);
  constBuffer.copyFromHostToDevice(ptr, size);
  return constBuffer;
}

// VBuffer <-> VImage
void copyFromBufferToImage(VBuffer& buffer, VImage& image) {
  auto device = vkContext->device();
  auto physicalDevice = vkContext->physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
  };
  ConstBlock constBlock{image.W(), image.H()};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  image.bind(
      descrSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
  buffer.bind(descrSet, 1);
  constBuffer.bind(descrSet, 2);
  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_nchw_to_image),
                          descrSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descrSet);

  image.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL);

  buffer.addBufferMemoryBarrier(
      computeUnit.commandBuffer(), 0, buffer.sizeBytes());
  computeUnit.dispatchCommandBuffer(
      UP_DIV(image.W(), workGroupSize.x),
      UP_DIV(image.H(), workGroupSize.y),
      UP_DIV(image.D(), workGroupSize.z));
  computeUnit.runCommandBuffer();

  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
}

void copyFromImageToBuffer(VImage& image, VBuffer& buffer) {
  auto device = vkContext->device();
  auto physicalDevice = vkContext->physicalDevice();
  TORCH_INTERNAL_ASSERT(
      buffer.sizeBytes() >= image.capacityBytes(),
      "VulkanBuffer's capacity is less than VulkanImage capacity to copy from");
  struct ConstBlock {
    int32_t W;
    int32_t H;
  };
  ConstBlock constBlock{image.W(), image.H()};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  image.bind(
      descrSet,
      0,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  buffer.bind(descrSet, 1);
  constBuffer.bind(descrSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_image_to_nchw),
                          descrSetLayout,
                          workGroupSize};

  computeUnit.createCommandBuffer(descrSet);
  image.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(image.W(), workGroupSize.x),
      UP_DIV(image.H(), workGroupSize.y),
      UP_DIV(image.D(), workGroupSize.z));
  computeUnit.runCommandBuffer();

  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
} // VBuffer <-> VImage

// VulkanTensor
class VulkanTensor::Impl {
  friend class VulkanTensor;

 public:
  Impl(std::vector<int64_t> sizes) : sizes_(std::move(sizes)) {
    int64_t numel = 1;
    for (const auto& d : sizes_) {
      numel *= d;
    }
    numel_ = numel;
  }

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  inline int64_t dim() const {
    return sizes_.size();
  }

  inline int64_t numel() const {
    return numel_;
  }

  inline bool hasBuffer() {
    return static_cast<bool>(buffer_);
  }

  inline VBuffer& buffer() {
    return *(buffer_.get());
  }

  inline bool hasImage() {
    return static_cast<bool>(image_);
  }

  VImage& image() {
    if (!image_ && buffer_) {
      auto W = sizes_[3];
      auto H = sizes_[2];
      auto C = sizes_[1] * sizes_[0];
      image_ = std::make_unique<VImage>(W, H, C);
      copyFromBufferToImage(*buffer_, *image_);
    }
    return *(image_.get());
  }

  inline bool hasStorage() {
    return hasBuffer();
  }

  void allocateStorage() {
    const auto bufferSize = sizeof(float) * sizes_[0] * ALIGN_UP4(sizes_[1]) *
        sizes_[2] * sizes_[3];
    const auto bufferSizeAligned = ROUND_UP(
        bufferSize, vkContext->limits().minStorageBufferOffsetAlignment);
    buffer_ = std::make_unique<VBuffer>(bufferSizeAligned);
  }

  void setDataFromHost(const float* inputData) {
    if (!hasStorage()) {
      allocateStorage();
    }
    buffer_->copyFromHostToDevice((void*)inputData, sizeof(float) * numel_);
  }

  void copyDataToHost(float* outputData) {
    if (hasImage()) {
      copyFromImageToBuffer(image(), buffer());
    }
    buffer_->copyFromDeviceToHost(outputData, sizeof(float) * numel_);
  }

 private:
  std::vector<int64_t> sizes_;
  int64_t numel_;
  std::unique_ptr<VBuffer> buffer_;
  std::unique_ptr<VImage> image_;
};

VulkanTensor::VulkanTensor(std::vector<int64_t> sizes)
    : pImpl(std::make_shared<Impl>(std::move(sizes))) {
  TORCH_CHECK(
      initVulkanContextOnce(), "Vulkan Failed to create Vulkan Context");
}

std::vector<int64_t> VulkanTensor::sizes() const {
  return impl()->sizes();
}

bool VulkanTensor::hasStorage() const {
  return impl()->hasBuffer();
}

void VulkanTensor::allocateStorage() {
  impl()->allocateStorage();
}

void VulkanTensor::setDataFromHost(const float* inputData) {
  impl()->setDataFromHost(inputData);
}

void VulkanTensor::copyDataToHost(float* outputData) {
  impl()->copyDataToHost(outputData);
}

// Ops
void upsample_nearest2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t _N,
    int64_t _C,
    float scaleH,
    float scaleW) {
  auto device = vkContext->device();
  auto physicalDevice = vkContext->physicalDevice();
  int64_t C = _N * _C;
  struct ConstBlock {
    int32_t IW;
    int32_t IH;
    int32_t OW;
    int32_t OH;
    float scaleX;
    float scaleY;
  };
  ConstBlock constBlock{IW, IH, OW, OH, scaleW, scaleH};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);
  output.impl()->image().bind(
      descrSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
  input.impl()->image().bind(
      descrSet,
      1,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  constBuffer.bind(descrSet, 2);

  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit{
      at::native::vulkan::GLSL_SPV(vulkan_upsampleNearest2d),
      descrSetLayout,
      workGroupSize};
  computeUnit.createCommandBuffer(descrSet);
  input.impl()->image().addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(OW, workGroupSize.x),
      UP_DIV(OH, workGroupSize.y),
      UP_DIV(C, workGroupSize.z));
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
}

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha) {
  TORCH_INTERNAL_ASSERT(
      output.impl()->dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, output is not 4-dim");
  TORCH_INTERNAL_ASSERT(
      input0.impl()->dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, input0 is not 4-dim");
  TORCH_INTERNAL_ASSERT(
      input1.impl()->dim() == 4,
      "Vulkan add is implemented for 4-dim tensors, input1 is not 4-dim");
  auto sizes = output.impl()->sizes();
  auto C = sizes[0] * sizes[1];
  auto H = sizes[2];
  auto W = sizes[3];

  auto device = vkContext->device();
  auto physicalDevice = vkContext->physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
    int32_t C;
    float alpha;
  };
  ConstBlock constBlock{W, H, C, alpha};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 4 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 4 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);
  output.impl()->image().bind(
      descrSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
  input0.impl()->image().bind(
      descrSet,
      1,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  input1.impl()->image().bind(
      descrSet,
      2,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  constBuffer.bind(descrSet, 3);

  WorkGroupSize workGroupSize{8, 8, 1};
  ComputeUnit computeUnit{
      at::native::vulkan::GLSL_SPV(vulkan_add), descrSetLayout, workGroupSize};
  computeUnit.createCommandBuffer(descrSet);
  output.impl()->image().addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL);
  input0.impl()->image().addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  input1.impl()->image().addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(W, workGroupSize.x),
      UP_DIV(H, workGroupSize.y),
      UP_DIV(C, workGroupSize.z));
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
}

VBuffer kernelNCHW_OCHW_repack_O4C4HWi4o4(
    const float* weights,
    const int OC,
    const int C,
    const int KH,
    const int KW) {
  const auto Cau4 = ALIGN_UP4(C);
  const auto C_4 = UP_DIV(C, 4);
  const auto kBufSizeNumel = ALIGN_UP4(OC) * Cau4 * KH * KW;
  auto size = sizeof(float) * kBufSizeNumel;
  auto sizeAligned =
      ROUND_UP(size, vkContext->limits().minStorageBufferOffsetAlignment);
  VBuffer kernelBuffer{sizeAligned};
  const int oc_4SizeNumel = KW * KH * C_4 * 16;
  auto mappedMemory = kernelBuffer.map();
  if (mappedMemory.ptr()) {
    float* basePtr = (float*)mappedMemory.ptr();
    memset(basePtr, 0, size);
    const float* src = weights;
    int ridx = 0;
    for (int oc = 0; oc < OC; ++oc) {
      int oc_4 = oc / 4;
      int oc_4_i = oc % 4;
      float* dst_oc = basePtr + oc_4 * oc_4SizeNumel;
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
  return kernelBuffer;
}

VImage conv2d_kernelImage_from_hostCHW(
    const float* data,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW) {
  auto device = vkContext->device();
  auto kernelBuffer = kernelNCHW_OCHW_repack_O4C4HWi4o4(data, OC, C, KH, KW);
  auto OC_4 = UP_DIV(OC, 4);
  auto C_4 = UP_DIV(C, 4);

  VImage kernelImage{C_4 * 4, OC_4, 4 * KH * KW};
  struct ConstBlock {
    int32_t KWxKH;
    int32_t C_4;
  };
  ConstBlock constBlock{KW * KH, C_4};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  kernelImage.bind(
      descrSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
  kernelBuffer.bind(descrSet, 1);
  constBuffer.bind(descrSet, 2);

  WorkGroupSize workGroupSize{1, 1, 1};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_KO4C4HW_to_image),
                          descrSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descrSet);
  kernelImage.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL);
  kernelBuffer.addBufferMemoryBarrier(
      computeUnit.commandBuffer(), 0, kernelBuffer.sizeBytes());
  computeUnit.dispatchCommandBuffer(
      UP_DIV(C_4, workGroupSize.x),
      UP_DIV(OC_4, workGroupSize.y),
      UP_DIV(KH * KW, workGroupSize.z));
  computeUnit.runCommandBuffer();
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
  return kernelImage;
}

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
    int64_t G) {
  auto device = vkContext->device();
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

  VImage inputImage{W, H, C};
  copyFromBufferToImage(input.impl()->buffer(), inputImage);
  VImage outputImage{OW, OH, OC};

  auto biasBufferSize = sizeof(float) * ALIGN_UP4(OC);
  auto biasBufferSizeAligned = ROUND_UP(
      biasBufferSize, vkContext->limits().minStorageBufferOffsetAlignment);
  VBuffer biasBuffer{biasBufferSizeAligned};
  biasBuffer.copyFromHostToDevice((void*)bias, biasBufferSize);

  struct ConstBlock {
    int32_t padding[2];
    int32_t kernelSize[2];
    int32_t stride[2];
    int32_t dilate[2];
    int32_t inputSize[4];
    int32_t outputSize[4];
  };
  ConstBlock constBlock{{PX, PY},
                        {KW, KH},
                        {SX, SY},
                        {DX, DY},
                        {OW, OH, OC_4, 0},
                        {W, H, C_4, 0}};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));
  VImage kernelImage = conv2d_kernelImage_from_hostCHW(weight, OC, C, KH, KW);

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 5 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 5 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);
  outputImage.bind(
      descrSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);
  inputImage.bind(
      descrSet,
      1,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  kernelImage.bind(
      descrSet,
      2,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  biasBuffer.bind(descrSet, 3);
  constBuffer.bind(descrSet, 4);
  WorkGroupSize workGroupSize{1, 1, OC_4};
  ComputeUnit computeUnit{at::native::vulkan::GLSL_SPV(vulkan_conv_tex_IKnc4hw),
                          descrSetLayout,
                          workGroupSize};
  computeUnit.createCommandBuffer(descrSet);

  outputImage.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL);
  inputImage.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  kernelImage.addImageMemoryBarrier(
      computeUnit.commandBuffer(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  computeUnit.dispatchCommandBuffer(
      UP_DIV(OW, 4 * workGroupSize.x),
      UP_DIV(OH, workGroupSize.y),
      UP_DIV(OC_4, workGroupSize.z));
  computeUnit.runCommandBuffer();

  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);

  copyFromImageToBuffer(outputImage, output.impl()->buffer());
}
// ---Ops

bool is_available() {
  return initVulkanContextOnce();
}

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
