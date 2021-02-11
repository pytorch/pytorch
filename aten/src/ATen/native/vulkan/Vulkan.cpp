#include <ATen/Utils.h>
#include <c10/util/accumulate.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#ifdef USE_VULKAN_WRAPPER
#include <vulkan_wrapper.h>
#else
#include <vulkan/vulkan.h>
#endif

#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanCommon.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#include <shaderc/shaderc.hpp>
#else
#include <ATen/native/vulkan/spv.h>
#endif

#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <unistd.h>


#define VK_CHECK(f)                                                \
  {                                                                \
    VkResult res = (f);                                            \
    TORCH_CHECK(res == VK_SUCCESS, "Vulkan error VkResult:", res); \
  }

namespace at {
namespace native {
namespace vulkan {
namespace detail {

VContext::VContext(const bool enableValidationLayers)
    : enableValidationLayers_(enableValidationLayers) {
  createInstance();
  findPhysicalDevice();
  createDevice();

  computeUnitFactory_ = std::make_unique<ComputeUnitFactory>(device_);
}

VContext::~VContext() {
  if (enableValidationLayers_) {
    const auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
        instance_, "vkDestroyDebugReportCallbackEXT");
    if (func) {
      func(instance_, debugReportCallback_, nullptr);
    }
  }

  // ComputeUnitFactory_ owns ComputeUnits and VkPipelineCache, need valid
  // VkDevice for destructing, destructing before vkDestroyDevice
  computeUnitFactory_.reset();

  vkDestroyCommandPool(device_, commandPool_, nullptr);
  vkDestroyDevice(device_, nullptr);
  vkDestroyInstance(instance_, nullptr);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
    const VkDebugReportFlagsEXT msgFlags,
    const VkDebugReportObjectTypeEXT objectType,
    const uint64_t object,
    const size_t location,
    const int32_t msgCode,
    const char* const pLayerPrefix,
    const char* const pMsg,
    void* const pUserData) {
  std::stringstream s;
  s << pLayerPrefix << " " << msgCode << " " << pMsg << std::endl;
  if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
    LOG(ERROR) << s.str();
  } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
    LOG(WARNING) << "WARN:" << s.str();
  } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
    LOG(WARNING) << "PERF_WARN:" << s.str();
  } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
    LOG(INFO) << s.str();
  }
  return VK_FALSE;
}

void VContext::createInstance() {
  std::vector<const char*> enabledExtensions;
  if (enableValidationLayers_) {
    uint32_t layerPresentCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerPresentCount, nullptr));
    std::vector<VkLayerProperties> layerProps(layerPresentCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerPresentCount, layerProps.data()));
    std::array<const char*, 6> instanceLayers{
        "VK_LAYER_GOOGLE_unique_objects",
        "VK_LAYER_GOOGLE_threading",
        "VK_LAYER_LUNARG_object_tracker",
        "VK_LAYER_LUNARG_core_validation",
        "VK_LAYER_LUNARG_parameter_validation",
        "VK_LAYER_KHRONOS_validation",
    };

    for (const auto& wantedLayer : instanceLayers) {
      for (const auto& presentLayer : layerProps) {
        if (strcmp(wantedLayer, presentLayer.layerName) == 0) {
          enabledValidationLayers_.push_back(wantedLayer);
          break;
        }
      }
    }

    uint32_t extCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr));
    std::vector<VkExtensionProperties> extProps(extCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extCount, extProps.data()));
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

  VkApplicationInfo applicationInfo{};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pApplicationName = "PyTorch";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "PyTorch";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.flags = 0;
  createInfo.pApplicationInfo = &applicationInfo;
  createInfo.enabledLayerCount = enabledValidationLayers_.size();
  createInfo.ppEnabledLayerNames = enabledValidationLayers_.data();
  createInfo.enabledExtensionCount = enabledExtensions.size();
  createInfo.ppEnabledExtensionNames = enabledExtensions.data();

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

  if (enableValidationLayers_) {
    VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{};
    debugReportCallbackCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debugReportCallbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
        VK_DEBUG_REPORT_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    debugReportCallbackCreateInfo.pfnCallback = &debugReportCallbackFn;

    const auto vkCreateDebugReportCallbackEXT =
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

void VContext::findPhysicalDevice() {
  uint32_t deviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr));
  TORCH_CHECK(
      deviceCount > 0, "Vulkan: Could not find a device with vulkan support");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()));
  physicalDevice_ = devices[0];
}

uint32_t VContext::getComputeQueueFamilyIndex() {
  uint32_t queueFamilyCount = 0;

  vkGetPhysicalDeviceQueueFamilyProperties(
      physicalDevice_, &queueFamilyCount, nullptr);
  TORCH_CHECK(
      queueFamilyCount > 0, "Vulkan: Invalid number of queue families");
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physicalDevice_, &queueFamilyCount, queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
    VkQueueFamilyProperties props = queueFamilies[i];
    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      return i;
    }
  }

  TORCH_CHECK(
      false, "Vulkan: Could not find a queue family that supports operations");
}

void VContext::createDevice() {
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueFamilyIndex_ = getComputeQueueFamilyIndex();
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
  queueCreateInfo.queueCount = 1;
  const float queuePriorities = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriorities;
  VkDeviceCreateInfo deviceCreateInfo{};
  VkPhysicalDeviceFeatures deviceFeatures{};

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

  VkPhysicalDeviceProperties physicalDeviceProperties{};
  vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);

  VkCommandPoolCreateInfo commandPoolCreateInfo{};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
  VK_CHECK(vkCreateCommandPool(
      device_, &commandPoolCreateInfo, nullptr, &commandPool_));
  physicalDeviceLimits_ = physicalDeviceProperties.limits;
}

static std::unique_ptr<VContext> gContext;
const VContext& context() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(gContext);
  return *gContext;
}

bool initVulkanContextOnce() {
  static const int once = []() {
#ifdef USE_VULKAN_WRAPPER
    if (!InitVulkan()) {
      TORCH_WARN("Vulkan Wrapper Failed to InitVulkan");
      return 1;
    }
#endif
    gContext = std::make_unique<VContext>(kEnableValidationLayers);
    if (!gContext) {
      TORCH_WARN("Vulkan Failed to create Vulkan Context");
      return 2;
    }
    return 0;
  }();
  ((void)once);
  return static_cast<bool>(gContext);
}

bool is_available() {
  return initVulkanContextOnce();
}

uint32_t findMemoryType(
    const VkPhysicalDevice physicalDevice,
    const uint32_t memoryTypeBits,
    const VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if ((memoryTypeBits & (1 << i)) &&
        ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
         properties)) {
      return i;
    }
  }
  return -1;
}

void VBuffer::MapMemory::flushWriteToDevice() {
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = deviceMemory_;
  range.offset = offset_;
  range.size = size_;
  range.pNext = nullptr;

  VK_CHECK(vkFlushMappedMemoryRanges(context().device(), 1, &range));
}

void VBuffer::MapMemory::flushWriteToHost() {
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = deviceMemory_;
  range.offset = offset_;
  range.size = size_;
  range.pNext = nullptr;

  VK_CHECK(vkInvalidateMappedMemoryRanges(context().device(), 1, &range));
}

VBuffer::VBuffer(
    const VkDeviceSize bufferSizeBytes,
    const VkBufferUsageFlags bufferUsageFlags,
    const VkDescriptorType descriptorType)
    : bufferSizeBytes_(bufferSizeBytes), descriptorType_(descriptorType) {
  const auto device = context().device();
  const auto physicalDevice = context().physicalDevice();
  VkBufferCreateInfo bufferCreateInfo{};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size = bufferSizeBytes_;
  bufferCreateInfo.usage = bufferUsageFlags;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer_));
  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(device, buffer_, &memoryRequirements);
  VkMemoryAllocateInfo allocateInfo{};
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

VBuffer::~VBuffer() {
  vkFreeMemory(context().device(), bufferMemory_, nullptr);
  vkDestroyBuffer(context().device(), buffer_, nullptr);
}

void VBuffer::copy_from_device_to_host(
    void* const outputData, const int64_t size) const {
  auto mm = map();
  TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
  ::memcpy(outputData, mm.ptr(), size);
  mm.flushWriteToHost();
}

void VBuffer::copy_from_host_to_device(
    const void* const data, const int64_t size) {
  auto mm = map();
  TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
  ::memcpy(mm.ptr(), data, size);
  mm.flushWriteToDevice();
}

void VBuffer::set_zeros() {
  auto mm = map();
  TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
  ::memset(mm.ptr(), 0, bufferSizeBytes_);
}

VkDescriptorBufferInfo VBuffer::makeDescriptorBufferInfo() const {
  VkDescriptorBufferInfo info{};
  info.buffer = buffer_;
  info.offset = 0;
  info.range = bufferSizeBytes_;
  return info;
}

VkWriteDescriptorSet VBuffer::makeWriteDescriptorSet(
    const VkDescriptorSet descriptorSet,
    const uint32_t binding,
    const VkDescriptorBufferInfo* const bufferInfo) const {
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

void VBuffer::bind(const VkDescriptorSet descriptorSet, const uint32_t binding) const {
  const auto descrBufferInfo = makeDescriptorBufferInfo();
  const auto writeDescrSet =
      makeWriteDescriptorSet(descriptorSet, binding, &descrBufferInfo);
  vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
}

void VBuffer::addBufferMemoryBarrier(
    const VkCommandBuffer commandBuffer,
    const VkDeviceSize offset,
    const VkDeviceSize size) const {
  VkBufferMemoryBarrier barrier{};
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

VImage::VImage(const ImageSize imageSize, const ImageSize dataSize)
    : imageSize_(imageSize), dataSize_(dataSize) {
  const auto device = context().device();
  const auto physicalDevice = context().physicalDevice();

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = kImageType;
  imageInfo.extent.width = imageSize_[0];
  imageInfo.extent.height = imageSize_[1];
  imageInfo.extent.depth = imageSize_[2];

  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = kFormat;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.pNext = nullptr;
  imageInfo.flags = 0;
  imageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

  VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image_));

  VkMemoryRequirements memReqs{};
  vkGetImageMemoryRequirements(device, image_, &memReqs);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice,
      memReqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory_));
  VK_CHECK(vkBindImageMemory(device, image_, imageMemory_, 0));

  const VkImageViewCreateInfo imageViewCreateInfo = makeImageViewCreateInfo();
  VK_CHECK(
      vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView_));

  const VkSamplerCreateInfo samplerCreateInfo = makeSamplerCreateInfo();
  VK_CHECK(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler_));
}

VImage::~VImage() {
  vkFreeMemory(context().device(), imageMemory_, nullptr);
  vkDestroySampler(context().device(), sampler_, nullptr);
  vkDestroyImageView(context().device(), imageView_, nullptr);
  vkDestroyImage(context().device(), image_, nullptr);
}

VkImageViewCreateInfo VImage::makeImageViewCreateInfo() const {
  VkImageViewCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.image = image_;
  info.viewType = kImageViewType;
  info.format = kFormat;
  info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  return info;
}

VkSamplerCreateInfo VImage::makeSamplerCreateInfo() const {
  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = kFilter;
  info.minFilter = kFilter;
  info.addressModeU = kSamplerAddressMode;
  info.addressModeV = kSamplerAddressMode;
  info.addressModeW = kSamplerAddressMode;
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

VkDescriptorImageInfo VImage::makeDescriptorImageInfo(
    const VkImageLayout imageLayout) const {
  VkDescriptorImageInfo info{};
  info.sampler = sampler_;
  info.imageView = imageView_;
  info.imageLayout = imageLayout;
  return info;
}

VkWriteDescriptorSet VImage::makeWriteDescriptorSet(
    const VkDescriptorSet descriptorSet,
    const uint32_t binding,
    const VkDescriptorType descriptorType,
    const VkDescriptorImageInfo* const imageInfo) const {
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

void VImage::bind(
    const VkDescriptorSet descriptorSet,
    const uint32_t binding,
    const VkDescriptorType descriptorType,
    const VkImageLayout imageLayout) const {
  const auto descrImageInfo = makeDescriptorImageInfo(imageLayout);
  const auto writeDescrSet = makeWriteDescriptorSet(
      descriptorSet, binding, descriptorType, &descrImageInfo);
  vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
}

void VImage::bindShaderRead(
    const VkDescriptorSet descriptorSet, const uint32_t binding) const {
  bind(
      descriptorSet,
      binding,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VImage::bindStorageImage(
    const VkDescriptorSet descriptorSet, const uint32_t binding) const {
  bind(
      descriptorSet,
      binding,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_IMAGE_LAYOUT_GENERAL);
}

void VImage::addImageMemoryBarrier(
    const VkCommandBuffer commandBuffer,
    const VkImageLayout newLayout) const {
  const VkImageLayout oldLayout = imageLayout_;
  if (oldLayout == newLayout) {
    return;
  }

  VkImageMemoryBarrier barrier{};
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

  VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_GENERAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  } else if (
      oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
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
      srcStageMask,
      dstStageMask,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &barrier);
  imageLayout_ = newLayout;
}

void VImage::addImageMemoryBarrierToGeneral(
    const VkCommandBuffer commandBuffer) const {
  addImageMemoryBarrier(commandBuffer, VK_IMAGE_LAYOUT_GENERAL);
}

void VImage::addImageMemoryBarrierToShaderRead(
    const VkCommandBuffer commandBuffer) const {
  addImageMemoryBarrier(
      commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    const uint32_t binding,
    const VkDescriptorType descriptorType) {
  return {binding, descriptorType, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
}

void createDescriptorSetLayout(
    const VkDevice device,
    const VkDescriptorSetLayoutBinding* const bindings,
    const uint32_t bindingCount,
    VkDescriptorSetLayout* const setLayout) {
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
    const VkDevice device,
    const VkDescriptorPoolSize* poolSizes,
    const uint32_t poolSizeCount,
    const uint32_t maxSets,
    VkDescriptorPool* const descriptorPool) {
  VkDescriptorPoolCreateInfo createInfo{};
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
    const VkDevice device,
    const VkDescriptorPool descriptorPool,
    const VkDescriptorSetLayout* const descriptorSetLayout,
    VkDescriptorSet* const descriptorSet) {
  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.pNext = nullptr;
  allocateInfo.descriptorPool = descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = descriptorSetLayout;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSet));
}

void createDescriptorSetLayoutSinglePool(
    const VkDevice device,
    const std::vector<VkDescriptorType>& descrTypes,
    VkDescriptorSetLayout* const descrSetLayout,
    VkDescriptorPool* const descrPool,
    VkDescriptorSet* const descrSet) {
  const auto size = descrTypes.size();
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  std::vector<VkDescriptorPoolSize> poolSizes;
  uint32_t i = 0;
  for (const auto& descrType : descrTypes) {
    bindings.push_back(descriptorSetLayoutBinding(i, descrType));
    poolSizes.push_back(VkDescriptorPoolSize{descrType, 1});
    i++;
  }
  createDescriptorSetLayout(device, bindings.data(), size, descrSetLayout);
  createDescriptorPool(
      device, poolSizes.data(), size, 1 /* maxSets */, descrPool);
  allocateDescriptorSet(device, *descrPool, descrSetLayout, descrSet);
}

void allocateCommandBuffer(VkDevice device, VkCommandBuffer* commandBuffer) {
  VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.commandPool = context().commandPool();
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;

  VK_CHECK(vkAllocateCommandBuffers(
      device, &commandBufferAllocateInfo, commandBuffer));
}

void beginCommandBuffer(VkCommandBuffer commandBuffer) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
}

void endCommandBuffer(VkCommandBuffer commandBuffer) {
  VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void submitAndWaitCommandBuffer(
    VkDevice device,
    VkQueue queue,
    VkCommandBuffer commandBuffer) {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  VkFence fence;
  VkFenceCreateInfo fenceCreateInfo{};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  VK_CHECK(vkCreateFence(device, &fenceCreateInfo, NULL, &fence))

  VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
  vkWaitForFences(device, 1, &fence, VK_TRUE, ComputeUnit::kFenceTimeoutNanos);

  vkDestroyFence(device, fence, NULL);
}

ComputeUnit::~ComputeUnit() {
  vkDestroyShaderModule(context().device(), computeShaderModule_, nullptr);
  vkDestroyPipelineLayout(context().device(), pipelineLayout_, nullptr);
  vkDestroyPipeline(context().device(), pipeline_, nullptr);
}

void ComputeUnit::createComputePipeline(
    const uint32_t* const code,
    const uint32_t codeSize,
    const VkPipelineCache pipelineCache,
    const VkDescriptorSetLayout descrSetLayout,
    const WorkGroupSize workGroupSize) {
  const auto device = context().device();
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode = code;
  createInfo.codeSize = codeSize;

  VK_CHECK(vkCreateShaderModule(
      device, &createInfo, nullptr, &computeShaderModule_));

  VkSpecializationMapEntry spMapEntries[3];
  {
    uint32_t offset = 0;
    size_t size = sizeof(WorkGroupSize::x);
    spMapEntries[0].constantID = 0;
    spMapEntries[0].offset = offset;
    spMapEntries[0].size = size;
    offset += size;
    size = sizeof(WorkGroupSize::y);
    spMapEntries[1].constantID = 1;
    spMapEntries[1].offset = offset;
    spMapEntries[1].size = size;
    offset += size;
    size = sizeof(WorkGroupSize::z);
    spMapEntries[2].constantID = 2;
    spMapEntries[2].offset = offset;
    spMapEntries[2].size = size;
  }
  VkSpecializationInfo spInfo;
  spInfo.mapEntryCount = 3;
  spInfo.pMapEntries = spMapEntries;
  spInfo.dataSize = sizeof(workGroupSize);
  spInfo.pData = &workGroupSize;

  VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
  shaderStageCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module = computeShaderModule_;
  shaderStageCreateInfo.pName = "main";
  shaderStageCreateInfo.pSpecializationInfo = &spInfo;

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &descrSetLayout;

  VK_CHECK(vkCreatePipelineLayout(
      device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));

  VkComputePipelineCreateInfo pipelineCreateInfo{};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stage = shaderStageCreateInfo;
  pipelineCreateInfo.layout = pipelineLayout_;

  VK_CHECK(vkCreateComputePipelines(
      device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline_));
}

#ifdef USE_VULKAN_SHADERC_RUNTIME
void ComputeUnit::createComputePipelineCompile(
    const std::string& glslSrc,
    const VkPipelineCache pipelineCache,
    const VkDescriptorSetLayout descrSetLayout,
    const WorkGroupSize workGroupSize) {
  shaderc::Compiler compiler{};
  shaderc::CompileOptions options{};
#ifdef DEBUG
  options.SetGenerateDebugInfo();
#endif
  options.SetTargetEnvironment(
      shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
  options.SetForcedVersionProfile(450, shaderc_profile_core);
  const shaderc::SpvCompilationResult compilationResult = compiler.CompileGlslToSpv(
      glslSrc.c_str(),
      glslSrc.size(),
      shaderc_compute_shader,
      "vulkan_shader.comp",
      "main",
      options);
  const auto compilationStatus = compilationResult.GetCompilationStatus();
  TORCH_INTERNAL_ASSERT(
      compilationStatus == shaderc_compilation_status_success,
      "Shader compilation error: status:",
      compilationStatus,
      compilationResult.GetErrorMessage());
  const std::vector<uint32_t> shaderSpvCode(
      compilationResult.cbegin(), compilationResult.cend());
  const auto codeSizeBytes = 4 * shaderSpvCode.size();
  createComputePipeline(
      shaderSpvCode.data(),
      codeSizeBytes,
      pipelineCache,
      descrSetLayout,
      workGroupSize);
}
#endif

void ComputeUnit::createCommandBuffer(VkDescriptorSet& descriptorSet) {
  const auto device = context().device();
  VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.commandPool = context().commandPool();
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;

  VK_CHECK(vkAllocateCommandBuffers(
      device, &commandBufferAllocateInfo, &commandBuffer_));

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

  vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
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

void ComputeUnit::addMemoryBarrier(
    const VkPipelineStageFlags srcStageMask,
    const VkAccessFlags srcAccessMask,
    const VkPipelineStageFlags dstStageMask,
    const VkAccessFlags dstAccessMask) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask = srcAccessMask;
  barrier.dstAccessMask = dstAccessMask;
  vkCmdPipelineBarrier(
      commandBuffer_,
      srcStageMask,
      dstStageMask,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

void ComputeUnit::dispatchCommandBuffer(
    const uint32_t groupCountX,
    const uint32_t groupCountY,
    const uint32_t groupCountZ) {
  vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
}

void ComputeUnit::endCommandBuffer() {
  at::native::vulkan::detail::endCommandBuffer(commandBuffer_);
}

void ComputeUnit::dispatchCommandBuffer(
    const uint32_t gridX,
    const uint32_t gridY,
    const uint32_t gridZ,
    const WorkGroupSize workGroupSize) {
  dispatchCommandBuffer(
      UP_DIV(gridX, workGroupSize.x),
      UP_DIV(gridY, workGroupSize.y),
      UP_DIV(gridZ, workGroupSize.z));
}

void ComputeUnit::submitAndWaitCommandBuffer() {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  VkFence fence{};
  VkFenceCreateInfo fenceCreateInfo{};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  VK_CHECK(vkCreateFence(context().device(), &fenceCreateInfo, NULL, &fence))

  VK_CHECK(vkQueueSubmit(context().queue(), 1, &submitInfo, fence));
  vkWaitForFences(context().device(), 1, &fence, VK_TRUE, kFenceTimeoutNanos);

  vkDestroyFence(context().device(), fence, NULL);
}

VBuffer makeUniformConstBuffer(const void* const ptr, const VkDeviceSize size) {
  VBuffer constBuffer = VBuffer::makeUniformBuffer(size);
  constBuffer.copy_from_host_to_device(ptr, size);
  return constBuffer;
}

ComputeUnitFactory::ComputeUnitFactory(const VkDevice device)
    : device_(device) {
  VkPipelineCacheCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.initialDataSize = 0;
  createInfo.pInitialData = nullptr;
  VK_CHECK(vkCreatePipelineCache(
      device_, &createInfo, nullptr /* allocator */, &pipelineCache_));
}

ComputeUnitFactory::~ComputeUnitFactory() {
  vkDestroyPipelineCache(device_, pipelineCache_, nullptr /* allocator */);
}

std::string ComputeUnitFactory::getCacheKey(
    const char* const key,
    const WorkGroupSize workGroupSize) {
  std::stringstream ss;
  ss << key << ':' << workGroupSize.x << ':' << workGroupSize.y << ':'
     << workGroupSize.z;
  return ss.str();
}

ComputeUnit& ComputeUnitFactory::get(
    const std::string& cacheKey,
    const std::function<std::shared_ptr<ComputeUnit>()> factoryFn) {
  const auto it = computeUnits_.find(cacheKey);
  if (it != computeUnits_.end()) {
    return *(it->second.get());
  }
  auto computeUnit = factoryFn();
  computeUnits_.insert(std::make_pair(cacheKey, computeUnit));
  return *(computeUnit.get());
}

#ifdef USE_VULKAN_SHADERC_RUNTIME
ComputeUnit& ComputeUnitFactory::get(
    const char* const key,
    const char* const glslSrc,
    const VkDescriptorSetLayout descrSetLayout,
    const WorkGroupSize workGroupSize) {
  return get(
      getCacheKey(key, workGroupSize),
      [glslSrc,
       pipelineCache = pipelineCache_,
       descrSetLayout,
       workGroupSize]() {
        return std::make_shared<ComputeUnit>(
            glslSrc, pipelineCache, descrSetLayout, workGroupSize);
      });
}
#else
ComputeUnit& ComputeUnitFactory::get(
    const char* const key,
    const uint32_t* const code,
    const uint32_t codeSize,
    const VkDescriptorSetLayout descrSetLayout,
    const WorkGroupSize workGroupSize) {
  return get(
      getCacheKey(key, workGroupSize),
      [code,
       codeSize,
       pipelineCache = pipelineCache_,
       descrSetLayout,
       workGroupSize]() {
        return std::make_shared<ComputeUnit>(
            code, codeSize, pipelineCache, descrSetLayout, workGroupSize);
      });
}
#endif

// VBuffer <-> VImage
void copy_buffer_to_image(const VBuffer& buffer, VImage& image) {
  const auto device = context().device();

  VkDescriptorSetLayout descrSetLayout{};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool{};
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet{};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  image.bindStorageImage(descrSet, 0);
  buffer.bind(descrSet, 1);
  WorkGroupSize workGroupSize{8, 8, 1};

  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(nchw_to_image), descrSetLayout, workGroupSize);
  computeUnit.createCommandBuffer(descrSet);

  image.addImageMemoryBarrierToGeneral(computeUnit.commandBuffer());
  buffer.addBufferMemoryBarrier(
      computeUnit.commandBuffer(), 0, buffer.sizeBytes());
  computeUnit.addMemoryBarrier(
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
  computeUnit.dispatchCommandBuffer(
      image.w(), image.h(), image.d(), workGroupSize);
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();

  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
}

void copy_image_to_buffer(
    const VImage& image,
    VBuffer& buffer,
    bool addBufferMemoryBarrierForHost) {
  const auto device = context().device();
  TORCH_INTERNAL_ASSERT(
      buffer.sizeBytes() >= image.capacityBytes(),
      "VulkanBuffer's capacity is less than VulkanImage capacity to copy from");

  VkDescriptorSetLayout descrSetLayout{};
  const VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool{};
  const VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet{};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  image.bindShaderRead(descrSet, 0);
  buffer.bind(descrSet, 1);

  const WorkGroupSize workGroupSize{8, 8, 1};
  auto& computeUnit = context().computeUnitFactory().get(
      GLSL_SPV(image_to_nchw), descrSetLayout, workGroupSize);

  computeUnit.createCommandBuffer(descrSet);
  image.addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
  computeUnit.dispatchCommandBuffer(
      image.w(), image.h(), image.d(), workGroupSize);

  if (addBufferMemoryBarrierForHost) {
    computeUnit.addMemoryBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_HOST_READ_BIT);
  }
  computeUnit.endCommandBuffer();
  computeUnit.submitAndWaitCommandBuffer();

  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
} // VBuffer <-> VImage

void copy_buffer_to_buffer(
    const VBuffer& srcBuffer,
    VBuffer& dstBuffer,
    VkDeviceSize size,
    VkDeviceSize srcOffset,
    VkDeviceSize dstOffset) {
  auto device = context().device();
  VkCommandBuffer commandBuffer{};
  allocateCommandBuffer(device, &commandBuffer);
  beginCommandBuffer(commandBuffer);

  VkBufferCopy copyRegion{};
  copyRegion.srcOffset = srcOffset;
  copyRegion.dstOffset = dstOffset;
  copyRegion.size = size;
  vkCmdCopyBuffer(
      commandBuffer,
      srcBuffer.vkbuffer(),
      dstBuffer.vkbuffer(),
      1,
      &copyRegion);

  endCommandBuffer(commandBuffer);
  submitAndWaitCommandBuffer(device, context().queue(), commandBuffer);
}

// VulkanTensor

class VulkanTensor::Impl final {
 public:
  explicit Impl(std::vector<int64_t> sizes)
      : sizes_(std::move(sizes)),
        strides_(std::vector<int64_t>(sizes_.size())),
        numel_(c10::multiply_integers(sizes_)) {
    TORCH_CHECK(
        initVulkanContextOnce(), "Vulkan Failed to create Vulkan Context");
  }

  std::vector<int64_t> sizes() const {
    return sizes_;
  }

  std::vector<int64_t> strides() const {
    return strides_;
  }

  inline int64_t dim() const {
    return sizes_.size();
  }

  inline int64_t numel() const {
    return numel_;
  }

  inline bool has_buffer() const {
    return static_cast<bool>(buffer_);
  }

  inline VBuffer* buffer() {
    if (!has_buffer()) {
      buffer_ = std::make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
    }
    return buffer_.get();
  }

  const VBuffer* buffer() const {
    if (!has_buffer()) {
      buffer_ = std::make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
    }
    return buffer_.get();
  }

  inline bool can_be_image() const {
    return dim() <= 4;
  }

  inline bool has_image() const {
    return static_cast<bool>(image_);
  }

  inline bool has_storage() {
    return has_buffer();
  }

  ImageSizes imageSizes_W_H_NC4() {
    TORCH_INTERNAL_ASSERT(
        can_be_image(),
        "Vulkan: Only Tensors with dim <= 4 can be represented as Vulkam Image");
    auto d = dim();
    int64_t _wd = 1;
    int64_t _hd = 1;
    int64_t _dd = 1;
    if (d == 4) {
      _wd = sizes_[3];
      _hd = sizes_[2];
      _dd = sizes_[1] * sizes_[0];
    } else if (d == 3) {
      _wd = sizes_[2];
      _hd = sizes_[1];
      _dd = sizes_[0];
    } else if (d == 2) {
      _wd = sizes_[1];
      _hd = sizes_[0];
    } else if (d == 1) {
      _wd = sizes_[0];
    }
    int32_t wd = safe_downcast<int64_t>(_wd);
    int32_t hd = safe_downcast<int64_t>(_hd);
    int32_t dd = safe_downcast<int64_t>(_dd);
    return {{wd, hd, UP_DIV(dd, 4)}, {wd, hd, dd}};
  }

  VImage* image(const c10::optional<ImageSizes> imageSizes = c10::nullopt) {
    if (image_) {
      return image_.get();
    }

    if (imageSizes.has_value()) {
      image_ = std::make_unique<VImage>(*imageSizes);
      return image_.get();
    }

    image_ = std::make_unique<VImage>(imageSizes_W_H_NC4());
    if (buffer_) {
      copy_buffer_to_image(*buffer_, *image_);
    }
    return image_.get();
  }

  const VImage* image(
      c10::optional<ImageSizes> imageSizes = c10::nullopt) const {
    return const_cast<VulkanTensor::Impl*>(this)->image(imageSizes);
  }

  VkDeviceSize buffer_size_for_sizes(std::vector<int64_t> sizes) const {
    const auto d = sizes.size();
    const auto numel = c10::multiply_integers(sizes);
    VkDeviceSize bufferSize{sizeof(float) * numel};
    // alignment to be able to copy between image and buffer
    if (d == 4) {
      bufferSize =
          sizeof(float) * ALIGN_UP4(sizes[0] * sizes[1]) * sizes[2] * sizes[3];
    } else if (d == 3) {
      bufferSize = sizeof(float) * ALIGN_UP4(sizes[0]) * sizes[1] * sizes[2];
    } else if (d == 2) {
      bufferSize = sizeof(float) * 4 * sizes[0] * sizes[1];
    } else if (d == 1) {
      bufferSize = sizeof(float) * 4 * sizes[0];
    }
    return bufferSize;
  }

  void allocate_storage() {
    buffer_ = std::make_unique<VBuffer>(buffer_size_for_sizes(sizes_));
  }

  void set_data_from_host(const float* const inputData) {
    buffer()->copy_from_host_to_device(
        (const void*)inputData, sizeof(float) * numel_);
  }

  void copy_data_to_host(float* const outputData) const {
    sync_image_to_buffer();
    buffer()->copy_from_device_to_host(outputData, sizeof(float) * numel_);
  }

  void sync_image_to_buffer() const {
    if (has_image()) {
      copy_image_to_buffer(
          *image(),
          *(const_cast<VBuffer*>(buffer())),
          true /* memory barrier for host memory map */);
    }
  }

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t numel_;
  mutable std::unique_ptr<VBuffer> buffer_;
  std::unique_ptr<VImage> image_;
};

std::shared_ptr<VulkanTensor::Impl> VulkanTensor::impl() {
  return impl_;
}

std::shared_ptr<const VulkanTensor::Impl> VulkanTensor::impl() const {
  return impl_;
}

std::vector<int64_t> VulkanTensor::sizes() const {
  return impl()->sizes();
}

void VulkanTensor::sync_image_to_buffer() const {
  return impl()->sync_image_to_buffer();
}

std::vector<int64_t> VulkanTensor::strides() const {
  return impl()->strides();
}

int64_t VulkanTensor::dim() const {
  return impl()->dim();
}

int64_t VulkanTensor::numel() const {
  return impl()->numel();
}

bool VulkanTensor::has_storage() const {
  return impl()->has_buffer();
}

void VulkanTensor::allocate_storage() {
  impl()->allocate_storage();
}

void VulkanTensor::set_data_from_host(const float* const inputData) {
  impl()->set_data_from_host(inputData);
}

void VulkanTensor::copy_data_to_host(float* const outputData) const {
  impl()->copy_data_to_host(outputData);
}

bool VulkanTensor::has_buffer() const {
  return impl()->has_buffer();
}

VBuffer* VulkanTensor::buffer() {
  return impl()->buffer();
}

const VBuffer* VulkanTensor::buffer() const {
  return impl()->buffer();
}

bool VulkanTensor::can_be_image() const {
  return impl()->can_be_image();
}

bool VulkanTensor::has_image() const {
  return impl()->has_image();
}

VImage* VulkanTensor::image(const c10::optional<ImageSizes> imageSizes) {
  return impl()->image(imageSizes);
}

const VImage* VulkanTensor::image(const c10::optional<ImageSizes> imageSizes) const {
  return impl()->image(imageSizes);
}

VulkanTensor::VulkanTensor(std::vector<int64_t> sizes)
    : impl_(std::make_shared<Impl>(std::move(sizes))) {}

std::ostream& operator<<(std::ostream& s, const ImageSize& imageSize) {
  s << "ImageSize{" << imageSize[0] << ", " << imageSize[1] << ", "
    << imageSize[2] << "}";
  return s;
}
std::ostream& operator<<(std::ostream& s, const ImageSizes& imageSizes) {
  s << "ImageSizes{imageSize:" << imageSizes.imageSize
    << ", dataSize:" << imageSizes.dataSize << "}";
  return s;
}

std::ostream& operator<<(std::ostream& s, const WorkGroupSize& workGroupSize) {
  s << "WorkGroupSize{" << workGroupSize.x << " " << workGroupSize.y << " "
    << workGroupSize.z << "}";
  return s;
}

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
