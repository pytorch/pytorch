#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>

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
#else
#include <ATen/native/vulkan/spv.h>
#endif

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

VContext::VContext(bool enableValidationLayers)
    : enableValidationLayers_(enableValidationLayers) {
  createInstance();
  findPhysicalDevice();
  createDevice();
}

VContext::~VContext() {
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

void VContext::createInstance() {
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
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, extProps.data());
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
  applicationInfo.pApplicationName = "pytorch";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "compute";
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

  vkCreateInstance(&createInfo, nullptr, &instance_);

  if (enableValidationLayers_) {
    VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{};
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

void VContext::findPhysicalDevice() {
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

uint32_t VContext::getComputeQueueFamilyIndex() {
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

void VContext::createDevice() {
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueFamilyIndex_ = getComputeQueueFamilyIndex();
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
  queueCreateInfo.queueCount = 1;
  float queuePriorities = 1.0;
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

  VkPhysicalDeviceProperties physicalDeviceProperties;
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
  return *(gContext.get());
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

VBuffer::VBuffer(
    VkDeviceSize bufferSizeBytes,
    VkBufferUsageFlags bufferUsageFlags,
    VkDescriptorType descriptorType)
    : bufferSizeBytes_(bufferSizeBytes), descriptorType_(descriptorType) {
  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
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

VBuffer::~VBuffer() noexcept {
  vkFreeMemory(context().device(), bufferMemory_, nullptr);
  vkDestroyBuffer(context().device(), buffer_, nullptr);
}

void VBuffer::copyFromDeviceToHost(void* outputData, int64_t size) {
  auto mm = map();
  TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
  ::memcpy(outputData, mm.ptr(), size);
}

void VBuffer::copyFromHostToDevice(void* data, int64_t size) {
  auto mm = map();
  TORCH_INTERNAL_ASSERT(mm.ptr(), "Vulkan: Failed to map Vulkan Buffer memory");
  ::memcpy(mm.ptr(), data, size);
}

VkDescriptorBufferInfo VBuffer::makeDescriptorBufferInfo() {
  VkDescriptorBufferInfo info{};
  info.buffer = buffer_;
  info.offset = 0;
  info.range = bufferSizeBytes_;
  return info;
}

VkWriteDescriptorSet VBuffer::makeWriteDescriptorSet(
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

void VBuffer::bind(VkDescriptorSet descriptorSet, uint32_t binding) {
  auto descrBufferInfo = makeDescriptorBufferInfo();
  auto writeDescrSet =
      makeWriteDescriptorSet(descriptorSet, binding, &descrBufferInfo);
  vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
}

void VBuffer::addBufferMemoryBarrier(
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

VImage::VImage(uint32_t W, uint32_t H, uint32_t C)
    : W_(W), H_(H), C_(C), D_(UP_DIV(C, 4)) {
  auto device = context().device();
  auto physicalDevice = context().physicalDevice();

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = kImageType;
  imageInfo.extent.width = W_;
  imageInfo.extent.height = H_;
  imageInfo.extent.depth = D_;

  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = kFormat;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = kImageLayoutInitial;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.pNext = nullptr;
  imageInfo.flags = 0;

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

  VkImageViewCreateInfo imageViewCreateInfo = makeImageViewCreateInfo();
  VK_CHECK(
      vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView_));

  VkSamplerCreateInfo samplerCreateInfo = makeSamplerCreateInfo();
  VK_CHECK(vkCreateSampler(device, &samplerCreateInfo, nullptr, &sampler_));
}
VImage::~VImage() noexcept {
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
    VkImageLayout imageLayout) const {
  VkDescriptorImageInfo info{};
  info.sampler = sampler_;
  info.imageView = imageView_;
  info.imageLayout = imageLayout;
  return info;
}

VkWriteDescriptorSet VImage::makeWriteDescriptorSet(
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorImageInfo* imageInfo) const {
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
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    VkDescriptorType descriptorType,
    VkImageLayout imageLayout) const {
  auto descrImageInfo = makeDescriptorImageInfo(imageLayout);
  auto writeDescrSet = makeWriteDescriptorSet(
      descriptorSet, binding, descriptorType, &descrImageInfo);
  vkUpdateDescriptorSets(context().device(), 1, &writeDescrSet, 0, nullptr);
}

void VImage::bindShaderRead(VkDescriptorSet descriptorSet, uint32_t binding)
    const {
  bind(
      descriptorSet,
      binding,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VImage::bindStorageImage(VkDescriptorSet descriptorSet, uint32_t binding)
    const {
  bind(
      descriptorSet,
      binding,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_IMAGE_LAYOUT_GENERAL);
}

void VImage::addImageMemoryBarrier(
    VkCommandBuffer commandBuffer,
    VkImageLayout oldLayout,
    VkImageLayout newLayout) {
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

void VImage::addImageMemoryBarrierUndefinedToGeneral(
    VkCommandBuffer commandBuffer) {
  addImageMemoryBarrier(
      commandBuffer, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
}

void VImage::addImageMemoryBarrierGeneralToShaderRead(
    VkCommandBuffer commandBuffer) {
  addImageMemoryBarrier(
      commandBuffer,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

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
    VkDevice device,
    VkDescriptorPool descriptorPool,
    const VkDescriptorSetLayout* descriptorSetLayout,
    VkDescriptorSet* descriptorSet) {
  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.pNext = nullptr;
  allocateInfo.descriptorPool = descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = descriptorSetLayout;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSet));
}

void createDescriptorSetLayoutSinglePool(
    VkDevice device,
    std::vector<VkDescriptorType> descrTypes,
    VkDescriptorSetLayout* descrSetLayout,
    VkDescriptorPool* descrPool,
    VkDescriptorSet* descrSet) {
  auto size = descrTypes.size();
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

ComputeUnit::~ComputeUnit() {
  vkDestroyShaderModule(context().device(), computeShaderModule_, nullptr);
  vkDestroyPipelineLayout(context().device(), pipelineLayout_, nullptr);
  vkDestroyPipeline(context().device(), pipeline_, nullptr);
}

void ComputeUnit::createComputePipeline(
    const uint32_t* code,
    const uint32_t codeSize,
    const VkDescriptorSetLayout& descrSetLayout,
    WorkGroupSize& workGroupSize) {
  auto device = context().device();
  VkShaderModuleCreateInfo createInfo{};
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
      device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline_));
}

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
void ComputeUnit::createComputePipelineCompile(
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

void ComputeUnit::createCommandBuffer(VkDescriptorSet& descriptorSet) {
  auto device = context().device();
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

void ComputeUnit::dispatchCommandBuffer(
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ) {
  vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
  VK_CHECK(vkEndCommandBuffer(commandBuffer_));
}

void ComputeUnit::dispatchCommandBuffer(
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    WorkGroupSize workGroupSize) {
  dispatchCommandBuffer(
      UP_DIV(gridX, workGroupSize.x),
      UP_DIV(gridY, workGroupSize.y),
      UP_DIV(gridZ, workGroupSize.z));
}

void ComputeUnit::runCommandBuffer() {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer_;

  VkFence fence;
  VkFenceCreateInfo fenceCreateInfo{};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  VK_CHECK(vkCreateFence(context().device(), &fenceCreateInfo, NULL, &fence))

  VK_CHECK(vkQueueSubmit(context().queue(), 1, &submitInfo, fence));
  vkWaitForFences(context().device(), 1, &fence, VK_TRUE, 100000000000);

  vkDestroyFence(context().device(), fence, NULL);
}

VBuffer makeUniformConstBuffer(void* ptr, VkDeviceSize size) {
  auto sizeAligned =
      ROUND_UP(size, context().limits().minUniformBufferOffsetAlignment);
  VBuffer constBuffer = VBuffer::makeUniformBuffer(sizeAligned);
  constBuffer.copyFromHostToDevice(ptr, size);
  return constBuffer;
}

// VBuffer <-> VImage
void copyFromBufferToImage(VBuffer& buffer, VImage& image) {
  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
  struct ConstBlock {
    int32_t W;
    int32_t H;
  };
  ConstBlock constBlock{image.W(), image.H()};
  VBuffer constBuffer =
      makeUniformConstBuffer((void*)&constBlock, sizeof(constBlock));

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
  auto device = context().device();
  auto physicalDevice = context().physicalDevice();
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

  VkDescriptorSetLayout descrSetLayout{};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool{};
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet{};
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
 public:
  Impl(std::vector<int64_t> sizes) : sizes_(std::move(sizes)) {
    numel_ = std::accumulate(
        std::begin(sizes_), std::end(sizes_), 1, std::multiplies<int64_t>());
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

  inline bool hasBuffer() const {
    return static_cast<bool>(buffer_);
  }

  inline VBuffer& buffer() {
    return *(buffer_.get());
  }

  inline bool canBeImage() const {
    return dim() <= 4;
  }

  inline bool hasImage() const {
    return static_cast<bool>(image_);
  }

  inline bool hasStorage() {
    return hasBuffer();
  }

  VImage& image() {
    auto d = dim();
    TORCH_INTERNAL_ASSERT(
        d <= 4,
        "Vulkan: Only Tensors with dim <= 4 can be represented as Vulkam Image");
    if (!image_ && buffer_) {
      auto W = 0;
      auto H = 0;
      auto C = 0;
      if (d == 4) {
        W = sizes_[3];
        H = sizes_[2];
        C = sizes_[1] * sizes_[0];
      } else if (d == 3) {
        W = sizes_[2];
        H = sizes_[1];
        C = sizes_[0];
      } else if (d == 2) {
        W = sizes_[1];
        H = sizes_[0];
        C = 1;
      } else if (d == 1) {
        W = sizes_[0];
        H = 1;
        C = 1;
      }
      image_ = std::make_unique<VImage>(W, H, C);
      copyFromBufferToImage(*buffer_, *image_);
    }
    return *(image_.get());
  }

  VImage& image() const {
    return const_cast<VulkanTensor::Impl*>(this)->image();
  }

  void allocateStorage() {
    auto bufferSize = sizeof(float) * numel_;
    const auto d = dim();
    if (d == 4) {
      bufferSize = sizeof(float) * ALIGN_UP4(sizes_[0] * sizes_[1]) *
          sizes_[2] * sizes_[3];
    } else if (d == 3) {
      bufferSize = sizeof(float) * ALIGN_UP4(sizes_[0]) * sizes_[1] * sizes_[2];
    } else if (d == 2) {
      bufferSize = sizeof(float) * 4 * sizes_[0] * sizes_[1];
    } else if (d == 1) {
      bufferSize = sizeof(float) * 4 * sizes_[0];
    }
    const auto bufferSizeAligned = ROUND_UP(
        bufferSize, context().limits().minStorageBufferOffsetAlignment);
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

std::shared_ptr<VulkanTensor::Impl> VulkanTensor::impl() {
  return impl_;
}

std::shared_ptr<const VulkanTensor::Impl> VulkanTensor::impl() const {
  return impl_;
}

std::vector<int64_t> VulkanTensor::sizes() const {
  return impl()->sizes();
}

int64_t VulkanTensor::dim() const {
  return impl()->dim();
}

int64_t VulkanTensor::numel() const {
  return impl()->numel();
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

bool VulkanTensor::hasBuffer() const {
  return impl()->hasBuffer();
}

VBuffer& VulkanTensor::buffer() {
  return impl()->buffer();
}

bool VulkanTensor::canBeImage() const {
  return impl()->canBeImage();
}

bool VulkanTensor::hasImage() const {
  return impl()->hasImage();
}

VImage& VulkanTensor::image() const {
  return impl()->image();
}

VImage& VulkanTensor::image() {
  return impl()->image();
}

VulkanTensor::VulkanTensor(std::vector<int64_t> sizes)
    : impl_(std::make_shared<Impl>(std::move(sizes))) {
  TORCH_CHECK(
      initVulkanContextOnce(), "Vulkan Failed to create Vulkan Context");
}

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
