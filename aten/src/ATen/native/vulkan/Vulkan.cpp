#ifdef USE_VULKAN

#include <stdio.h>
#include <unistd.h>
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

#define VK_CHECK_RESULT(f)                                         \
  {                                                                \
    VkResult res = (f);                                            \
    TORCH_CHECK(res == VK_SUCCESS, "Vulkan error VkResult:", res); \
  }

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

class VContext {
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

  VkInstance instance_;
  VkDebugReportCallbackEXT debugReportCallback_;
  VkPhysicalDevice physicalDevice_;
  VkDevice device_;
  std::vector<const char*> enabledValidationLayers_;
  VkQueue queue_;
  uint32_t queueFamilyIndex_;
  uint64_t uboAlign_;
  uint64_t sboAlign_;
  bool enableValidationLayers_;

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
      VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(
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

    VK_CHECK_RESULT(
        vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_));
    queue_ = {};
    vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties);

    uboAlign_ = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
    sboAlign_ = physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;
  }

  void cleanup() {
    if (enableValidationLayers_) {
      auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
          instance_, "vkDestroyDebugReportCallbackEXT");
      TORCH_CHECK(func, "Could not load vkDestroyDebugReportCallbackEXT");
      func(instance_, debugReportCallback_, nullptr);
    }

    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
  }
}; // class VContext
static std::unique_ptr<VContext> vkContext;
static constexpr bool kEnableValidationLayers = true;

void initVulkanContextOnce() {
  static const int once = []() {
#ifdef USE_VULKAN_WRAPPER
    bool res = InitVulkan();
    TORCH_CHECK(res, "Vulkan Wrapper Failed to InitVulkan");
#endif
    vkContext = std::make_unique<VContext>(kEnableValidationLayers);
    TORCH_CHECK(vkContext, "Vulkan Failed to create Vulkan Context");
    return 0;
  }();
  ((void)once);
}

class VBuffer;
class VulkanTensor::Impl {
 public:
  Impl(std::vector<int64_t> sizes) : sizes_(std::move(sizes)) {
    int64_t numel = 1;
    for (const auto& d : sizes_) {
      numel *= d;
    }
    numel_ = numel;
  }

  std::vector<int64_t> sizes_;
  int64_t numel_;
  std::unique_ptr<VBuffer> vbuffer_;
};

VulkanTensor::VulkanTensor(std::vector<int64_t> sizes)
    : pImpl(std::make_shared<Impl>(std::move(sizes))) {
  initVulkanContextOnce();
}

std::vector<int64_t> VulkanTensor::sizes() {
  return pImpl->sizes_;
}

bool VulkanTensor::hasStorage() {
  return static_cast<bool>(pImpl->vbuffer_);
}

void VulkanTensor::allocateStorage() {
  const auto bufferSize = sizeof(float) * pImpl->numel_;
  const auto bufferSizeAligned = ROUND_UP(bufferSize, vkContext->sboAlign_);
  pImpl->vbuffer_ = std::make_unique<VBuffer>(
      bufferSizeAligned, vkContext->physicalDevice_, vkContext->device_);
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

class VBuffer {
 public:
  VBuffer(
      uint32_t bufferSize,
      VkPhysicalDevice physicalDevice,
      VkDevice device,
      bool isUniform = false)
      : bufferSize_(bufferSize),
        isUniform_(isUniform),
        physicalDevice_(physicalDevice),
        device_(device) {
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = isUniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                       : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(
        vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &buffer_));
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device_, buffer_, &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(
        physicalDevice_,
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VK_CHECK_RESULT(
        vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory_));
    VK_CHECK_RESULT(vkBindBufferMemory(device_, buffer_, bufferMemory_, 0));
  }

  void copyFromDeviceToHost(void* outputData, int64_t size) {
    void* mappedMemory = nullptr;
    vkMapMemory(device_, bufferMemory_, 0, size, 0, &mappedMemory);
    ::memcpy(outputData, mappedMemory, size);
    vkUnmapMemory(device_, bufferMemory_);
  }

  void copyFromHostToDevice(void* data, int64_t size) {
    void* mappedMemory = nullptr;
    vkMapMemory(device_, bufferMemory_, 0, size, 0, &mappedMemory);
    ::memcpy(mappedMemory, data, size);
    vkUnmapMemory(device_, bufferMemory_);
  }

  VkDescriptorBufferInfo makeDescriptorBufferInfo() {
    VkDescriptorBufferInfo info = {};
    info.buffer = buffer_;
    info.offset = 0;
    info.range = bufferSize_;
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
    writeSet.descriptorType = isUniform_ ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                         : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeSet.pImageInfo = nullptr;
    writeSet.pBufferInfo = bufferInfo;
    writeSet.pTexelBufferView = nullptr;
    return writeSet;
  }

  void bind(VkDescriptorSet descriptorSet, uint32_t binding) {
    auto descrBufferInfo = makeDescriptorBufferInfo();
    auto writeDescrSet =
        makeWriteDescriptorSet(descriptorSet, binding, &descrBufferInfo);
    vkUpdateDescriptorSets(device_, 1, &writeDescrSet, 0, nullptr);
  }

  ~VBuffer() {
    vkFreeMemory(device_, bufferMemory_, nullptr);
    vkDestroyBuffer(device_, buffer_, nullptr);
  }

  uint32_t bufferSize_;
  bool isUniform_;
  VkBuffer buffer_;
  VkDeviceMemory bufferMemory_;
  VkPhysicalDevice physicalDevice_;
  VkDevice device_;
}; // class VBuffer

class ComputeUnit {
 public:
  ComputeUnit() {}

  ~ComputeUnit() {
    vkDestroyShaderModule(vkContext->device_, computeShaderModule_, nullptr);
    vkDestroyPipelineLayout(vkContext->device_, pipelineLayout_, nullptr);
    vkDestroyPipeline(vkContext->device_, pipeline_, nullptr);
    vkDestroyCommandPool(vkContext->device_, commandPool_, nullptr);
  }

  void createComputePipeline(
      const uint32_t* code,
      const uint32_t codeSize,
      const VkDescriptorSetLayout& descrSetLayout) {
    auto device = vkContext->device_;
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = codeSize;

    VK_CHECK_RESULT(vkCreateShaderModule(
        device, &createInfo, nullptr, &computeShaderModule_));

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeShaderModule_;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descrSetLayout;

    VK_CHECK_RESULT(vkCreatePipelineLayout(
        device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout_;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline_));
  }

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
  void createComputePipelineCompile(
      std::string glslSrc,
      const VkDescriptorSetLayout& descrSetLayout) {
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
    const uint32_t codeSizeBytes = 4 * shaderSpvCode.size();
    createComputePipeline(shaderSpvCode.data(), codeSizeBytes, descrSetLayout);
  }
#endif

  void createCommandBuffer(
      VkDescriptorSet& descriptorSet,
      uint32_t groupCountX,
      uint32_t groupCountY,
      uint32_t groupCountZ) {
    auto device = vkContext->device_;
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = vkContext->queueFamilyIndex_;
    VK_CHECK_RESULT(vkCreateCommandPool(
        device, &commandPoolCreateInfo, nullptr, &commandPool_));
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool_;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VK_CHECK_RESULT(vkAllocateCommandBuffers(
        device, &commandBufferAllocateInfo, &commandBuffer_));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

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
    vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer_));
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
    VK_CHECK_RESULT(
        vkCreateFence(vkContext->device_, &fenceCreateInfo, NULL, &fence))

    VK_CHECK_RESULT(vkQueueSubmit(vkContext->queue_, 1, &submitInfo, fence));
    vkWaitForFences(vkContext->device_, 1, &fence, VK_TRUE, 100000000000);

    vkDestroyFence(vkContext->device_, fence, NULL);
  }

 private:
  VkCommandPool commandPool_;
  VkCommandBuffer commandBuffer_;
  VkPipeline pipeline_;
  VkPipelineLayout pipelineLayout_;
  VkShaderModule computeShaderModule_;
};

#ifdef USE_VULKAN_GLES_SHADERC_RUNTIME
auto makeComputeUnit(
    const char* glslSrc,
    const VkDescriptorSetLayout& descrSetLayout) {
  auto computeUnit = std::make_unique<ComputeUnit>();
  computeUnit->createComputePipelineCompile(
      std::string{glslSrc, std::strlen(glslSrc)}, descrSetLayout);
  return computeUnit;
}
#else
auto makeComputeUnit(
    const unsigned char* spvCode,
    const unsigned int spvCodeSize,
    const VkDescriptorSetLayout& descrSetLayout) {
  auto computeUnit = std::make_unique<ComputeUnit>();
  const uint32_t* code = reinterpret_cast<const uint32_t*>(spvCode);
  const uint32_t codeSize = spvCodeSize;
  computeUnit->createComputePipeline(code, codeSize, descrSetLayout);
  return computeUnit;
}
#endif

void VulkanTensor::setDataFromHost(const float* inputData) {
  initVulkanContextOnce();

  const auto inputDataSize = sizeof(float) * pImpl->numel_;
  if (!hasStorage()) {
    allocateStorage();
  }
  pImpl->vbuffer_->copyFromHostToDevice((void*)inputData, inputDataSize);
}

void VulkanTensor::copyDataToHost(float* output) {
  initVulkanContextOnce();
  auto bufferDataSize = sizeof(float) * pImpl->numel_;
  pImpl->vbuffer_->copyFromDeviceToHost(output, bufferDataSize);
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
  VK_CHECK_RESULT(
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
  VK_CHECK_RESULT(
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
  VK_CHECK_RESULT(
      vkAllocateDescriptorSets(device, &allocateInfo, descriptorSet));
}

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
  auto device = vkContext->device_;
  auto physicalDevice = vkContext->physicalDevice_;
  int64_t C = _N * _C;
  struct ConstBlock {
    int32_t IW;
    int32_t IH;
    int32_t OW;
    int32_t OH;
    float scaleX;
    float scaleY;
  };
  ConstBlock u{IW, IH, OW, OH, scaleW, scaleH};
  int64_t bufferConstSize = sizeof(u);
  int64_t bufferConstSizeAligned =
      ROUND_UP(bufferConstSize, vkContext->uboAlign_);
  VBuffer bufferConst{bufferConstSizeAligned, physicalDevice, device, true};
  bufferConst.copyFromHostToDevice((void*)&u, bufferConstSize);

  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorSetLayoutBinding bindings[] = {
      descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
  createDescriptorSetLayout(
      device, bindings, 3 /* bindingsCount */, &descrSetLayout);

  VkDescriptorPool descrPool = {};
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  createDescriptorPool(
      device, poolSizes, 3 /* poolSizeCount */, 1 /* maxSets */, &descrPool);

  VkDescriptorSet descrSet = {};
  allocateDescriptorSet(device, descrPool, &descrSetLayout, &descrSet);

  output.pImpl->vbuffer_->bind(descrSet, 0);
  input.pImpl->vbuffer_->bind(descrSet, 1);
  bufferConst.bind(descrSet, 2);

  auto computeUnit = makeComputeUnit(
      at::native::vulkan::GLSL_SPV(vulkan_upsampleNearest2d), descrSetLayout);
  computeUnit->createCommandBuffer(descrSet, UP_DIV(OW, 8), UP_DIV(OH, 8), C);
  computeUnit->runCommandBuffer();
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
}

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
