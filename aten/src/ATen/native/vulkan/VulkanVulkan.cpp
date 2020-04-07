#ifdef USE_VULKAN

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <ATen/native/vulkan/VulkanDebugUtils.h>
#include <ATen/native/vulkan/VulkanVulkan.h>
#include "vulkan_wrapper.h"

#include <ATen/native/vulkan/glsl.h>
#include <ATen/native/vulkan/spv.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include "shaderc/shaderc.hpp"
#define GLSL_SPV(name) name##_glsl
#else
#define GLSL_SPV(name) name##_spv, name##_spv_len
#endif

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

#define VK_CHECK_RESULT(f)                                          \
  {                                                                 \
    VkResult res = (f);                                             \
    if (res != VK_SUCCESS) {                                        \
      std::cout << FLF << " VK_CHECK_RESULT Fatal VkResult:" << res \
                << std::endl;                                       \
      assert(res == VK_SUCCESS);                                    \
      throw std::runtime_error("VK_CHECK_RESULT Fail");             \
    }                                                               \
  }

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

static const bool enableValidationLayers = true;

class AVKContext;
static std::unique_ptr<AVKContext> vkContext;

void initVulkanContextOnce() {
  COUT_FLF;

  static const int once = []() {
    bool res = InitVulkan();
    if (!res) {
      std::cout << FLF << " ERROR Failed to InitVulkan" << std::endl;
      assert(false);
    }
    std::cout << FLF << "InitVulkan ok" << std::endl;

    vkContext = std::make_unique<AVKContext>();
    if (!vkContext) {
      std::cout << FLF << " ERROR Failed to create AVKContext" << std::endl;
      assert(false);
    }
    std::cout << FLF << " AVKContext created ok" << std::endl;
    return 0;
  }();
  ((void)once);
}

class AVKContext {
 public:
  AVKContext() {
    COUT_FLF;
    createInstance();
    findPhysicalDevice();
    createDevice();
  }
  ~AVKContext() {
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

 private:
  void createInstance() {
    std::vector<const char*> enabledExtensions;
    if (enableValidationLayers) {
      uint32_t layer_present_count;
      vkEnumerateInstanceLayerProperties(&layer_present_count, nullptr);
      std::cout << "validation_layer_present_count:" << layer_present_count
                << std::endl;

      std::vector<VkLayerProperties> layer_props(layer_present_count);
      vkEnumerateInstanceLayerProperties(
          &layer_present_count, layer_props.data());
      for (uint32_t i = 0; i < layer_present_count; i++) {
        printf(
            "validation_layer_present[%u]:%s\n", i, layer_props[i].layerName);
      }

      const char* instance_layers[] = {
          "VK_LAYER_GOOGLE_unique_objects",
          "VK_LAYER_GOOGLE_threading",
          "VK_LAYER_LUNARG_object_tracker",
          "VK_LAYER_LUNARG_core_validation",
          "VK_LAYER_LUNARG_parameter_validation",
          "VK_LAYER_KHRONOS_validation",
      };

      uint32_t instance_layer_request_count =
          sizeof(instance_layers) / sizeof(instance_layers[0]);
      for (uint32_t i = 0; i < instance_layer_request_count; i++) {
        bool found = false;
        for (uint32_t j = 0; j < layer_present_count; j++) {
          if (strcmp(instance_layers[i], layer_props[j].layerName) == 0) {
            found = true;
          }
        }

        if (found) {
          enabledValidationLayers_.push_back(instance_layers[i]);
        } else {
          std::cout << "Validation layer not supported " << instance_layers[i]
                    << std::endl;
        }
      }

      uint32_t extension_count;
      vkEnumerateInstanceExtensionProperties(
          nullptr, &extension_count, nullptr);
      std::vector<VkExtensionProperties> extension_props(extension_count);
      vkEnumerateInstanceExtensionProperties(
          nullptr, &extension_count, extension_props.data());
      for (uint32_t i = 0; i < extension_count; i++) {
        std::cout << "extension_present " << i << " "
                  << extension_props[i].extensionName << std::endl;
      }

      bool foundExtension = false;
      for (VkExtensionProperties prop : extension_props) {
        if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) ==
            0) {
          foundExtension = true;
          break;
        }
      }

      if (foundExtension) {
        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
      } else {
        std::cout
            << "Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported"
            << std::endl;
      }
    }

    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "vkphappname";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "Compute";
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

    if (enableValidationLayers) {
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
      if (vkCreateDebugReportCallbackEXT == nullptr) {
        throw std::runtime_error(
            "Could not load vkCreateDebugReportCallbackEXT");
      }

      VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(
          instance_,
          &debugReportCallbackCreateInfo,
          nullptr,
          &debugReportCallback_));
    }
    COUT_FLF;
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
    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
      std::cout << "ERROR: " << pLayerPrefix << " " << msgCode << " " << pMsg
                << std::endl;
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
      std::cout << "WARNING: " << pLayerPrefix << " " << msgCode << " " << pMsg
                << std::endl;
    } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
      std::cout << "PERF_WARNING: " << pLayerPrefix << " " << msgCode << " "
                << pMsg << std::endl;
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
      std::cout << "INFO: " << pLayerPrefix << " " << msgCode << " " << pMsg
                << std::endl;
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
      std::cout << "DEBUG: " << pLayerPrefix << " " << msgCode << " " << pMsg
                << std::endl;
    }
    return VK_FALSE;
  }

  void findPhysicalDevice() {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("could not find a device with vulkan support");
    }

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
    for (; i < queueFamilies.size(); ++i) {
      VkQueueFamilyProperties props = queueFamilies[i];
      if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        break;
      }
    }

    if (i == queueFamilies.size()) {
      assert(false);
      throw std::runtime_error(
          "could not find a queue family that supports operations");
    }

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
    if (enableValidationLayers) {
      auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
          instance_, "vkDestroyDebugReportCallbackEXT");
      if (func == nullptr) {
        throw std::runtime_error(
            "Could not load vkDestroyDebugReportCallbackEXT");
      }
      func(instance_, debugReportCallback_, nullptr);
    }

    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
  }
};

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

class AVKBuffer {
 public:
  AVKBuffer(
      uint32_t bufferSize,
      VkPhysicalDevice physicalDevice,
      VkDevice device,
      bool isUniform)
      : bufferSize_(bufferSize),
        physicalDevice_(physicalDevice),
        device_(device) {
    COUT_FLF;
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = isUniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                       : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    COUT_FLF;

    VK_CHECK_RESULT(
        vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &buffer_));
    COUT_FLF;
    VkMemoryRequirements memoryRequirements;
    COUT_FLF;
    vkGetBufferMemoryRequirements(device_, buffer_, &memoryRequirements);
    COUT_FLF;
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(
        physicalDevice_,
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    COUT_FLF;
    VK_CHECK_RESULT(
        vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory_));
    COUT_FLF;
    VK_CHECK_RESULT(vkBindBufferMemory(device_, buffer_, bufferMemory_, 0));
    COUT_FLF;
  }

  void toHost(void* outputData, int64_t size) {
    void* mappedMemory = nullptr;
    vkMapMemory(device_, bufferMemory_, 0, size, 0, &mappedMemory);
    ::memcpy(outputData, mappedMemory, size);
    vkUnmapMemory(device_, bufferMemory_);
  }

  void toDevice(void* data, int64_t size) {
    void* mappedMemory = nullptr;
    vkMapMemory(device_, bufferMemory_, 0, size, 0, &mappedMemory);
    ::memcpy(mappedMemory, data, size);
    vkUnmapMemory(device_, bufferMemory_);
  }

  ~AVKBuffer() {
    COUT_FLF;
    vkFreeMemory(device_, bufferMemory_, nullptr);
    COUT_FLF;
    vkDestroyBuffer(device_, buffer_, nullptr);
    COUT_FLF;
  }

  uint32_t bufferSize_;
  VkBuffer buffer_;
  VkDeviceMemory bufferMemory_;
  VkPhysicalDevice physicalDevice_;
  VkDevice device_;
};

AVKImage::AVKImage(int64_t W, int64_t H, int64_t C) {
  auto& device = vkContext->device_;
  auto& physicalDevice = vkContext->physicalDevice_;
  int32_t C_4 = UP_DIV(C, 4);

  initialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  filter_ = VK_FILTER_NEAREST;
  samplerAddressMode_ = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  format_ = VK_FORMAT_R16G16B16A16_SFLOAT;

  COUT_FLF;

  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_3D;
  imageInfo.extent.width = static_cast<uint32_t>(W);
  imageInfo.extent.height = static_cast<uint32_t>(H);
  imageInfo.extent.depth = static_cast<uint32_t>(C_4);

  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format_;
  imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
  imageInfo.initialLayout = initialLayout_;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.pNext = nullptr;
  imageInfo.flags = 0;

  COUT_FLF;
  VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &image_));
  COUT_FLF;

  VkMemoryRequirements memReqs = {};
  COUT_FLF;
  vkGetImageMemoryRequirements(device, image_, &memReqs);
  COUT_FLF;
  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice,
      memReqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  COUT_FLF;
  VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory_));
  COUT_FLF;
  VK_CHECK_RESULT(vkBindImageMemory(device, image_, imageMemory_, 0));
  COUT_FLF;

  COUT_FLF;
  VkImageViewCreateInfo _imageViewCreateInfo = imageViewCreateInfo();
  COUT_FLF;
  VK_CHECK_RESULT(
      vkCreateImageView(device, &_imageViewCreateInfo, nullptr, &imageView_));
  COUT_FLF;

  VkSamplerCreateInfo _samplerCreateInfo = samplerCreateInfo();
  COUT_FLF;
  VK_CHECK_RESULT(
      vkCreateSampler(device, &_samplerCreateInfo, nullptr, &sampler_));
  COUT_FLF;
}

AVKImage::~AVKImage() {
  COUT_FLF;
  vkDestroySampler(vkContext->device_, sampler_, nullptr);
  COUT_FLF;
  vkDestroyImageView(vkContext->device_, imageView_, nullptr);
  COUT_FLF;
  vkFreeMemory(vkContext->device_, imageMemory_, nullptr);
  COUT_FLF;
  vkDestroyImage(vkContext->device_, image_, nullptr);
  COUT_FLF;
}

VkImageViewCreateInfo AVKImage::imageViewCreateInfo() {
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

VkSamplerCreateInfo AVKImage::samplerCreateInfo() {
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
    COUT_FLF;
    auto& device = vkContext->device_;
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = codeSize;

    COUT_FLF;
    VK_CHECK_RESULT(vkCreateShaderModule(
        device, &createInfo, nullptr, &computeShaderModule_));
    COUT_FLF;

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

    COUT_FLF;
    VK_CHECK_RESULT(vkCreatePipelineLayout(
        device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout_));
    COUT_FLF;

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout_;

    COUT_FLF;
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline_));
    COUT_FLF;
  }

#ifdef USE_VULKAN_SHADERC_RUNTIME
  void createComputePipelineCompile(
      std::string glslSrc,
      const VkDescriptorSetLayout& descrSetLayout) {
    COUT_FLF;
    std::cout << "\nGLSL{\n"
              << glslSrc << "\n}GLSL size:" << glslSrc.size() << std::endl;
    shaderc::Compiler compiler;
    COUT_FLF;
    shaderc::CompileOptions options;
    options.SetGenerateDebugInfo();
    options.SetTargetEnvironment(
        shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
    options.SetForcedVersionProfile(450, shaderc_profile_core);

    shaderc::PreprocessedSourceCompilationResult ppRes =
        compiler.PreprocessGlsl(
            glslSrc.c_str(),
            glslSrc.size(),
            shaderc_compute_shader,
            "vulkan_shader.comp",
            options);

    auto ppCompStat = ppRes.GetCompilationStatus();
    auto numErrors = ppRes.GetNumErrors();
    auto numWarnings = ppRes.GetNumWarnings();
    std::cout << "shaderc compilation numError:" << numError
              << " numWarnings:" << numWarnings << std::endl;
    if (ppCompStat != shaderc_compilation_status_success) {
      std::cout << "Shader preproc compilation error ppCompStat:" << ppCompStat
                << ppRes.GetErrorMessage() << std::endl;
      assert(false);
    } else {
      COUT_FLF;
      std::vector<char> preproc{ppRes.cbegin(), ppRes.cend()};
      std::cout << "GLSL Preproc{" << preproc.data() << "}GLSL Preproc"
                << std::endl;
      COUT_FLF;
    }
    COUT_FLF;
    shaderc::SpvCompilationResult compilationResult = compiler.CompileGlslToSpv(
        glslSrc.c_str(),
        glslSrc.size(),
        shaderc_compute_shader,
        "vulkan_shader.comp",
        "main",
        options);
    COUT_FLF;
    auto compStat = compilationResult.GetCompilationStatus();
    if (compStat != shaderc_compilation_status_success) {
      std::cout << "Shader compilation error compStat:" << compStat
                << compilationResult.GetErrorMessage() << std::endl;
      assert(false);
    }
    COUT_FLF;
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
    COUT_FLF;
    auto& device = vkContext->device_;
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = vkContext->queueFamilyIndex_;
    COUT_FLF;
    VK_CHECK_RESULT(vkCreateCommandPool(
        device, &commandPoolCreateInfo, nullptr, &commandPool_));
    COUT_FLF;
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool_;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    COUT_FLF;

    VK_CHECK_RESULT(vkAllocateCommandBuffers(
        device, &commandBufferAllocateInfo, &commandBuffer_));

    COUT_FLF;
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    COUT_FLF;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer_, &beginInfo));
    COUT_FLF;

    vkCmdBindPipeline(
        commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    COUT_FLF;
    vkCmdBindDescriptorSets(
        commandBuffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout_,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    COUT_FLF;
    vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
    COUT_FLF;
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer_));
    COUT_FLF;
  }
  void runCommandBuffer() {
    COUT_FLF;
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;
    COUT_FLF;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(
        vkCreateFence(vkContext->device_, &fenceCreateInfo, NULL, &fence))

    COUT_FLF;
    VK_CHECK_RESULT(vkQueueSubmit(vkContext->queue_, 1, &submitInfo, fence));
    COUT_FLF;
    vkWaitForFences(vkContext->device_, 1, &fence, VK_TRUE, 100000000000);
    COUT_FLF;

    vkDestroyFence(vkContext->device_, fence, NULL);
    COUT_FLF;
  }

 private:
  VkCommandPool commandPool_; // own
  VkCommandBuffer commandBuffer_; // own
  VkPipeline pipeline_;
  VkPipelineLayout pipelineLayout_; // own
  VkShaderModule computeShaderModule_; // own
};

VulkanVulkanTensor::VulkanVulkanTensor(std::vector<int64_t> sizes)
    : sizes_(sizes) {
  COUT_FLF;
  assert(sizes_.size() == 4);
  initVulkanContextOnce();
}

#ifdef USE_VULKAN_SHADERC_RUNTIME
auto makeComputeUnit(
    const char* glslSrc,
    const VkDescriptorSetLayout& descrSetLayout) {
  COUT_FLPF;
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
  COUT_FLPF;
  auto computeUnit = std::make_unique<ComputeUnit>();
  const uint32_t* code = reinterpret_cast<const uint32_t*>(spvCode);
  const uint32_t codeSize = spvCodeSize;
  computeUnit->createComputePipeline(code, codeSize, descrSetLayout);
  return computeUnit;
}
#endif

struct uniforms {
  int32_t w;
  int32_t h;
};

void VulkanVulkanTensor::setDataFromHost(const float* inputData) {
  COUT_FLF;
  initVulkanContextOnce();

  auto& device = vkContext->device_;
  auto& physicalDevice = vkContext->physicalDevice_;

  int64_t numel = 1;
  for (const auto& d : sizes_) {
    numel *= d;
  }

  int32_t N = sizes_[0];
  int32_t C = sizes_[1];
  int32_t H = sizes_[2];
  int32_t W = sizes_[3];
  int32_t C_4 = UP_DIV(C, 4);

  tensorImage_ = std::make_unique<AVKImage>(W, H, C);

  COUT_FLF;
  int64_t inputDataSize = sizeof(float) * numel;
  int64_t bufferDataSizeAligned = ROUND_UP(inputDataSize, vkContext->sboAlign_);
  AVKBuffer bufferData{bufferDataSizeAligned, physicalDevice, device, false};
  bufferData.toDevice((void*)inputData, inputDataSize);
  COUT_FLF;

  uniforms wh{W, H};
  int64_t bufferConstSize = sizeof(wh);
  int64_t bufferConstSizeAligned =
      ROUND_UP(bufferConstSize, vkContext->uboAlign_);
  AVKBuffer bufferConst{bufferConstSizeAligned, physicalDevice, device, true};
  bufferConst.toDevice((void*)&wh, bufferConstSize);

  COUT_FLF;
  VkDescriptorSet descriptorSet = {};
  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorPool descrPool = {};

  COUT_FLF;
  VkDescriptorSetLayoutBinding descrSetLayoutBinding[] = {
      {
          0, // binding
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      },
      {
          1, // binding
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // descriptorType
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      },
      {
          2, // binding
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // descriptorType
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      }};

  COUT_FLF;
  VkDescriptorSetLayoutCreateInfo descrSetLayoutCreateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0, // flags
      3, // bindingCount
      descrSetLayoutBinding // pBindings
  };

  COUT_FLF;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
      device, &descrSetLayoutCreateInfo, nullptr, &descrSetLayout));
  COUT_FLF;

  VkDescriptorPoolSize descrPoolSize[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};

  COUT_FLF;
  VkDescriptorPoolCreateInfo descrPoolCreateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0, // flags
      1, // maxSets
      3, // poolSizeCount
      descrPoolSize // pPoolSizes
  };

  COUT_FLF;
  VK_CHECK_RESULT(vkCreateDescriptorPool(
      device, &descrPoolCreateInfo, nullptr, &descrPool));
  COUT_FLF;

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      descrPool, // descrPool
      1, // descriptorSetCount
      &descrSetLayout // pSetLayouts
  };
  COUT_FLF;
  VK_CHECK_RESULT(vkAllocateDescriptorSets(
      device, &descriptorSetAllocateInfo, &descriptorSet));
  COUT_FLF;

  VkDescriptorBufferInfo descrBufferData = {};
  descrBufferData.buffer = bufferData.buffer_;
  descrBufferData.offset = 0;
  descrBufferData.range = bufferData.bufferSize_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrBufferData = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      1, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr, // pImageInfo
      &descrBufferData, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrBufferData, 0, nullptr);
  COUT_FLF;

  VkDescriptorBufferInfo descrBufferConst = {};
  descrBufferConst.buffer = bufferConst.buffer_;
  descrBufferConst.offset = 0;
  descrBufferConst.range = bufferConst.bufferSize_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrBufferConst = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      2, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      nullptr, // pImageInfo
      &descrBufferConst, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrBufferConst, 0, nullptr);
  COUT_FLF;

  VkDescriptorImageInfo descrImage = {};
  descrImage.sampler = tensorImage_->sampler_;
  descrImage.imageLayout = tensorImage_->imageLayout_;
  descrImage.imageView = tensorImage_->imageView_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrImage = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      0, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      &descrImage, // pImageInfo
      nullptr, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrImage, 0, nullptr);
  COUT_FLF;
  auto computeUnit = makeComputeUnit(
      at::native::vulkan::GLSL_SPV(vulkan_nchw_buf_to_tex), descrSetLayout);
  COUT_FLF;
  computeUnit->createCommandBuffer(
      descriptorSet, UP_DIV(W, 8), UP_DIV(H, 8), C_4);
  COUT_FLF;
  computeUnit->runCommandBuffer();
  COUT_FLF;
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  COUT_FLF;
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
  COUT_FLF;
}

void VulkanVulkanTensor::copyDataToHost(float* output) {
  COUT_FLF;
  initVulkanContextOnce();

  auto& device = vkContext->device_;
  auto& physicalDevice = vkContext->physicalDevice_;

  int64_t numel = 1;
  for (const auto& d : sizes_) {
    numel *= d;
  }

  int N = sizes_[0];
  int C = sizes_[1];
  int H = sizes_[2];
  int W = sizes_[3];
  int C_4 = UP_DIV(C, 4);

  COUT_FLF;
  int64_t bufferDataSize = sizeof(float) * numel;
  int64_t bufferDataSizeAligned =
      ROUND_UP(bufferDataSize, vkContext->sboAlign_);
  AVKBuffer bufferData{bufferDataSizeAligned, physicalDevice, device, false};

  COUT_FLF;

  uniforms wh{W, H};
  int64_t bufferConstSize = sizeof(wh);
  int64_t bufferConstSizeAligned =
      ROUND_UP(bufferConstSize, vkContext->uboAlign_);
  AVKBuffer bufferConst{bufferConstSizeAligned, physicalDevice, device, true};
  bufferConst.toDevice((void*)&wh, bufferConstSize);

  COUT_FLF;
  VkDescriptorSet descriptorSet = {};
  VkDescriptorSetLayout descrSetLayout = {};
  VkDescriptorPool descrPool = {};

  COUT_FLF;
  VkDescriptorSetLayoutBinding descrSetLayoutBinding[] = {
      {
          0, // binding
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      },
      {
          1, // binding
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // descriptorType
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      },
      {
          2, // binding
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // descriptorType
          1, // descriptorCount
          VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
          nullptr // pImmutableSamplers
      }};

  COUT_FLF;
  VkDescriptorSetLayoutCreateInfo descrSetLayoutCreateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0, // flags
      3, // bindingCount
      descrSetLayoutBinding // pBindings
  };

  COUT_FLF;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
      device, &descrSetLayoutCreateInfo, nullptr, &descrSetLayout));
  COUT_FLF;

  VkDescriptorPoolSize descrPoolSize[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};

  COUT_FLF;
  VkDescriptorPoolCreateInfo descrPoolCreateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0, // flags
      1, // maxSets
      3, // poolSizeCount
      descrPoolSize // pPoolSizes
  };

  COUT_FLF;
  VK_CHECK_RESULT(vkCreateDescriptorPool(
      device, &descrPoolCreateInfo, nullptr, &descrPool));
  COUT_FLF;

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      descrPool, // descrPool
      1, // descriptorSetCount
      &descrSetLayout // pSetLayouts
  };
  COUT_FLF;
  VK_CHECK_RESULT(vkAllocateDescriptorSets(
      device, &descriptorSetAllocateInfo, &descriptorSet));
  COUT_FLF;

  VkDescriptorBufferInfo descrBufferData = {};
  descrBufferData.buffer = bufferData.buffer_;
  descrBufferData.offset = 0;
  descrBufferData.range = bufferData.bufferSize_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrBufferData = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      1, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      nullptr, // pImageInfo
      &descrBufferData, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrBufferData, 0, nullptr);
  COUT_FLF;

  VkDescriptorBufferInfo descrBufferConst = {};
  descrBufferConst.buffer = bufferConst.buffer_;
  descrBufferConst.offset = 0;
  descrBufferConst.range = bufferConst.bufferSize_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrBufferConst = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      2, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      nullptr, // pImageInfo
      &descrBufferConst, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrBufferConst, 0, nullptr);
  COUT_FLF;

  VkDescriptorImageInfo descrImage = {};
  descrImage.sampler = tensorImage_->sampler_;
  descrImage.imageView = tensorImage_->imageView_;
  descrImage.imageLayout = tensorImage_->imageLayout_;
  COUT_FLF;
  VkWriteDescriptorSet writeDescrImage = {
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
      nullptr, // pNext
      descriptorSet, // dstSet
      0, // dstBinding
      0, // dstArrayElement
      1, // descriptorCount
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      &descrImage, // pImageInfo
      nullptr, // pBufferInfo
      nullptr, // pTexelBufferView
  };
  COUT_FLF;
  vkUpdateDescriptorSets(device, 1, &writeDescrImage, 0, nullptr);
  COUT_FLF;
  auto computeUnit =
      makeComputeUnit(GLSL_SPV(vulkan_tex_to_nchw_buf), descrSetLayout);
  COUT_FLF;
  computeUnit->createCommandBuffer(
      descriptorSet, UP_DIV(W, 8), UP_DIV(H, 8), C_4);
  COUT_FLF;
  computeUnit->runCommandBuffer();
  COUT_FLF;

  bufferData.toHost(output, bufferDataSize);

  COUT_FLF;
  vkDestroyDescriptorPool(device, descrPool, nullptr);
  COUT_FLF;
  vkDestroyDescriptorSetLayout(device, descrSetLayout, nullptr);
  COUT_FLF;
  at::native::vulkan::debug::vk_print4d("copyDataToHost", output, N, C, H, W);
}

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
#endif
