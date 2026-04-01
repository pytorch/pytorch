#include "vulkan-api.h"

#include "slang.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if _DEBUG
#define ENABLE_VALIDATION_LAYER 1
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
    VkDebugReportFlagsEXT /*flags*/,
    VkDebugReportObjectTypeEXT /*objType*/,
    uint64_t /*srcObject*/,
    size_t /*location*/,
    int32_t /*msgCode*/,
    const char* pLayerPrefix,
    const char* pMsg,
    void* /*pUserData*/
)
{
    printf("[%s]: %s\n", pLayerPrefix, pMsg);
    return 1;
}

int initializeVulkanDevice(VulkanAPI& api)
{
    // Load vulkan library.
    const char* dynamicLibraryName = "Unknown";

#if SLANG_WINDOWS_FAMILY
    dynamicLibraryName = "vulkan-1.dll";
    HMODULE module = ::LoadLibraryA(dynamicLibraryName);
    api.vulkanLibraryHandle = (void*)module;
#define VK_API_GET_GLOBAL_PROC(x) api.x = (PFN_##x)GetProcAddress(module, #x);
#elif SLANG_APPLE_FAMILY
    dynamicLibraryName = "libvulkan.dylib";
    api.vulkanLibraryHandle = dlopen(dynamicLibraryName, RTLD_NOW);
#define VK_API_GET_GLOBAL_PROC(x) api.x = (PFN_##x)dlsym(api.vulkanLibraryHandle, #x);
#else
    dynamicLibraryName = "libvulkan.so.1";
    api.vulkanLibraryHandle = dlopen(dynamicLibraryName, RTLD_NOW);
#define VK_API_GET_GLOBAL_PROC(x) api.x = (PFN_##x)dlsym(api.vulkanLibraryHandle, #x);
#endif

    // Initialize all the global functions.
    VK_API_ALL_GLOBAL_PROCS(VK_API_GET_GLOBAL_PROC)
    if (!api.vkCreateInstance)
        return -1;

    // Enable validation layer if available.
    std::vector<const char*> layers;
#ifdef ENABLE_VALIDATION_LAYER
    uint32_t propertyCount;
    if (api.vkEnumerateInstanceLayerProperties(&propertyCount, nullptr) != 0)
        return -1;
    std::vector<VkLayerProperties> properties(propertyCount);
    if (api.vkEnumerateInstanceLayerProperties(&propertyCount, properties.data()) != 0)
        return -1;
    for (const auto& p : properties)
    {
        if (strcmp(p.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
#endif

    // Create Vulkan Instance.
    VkApplicationInfo applicationInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    applicationInfo.pApplicationName = "slang-hello-world";
    applicationInfo.pEngineName = "slang-hello-world";
    applicationInfo.apiVersion = VK_API_VERSION_1_2;
    applicationInfo.engineVersion = 1;
    applicationInfo.applicationVersion = 1;
    const char* instanceExtensions[] = {
#if SLANG_APPLE_FAMILY
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
    };
    VkInstanceCreateInfo instanceCreateInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#if SLANG_APPLE_FAMILY
    instanceCreateInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledExtensionCount = SLANG_COUNT_OF(instanceExtensions);
    instanceCreateInfo.ppEnabledExtensionNames = &instanceExtensions[0];
    if (layers.size())
    {
        instanceCreateInfo.ppEnabledLayerNames = &layers[0];
        instanceCreateInfo.enabledLayerCount = (uint32_t)layers.size();
    }
    if (api.vkCreateInstance(&instanceCreateInfo, nullptr, &api.instance) != 0)
        return -1;

    // Load instance functions.
    api.initInstanceProcs();

    // Create debug report callback.
    if (api.vkCreateDebugReportCallbackEXT)
    {
        VkDebugReportFlagsEXT debugFlags =
            VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;

        VkDebugReportCallbackCreateInfoEXT debugCreateInfo = {
            VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT};
        debugCreateInfo.pfnCallback = &debugMessageCallback;
        debugCreateInfo.pUserData = nullptr;
        debugCreateInfo.flags = debugFlags;

        RETURN_ON_FAIL(api.vkCreateDebugReportCallbackEXT(
            api.instance,
            &debugCreateInfo,
            nullptr,
            &api.debugReportCallback));
    }

    // Enumerate physical devices.
    uint32_t numPhysicalDevices = 0;
    RETURN_ON_FAIL(api.vkEnumeratePhysicalDevices(api.instance, &numPhysicalDevices, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices;
    physicalDevices.resize(numPhysicalDevices);
    RETURN_ON_FAIL(
        api.vkEnumeratePhysicalDevices(api.instance, &numPhysicalDevices, &physicalDevices[0]));

    // We will use device 0.
    api.initPhysicalDevice(physicalDevices[0]);

    VkDeviceCreateInfo deviceCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &api.deviceFeatures;

    // Find proper queue family index.
    uint32_t numQueueFamilies = 0;
    api.vkGetPhysicalDeviceQueueFamilyProperties(api.physicalDevice, &numQueueFamilies, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies;
    queueFamilies.resize(numQueueFamilies);
    api.vkGetPhysicalDeviceQueueFamilyProperties(
        api.physicalDevice,
        &numQueueFamilies,
        &queueFamilies[0]);

    // Find a queue that can service our needs.
    auto requiredQueueFlags = VK_QUEUE_COMPUTE_BIT;
    for (int i = 0; i < int(numQueueFamilies); ++i)
    {
        if ((queueFamilies[i].queueFlags & requiredQueueFlags) == requiredQueueFlags)
        {
            api.queueFamilyIndex = i;
            break;
        }
    }
    if (api.queueFamilyIndex == -1)
        return -1;

#if SLANG_APPLE_FAMILY
    const char* deviceExtensions[] = {
        "VK_KHR_portability_subset",
    };
#endif

    VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    float queuePriority = 0.0f;
    queueCreateInfo.queueFamilyIndex = api.queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
#if SLANG_APPLE_FAMILY
    deviceCreateInfo.enabledExtensionCount = SLANG_COUNT_OF(deviceExtensions);
    deviceCreateInfo.ppEnabledExtensionNames = &deviceExtensions[0];
#endif
    RETURN_ON_FAIL(api.vkCreateDevice(api.physicalDevice, &deviceCreateInfo, nullptr, &api.device));

    // Load device functions.
    api.initDeviceProcs();

    return 0;
}

int VulkanAPI::initInstanceProcs()
{
    assert(instance && vkGetInstanceProcAddr != nullptr);

#define VK_API_GET_INSTANCE_PROC(x) x = (PFN_##x)vkGetInstanceProcAddr(instance, #x);

    VK_API_ALL_INSTANCE_PROCS(VK_API_GET_INSTANCE_PROC)
    // Get optional
    VK_API_INSTANCE_PROCS_OPT(VK_API_GET_INSTANCE_PROC)

#undef VK_API_GET_INSTANCE_PROC

    return 0;
}

int VulkanAPI::initPhysicalDevice(VkPhysicalDevice inPhysicalDevice)
{
    assert(physicalDevice == VK_NULL_HANDLE);
    physicalDevice = inPhysicalDevice;

    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

    return 0;
}

int VulkanAPI::initDeviceProcs()
{
    assert(instance && device && vkGetDeviceProcAddr != nullptr);

#define VK_API_GET_DEVICE_PROC(x) x = (PFN_##x)vkGetDeviceProcAddr(device, #x);
    VK_API_DEVICE_PROCS(VK_API_GET_DEVICE_PROC)
#undef VK_API_GET_DEVICE_PROC

    return 0;
}

int VulkanAPI::findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
{
    assert(typeBits);

    const int numMemoryTypes = int(deviceMemoryProperties.memoryTypeCount);

    // bit holds current test bit against typeBits. Ie bit == 1 << typeBits

    uint32_t bit = 1;
    for (int i = 0; i < numMemoryTypes; ++i, bit += bit)
    {
        auto const& memoryType = deviceMemoryProperties.memoryTypes[i];
        if ((typeBits & bit) && (memoryType.propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    // assert(!"failed to find a usable memory type");
    return -1;
}

VulkanAPI::~VulkanAPI()
{
    if (vkDestroyDevice)
    {
        vkDestroyDevice(device, nullptr);
    }
    if (vkDestroyDebugReportCallbackEXT)
    {
        vkDestroyDebugReportCallbackEXT(instance, debugReportCallback, nullptr);
    }
    if (vkDestroyInstance)
    {
        vkDestroyInstance(instance, nullptr);
    }
}
