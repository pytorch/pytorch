//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#ifdef _WIN32

#include "SparseBindingTest.h"
#include "Tests.h"
#include "VmaUsage.h"
#include "Common.h"
#include <atomic>
#include <Shlwapi.h>
#include <unordered_set>

#pragma comment(lib, "shlwapi.lib")

static const char* const SHADER_PATH1 = "./Shaders/";
static const char* const SHADER_PATH2 = "../bin/";
static const wchar_t* const WINDOW_CLASS_NAME = L"VULKAN_MEMORY_ALLOCATOR_SAMPLE";
static const char* const VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
static const char* const APP_TITLE_A =     "Vulkan Memory Allocator Sample 3.3.0";
static const wchar_t* const APP_TITLE_W = L"Vulkan Memory Allocator Sample 3.3.0";

static const bool VSYNC = true;
static const uint32_t COMMAND_BUFFER_COUNT = 2;
static void* const CUSTOM_CPU_ALLOCATION_CALLBACK_USER_DATA = (void*)(intptr_t)43564544;
static const bool USE_CUSTOM_CPU_ALLOCATION_CALLBACKS = true;

enum class ExitCode : int
{
    GPUList = 2,
    Help = 1,
    Success = 0,
    RuntimeError = -1,
    CommandLineError = -2,
};

VkPhysicalDevice g_hPhysicalDevice;
VkDevice g_hDevice;
VmaAllocator g_hAllocator;
VkInstance g_hVulkanInstance;

bool g_EnableValidationLayer = true;
bool VK_KHR_get_memory_requirements2_enabled = false;
bool VK_KHR_get_physical_device_properties2_enabled = false;
bool VK_KHR_dedicated_allocation_enabled = false;
bool VK_KHR_bind_memory2_enabled = false;
bool VK_EXT_memory_budget_enabled = false;
bool VK_AMD_device_coherent_memory_enabled = false;
bool VK_KHR_buffer_device_address_enabled = false;
bool VK_EXT_memory_priority_enabled = false;
bool VK_EXT_debug_utils_enabled = false;
bool VK_KHR_maintenance5_enabled = false;
bool VK_KHR_external_memory_win32_enabled = false;
bool g_SparseBindingEnabled = false;

// # Pointers to functions from extensions
PFN_vkGetBufferDeviceAddressKHR g_vkGetBufferDeviceAddressKHR;

static HINSTANCE g_hAppInstance;
static HWND g_hWnd;
static LONG g_SizeX = 1280, g_SizeY = 720;
static VkSurfaceKHR g_hSurface;
static VkQueue g_hPresentQueue;
static VkSurfaceFormatKHR g_SurfaceFormat;
static VkExtent2D g_Extent;
static VkSwapchainKHR g_hSwapchain;
static std::vector<VkImage> g_SwapchainImages;
static std::vector<VkImageView> g_SwapchainImageViews;
static std::vector<VkFramebuffer> g_Framebuffers;
static VkCommandPool g_hCommandPool;
static VkCommandBuffer g_MainCommandBuffers[COMMAND_BUFFER_COUNT];
static VkFence g_MainCommandBufferExecutedFences[COMMAND_BUFFER_COUNT];
VkFence g_ImmediateFence;
static uint32_t g_NextCommandBufferIndex;
// Signaled by vkAcquireNextImageKHR.
static VkSemaphore g_hImageAvailableSemaphores[COMMAND_BUFFER_COUNT];
// Notice we need as many semaphores as there are swapchain images.
static std::vector<VkSemaphore> g_hRenderFinishedSemaphores;
static uint32_t g_SwapchainImageCount = 0;
static uint32_t g_SwapchainImageIndex = 0;
static uint32_t g_GraphicsQueueFamilyIndex = UINT_MAX;
static uint32_t g_PresentQueueFamilyIndex = UINT_MAX;
static uint32_t g_SparseBindingQueueFamilyIndex = UINT_MAX;
static VkDescriptorSetLayout g_hDescriptorSetLayout;
static VkDescriptorPool g_hDescriptorPool;
static VkDescriptorSet g_hDescriptorSet; // Automatically destroyed with m_DescriptorPool.
static VkSampler g_hSampler;
static VkFormat g_DepthFormat;
static VkImage g_hDepthImage;
static VmaAllocation g_hDepthImageAlloc;
static VkImageView g_hDepthImageView;

static VkSurfaceCapabilitiesKHR g_SurfaceCapabilities;
static std::vector<VkSurfaceFormatKHR> g_SurfaceFormats;
static std::vector<VkPresentModeKHR> g_PresentModes;

static const VkDebugUtilsMessageSeverityFlagsEXT DEBUG_UTILS_MESSENGER_MESSAGE_SEVERITY =
    //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
static const VkDebugUtilsMessageTypeFlagsEXT DEBUG_UTILS_MESSENGER_MESSAGE_TYPE =
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT_Func;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT_Func;
static PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT_Func;

static VkQueue g_hGraphicsQueue;
VkQueue g_hSparseBindingQueue;
VkCommandBuffer g_hTemporaryCommandBuffer;

static VkPipelineLayout g_hPipelineLayout;
static VkRenderPass g_hRenderPass;
static VkPipeline g_hPipeline;

static VkBuffer g_hVertexBuffer;
static VmaAllocation g_hVertexBufferAlloc;
static VkBuffer g_hIndexBuffer;
static VmaAllocation g_hIndexBufferAlloc;
static uint32_t g_VertexCount;
static uint32_t g_IndexCount;

static VkImage g_hTextureImage;
static VmaAllocation g_hTextureImageAlloc;
static VkImageView g_hTextureImageView;

static std::atomic_uint32_t g_CpuAllocCount;

static void* CustomCpuAllocation(
    void* pUserData, size_t size, size_t alignment,
    VkSystemAllocationScope allocationScope)
{
    assert(pUserData == CUSTOM_CPU_ALLOCATION_CALLBACK_USER_DATA);
    void* const result = _aligned_malloc(size, alignment);
    if(result)
    {
        ++g_CpuAllocCount;
    }
    return result;
}

static void* CustomCpuReallocation(
    void* pUserData, void* pOriginal, size_t size, size_t alignment,
    VkSystemAllocationScope allocationScope)
{
    assert(pUserData == CUSTOM_CPU_ALLOCATION_CALLBACK_USER_DATA);
    void* const result = _aligned_realloc(pOriginal, size, alignment);
    if(pOriginal && !result)
    {
        --g_CpuAllocCount;
    }
    else if(!pOriginal && result)
    {
        ++g_CpuAllocCount;
    }
    return result;
}

static void CustomCpuFree(void* pUserData, void* pMemory)
{
    assert(pUserData == CUSTOM_CPU_ALLOCATION_CALLBACK_USER_DATA);
    if(pMemory)
    {
        const uint32_t oldAllocCount = g_CpuAllocCount.fetch_sub(1);
        TEST(oldAllocCount > 0);
        _aligned_free(pMemory);
    }
}

static const VkAllocationCallbacks g_CpuAllocationCallbacks = {
    CUSTOM_CPU_ALLOCATION_CALLBACK_USER_DATA, // pUserData
    &CustomCpuAllocation, // pfnAllocation
    &CustomCpuReallocation, // pfnReallocation
    &CustomCpuFree // pfnFree
};

const VkAllocationCallbacks* g_Allocs;

struct GPUSelection
{
    uint32_t Index = UINT32_MAX;
    std::wstring Substring;
};

class VulkanUsage
{
public:
    void Init();
    ~VulkanUsage();
    void PrintPhysicalDeviceList() const;
    // If failed, returns VK_NULL_HANDLE.
    VkPhysicalDevice SelectPhysicalDevice(const GPUSelection& GPUSelection) const;

private:
    VkDebugUtilsMessengerEXT m_DebugUtilsMessenger = VK_NULL_HANDLE;

    void RegisterDebugCallbacks();
    static bool IsLayerSupported(const VkLayerProperties* pProps, size_t propCount, const char* pLayerName);
};

struct CommandLineParameters
{
    bool m_Help = false;
    bool m_List = false;
    bool m_Test = false;
    bool m_TestSparseBinding = false;
    GPUSelection m_GPUSelection;

    bool Parse(int argc, wchar_t** argv)
    {
        for(int i = 1; i < argc; ++i)
        {
            if(_wcsicmp(argv[i], L"-h") == 0 || _wcsicmp(argv[i], L"--Help") == 0)
            {
                m_Help = true;
            }
            else if(_wcsicmp(argv[i], L"-l") == 0 || _wcsicmp(argv[i], L"--List") == 0)
            {
                m_List = true;
            }
            else if((_wcsicmp(argv[i], L"-g") == 0 || _wcsicmp(argv[i], L"--GPU") == 0) && i + 1 < argc)
            {
                m_GPUSelection.Substring = argv[i + 1];
                ++i;
            }
            else if((_wcsicmp(argv[i], L"-i") == 0 || _wcsicmp(argv[i], L"--GPUIndex") == 0) && i + 1 < argc)
            {
                m_GPUSelection.Index = _wtoi(argv[i + 1]);
                ++i;
            }
            else if (_wcsicmp(argv[i], L"-t") == 0 || _wcsicmp(argv[i], L"--Test") == 0)
            {
                m_Test = true;
            }
            else if (_wcsicmp(argv[i], L"-s") == 0 || _wcsicmp(argv[i], L"--TestSparseBinding") == 0)
            {
                m_TestSparseBinding = true;
            }
            else
                return false;
        }
        return true;
    }
} g_CommandLineParameters;

void SetDebugUtilsObjectName(VkObjectType type, uint64_t handle, const std::string &name)
{
    if (vkSetDebugUtilsObjectNameEXT_Func == nullptr)
        return;

    VkDebugUtilsObjectNameInfoEXT info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name.c_str();
    ERR_GUARD_VULKAN( vkSetDebugUtilsObjectNameEXT_Func(g_hDevice, &info) );
}

void BeginSingleTimeCommands()
{
    VkCommandBufferBeginInfo cmdBufBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ERR_GUARD_VULKAN( vkBeginCommandBuffer(g_hTemporaryCommandBuffer, &cmdBufBeginInfo) );
}

void EndSingleTimeCommands()
{
    ERR_GUARD_VULKAN( vkEndCommandBuffer(g_hTemporaryCommandBuffer) );

    SetDebugUtilsObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, reinterpret_cast<std::uint64_t>(g_hTemporaryCommandBuffer), "g_hTemporaryCommandBuffer");

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &g_hTemporaryCommandBuffer;

    ERR_GUARD_VULKAN( vkQueueSubmit(g_hGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) );
    ERR_GUARD_VULKAN( vkQueueWaitIdle(g_hGraphicsQueue) );
}

void LoadShader(std::vector<char>& out, const char* fileName)
{
    std::ifstream file(std::string(SHADER_PATH1) + fileName, std::ios::ate | std::ios::binary);
    if(file.is_open() == false)
        file.open(std::string(SHADER_PATH2) + fileName, std::ios::ate | std::ios::binary);
    assert(file.is_open());
    size_t fileSize = (size_t)file.tellg();
    if(fileSize > 0)
    {
        out.resize(fileSize);
        file.seekg(0);
        file.read(out.data(), fileSize);
        file.close();
    }
    else
        out.clear();
}

static VkBool32 VKAPI_PTR MyDebugReportCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
    void*                                            pUserData)
{
    assert(pCallbackData && pCallbackData->pMessageIdName && pCallbackData->pMessage);

    switch(messageSeverity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        SetConsoleColor(CONSOLE_COLOR::WARNING);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        SetConsoleColor(CONSOLE_COLOR::ERROR_);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        SetConsoleColor(CONSOLE_COLOR::NORMAL);
        break;
    default: // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        SetConsoleColor(CONSOLE_COLOR::INFO);
    }

    printf("%s \xBA %s\n", pCallbackData->pMessageIdName, pCallbackData->pMessage);

    SetConsoleColor(CONSOLE_COLOR::NORMAL);

    if(messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ||
        messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        OutputDebugStringA(pCallbackData->pMessage);
        OutputDebugStringA("\n");
    }

    return VK_FALSE;
}

static VkSurfaceFormatKHR ChooseSurfaceFormat()
{
    assert(!g_SurfaceFormats.empty());

    if((g_SurfaceFormats.size() == 1) && (g_SurfaceFormats[0].format == VK_FORMAT_UNDEFINED))
    {
        VkSurfaceFormatKHR result = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        return result;
    }

    for(const auto& format : g_SurfaceFormats)
    {
        if((format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
            (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR))
        {
            return format;
        }
    }

    return g_SurfaceFormats[0];
}

VkPresentModeKHR ChooseSwapPresentMode()
{
    VkPresentModeKHR preferredMode = VSYNC ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR;

    if(std::find(g_PresentModes.begin(), g_PresentModes.end(), preferredMode) !=
        g_PresentModes.end())
    {
        return preferredMode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D ChooseSwapExtent()
{
    if(g_SurfaceCapabilities.currentExtent.width != UINT_MAX)
        return g_SurfaceCapabilities.currentExtent;

    VkExtent2D result = {
        std::max(g_SurfaceCapabilities.minImageExtent.width,
            std::min(g_SurfaceCapabilities.maxImageExtent.width, (uint32_t)g_SizeX)),
        std::max(g_SurfaceCapabilities.minImageExtent.height,
            std::min(g_SurfaceCapabilities.maxImageExtent.height, (uint32_t)g_SizeY)) };
    return result;
}

static constexpr uint32_t GetVulkanApiVersion()
{
#if VMA_VULKAN_VERSION == 1004000
    return VK_API_VERSION_1_4;
#elif VMA_VULKAN_VERSION == 1003000
    return VK_API_VERSION_1_3;
#elif VMA_VULKAN_VERSION == 1002000
    return VK_API_VERSION_1_2;
#elif VMA_VULKAN_VERSION == 1001000
    return VK_API_VERSION_1_1;
#elif VMA_VULKAN_VERSION == 1000000
    return VK_API_VERSION_1_0;
#else
#error Invalid VMA_VULKAN_VERSION.
    return UINT32_MAX;
#endif
}

void VulkanUsage::Init()
{
    g_hAppInstance = (HINSTANCE)GetModuleHandle(NULL);

    if(USE_CUSTOM_CPU_ALLOCATION_CALLBACKS)
    {
        g_Allocs = &g_CpuAllocationCallbacks;
    }

#ifdef VOLK_HEADER_VERSION
    ERR_GUARD_VULKAN(volkInitialize());
#endif

    uint32_t instanceLayerPropCount = 0;
    ERR_GUARD_VULKAN( vkEnumerateInstanceLayerProperties(&instanceLayerPropCount, nullptr) );
    std::vector<VkLayerProperties> instanceLayerProps(instanceLayerPropCount);
    if(instanceLayerPropCount > 0)
    {
        ERR_GUARD_VULKAN( vkEnumerateInstanceLayerProperties(&instanceLayerPropCount, instanceLayerProps.data()) );
    }

    if(g_EnableValidationLayer)
    {
        if(IsLayerSupported(instanceLayerProps.data(), instanceLayerProps.size(), VALIDATION_LAYER_NAME) == false)
        {
            wprintf(L"Layer \"%hs\" not supported.", VALIDATION_LAYER_NAME);
            g_EnableValidationLayer = false;
        }
    }

    uint32_t availableInstanceExtensionCount = 0;
    ERR_GUARD_VULKAN( vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, nullptr) );
    std::vector<VkExtensionProperties> availableInstanceExtensions(availableInstanceExtensionCount);
    if(availableInstanceExtensionCount > 0)
    {
        ERR_GUARD_VULKAN( vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, availableInstanceExtensions.data()) );
    }

    std::vector<const char*> enabledInstanceExtensions;
    enabledInstanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    enabledInstanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

    std::vector<const char*> instanceLayers;
    if(g_EnableValidationLayer)
    {
        instanceLayers.push_back(VALIDATION_LAYER_NAME);
    }

    for(const auto& extensionProperties : availableInstanceExtensions)
    {
        if(strcmp(extensionProperties.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0)
        {
            if(GetVulkanApiVersion() == VK_API_VERSION_1_0)
            {
                enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
                VK_KHR_get_physical_device_properties2_enabled = true;
            }
        }
        else if(strcmp(extensionProperties.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
        {
            enabledInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            VK_EXT_debug_utils_enabled = true;
        }
    }

    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = APP_TITLE_A;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = APP_TITLE_A;
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = GetVulkanApiVersion();

    VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;
    instInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
    instInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();
    instInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
    instInfo.ppEnabledLayerNames = instanceLayers.data();

    wprintf(L"Vulkan API version used: ");
    switch(appInfo.apiVersion)
    {
    case VK_API_VERSION_1_0: wprintf(L"1.0\n"); break;
#ifdef VK_VERSION_1_1
    case VK_API_VERSION_1_1: wprintf(L"1.1\n"); break;
#endif
#ifdef VK_VERSION_1_2
    case VK_API_VERSION_1_2: wprintf(L"1.2\n"); break;
#endif
#ifdef VK_VERSION_1_3
    case VK_API_VERSION_1_3: wprintf(L"1.3\n"); break;
#endif
#ifdef VK_VERSION_1_4
    case VK_API_VERSION_1_4: wprintf(L"1.4\n"); break;
#endif
    default: assert(0);
    }

    ERR_GUARD_VULKAN( vkCreateInstance(&instInfo, g_Allocs, &g_hVulkanInstance) );

#ifdef VOLK_HEADER_VERSION
    volkLoadInstance(g_hVulkanInstance);
#endif

    if(VK_EXT_debug_utils_enabled)
    {
        RegisterDebugCallbacks();
    }
}

VulkanUsage::~VulkanUsage()
{
    if(m_DebugUtilsMessenger)
    {
        vkDestroyDebugUtilsMessengerEXT_Func(g_hVulkanInstance, m_DebugUtilsMessenger, g_Allocs);
    }

    if(g_hVulkanInstance)
    {
        vkDestroyInstance(g_hVulkanInstance, g_Allocs);
        g_hVulkanInstance = VK_NULL_HANDLE;
    }
}

void VulkanUsage::PrintPhysicalDeviceList() const
{
    uint32_t deviceCount = 0;
    ERR_GUARD_VULKAN(vkEnumeratePhysicalDevices(g_hVulkanInstance, &deviceCount, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    if(deviceCount > 0)
    {
        ERR_GUARD_VULKAN(vkEnumeratePhysicalDevices(g_hVulkanInstance, &deviceCount, physicalDevices.data()));
    }

    for(size_t i = 0; i < deviceCount; ++i)
    {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
        wprintf(L"Physical device %zu: %hs\n", i, props.deviceName);
    }
}

VkPhysicalDevice VulkanUsage::SelectPhysicalDevice(const GPUSelection& GPUSelection) const
{
    uint32_t deviceCount = 0;
    ERR_GUARD_VULKAN(vkEnumeratePhysicalDevices(g_hVulkanInstance, &deviceCount, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    if(deviceCount > 0)
    {
        ERR_GUARD_VULKAN(vkEnumeratePhysicalDevices(g_hVulkanInstance, &deviceCount, physicalDevices.data()));
    }

    if(GPUSelection.Index != UINT32_MAX)
    {
        // Cannot specify both index and name.
        if(!GPUSelection.Substring.empty())
        {
            return VK_NULL_HANDLE;
        }

        return GPUSelection.Index < deviceCount ? physicalDevices[GPUSelection.Index] : VK_NULL_HANDLE;
    }

    if(!GPUSelection.Substring.empty())
    {
        VkPhysicalDevice result = VK_NULL_HANDLE;
        std::wstring name;
        for(uint32_t i = 0; i < deviceCount; ++i)
        {
            VkPhysicalDeviceProperties props = {};
            vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
            if(ConvertCharsToUnicode(&name, props.deviceName, strlen(props.deviceName), CP_UTF8) &&
                StrStrI(name.c_str(), GPUSelection.Substring.c_str()))
            {
                // Second matching device found - error.
                if(result != VK_NULL_HANDLE)
                {
                    return VK_NULL_HANDLE;
                }
                // First matching device found.
                result = physicalDevices[i];
            }
        }
        // Found or not, return it.
        return result;
    }

    // Select first one.
    return deviceCount > 0 ? physicalDevices[0] : VK_NULL_HANDLE;
}

void VulkanUsage::RegisterDebugCallbacks()
{
    vkCreateDebugUtilsMessengerEXT_Func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        g_hVulkanInstance, "vkCreateDebugUtilsMessengerEXT");
    vkDestroyDebugUtilsMessengerEXT_Func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        g_hVulkanInstance, "vkDestroyDebugUtilsMessengerEXT");
    vkSetDebugUtilsObjectNameEXT_Func = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(
        g_hVulkanInstance, "vkSetDebugUtilsObjectNameEXT");
    assert(vkCreateDebugUtilsMessengerEXT_Func);
    assert(vkDestroyDebugUtilsMessengerEXT_Func);
    assert(vkSetDebugUtilsObjectNameEXT_Func);

    VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    messengerCreateInfo.messageSeverity = DEBUG_UTILS_MESSENGER_MESSAGE_SEVERITY;
    messengerCreateInfo.messageType = DEBUG_UTILS_MESSENGER_MESSAGE_TYPE;
    messengerCreateInfo.pfnUserCallback = MyDebugReportCallback;
    ERR_GUARD_VULKAN( vkCreateDebugUtilsMessengerEXT_Func(g_hVulkanInstance, &messengerCreateInfo, g_Allocs, &m_DebugUtilsMessenger) );
}

bool VulkanUsage::IsLayerSupported(const VkLayerProperties* pProps, size_t propCount, const char* pLayerName)
{
    const VkLayerProperties* propsEnd = pProps + propCount;
    return std::find_if(
        pProps,
        propsEnd,
        [pLayerName](const VkLayerProperties& prop) -> bool {
        return strcmp(pLayerName, prop.layerName) == 0;
    }) != propsEnd;
}

struct Vertex
{
    float pos[3];
    float color[3];
    float texCoord[2];
};

static void CreateMesh()
{
    assert(g_hAllocator);

    static Vertex vertices[] = {
        // -X
        { { -1.f, -1.f, -1.f}, {1.0f, 0.0f, 0.0f}, {0.f, 0.f} },
        { { -1.f, -1.f,  1.f}, {1.0f, 0.0f, 0.0f}, {1.f, 0.f} },
        { { -1.f,  1.f, -1.f}, {1.0f, 0.0f, 0.0f}, {0.f, 1.f} },
        { { -1.f,  1.f,  1.f}, {1.0f, 0.0f, 0.0f}, {1.f, 1.f} },
        // +X
        { { 1.f, -1.f,  1.f}, {0.0f, 1.0f, 0.0f}, {0.f, 0.f} },
        { { 1.f, -1.f, -1.f}, {0.0f, 1.0f, 0.0f}, {1.f, 0.f} },
        { { 1.f,  1.f,  1.f}, {0.0f, 1.0f, 0.0f}, {0.f, 1.f} },
        { { 1.f,  1.f, -1.f}, {0.0f, 1.0f, 0.0f}, {1.f, 1.f} },
        // -Z
        { { 1.f, -1.f, -1.f}, {0.0f, 0.0f, 1.0f}, {0.f, 0.f} },
        { {-1.f, -1.f, -1.f}, {0.0f, 0.0f, 1.0f}, {1.f, 0.f} },
        { { 1.f,  1.f, -1.f}, {0.0f, 0.0f, 1.0f}, {0.f, 1.f} },
        { {-1.f,  1.f, -1.f}, {0.0f, 0.0f, 1.0f}, {1.f, 1.f} },
        // +Z
        { {-1.f, -1.f,  1.f}, {1.0f, 1.0f, 0.0f}, {0.f, 0.f} },
        { { 1.f, -1.f,  1.f}, {1.0f, 1.0f, 0.0f}, {1.f, 0.f} },
        { {-1.f,  1.f,  1.f}, {1.0f, 1.0f, 0.0f}, {0.f, 1.f} },
        { { 1.f,  1.f,  1.f}, {1.0f, 1.0f, 0.0f}, {1.f, 1.f} },
        // -Y
        { {-1.f, -1.f, -1.f}, {0.0f, 1.0f, 1.0f}, {0.f, 0.f} },
        { { 1.f, -1.f, -1.f}, {0.0f, 1.0f, 1.0f}, {1.f, 0.f} },
        { {-1.f, -1.f,  1.f}, {0.0f, 1.0f, 1.0f}, {0.f, 1.f} },
        { { 1.f, -1.f,  1.f}, {0.0f, 1.0f, 1.0f}, {1.f, 1.f} },
        // +Y
        { { 1.f,  1.f, -1.f}, {1.0f, 0.0f, 1.0f}, {0.f, 0.f} },
        { {-1.f,  1.f, -1.f}, {1.0f, 0.0f, 1.0f}, {1.f, 0.f} },
        { { 1.f,  1.f,  1.f}, {1.0f, 0.0f, 1.0f}, {0.f, 1.f} },
        { {-1.f,  1.f,  1.f}, {1.0f, 0.0f, 1.0f}, {1.f, 1.f} },
    };
    static uint16_t indices[] = {
         0,  1,  2,  3, USHRT_MAX,
         4,  5,  6,  7, USHRT_MAX,
         8,  9, 10, 11, USHRT_MAX,
        12, 13, 14, 15, USHRT_MAX,
        16, 17, 18, 19, USHRT_MAX,
        20, 21, 22, 23, USHRT_MAX,
    };

    size_t vertexBufferSize = sizeof(Vertex) * _countof(vertices);
    size_t indexBufferSize = sizeof(uint16_t) * _countof(indices);
    g_IndexCount = (uint32_t)_countof(indices);

    // Create vertex buffer

    VkBufferCreateInfo vbInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    vbInfo.size = vertexBufferSize;
    vbInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vbInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vbAllocCreateInfo = {};
    vbAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    vbAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingVertexBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingVertexBufferAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingVertexBufferAllocInfo = {};
    ERR_GUARD_VULKAN( vmaCreateBuffer(g_hAllocator, &vbInfo, &vbAllocCreateInfo, &stagingVertexBuffer, &stagingVertexBufferAlloc, &stagingVertexBufferAllocInfo) );

    memcpy(stagingVertexBufferAllocInfo.pMappedData, vertices, vertexBufferSize);

    // No need to flush stagingVertexBuffer memory because CPU_ONLY memory is always HOST_COHERENT.

    vbInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vbAllocCreateInfo.flags = 0;
    ERR_GUARD_VULKAN( vmaCreateBuffer(g_hAllocator, &vbInfo, &vbAllocCreateInfo, &g_hVertexBuffer, &g_hVertexBufferAlloc, nullptr) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<std::uint64_t>(g_hVertexBuffer), "g_hVertexBuffer");

    // Create index buffer

    VkBufferCreateInfo ibInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ibInfo.size = indexBufferSize;
    ibInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ibInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ibAllocCreateInfo = {};
    ibAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    ibAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingIndexBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingIndexBufferAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingIndexBufferAllocInfo = {};
    ERR_GUARD_VULKAN( vmaCreateBuffer(g_hAllocator, &ibInfo, &ibAllocCreateInfo, &stagingIndexBuffer, &stagingIndexBufferAlloc, &stagingIndexBufferAllocInfo) );

    memcpy(stagingIndexBufferAllocInfo.pMappedData, indices, indexBufferSize);

    // No need to flush stagingIndexBuffer memory because CPU_ONLY memory is always HOST_COHERENT.

    ibInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    ibAllocCreateInfo.flags = 0;
    ERR_GUARD_VULKAN( vmaCreateBuffer(g_hAllocator, &ibInfo, &ibAllocCreateInfo, &g_hIndexBuffer, &g_hIndexBufferAlloc, nullptr) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<std::uint64_t>(g_hIndexBuffer), "g_hIndexBuffer");

    // Copy buffers

    BeginSingleTimeCommands();

    VkBufferCopy vbCopyRegion = {};
    vbCopyRegion.srcOffset = 0;
    vbCopyRegion.dstOffset = 0;
    vbCopyRegion.size = vbInfo.size;
    vkCmdCopyBuffer(g_hTemporaryCommandBuffer, stagingVertexBuffer, g_hVertexBuffer, 1, &vbCopyRegion);

    VkBufferCopy ibCopyRegion = {};
    ibCopyRegion.srcOffset = 0;
    ibCopyRegion.dstOffset = 0;
    ibCopyRegion.size = ibInfo.size;
    vkCmdCopyBuffer(g_hTemporaryCommandBuffer, stagingIndexBuffer, g_hIndexBuffer, 1, &ibCopyRegion);

    EndSingleTimeCommands();

    vmaDestroyBuffer(g_hAllocator, stagingIndexBuffer, stagingIndexBufferAlloc);
    vmaDestroyBuffer(g_hAllocator, stagingVertexBuffer, stagingVertexBufferAlloc);
}

static void CreateTexture(uint32_t sizeX, uint32_t sizeY)
{
    // Create staging buffer.

    const VkDeviceSize imageSize = sizeX * sizeY * 4;

    VkBufferCreateInfo stagingBufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingBufInfo.size = imageSize;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingBufAllocCreateInfo = {};
    stagingBufAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingBufAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VmaAllocation stagingBufAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingBufAllocInfo = {};
    ERR_GUARD_VULKAN( vmaCreateBuffer(g_hAllocator, &stagingBufInfo, &stagingBufAllocCreateInfo, &stagingBuf, &stagingBufAlloc, &stagingBufAllocInfo) );

    char* const pImageData = (char*)stagingBufAllocInfo.pMappedData;
    uint8_t* pRowData = (uint8_t*)pImageData;
    for(uint32_t y = 0; y < sizeY; ++y)
    {
        uint32_t* pPixelData = (uint32_t*)pRowData;
        for(uint32_t x = 0; x < sizeX; ++x)
        {
            *pPixelData =
                ((x & 0x18) == 0x08 ? 0x000000FF : 0x00000000) |
                ((x & 0x18) == 0x10 ? 0x0000FFFF : 0x00000000) |
                ((y & 0x18) == 0x08 ? 0x0000FF00 : 0x00000000) |
                ((y & 0x18) == 0x10 ? 0x00FF0000 : 0x00000000);
            ++pPixelData;
        }
        pRowData += sizeX * 4;
    }

    // No need to flush stagingImage memory because CPU_ONLY memory is always HOST_COHERENT.

    // Create g_hTextureImage in GPU memory.

    VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = sizeX;
    imageInfo.extent.height = sizeY;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0;

    VmaAllocationCreateInfo imageAllocCreateInfo = {};
    imageAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo textureImageAllocInfo = {};

    ERR_GUARD_VULKAN( vmaCreateImage(g_hAllocator, &imageInfo, &imageAllocCreateInfo, &g_hTextureImage, &g_hTextureImageAlloc, &textureImageAllocInfo) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<std::uint64_t>(g_hTextureImage), "g_hTextureImage");
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, reinterpret_cast<std::uint64_t>(textureImageAllocInfo.deviceMemory), "textureImageAllocInfo.deviceMemory");

    // Transition image layouts, copy image.

    BeginSingleTimeCommands();

    VkImageMemoryBarrier imgMemBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    imgMemBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgMemBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgMemBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imgMemBarrier.subresourceRange.baseMipLevel = 0;
    imgMemBarrier.subresourceRange.levelCount = 1;
    imgMemBarrier.subresourceRange.baseArrayLayer = 0;
    imgMemBarrier.subresourceRange.layerCount = 1;
    imgMemBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgMemBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgMemBarrier.image = g_hTextureImage;
    imgMemBarrier.srcAccessMask = 0;
    imgMemBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        g_hTemporaryCommandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &imgMemBarrier);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = sizeX;
    region.imageExtent.height = sizeY;
    region.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(g_hTemporaryCommandBuffer, stagingBuf, g_hTextureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    imgMemBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgMemBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imgMemBarrier.image = g_hTextureImage;
    imgMemBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgMemBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        g_hTemporaryCommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &imgMemBarrier);

    EndSingleTimeCommands();

    vmaDestroyBuffer(g_hAllocator, stagingBuf, stagingBufAlloc);

    // Create ImageView

    VkImageViewCreateInfo textureImageViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    textureImageViewInfo.image = g_hTextureImage;
    textureImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    textureImageViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    textureImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    textureImageViewInfo.subresourceRange.baseMipLevel = 0;
    textureImageViewInfo.subresourceRange.levelCount = 1;
    textureImageViewInfo.subresourceRange.baseArrayLayer = 0;
    textureImageViewInfo.subresourceRange.layerCount = 1;
    ERR_GUARD_VULKAN( vkCreateImageView(g_hDevice, &textureImageViewInfo, g_Allocs, &g_hTextureImageView) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<std::uint64_t>(g_hTextureImageView), "g_hTextureImageView");
}

struct UniformBufferObject
{
    mat4 ModelViewProj;
};

static VkFormat FindSupportedFormat(
    const std::vector<VkFormat>& candidates,
    VkImageTiling tiling,
    VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(g_hPhysicalDevice, format, &props);

        if ((tiling == VK_IMAGE_TILING_LINEAR) &&
            ((props.linearTilingFeatures & features) == features))
        {
            return format;
        }
        else if ((tiling == VK_IMAGE_TILING_OPTIMAL) &&
            ((props.optimalTilingFeatures & features) == features))
        {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

static VkFormat FindDepthFormat()
{
    std::vector<VkFormat> formats;
    formats.push_back(VK_FORMAT_D32_SFLOAT);
    formats.push_back(VK_FORMAT_D32_SFLOAT_S8_UINT);
    formats.push_back(VK_FORMAT_D24_UNORM_S8_UINT);

    return FindSupportedFormat(
        formats,
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

static void CreateSwapchain()
{
    // Query surface formats.

    ERR_GUARD_VULKAN( vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_hPhysicalDevice, g_hSurface, &g_SurfaceCapabilities) );

    uint32_t formatCount = 0;
    ERR_GUARD_VULKAN( vkGetPhysicalDeviceSurfaceFormatsKHR(g_hPhysicalDevice, g_hSurface, &formatCount, nullptr) );
    g_SurfaceFormats.resize(formatCount);
    ERR_GUARD_VULKAN( vkGetPhysicalDeviceSurfaceFormatsKHR(g_hPhysicalDevice, g_hSurface, &formatCount, g_SurfaceFormats.data()) );

    uint32_t presentModeCount = 0;
    ERR_GUARD_VULKAN( vkGetPhysicalDeviceSurfacePresentModesKHR(g_hPhysicalDevice, g_hSurface, &presentModeCount, nullptr) );
    g_PresentModes.resize(presentModeCount);
    ERR_GUARD_VULKAN( vkGetPhysicalDeviceSurfacePresentModesKHR(g_hPhysicalDevice, g_hSurface, &presentModeCount, g_PresentModes.data()) );

    // Create swap chain

    g_SurfaceFormat = ChooseSurfaceFormat();
    VkPresentModeKHR presentMode = ChooseSwapPresentMode();
    g_Extent = ChooseSwapExtent();

    g_SwapchainImageCount = g_SurfaceCapabilities.minImageCount + 1;
    if((g_SurfaceCapabilities.maxImageCount > 0) &&
        (g_SwapchainImageCount > g_SurfaceCapabilities.maxImageCount))
    {
        g_SwapchainImageCount = g_SurfaceCapabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapChainInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swapChainInfo.surface = g_hSurface;
    swapChainInfo.minImageCount = g_SwapchainImageCount;
    swapChainInfo.imageFormat = g_SurfaceFormat.format;
    swapChainInfo.imageColorSpace = g_SurfaceFormat.colorSpace;
    swapChainInfo.imageExtent = g_Extent;
    swapChainInfo.imageArrayLayers = 1;
    swapChainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapChainInfo.preTransform = g_SurfaceCapabilities.currentTransform;
    swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainInfo.presentMode = presentMode;
    swapChainInfo.clipped = VK_TRUE;
    swapChainInfo.oldSwapchain = g_hSwapchain;

    uint32_t queueFamilyIndices[] = { g_GraphicsQueueFamilyIndex, g_PresentQueueFamilyIndex };
    if(g_PresentQueueFamilyIndex != g_GraphicsQueueFamilyIndex)
    {
        swapChainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapChainInfo.queueFamilyIndexCount = 2;
        swapChainInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VkSwapchainKHR hNewSwapchain = VK_NULL_HANDLE;
    ERR_GUARD_VULKAN( vkCreateSwapchainKHR(g_hDevice, &swapChainInfo, g_Allocs, &hNewSwapchain) );
    if(g_hSwapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(g_hDevice, g_hSwapchain, g_Allocs);
    g_hSwapchain = hNewSwapchain;

    SetDebugUtilsObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, reinterpret_cast<std::uint64_t>(g_hSwapchain), "g_hSwapchain");

    // Retrieve swapchain images.

    uint32_t swapchainImageCount = 0;
    ERR_GUARD_VULKAN( vkGetSwapchainImagesKHR(g_hDevice, g_hSwapchain, &swapchainImageCount, nullptr) );
    g_SwapchainImages.resize(swapchainImageCount);
    ERR_GUARD_VULKAN( vkGetSwapchainImagesKHR(g_hDevice, g_hSwapchain, &swapchainImageCount, g_SwapchainImages.data()) );

    for (size_t i = 0; i < swapchainImageCount; i++) {
        std::string swapchainImgName = "g_SwapchainImages[" + std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<std::uint64_t>(g_SwapchainImages[i]), swapchainImgName);
    }

    // Create swapchain image views.

    for(size_t i = g_SwapchainImageViews.size(); i--; )
        vkDestroyImageView(g_hDevice, g_SwapchainImageViews[i], g_Allocs);
    g_SwapchainImageViews.clear();

    VkImageViewCreateInfo swapchainImageViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    g_SwapchainImageViews.resize(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; ++i)
    {
        swapchainImageViewInfo.image = g_SwapchainImages[i];
        swapchainImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        swapchainImageViewInfo.format = g_SurfaceFormat.format;
        swapchainImageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        swapchainImageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        swapchainImageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        swapchainImageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        swapchainImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        swapchainImageViewInfo.subresourceRange.baseMipLevel = 0;
        swapchainImageViewInfo.subresourceRange.levelCount = 1;
        swapchainImageViewInfo.subresourceRange.baseArrayLayer = 0;
        swapchainImageViewInfo.subresourceRange.layerCount = 1;
        ERR_GUARD_VULKAN( vkCreateImageView(g_hDevice, &swapchainImageViewInfo, g_Allocs, &g_SwapchainImageViews[i]) );
        std::string imgViewName = "g_SwapchainImageViews["+ std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<std::uint64_t>(g_SwapchainImageViews[i]), imgViewName);
    }

    // Create depth buffer

    g_DepthFormat = FindDepthFormat();
    assert(g_DepthFormat != VK_FORMAT_UNDEFINED);

    VkImageCreateInfo depthImageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageInfo.extent.width = g_Extent.width;
    depthImageInfo.extent.height = g_Extent.height;
    depthImageInfo.extent.depth = 1;
    depthImageInfo.mipLevels = 1;
    depthImageInfo.arrayLayers = 1;
    depthImageInfo.format = g_DepthFormat;
    depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImageInfo.flags = 0;

    VmaAllocationCreateInfo depthImageAllocCreateInfo = {};
    depthImageAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo depthImageAllocInfo = {};

    ERR_GUARD_VULKAN( vmaCreateImage(g_hAllocator, &depthImageInfo, &depthImageAllocCreateInfo, &g_hDepthImage, &g_hDepthImageAlloc, &depthImageAllocInfo) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<std::uint64_t>(g_hDepthImage), "g_hDepthImage");
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, reinterpret_cast<std::uint64_t>(depthImageAllocInfo.deviceMemory), "depthImageAllocInfo.deviceMemory");

    VkImageViewCreateInfo depthImageViewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    depthImageViewInfo.image = g_hDepthImage;
    depthImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthImageViewInfo.format = g_DepthFormat;
    depthImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthImageViewInfo.subresourceRange.baseMipLevel = 0;
    depthImageViewInfo.subresourceRange.levelCount = 1;
    depthImageViewInfo.subresourceRange.baseArrayLayer = 0;
    depthImageViewInfo.subresourceRange.layerCount = 1;

    ERR_GUARD_VULKAN( vkCreateImageView(g_hDevice, &depthImageViewInfo, g_Allocs, &g_hDepthImageView) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<std::uint64_t>(g_hDepthImageView), "g_hDepthImageView");

    // Create pipeline layout
    {
        if(g_hPipelineLayout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(g_hDevice, g_hPipelineLayout, g_Allocs);
            g_hPipelineLayout = VK_NULL_HANDLE;
        }

        VkPushConstantRange pushConstantRanges[1];
        ZeroMemory(&pushConstantRanges, sizeof pushConstantRanges);
        pushConstantRanges[0].offset = 0;
        pushConstantRanges[0].size = sizeof(UniformBufferObject);
        pushConstantRanges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayout descriptorSetLayouts[] = { g_hDescriptorSetLayout };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges;
        ERR_GUARD_VULKAN( vkCreatePipelineLayout(g_hDevice, &pipelineLayoutInfo, g_Allocs, &g_hPipelineLayout) );
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, reinterpret_cast<std::uint64_t>(g_hPipelineLayout), "g_hPipelineLayout");
    }

    // Create render pass
    {
        if(g_hRenderPass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(g_hDevice, g_hRenderPass, g_Allocs);
            g_hRenderPass = VK_NULL_HANDLE;
        }

        VkAttachmentDescription attachments[2];
        ZeroMemory(attachments, sizeof(attachments));

        attachments[0].format = g_SurfaceFormat.format;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        attachments[1].format = g_DepthFormat;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthStencilAttachmentRef = {};
        depthStencilAttachmentRef.attachment = 1;
        depthStencilAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDesc = {};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorAttachmentRef;
        subpassDesc.pDepthStencilAttachment = &depthStencilAttachmentRef;

        VkSubpassDependency dependencies[1] = {};
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = 0;

        VkRenderPassCreateInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        renderPassInfo.attachmentCount = (uint32_t)_countof(attachments);
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDesc;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = dependencies;
        ERR_GUARD_VULKAN( vkCreateRenderPass(g_hDevice, &renderPassInfo, g_Allocs, &g_hRenderPass) );
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_RENDER_PASS, reinterpret_cast<std::uint64_t>(g_hRenderPass), "g_hRenderPass");
    }

    // Create pipeline
    {
        std::vector<char> vertShaderCode;
        LoadShader(vertShaderCode, "Shader.vert.spv");
        VkShaderModuleCreateInfo shaderModuleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        shaderModuleInfo.codeSize = vertShaderCode.size();
        shaderModuleInfo.pCode = (const uint32_t*)vertShaderCode.data();
        VkShaderModule hVertShaderModule = VK_NULL_HANDLE;
        ERR_GUARD_VULKAN( vkCreateShaderModule(g_hDevice, &shaderModuleInfo, g_Allocs, &hVertShaderModule) );
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_SHADER_MODULE, reinterpret_cast<std::uint64_t>(hVertShaderModule), "hVertShaderModule");

        std::vector<char> hFragShaderCode;
        LoadShader(hFragShaderCode, "Shader.frag.spv");
        shaderModuleInfo.codeSize = hFragShaderCode.size();
        shaderModuleInfo.pCode = (const uint32_t*)hFragShaderCode.data();
        VkShaderModule fragShaderModule = VK_NULL_HANDLE;
        ERR_GUARD_VULKAN( vkCreateShaderModule(g_hDevice, &shaderModuleInfo, g_Allocs, &fragShaderModule) );
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_SHADER_MODULE, reinterpret_cast<std::uint64_t>(fragShaderModule), "fragShaderModule");

        VkPipelineShaderStageCreateInfo vertPipelineShaderStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        vertPipelineShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertPipelineShaderStageInfo.module = hVertShaderModule;
        vertPipelineShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragPipelineShaderStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        fragPipelineShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragPipelineShaderStageInfo.module = fragShaderModule;
        fragPipelineShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo pipelineShaderStageInfos[] = {
            vertPipelineShaderStageInfo,
            fragPipelineShaderStageInfo
        };

        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributeDescriptions[3];
        ZeroMemory(attributeDescriptions, sizeof(attributeDescriptions));

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        pipelineVertexInputStateInfo.vertexBindingDescriptionCount = 1;
        pipelineVertexInputStateInfo.pVertexBindingDescriptions = &bindingDescription;
        pipelineVertexInputStateInfo.vertexAttributeDescriptionCount = _countof(attributeDescriptions);
        pipelineVertexInputStateInfo.pVertexAttributeDescriptions = attributeDescriptions;

        VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        pipelineInputAssemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        pipelineInputAssemblyStateInfo.primitiveRestartEnable = VK_TRUE;

        VkViewport viewport = {};
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width = (float)g_Extent.width;
        viewport.height = (float)g_Extent.height;
        viewport.minDepth = 0.f;
        viewport.maxDepth = 1.f;

        VkRect2D scissor = {};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent = g_Extent;

        VkPipelineViewportStateCreateInfo pipelineViewportStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        pipelineViewportStateInfo.viewportCount = 1;
        pipelineViewportStateInfo.pViewports = &viewport;
        pipelineViewportStateInfo.scissorCount = 1;
        pipelineViewportStateInfo.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        pipelineRasterizationStateInfo.depthClampEnable = VK_FALSE;
        pipelineRasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
        pipelineRasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
        pipelineRasterizationStateInfo.lineWidth = 1.f;
        pipelineRasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        pipelineRasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        pipelineRasterizationStateInfo.depthBiasEnable = VK_FALSE;
        pipelineRasterizationStateInfo.depthBiasConstantFactor = 0.f;
        pipelineRasterizationStateInfo.depthBiasClamp = 0.f;
        pipelineRasterizationStateInfo.depthBiasSlopeFactor = 0.f;

        VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        pipelineMultisampleStateInfo.sampleShadingEnable = VK_FALSE;
        pipelineMultisampleStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        pipelineMultisampleStateInfo.minSampleShading = 1.f;
        pipelineMultisampleStateInfo.pSampleMask = nullptr;
        pipelineMultisampleStateInfo.alphaToCoverageEnable = VK_FALSE;
        pipelineMultisampleStateInfo.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState = {};
        pipelineColorBlendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;
        pipelineColorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        pipelineColorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        pipelineColorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        pipelineColorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        pipelineColorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        pipelineColorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        pipelineColorBlendStateInfo.logicOpEnable = VK_FALSE;
        pipelineColorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY;
        pipelineColorBlendStateInfo.attachmentCount = 1;
        pipelineColorBlendStateInfo.pAttachments = &pipelineColorBlendAttachmentState;

        VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencilStateInfo.depthTestEnable = VK_TRUE;
        depthStencilStateInfo.depthWriteEnable = VK_TRUE;
        depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
        depthStencilStateInfo.stencilTestEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = pipelineShaderStageInfos;
        pipelineInfo.pVertexInputState = &pipelineVertexInputStateInfo;
        pipelineInfo.pInputAssemblyState = &pipelineInputAssemblyStateInfo;
        pipelineInfo.pViewportState = &pipelineViewportStateInfo;
        pipelineInfo.pRasterizationState = &pipelineRasterizationStateInfo;
        pipelineInfo.pMultisampleState = &pipelineMultisampleStateInfo;
        pipelineInfo.pDepthStencilState = &depthStencilStateInfo;
        pipelineInfo.pColorBlendState = &pipelineColorBlendStateInfo;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = g_hPipelineLayout;
        pipelineInfo.renderPass = g_hRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;
        ERR_GUARD_VULKAN( vkCreateGraphicsPipelines(
            g_hDevice,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            g_Allocs,
            &g_hPipeline) );

        SetDebugUtilsObjectName(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(g_hPipeline), "g_hPipeline");

        vkDestroyShaderModule(g_hDevice, fragShaderModule, g_Allocs);
        vkDestroyShaderModule(g_hDevice, hVertShaderModule, g_Allocs);
    }

    // Create frambuffers

    for(size_t i = g_Framebuffers.size(); i--; )
        vkDestroyFramebuffer(g_hDevice, g_Framebuffers[i], g_Allocs);
    g_Framebuffers.clear();

    g_Framebuffers.resize(g_SwapchainImageViews.size());
    for(size_t i = 0; i < g_SwapchainImages.size(); ++i)
    {
        VkImageView attachments[] = { g_SwapchainImageViews[i], g_hDepthImageView };

        VkFramebufferCreateInfo framebufferInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        framebufferInfo.renderPass = g_hRenderPass;
        framebufferInfo.attachmentCount = (uint32_t)_countof(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = g_Extent.width;
        framebufferInfo.height = g_Extent.height;
        framebufferInfo.layers = 1;
        ERR_GUARD_VULKAN( vkCreateFramebuffer(g_hDevice, &framebufferInfo, g_Allocs, &g_Framebuffers[i]) );
        std::string framebufName = "g_Framebuffers["+ std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_FRAMEBUFFER, reinterpret_cast<std::uint64_t>(g_Framebuffers[i]), framebufName);
    }

    // Destroy the old semaphores and create new ones

    if (g_hRenderFinishedSemaphores.size() < g_SwapchainImageCount) {
        g_hRenderFinishedSemaphores.resize(g_SwapchainImageCount);
    }

    VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    for (std::size_t i = COMMAND_BUFFER_COUNT; i--; )
    {
        if (g_hImageAvailableSemaphores[i] != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(g_hDevice, g_hImageAvailableSemaphores[i], g_Allocs);
            g_hImageAvailableSemaphores[i] = VK_NULL_HANDLE;
        }
    }
    for (std::size_t swapchain_img_index = 0; swapchain_img_index < g_SwapchainImageCount; swapchain_img_index++) {
        if (g_hRenderFinishedSemaphores.at(swapchain_img_index) != VK_NULL_HANDLE) {
            vkDestroySemaphore(g_hDevice, g_hRenderFinishedSemaphores[swapchain_img_index], g_Allocs);
            g_hRenderFinishedSemaphores[swapchain_img_index] = VK_NULL_HANDLE;
        }
    }

    for (std::size_t i = 0; i < COMMAND_BUFFER_COUNT; ++i)
    {
        ERR_GUARD_VULKAN(vkCreateSemaphore(g_hDevice, &semaphoreInfo, g_Allocs, &g_hImageAvailableSemaphores[i]));
        std::string semaphoreName = "g_hImageAvailableSemaphores[" + std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_SEMAPHORE,
            reinterpret_cast<std::uint64_t>(g_hImageAvailableSemaphores[i]), semaphoreName);
    }

    for (std::size_t swapchain_img_index = 0; swapchain_img_index < g_SwapchainImageCount; swapchain_img_index++)
    {
        ERR_GUARD_VULKAN(vkCreateSemaphore(g_hDevice, &semaphoreInfo, g_Allocs, &g_hRenderFinishedSemaphores[swapchain_img_index]));
        std::string semaphoreName = "g_hRenderFinishedSemaphores[" + std::to_string(swapchain_img_index) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_SEMAPHORE,
            reinterpret_cast<std::uint64_t>(g_hRenderFinishedSemaphores[swapchain_img_index]), semaphoreName);
    }
}

static void DestroySwapchain(bool destroyActualSwapchain)
{
    for (std::size_t i = 0; i < COMMAND_BUFFER_COUNT; i++)
    {
        if (g_hImageAvailableSemaphores[i] != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(g_hDevice, g_hImageAvailableSemaphores[i], g_Allocs);
            g_hImageAvailableSemaphores[i] = VK_NULL_HANDLE;
        }
    }
    for (std::size_t swapchain_img_index = 0; swapchain_img_index < g_SwapchainImageCount; swapchain_img_index++)
    {
        if (g_hRenderFinishedSemaphores.at(swapchain_img_index) != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(g_hDevice, g_hRenderFinishedSemaphores[swapchain_img_index], g_Allocs);
            g_hRenderFinishedSemaphores[swapchain_img_index] = VK_NULL_HANDLE;
        }
    }

    for(size_t i = g_Framebuffers.size(); i--; )
        vkDestroyFramebuffer(g_hDevice, g_Framebuffers[i], g_Allocs);
    g_Framebuffers.clear();

    if(g_hDepthImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(g_hDevice, g_hDepthImageView, g_Allocs);
        g_hDepthImageView = VK_NULL_HANDLE;
    }
    if(g_hDepthImage != VK_NULL_HANDLE)
    {
        vmaDestroyImage(g_hAllocator, g_hDepthImage, g_hDepthImageAlloc);
        g_hDepthImage = VK_NULL_HANDLE;
    }

    if(g_hPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(g_hDevice, g_hPipeline, g_Allocs);
        g_hPipeline = VK_NULL_HANDLE;
    }

    if(g_hRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(g_hDevice, g_hRenderPass, g_Allocs);
        g_hRenderPass = VK_NULL_HANDLE;
    }

    if(g_hPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(g_hDevice, g_hPipelineLayout, g_Allocs);
        g_hPipelineLayout = VK_NULL_HANDLE;
    }

    for(size_t i = g_SwapchainImageViews.size(); i--; )
        vkDestroyImageView(g_hDevice, g_SwapchainImageViews[i], g_Allocs);
    g_SwapchainImageViews.clear();

    if(destroyActualSwapchain && (g_hSwapchain != VK_NULL_HANDLE))
    {
        vkDestroySwapchainKHR(g_hDevice, g_hSwapchain, g_Allocs);
        g_hSwapchain = VK_NULL_HANDLE;
    }
}

static void PrintEnabledFeatures()
{
    wprintf(L"Enabled extensions and features:\n");
    wprintf(L"Validation layer: %d\n", g_EnableValidationLayer ? 1 : 0);
    wprintf(L"Sparse binding: %d\n", g_SparseBindingEnabled ? 1 : 0);
    if(GetVulkanApiVersion() == VK_API_VERSION_1_0)
    {
        wprintf(L"VK_KHR_get_memory_requirements2: %d\n", VK_KHR_get_memory_requirements2_enabled ? 1 : 0);
        wprintf(L"VK_KHR_get_physical_device_properties2: %d\n", VK_KHR_get_physical_device_properties2_enabled ? 1 : 0);
        wprintf(L"VK_KHR_dedicated_allocation: %d\n", VK_KHR_dedicated_allocation_enabled ? 1 : 0);
        wprintf(L"VK_KHR_bind_memory2: %d\n", VK_KHR_bind_memory2_enabled ? 1 : 0);
    }
    wprintf(L"VK_EXT_memory_budget: %d\n", VK_EXT_memory_budget_enabled ? 1 : 0);
    wprintf(L"VK_AMD_device_coherent_memory: %d\n", VK_AMD_device_coherent_memory_enabled ? 1 : 0);
    if(GetVulkanApiVersion() < VK_API_VERSION_1_2)
    {
        wprintf(L"VK_KHR_buffer_device_address: %d\n", VK_KHR_buffer_device_address_enabled ? 1 : 0);
    }
    else
    {
        wprintf(L"bufferDeviceAddress: %d\n", VK_KHR_buffer_device_address_enabled ? 1 : 0);
    }
    wprintf(L"VK_EXT_memory_priority: %d\n", VK_EXT_memory_priority_enabled ? 1 : 0);
    wprintf(L"VK_KHR_maintenance5: %d\n", VK_KHR_maintenance5_enabled? 1 : 0);
    wprintf(L"VK_KHR_external_memory_win32: %d\n", VK_KHR_external_memory_win32_enabled ? 1 : 0);
}

void SetAllocatorCreateInfo(VmaAllocatorCreateInfo& outInfo)
{
    outInfo = {};

    outInfo.physicalDevice = g_hPhysicalDevice;
    outInfo.device = g_hDevice;
    outInfo.instance = g_hVulkanInstance;
    outInfo.vulkanApiVersion = GetVulkanApiVersion();

    if(VK_KHR_dedicated_allocation_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }
    if(VK_KHR_bind_memory2_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    }
#if !defined(VMA_MEMORY_BUDGET) || VMA_MEMORY_BUDGET == 1
    if(VK_EXT_memory_budget_enabled && (
        GetVulkanApiVersion() >= VK_API_VERSION_1_1 || VK_KHR_get_physical_device_properties2_enabled))
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
#endif
    if(VK_AMD_device_coherent_memory_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
    }
    if(VK_KHR_buffer_device_address_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
#if !defined(VMA_MEMORY_PRIORITY) || VMA_MEMORY_PRIORITY == 1
    if(VK_EXT_memory_priority_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
#endif
    if(VK_KHR_maintenance5_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    }

    if(VK_KHR_external_memory_win32_enabled)
    {
        outInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;
    }

    if(USE_CUSTOM_CPU_ALLOCATION_CALLBACKS)
    {
        outInfo.pAllocationCallbacks = &g_CpuAllocationCallbacks;
    }

#ifdef VOLK_HEADER_VERSION
    static VmaVulkanFunctions vulkanFunctions = {};
    vmaImportVulkanFunctionsFromVolk(&outInfo, &vulkanFunctions);
    outInfo.pVulkanFunctions = &vulkanFunctions;
#endif // #ifdef VOLK_HEADER_VERSION

#if VMA_DYNAMIC_VULKAN_FUNCTIONS
    static VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    outInfo.pVulkanFunctions = &vulkanFunctions;
#endif // #if VMA_DYNAMIC_VULKAN_FUNCTIONS

    // Uncomment to enable HeapSizeLimit.
    /*
    static std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heapSizeLimit;
    std::fill(heapSizeLimit.begin(), heapSizeLimit.end(), VK_WHOLE_SIZE);
    heapSizeLimit[0] = 512ull * 1024 * 1024;
    outInfo.pHeapSizeLimit = heapSizeLimit.data();
    */
}

static void PrintPhysicalDeviceProperties(const VkPhysicalDeviceProperties& properties)
{
    wprintf(L"physicalDeviceProperties:\n");
    wprintf(L"    driverVersion: 0x%X\n", properties.driverVersion);
    wprintf(L"    vendorID: 0x%X (%s)\n", properties.vendorID, VendorIDToStr(properties.vendorID));
    wprintf(L"    deviceID: 0x%X\n", properties.deviceID);
    wprintf(L"    deviceType: %u (%s)\n", properties.deviceType, PhysicalDeviceTypeToStr(properties.deviceType));
    wprintf(L"    deviceName: %hs\n", properties.deviceName);
    wprintf(L"    limits:\n");
    wprintf(L"        maxMemoryAllocationCount: %u\n", properties.limits.maxMemoryAllocationCount);
    wprintf(L"        bufferImageGranularity: %llu B\n", properties.limits.bufferImageGranularity);
    wprintf(L"        nonCoherentAtomSize: %llu B\n", properties.limits.nonCoherentAtomSize);
}

#if VMA_VULKAN_VERSION >= 1002000
static void PrintPhysicalDeviceVulkan11Properties(const VkPhysicalDeviceVulkan11Properties& properties)
{
    wprintf(L"physicalDeviceVulkan11Properties:\n");
    std::wstring sizeStr = SizeToStr(properties.maxMemoryAllocationSize);
    wprintf(L"    maxMemoryAllocationSize: %llu B (%s)\n", properties.maxMemoryAllocationSize, sizeStr.c_str());
}
static void PrintPhysicalDeviceVulkan12Properties(const VkPhysicalDeviceVulkan12Properties& properties)
{
    wprintf(L"physicalDeviceVulkan12Properties:\n");
    std::wstring str = DriverIDToStr(properties.driverID);
    wprintf(L"    driverID: %u (%s)\n", properties.driverID, str.c_str());
    wprintf(L"    driverName: %hs\n", properties.driverName);
    wprintf(L"    driverInfo: %hs\n", properties.driverInfo);
}
#endif // #if VMA_VULKAN_VERSION > 1002000

static void AddFlagToStr(std::wstring& inout, const wchar_t* flagStr)
{
    if(!inout.empty())
        inout += L", ";
    inout += flagStr;
}

static std::wstring HeapFlagsToStr(VkMemoryHeapFlags flags)
{
    std::wstring result;
    if(flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        AddFlagToStr(result, L"DEVICE_LOCAL");
    if(flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
        AddFlagToStr(result, L"MULTI_INSTANCE");
    return result;
}

static std::wstring PropertyFlagsToStr(VkMemoryPropertyFlags flags)
{
    std::wstring result;
    if(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        AddFlagToStr(result, L"DEVICE_LOCAL");
    if(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        AddFlagToStr(result, L"HOST_VISIBLE");
    if(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        AddFlagToStr(result, L"HOST_COHERENT");
    if(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        AddFlagToStr(result, L"HOST_CACHED");
    if(flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
        AddFlagToStr(result, L"LAZILY_ALLOCATED");

#if VMA_VULKAN_VERSION >= 1001000
    if(flags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
        AddFlagToStr(result, L"PROTECTED");
#endif

#if VK_AMD_device_coherent_memory
    if(flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD)
        AddFlagToStr(result, L"DEVICE_COHERENT (AMD)");
    if(flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)
        AddFlagToStr(result, L"DEVICE_UNCACHED (AMD)");
#endif

    return result;
}

static void PrintMemoryTypes()
{
    wprintf(L"MEMORY HEAPS:\n");
    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetMemoryProperties(g_hAllocator, &memProps);

    wprintf(L"heapCount=%u, typeCount=%u\n", memProps->memoryHeapCount, memProps->memoryTypeCount);

    std::wstring sizeStr, flagsStr;
    for(uint32_t heapIndex = 0; heapIndex < memProps->memoryHeapCount; ++heapIndex)
    {
        const VkMemoryHeap& heap = memProps->memoryHeaps[heapIndex];
        sizeStr = SizeToStr(heap.size);
        flagsStr = HeapFlagsToStr(heap.flags);
        wprintf(L"Heap %u: %llu B (%s) %s\n", heapIndex, heap.size, sizeStr.c_str(), flagsStr.c_str());

        for(uint32_t typeIndex = 0; typeIndex < memProps->memoryTypeCount; ++typeIndex)
        {
            const VkMemoryType& type = memProps->memoryTypes[typeIndex];
            if(type.heapIndex == heapIndex)
            {
                flagsStr = PropertyFlagsToStr(type.propertyFlags);
                wprintf(L"    Type %u: %s\n", typeIndex, flagsStr.c_str());
            }
        }
    }
}

static bool CanCreateVertexBuffer(uint32_t allowedMemoryTypeBits)
{
    VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufCreateInfo.size = 0x10000;
    bufCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkBuffer buf = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(g_hDevice, &bufCreateInfo, g_Allocs, &buf);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements memReq = {};
    vkGetBufferMemoryRequirements(g_hDevice, buf, &memReq);

    vkDestroyBuffer(g_hDevice, buf, g_Allocs);

    return (memReq.memoryTypeBits & allowedMemoryTypeBits) != 0;
}

static bool CanCreateOptimalSampledImage(uint32_t allowedMemoryTypeBits)
{
    VkImageCreateInfo imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imgCreateInfo.extent.width = 256;
    imgCreateInfo.extent.height = 256;
    imgCreateInfo.extent.depth = 1;
    imgCreateInfo.mipLevels = 1;
    imgCreateInfo.arrayLayers = 1;
    imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    imgCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkImage img = VK_NULL_HANDLE;
    VkResult res = vkCreateImage(g_hDevice, &imgCreateInfo, g_Allocs, &img);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements memReq = {};
    vkGetImageMemoryRequirements(g_hDevice, img, &memReq);

    vkDestroyImage(g_hDevice, img, g_Allocs);

    return (memReq.memoryTypeBits & allowedMemoryTypeBits) != 0;
}

static void PrintMemoryConclusions()
{
    wprintf(L"Conclusions:\n");

    const VkPhysicalDeviceProperties* props = nullptr;
    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetPhysicalDeviceProperties(g_hAllocator, &props);
    vmaGetMemoryProperties(g_hAllocator, &memProps);

    const uint32_t heapCount = memProps->memoryHeapCount;

    uint32_t deviceLocalHeapCount = 0;
    uint32_t hostVisibleHeapCount = 0;
    uint32_t deviceLocalAndHostVisibleHeapCount = 0;
    VkDeviceSize deviceLocalHeapSumSize = 0;
    VkDeviceSize hostVisibleHeapSumSize = 0;
    VkDeviceSize deviceLocalAndHostVisibleHeapSumSize = 0;

    for(uint32_t heapIndex = 0; heapIndex < heapCount; ++heapIndex)
    {
        const VkMemoryHeap& heap = memProps->memoryHeaps[heapIndex];
        const bool isDeviceLocal = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        bool isHostVisible = false;
        for(uint32_t typeIndex = 0; typeIndex < memProps->memoryTypeCount; ++typeIndex)
        {
            const VkMemoryType& type = memProps->memoryTypes[typeIndex];
            if(type.heapIndex == heapIndex && (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            {
                isHostVisible = true;
                break;
            }
        }
        if(isDeviceLocal)
        {
            ++deviceLocalHeapCount;
            deviceLocalHeapSumSize += heap.size;
        }
        if(isHostVisible)
        {
            ++hostVisibleHeapCount;
            hostVisibleHeapSumSize += heap.size;
            if(isDeviceLocal)
            {
                ++deviceLocalAndHostVisibleHeapCount;
                deviceLocalAndHostVisibleHeapSumSize += heap.size;
            }
        }
    }

    uint32_t hostVisibleNotHostCoherentTypeCount = 0;
    uint32_t notDeviceLocalNotHostVisibleTypeCount = 0;
    uint32_t amdSpecificTypeCount = 0;
    uint32_t lazilyAllocatedTypeCount = 0;
    uint32_t allTypeBits = 0;
    uint32_t deviceLocalTypeBits = 0;
    for(uint32_t typeIndex = 0; typeIndex < memProps->memoryTypeCount; ++typeIndex)
    {
        const VkMemoryType& type = memProps->memoryTypes[typeIndex];
        allTypeBits |= 1u << typeIndex;
        if(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            deviceLocalTypeBits |= 1u << typeIndex;
        }
        if((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
        {
            ++hostVisibleNotHostCoherentTypeCount;
        }
        if((type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0 &&
            (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            ++notDeviceLocalNotHostVisibleTypeCount;
        }
        if(type.propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD))
        {
            ++amdSpecificTypeCount;
        }
        if(type.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
        {
            ++lazilyAllocatedTypeCount;
        }
    }

    assert(deviceLocalHeapCount > 0);
    if(deviceLocalHeapCount == heapCount)
        wprintf(L"- All heaps are DEVICE_LOCAL.\n");
    else
        wprintf(L"- %u heaps are DEVICE_LOCAL, total %s.\n", deviceLocalHeapCount, SizeToStr(deviceLocalHeapSumSize).c_str());

    assert(hostVisibleHeapCount > 0);
    if(hostVisibleHeapCount == heapCount)
        wprintf(L"- All heaps are HOST_VISIBLE.\n");
    else
        wprintf(L"- %u heaps are HOST_VISIBLE, total %s.\n", deviceLocalHeapCount, SizeToStr(hostVisibleHeapSumSize).c_str());

    if(deviceLocalHeapCount < heapCount && hostVisibleHeapCount < heapCount)
    {
        if(deviceLocalAndHostVisibleHeapCount == 0)
            wprintf(L"- No heaps are DEVICE_LOCAL and HOST_VISIBLE.\n");
        if(deviceLocalAndHostVisibleHeapCount == heapCount)
            wprintf(L"- All heaps are DEVICE_LOCAL and HOST_VISIBLE.\n");
        else
            wprintf(L"- %u heaps are DEVICE_LOCAL and HOST_VISIBLE, total %s.\n", deviceLocalAndHostVisibleHeapCount, SizeToStr(deviceLocalAndHostVisibleHeapSumSize).c_str());
    }

    if(hostVisibleNotHostCoherentTypeCount == 0)
        wprintf(L"- No types are HOST_VISIBLE but not HOST_COHERENT.\n");
    else
        wprintf(L"- %u types are HOST_VISIBLE but not HOST_COHERENT.\n", hostVisibleNotHostCoherentTypeCount);

    if(notDeviceLocalNotHostVisibleTypeCount == 0)
        wprintf(L"- No types are not DEVICE_LOCAL and not HOST_VISIBLE.\n");
    else
        wprintf(L"- %u types are not DEVICE_LOCAL and not HOST_VISIBLE.\n", notDeviceLocalNotHostVisibleTypeCount);

    if(amdSpecificTypeCount == 0)
        wprintf(L"- No types are AMD-specific DEVICE_COHERENT or DEVICE_UNCACHED.\n");
    else
        wprintf(L"- %u types are AMD-specific DEVICE_COHERENT or DEVICE_UNCACHED.\n", amdSpecificTypeCount);

    if(lazilyAllocatedTypeCount == 0)
        wprintf(L"- No types are LAZILY_ALLOCATED.\n");
    else
        wprintf(L"- %u types are LAZILY_ALLOCATED.\n", lazilyAllocatedTypeCount);

    if(props->vendorID == VENDOR_ID_AMD &&
        props->deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        deviceLocalAndHostVisibleHeapSumSize > 256llu * 1024 * 1024)
    {
        wprintf(L"- AMD Smart Access Memory (SAM) is enabled!\n");
    }

    if(deviceLocalHeapCount < heapCount)
    {
        const uint32_t nonDeviceLocalTypeBits = ~deviceLocalTypeBits & allTypeBits;

        if(CanCreateVertexBuffer(nonDeviceLocalTypeBits))
            wprintf(L"- A buffer with VERTEX_BUFFER usage can be created in some non-DEVICE_LOCAL type.\n");
        else
            wprintf(L"- A buffer with VERTEX_BUFFER usage cannot be created in some non-DEVICE_LOCAL type.\n");

        if(CanCreateOptimalSampledImage(nonDeviceLocalTypeBits))
            wprintf(L"- An image with OPTIMAL tiling and SAMPLED usage can be created in some non-DEVICE_LOCAL type.\n");
        else
            wprintf(L"- An image with OPTIMAL tiling and SAMPLED usage cannot be created in some non-DEVICE_LOCAL type.\n");
    }

    //wprintf(L"\n");
}

static void InitializeApplication()
{
    // Create VkSurfaceKHR.
    VkWin32SurfaceCreateInfoKHR surfaceInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    surfaceInfo.hinstance = g_hAppInstance;
    surfaceInfo.hwnd = g_hWnd;
    VkResult result = vkCreateWin32SurfaceKHR(g_hVulkanInstance, &surfaceInfo, g_Allocs, &g_hSurface);
    assert(result == VK_SUCCESS);

    // Query for device extensions

    uint32_t physicalDeviceExtensionPropertyCount = 0;
    ERR_GUARD_VULKAN( vkEnumerateDeviceExtensionProperties(g_hPhysicalDevice, nullptr, &physicalDeviceExtensionPropertyCount, nullptr) );
    std::vector<VkExtensionProperties> physicalDeviceExtensionProperties{physicalDeviceExtensionPropertyCount};
    if(physicalDeviceExtensionPropertyCount)
    {
        ERR_GUARD_VULKAN( vkEnumerateDeviceExtensionProperties(
            g_hPhysicalDevice,
            nullptr,
            &physicalDeviceExtensionPropertyCount,
            physicalDeviceExtensionProperties.data()) );
    }

    for(uint32_t i = 0; i < physicalDeviceExtensionPropertyCount; ++i)
    {
        if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) == 0)
        {
            if(GetVulkanApiVersion() == VK_API_VERSION_1_0)
            {
                VK_KHR_get_memory_requirements2_enabled = true;
            }
        }
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) == 0)
        {
            if(GetVulkanApiVersion() == VK_API_VERSION_1_0)
            {
                VK_KHR_dedicated_allocation_enabled = true;
            }
        }
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) == 0)
        {
            if(GetVulkanApiVersion() == VK_API_VERSION_1_0)
            {
                VK_KHR_bind_memory2_enabled = true;
            }
        }
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0)
            VK_EXT_memory_budget_enabled = true;
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) == 0)
            VK_AMD_device_coherent_memory_enabled = true;
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0)
        {
            if(GetVulkanApiVersion() < VK_API_VERSION_1_2)
            {
                VK_KHR_buffer_device_address_enabled = true;
            }
        }
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) == 0)
            VK_EXT_memory_priority_enabled = true;
        else if(strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0)
            VK_KHR_maintenance5_enabled = true;
        else if (strcmp(physicalDeviceExtensionProperties[i].extensionName, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0)
            VK_KHR_external_memory_win32_enabled = VMA_DYNAMIC_VULKAN_FUNCTIONS;
    }

    if(GetVulkanApiVersion() >= VK_API_VERSION_1_2)
        VK_KHR_buffer_device_address_enabled = true; // Promoted to core Vulkan 1.2.

    // Query for features

#if VMA_VULKAN_VERSION >= 1001000
    VkPhysicalDeviceProperties2 physicalDeviceProperties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };

#if VMA_VULKAN_VERSION >= 1002000
    // Vulkan spec says structure VkPhysicalDeviceVulkan11Properties is "Provided by VK_VERSION_1_2" - is this a mistake? Assuming not...
    VkPhysicalDeviceVulkan11Properties physicalDeviceVulkan11Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
    VkPhysicalDeviceVulkan12Properties physicalDeviceVulkan12Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
    PnextChainPushFront(&physicalDeviceProperties2, &physicalDeviceVulkan11Properties);
    PnextChainPushFront(&physicalDeviceProperties2, &physicalDeviceVulkan12Properties);
#endif

    vkGetPhysicalDeviceProperties2(g_hPhysicalDevice, &physicalDeviceProperties2);

    PrintPhysicalDeviceProperties(physicalDeviceProperties2.properties);
#if VMA_VULKAN_VERSION >= 1002000
    PrintPhysicalDeviceVulkan11Properties(physicalDeviceVulkan11Properties);
    PrintPhysicalDeviceVulkan12Properties(physicalDeviceVulkan12Properties);
#endif

#else // #if VMA_VULKAN_VERSION >= 1001000
    VkPhysicalDeviceProperties physicalDeviceProperties = {};
    vkGetPhysicalDeviceProperties(g_hPhysicalDevice, &physicalDeviceProperties);
    PrintPhysicalDeviceProperties(physicalDeviceProperties);

#endif // #if VMA_VULKAN_VERSION >= 1001000

    wprintf(L"\n");

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };

    VkPhysicalDeviceCoherentMemoryFeaturesAMD physicalDeviceCoherentMemoryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD };
    if(VK_AMD_device_coherent_memory_enabled)
    {
        PnextChainPushFront(&physicalDeviceFeatures, &physicalDeviceCoherentMemoryFeatures);
    }

    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR physicalDeviceBufferDeviceAddressFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR };
    if(VK_KHR_buffer_device_address_enabled)
    {
        PnextChainPushFront(&physicalDeviceFeatures, &physicalDeviceBufferDeviceAddressFeatures);
    }

    VkPhysicalDeviceMemoryPriorityFeaturesEXT physicalDeviceMemoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT };
    if(VK_EXT_memory_priority_enabled)
    {
        PnextChainPushFront(&physicalDeviceFeatures, &physicalDeviceMemoryPriorityFeatures);
    }

    vkGetPhysicalDeviceFeatures2(g_hPhysicalDevice, &physicalDeviceFeatures);

    g_SparseBindingEnabled = physicalDeviceFeatures.features.sparseBinding != 0;

    // The extension is supported as fake with no real support for this feature? Don't use it.
    if(VK_AMD_device_coherent_memory_enabled && !physicalDeviceCoherentMemoryFeatures.deviceCoherentMemory)
        VK_AMD_device_coherent_memory_enabled = false;
    if(VK_KHR_buffer_device_address_enabled && !physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress)
        VK_KHR_buffer_device_address_enabled = false;
    if(VK_EXT_memory_priority_enabled && !physicalDeviceMemoryPriorityFeatures.memoryPriority)
        VK_EXT_memory_priority_enabled = false;

    // Find queue family index

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_hPhysicalDevice, &queueFamilyCount, nullptr);
    assert(queueFamilyCount > 0);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(g_hPhysicalDevice, &queueFamilyCount, queueFamilies.data());
    for(uint32_t i = 0;
        (i < queueFamilyCount) &&
            (g_GraphicsQueueFamilyIndex == UINT_MAX ||
                g_PresentQueueFamilyIndex == UINT_MAX ||
                (g_SparseBindingEnabled && g_SparseBindingQueueFamilyIndex == UINT_MAX));
        ++i)
    {
        if(queueFamilies[i].queueCount > 0)
        {
            const uint32_t flagsForGraphicsQueue = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if((g_GraphicsQueueFamilyIndex != 0) &&
                ((queueFamilies[i].queueFlags & flagsForGraphicsQueue) == flagsForGraphicsQueue))
            {
                g_GraphicsQueueFamilyIndex = i;
            }

            VkBool32 surfaceSupported = 0;
            VkResult res = vkGetPhysicalDeviceSurfaceSupportKHR(g_hPhysicalDevice, i, g_hSurface, &surfaceSupported);
            if((res >= 0) && (surfaceSupported == VK_TRUE))
            {
                g_PresentQueueFamilyIndex = i;
            }

            if(g_SparseBindingEnabled &&
                g_SparseBindingQueueFamilyIndex == UINT32_MAX &&
                (queueFamilies[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0)
            {
                g_SparseBindingQueueFamilyIndex = i;
            }
        }
    }
    assert(g_GraphicsQueueFamilyIndex != UINT_MAX);

    g_SparseBindingEnabled = g_SparseBindingEnabled && g_SparseBindingQueueFamilyIndex != UINT32_MAX;

    // Create logical device

    const float queuePriority = 1.f;

    VkDeviceQueueCreateInfo queueCreateInfo[3] = {};
    uint32_t queueCount = 1;
    queueCreateInfo[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo[0].queueFamilyIndex = g_GraphicsQueueFamilyIndex;
    queueCreateInfo[0].queueCount = 1;
    queueCreateInfo[0].pQueuePriorities = &queuePriority;

    if(g_PresentQueueFamilyIndex != g_GraphicsQueueFamilyIndex)
    {

        queueCreateInfo[queueCount].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo[queueCount].queueFamilyIndex = g_PresentQueueFamilyIndex;
        queueCreateInfo[queueCount].queueCount = 1;
        queueCreateInfo[queueCount].pQueuePriorities = &queuePriority;
        ++queueCount;
    }

    if(g_SparseBindingEnabled &&
        g_SparseBindingQueueFamilyIndex != g_GraphicsQueueFamilyIndex &&
        g_SparseBindingQueueFamilyIndex != g_PresentQueueFamilyIndex)
    {

        queueCreateInfo[queueCount].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo[queueCount].queueFamilyIndex = g_SparseBindingQueueFamilyIndex;
        queueCreateInfo[queueCount].queueCount = 1;
        queueCreateInfo[queueCount].pQueuePriorities = &queuePriority;
        ++queueCount;
    }

    std::vector<const char*> enabledDeviceExtensions;
    enabledDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if(VK_KHR_get_memory_requirements2_enabled)
        enabledDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    if(VK_KHR_dedicated_allocation_enabled)
        enabledDeviceExtensions.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    if(VK_KHR_bind_memory2_enabled)
        enabledDeviceExtensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
    if(VK_EXT_memory_budget_enabled)
        enabledDeviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if(VK_AMD_device_coherent_memory_enabled)
        enabledDeviceExtensions.push_back(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    if(VK_KHR_buffer_device_address_enabled && GetVulkanApiVersion() < VK_API_VERSION_1_2)
        enabledDeviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if(VK_EXT_memory_priority_enabled)
        enabledDeviceExtensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    if(VK_KHR_maintenance5_enabled)
        enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    if (VK_KHR_external_memory_win32_enabled)
        enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);

    VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    deviceFeatures.features.samplerAnisotropy = VK_TRUE;
    deviceFeatures.features.sparseBinding = g_SparseBindingEnabled ? VK_TRUE : VK_FALSE;

    if(VK_AMD_device_coherent_memory_enabled)
    {
        physicalDeviceCoherentMemoryFeatures.deviceCoherentMemory = VK_TRUE;
        PnextChainPushBack(&deviceFeatures, &physicalDeviceCoherentMemoryFeatures);
    }
    if(VK_KHR_buffer_device_address_enabled)
    {
        physicalDeviceBufferDeviceAddressFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR };
        physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
        PnextChainPushBack(&deviceFeatures, &physicalDeviceBufferDeviceAddressFeatures);
    }
    if(VK_EXT_memory_priority_enabled)
    {
        PnextChainPushBack(&deviceFeatures, &physicalDeviceMemoryPriorityFeatures);
    }

    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.pNext = &deviceFeatures;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)enabledDeviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = !enabledDeviceExtensions.empty() ? enabledDeviceExtensions.data() : nullptr;
    deviceCreateInfo.queueCreateInfoCount = queueCount;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfo;

    ERR_GUARD_VULKAN( vkCreateDevice(g_hPhysicalDevice, &deviceCreateInfo, g_Allocs, &g_hDevice) );

#ifdef VOLK_HEADER_VERSION
    volkLoadDevice(g_hDevice);
#endif

    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DEVICE, reinterpret_cast<std::uint64_t>(g_hDevice), "g_hDevice");
    // Only now that SetDebugUtilsObjectName is loaded, we can assign a name to g_hVulkanInstance as well
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_INSTANCE, reinterpret_cast<std::uint64_t>(g_hVulkanInstance), "g_hVulkanInstance");
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_PHYSICAL_DEVICE, reinterpret_cast<std::uint64_t>(g_hPhysicalDevice), "g_hPhysicalDevice");

    // Fetch pointers to extension functions
    if(VK_KHR_buffer_device_address_enabled)
    {
        if(GetVulkanApiVersion() >= VK_API_VERSION_1_2)
        {
            g_vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(g_hDevice, "vkGetBufferDeviceAddress");
        }
        else if(VK_KHR_buffer_device_address_enabled)
        {
            g_vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(g_hDevice, "vkGetBufferDeviceAddressKHR");
        }
        assert(g_vkGetBufferDeviceAddressKHR != nullptr);
    }

    // Create memory allocator

    VmaAllocatorCreateInfo allocatorInfo = {};
    SetAllocatorCreateInfo(allocatorInfo);
    ERR_GUARD_VULKAN( vmaCreateAllocator(&allocatorInfo, &g_hAllocator) );

    PrintMemoryTypes();
    wprintf(L"\n");
    PrintMemoryConclusions();
    wprintf(L"\n");
    PrintEnabledFeatures();
    wprintf(L"\n");

    // Retrieve queues (don't need to be destroyed).

    vkGetDeviceQueue(g_hDevice, g_GraphicsQueueFamilyIndex, 0, &g_hGraphicsQueue);
    vkGetDeviceQueue(g_hDevice, g_PresentQueueFamilyIndex, 0, &g_hPresentQueue);
    assert(g_hGraphicsQueue);
    assert(g_hPresentQueue);
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<std::uint64_t>(g_hGraphicsQueue), "g_hGraphicsQueue");
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<std::uint64_t>(g_hPresentQueue), "g_hPresentQueue");

    if(g_SparseBindingEnabled)
    {
        vkGetDeviceQueue(g_hDevice, g_SparseBindingQueueFamilyIndex, 0, &g_hSparseBindingQueue);
        assert(g_hSparseBindingQueue);
    }

    // Create command pool

    VkCommandPoolCreateInfo commandPoolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    commandPoolInfo.queueFamilyIndex = g_GraphicsQueueFamilyIndex;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ERR_GUARD_VULKAN( vkCreateCommandPool(g_hDevice, &commandPoolInfo, g_Allocs, &g_hCommandPool) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_COMMAND_POOL, reinterpret_cast<std::uint64_t>(g_hCommandPool), "g_hCommandPool");

    VkCommandBufferAllocateInfo commandBufferInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    commandBufferInfo.commandPool = g_hCommandPool;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = COMMAND_BUFFER_COUNT;
    ERR_GUARD_VULKAN( vkAllocateCommandBuffers(g_hDevice, &commandBufferInfo, g_MainCommandBuffers) );
    for (size_t i = 0; i < COMMAND_BUFFER_COUNT; i++) {
        std::string cmdBufName = "g_MainCommandBuffers[" + std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, reinterpret_cast<std::uint64_t>(g_MainCommandBuffers[i]), cmdBufName);
    }

    VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for(size_t i = 0; i < COMMAND_BUFFER_COUNT; ++i)
    {
        ERR_GUARD_VULKAN( vkCreateFence(g_hDevice, &fenceInfo, g_Allocs, &g_MainCommandBufferExecutedFences[i]) );
        std::string fenceName = "g_MainCommandBufferExecutedFences[" + std::to_string(i) + "]";
        SetDebugUtilsObjectName(VK_OBJECT_TYPE_FENCE, reinterpret_cast<std::uint64_t>(g_MainCommandBufferExecutedFences[i]), fenceName);
    }

    ERR_GUARD_VULKAN( vkCreateFence(g_hDevice, &fenceInfo, g_Allocs, &g_ImmediateFence) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_FENCE, reinterpret_cast<std::uint64_t>(g_ImmediateFence), "g_ImmediateFence");

    commandBufferInfo.commandBufferCount = 1;
    ERR_GUARD_VULKAN( vkAllocateCommandBuffers(g_hDevice, &commandBufferInfo, &g_hTemporaryCommandBuffer) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, reinterpret_cast<std::uint64_t>(g_hTemporaryCommandBuffer), "g_hTemporaryCommandBuffer");

    // Create texture sampler

    VkSamplerCreateInfo samplerInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.f;
    samplerInfo.minLod = 0.f;
    samplerInfo.maxLod = FLT_MAX;
    ERR_GUARD_VULKAN( vkCreateSampler(g_hDevice, &samplerInfo, g_Allocs, &g_hSampler) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<std::uint64_t>(g_hSampler), "g_hSampler");

    CreateTexture(128, 128);
    CreateMesh();

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descriptorSetLayoutInfo.bindingCount = 1;
    descriptorSetLayoutInfo.pBindings = &samplerLayoutBinding;
    ERR_GUARD_VULKAN( vkCreateDescriptorSetLayout(g_hDevice, &descriptorSetLayoutInfo, g_Allocs, &g_hDescriptorSetLayout) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, reinterpret_cast<std::uint64_t>(g_hDescriptorSetLayout), "g_hDescriptorSetLayout");

    // Create descriptor pool

    VkDescriptorPoolSize descriptorPoolSizes[2];
    ZeroMemory(descriptorPoolSizes, sizeof(descriptorPoolSizes));
    descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorPoolSizes[0].descriptorCount = 1;
    descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorPoolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptorPoolInfo.poolSizeCount = (uint32_t)_countof(descriptorPoolSizes);
    descriptorPoolInfo.pPoolSizes = descriptorPoolSizes;
    descriptorPoolInfo.maxSets = 1;
    ERR_GUARD_VULKAN( vkCreateDescriptorPool(g_hDevice, &descriptorPoolInfo, g_Allocs, &g_hDescriptorPool) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, reinterpret_cast<std::uint64_t>(g_hDescriptorPool), "g_hDescriptorPool");

    // Create descriptor set layout

    VkDescriptorSetLayout descriptorSetLayouts[] = { g_hDescriptorSetLayout };
    VkDescriptorSetAllocateInfo descriptorSetInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    descriptorSetInfo.descriptorPool = g_hDescriptorPool;
    descriptorSetInfo.descriptorSetCount = 1;
    descriptorSetInfo.pSetLayouts = descriptorSetLayouts;
    ERR_GUARD_VULKAN( vkAllocateDescriptorSets(g_hDevice, &descriptorSetInfo, &g_hDescriptorSet) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, reinterpret_cast<std::uint64_t>(g_hDescriptorSet), "g_hDescriptorSet");

    VkDescriptorImageInfo descriptorImageInfo = {};
    descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptorImageInfo.imageView = g_hTextureImageView;
    descriptorImageInfo.sampler = g_hSampler;

    VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writeDescriptorSet.dstSet = g_hDescriptorSet;
    writeDescriptorSet.dstBinding = 1;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.pImageInfo = &descriptorImageInfo;

    vkUpdateDescriptorSets(g_hDevice, 1, &writeDescriptorSet, 0, nullptr);

    CreateSwapchain();
}

static void FinalizeApplication()
{
    vkDeviceWaitIdle(g_hDevice);

    DestroySwapchain(true);

    if(g_hDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(g_hDevice, g_hDescriptorPool, g_Allocs);
        g_hDescriptorPool = VK_NULL_HANDLE;
    }

    if(g_hDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(g_hDevice, g_hDescriptorSetLayout, g_Allocs);
        g_hDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if(g_hTextureImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(g_hDevice, g_hTextureImageView, g_Allocs);
        g_hTextureImageView = VK_NULL_HANDLE;
    }
    if(g_hTextureImage != VK_NULL_HANDLE)
    {
        vmaDestroyImage(g_hAllocator, g_hTextureImage, g_hTextureImageAlloc);
        g_hTextureImage = VK_NULL_HANDLE;
    }

    if(g_hIndexBuffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(g_hAllocator, g_hIndexBuffer, g_hIndexBufferAlloc);
        g_hIndexBuffer = VK_NULL_HANDLE;
    }
    if(g_hVertexBuffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(g_hAllocator, g_hVertexBuffer, g_hVertexBufferAlloc);
        g_hVertexBuffer = VK_NULL_HANDLE;
    }

    if(g_hSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(g_hDevice, g_hSampler, g_Allocs);
        g_hSampler = VK_NULL_HANDLE;
    }

    if(g_ImmediateFence)
    {
        vkDestroyFence(g_hDevice, g_ImmediateFence, g_Allocs);
        g_ImmediateFence = VK_NULL_HANDLE;
    }

    for(size_t i = COMMAND_BUFFER_COUNT; i--; )
    {
        if(g_MainCommandBufferExecutedFences[i] != VK_NULL_HANDLE)
        {
            vkDestroyFence(g_hDevice, g_MainCommandBufferExecutedFences[i], g_Allocs);
            g_MainCommandBufferExecutedFences[i] = VK_NULL_HANDLE;
        }
    }
    if(g_MainCommandBuffers[0] != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(g_hDevice, g_hCommandPool, COMMAND_BUFFER_COUNT, g_MainCommandBuffers);
        ZeroMemory(g_MainCommandBuffers, sizeof(g_MainCommandBuffers));
    }
    if(g_hTemporaryCommandBuffer != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(g_hDevice, g_hCommandPool, 1, &g_hTemporaryCommandBuffer);
        g_hTemporaryCommandBuffer = VK_NULL_HANDLE;
    }

    if(g_hCommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(g_hDevice, g_hCommandPool, g_Allocs);
        g_hCommandPool = VK_NULL_HANDLE;
    }

    if(g_hAllocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(g_hAllocator);
        g_hAllocator = nullptr;
    }

    if(g_hDevice != VK_NULL_HANDLE)
    {
        vkDestroyDevice(g_hDevice, g_Allocs);
        g_hDevice = nullptr;
    }

    if(g_hSurface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(g_hVulkanInstance, g_hSurface, g_Allocs);
        g_hSurface = VK_NULL_HANDLE;
    }
}

static void PrintAllocatorStats()
{
#if VMA_STATS_STRING_ENABLED
    char* statsString = nullptr;
    vmaBuildStatsString(g_hAllocator, &statsString, true);
    printf("%s\n", statsString);
    vmaFreeStatsString(g_hAllocator, statsString);
#endif
}

static void RecreateSwapChain()
{
    vkDeviceWaitIdle(g_hDevice);
    DestroySwapchain(false);
    CreateSwapchain();
}

static void DrawFrame()
{
    // Begin main command buffer
    size_t cmdBufIndex = (g_NextCommandBufferIndex++) % COMMAND_BUFFER_COUNT;
    VkCommandBuffer hCommandBuffer = g_MainCommandBuffers[cmdBufIndex];
    VkFence hCommandBufferExecutedFence = g_MainCommandBufferExecutedFences[cmdBufIndex];

    ERR_GUARD_VULKAN( vkWaitForFences(g_hDevice, 1, &hCommandBufferExecutedFence, VK_TRUE, UINT64_MAX) );
    ERR_GUARD_VULKAN( vkResetFences(g_hDevice, 1, &hCommandBufferExecutedFence) );

    VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ERR_GUARD_VULKAN( vkBeginCommandBuffer(hCommandBuffer, &commandBufferBeginInfo) );
    SetDebugUtilsObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, reinterpret_cast<std::uint64_t>(hCommandBuffer), "hCommandBuffer");

    const VkSemaphore imageAvailableSemaphore = g_hImageAvailableSemaphores[cmdBufIndex];

    // Acquire swapchain image
    uint32_t imageIndex = 0;
    VkResult res = vkAcquireNextImageKHR(g_hDevice, g_hSwapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
    if(res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        RecreateSwapChain();
        return;
    }
    else if(res < 0)
    {
        ERR_GUARD_VULKAN(res);
    }

    // Record geometry pass

    VkClearValue clearValues[2];
    ZeroMemory(clearValues, sizeof(clearValues));
    clearValues[0].color.float32[0] = 0.25f;
    clearValues[0].color.float32[1] = 0.25f;
    clearValues[0].color.float32[2] = 0.5f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil.depth = 1.0f;

    VkRenderPassBeginInfo renderPassBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    renderPassBeginInfo.renderPass = g_hRenderPass;
    renderPassBeginInfo.framebuffer = g_Framebuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent = g_Extent;
    renderPassBeginInfo.clearValueCount = (uint32_t)_countof(clearValues);
    renderPassBeginInfo.pClearValues = clearValues;
    vkCmdBeginRenderPass(hCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(
        hCommandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        g_hPipeline);

    mat4 view = mat4::LookAt(
        vec3(0.f, 0.f, 0.f),
        vec3(0.f, -2.f, 4.f),
        vec3(0.f, 1.f, 0.f));
    mat4 proj = mat4::Perspective(
        1.0471975511966f, // 60 degrees
        (float)g_Extent.width / (float)g_Extent.height,
        0.1f,
        1000.f);
    mat4 viewProj = view * proj;

    vkCmdBindDescriptorSets(
        hCommandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        g_hPipelineLayout,
        0,
        1,
        &g_hDescriptorSet,
        0,
        nullptr);

    float rotationAngle = (float)GetTickCount() * 0.001f * (float)PI * 0.2f;
    mat4 model = mat4::RotationY(rotationAngle);

    UniformBufferObject ubo = {};
    ubo.ModelViewProj = model * viewProj;
    vkCmdPushConstants(hCommandBuffer, g_hPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(UniformBufferObject), &ubo);

    VkBuffer vertexBuffers[] = { g_hVertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(hCommandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(hCommandBuffer, g_hIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdDrawIndexed(hCommandBuffer, g_IndexCount, 1, 0, 0, 0);

    vkCmdEndRenderPass(hCommandBuffer);

    vkEndCommandBuffer(hCommandBuffer);

    // Submit command buffer

    VkSemaphore submitWaitSemaphores[] = { imageAvailableSemaphore };
    VkPipelineStageFlags submitWaitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore submitSignalSemaphores[] = { g_hRenderFinishedSemaphores.at(g_SwapchainImageIndex)};
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = submitWaitSemaphores;
    submitInfo.pWaitDstStageMask = submitWaitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &hCommandBuffer;
    submitInfo.signalSemaphoreCount = _countof(submitSignalSemaphores);
    submitInfo.pSignalSemaphores = submitSignalSemaphores;
    ERR_GUARD_VULKAN( vkQueueSubmit(g_hGraphicsQueue, 1, &submitInfo, hCommandBufferExecutedFence) );

    VkSemaphore presentWaitSemaphores[] = { g_hRenderFinishedSemaphores.at(g_SwapchainImageIndex) };

    VkSwapchainKHR swapchains[] = { g_hSwapchain };
    VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = _countof(presentWaitSemaphores);
    presentInfo.pWaitSemaphores = presentWaitSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    res = vkQueuePresentKHR(g_hPresentQueue, &presentInfo);
    if(res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        RecreateSwapChain();
    }
    else
        ERR_GUARD_VULKAN(res);

    g_SwapchainImageIndex++;
    if (g_SwapchainImageIndex >= g_SwapchainImageCount) {
        g_SwapchainImageIndex = 0;
    }
}

static void HandlePossibleSizeChange()
{
    RECT clientRect;
    GetClientRect(g_hWnd, &clientRect);
    LONG newSizeX = clientRect.right - clientRect.left;
    LONG newSizeY = clientRect.bottom - clientRect.top;
    if((newSizeX > 0) &&
        (newSizeY > 0) &&
        ((newSizeX != g_SizeX) || (newSizeY != g_SizeY)))
    {
        g_SizeX = newSizeX;
        g_SizeY = newSizeY;

        RecreateSwapChain();
    }
}

#define CATCH_PRINT_ERROR(extraCatchCode) \
    catch(const std::exception& ex) \
    { \
        fwprintf(stderr, L"ERROR: %hs\n", ex.what()); \
        extraCatchCode \
    } \
    catch(...) \
    { \
        fwprintf(stderr, L"UNKNOWN ERROR.\n"); \
        extraCatchCode \
    }

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
    case WM_DESTROY:
        try
        {
            FinalizeApplication();
        }
        CATCH_PRINT_ERROR(;)
        PostQuitMessage(0);
        return 0;

    // This prevents app from freezing when left Alt is pressed
    // (which normally enters modal menu loop).
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
        return 0;

    case WM_SIZE:
        if((wParam == SIZE_MAXIMIZED) || (wParam == SIZE_RESTORED))
        {
            try
            {
                HandlePossibleSizeChange();
            }
            CATCH_PRINT_ERROR(DestroyWindow(hWnd);)
        }
        return 0;

    case WM_EXITSIZEMOVE:
        try
        {
            HandlePossibleSizeChange();
        }
        CATCH_PRINT_ERROR(DestroyWindow(hWnd);)
        return 0;

    case WM_KEYDOWN:
        switch(wParam)
        {
        case VK_ESCAPE:
            PostMessage(hWnd, WM_CLOSE, 0, 0);
            break;
        case 'T':
            try
            {
                Test();
            }
            CATCH_PRINT_ERROR(;)
            break;
        case 'S':
            if (g_SparseBindingEnabled)
            {
                try
                {
                    TestSparseBinding();
                }
                CATCH_PRINT_ERROR(;)
            }
            else
            {
                printf("Sparse binding not supported.\n");
            }
            break;
        }
        return 0;

    default:
        break;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

static void PrintLogo()
{
    wprintf(L"%s using VMA %d.%d.%d\n", APP_TITLE_W,
        VK_VERSION_MAJOR(VMA_VERSION),
        VK_VERSION_MINOR(VMA_VERSION),
        VK_VERSION_PATCH(VMA_VERSION));
}

static void PrintHelp()
{
    wprintf(
        L"Command line syntax:\n"
        L"-h, --Help   Print this information\n"
        L"-l, --List   Print list of GPUs\n"
        L"-g S, --GPU S   Select GPU with name containing S\n"
        L"-i N, --GPUIndex N   Select GPU index N\n"
        L"-t, --Test   Run tests and exit\n"
        L"-s, --TestSparseBinding   Run sparese binding tests and exit\n"
    );
}

int MainWindow()
{
    WNDCLASSEX wndClassDesc = { sizeof(WNDCLASSEX) };
    wndClassDesc.style = CS_VREDRAW | CS_HREDRAW | CS_DBLCLKS;
    wndClassDesc.hbrBackground = NULL;
    wndClassDesc.hCursor = LoadCursor(NULL, IDC_CROSS);
    wndClassDesc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wndClassDesc.hInstance = g_hAppInstance;
    wndClassDesc.lpfnWndProc = WndProc;
    wndClassDesc.lpszClassName = WINDOW_CLASS_NAME;

    const ATOM hWndClass = RegisterClassEx(&wndClassDesc);
    assert(hWndClass);

    const DWORD style = WS_VISIBLE | WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_THICKFRAME;
    const DWORD exStyle = 0;

    RECT rect = { 0, 0, g_SizeX, g_SizeY };
    AdjustWindowRectEx(&rect, style, FALSE, exStyle);

    g_hWnd = CreateWindowEx(
        exStyle, WINDOW_CLASS_NAME, APP_TITLE_W, style,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        NULL, NULL, g_hAppInstance, NULL);
    assert(g_hWnd);

    InitializeApplication();
    //PrintAllocatorStats();

    // Run tests and close program
    if(g_CommandLineParameters.m_Test)
        Test();
    if(g_CommandLineParameters.m_TestSparseBinding)
    {
        if(g_SparseBindingEnabled)
            TestSparseBinding();
        else
            printf("Sparse binding not supported.\n");
    }
    if(g_CommandLineParameters.m_Test || g_CommandLineParameters.m_TestSparseBinding)
        PostMessage(g_hWnd, WM_CLOSE, 0, 0);

    MSG msg;
    for(;;)
    {
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT)
                break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
		{
            DrawFrame();
		}
    }

    return (int)msg.wParam;;
}

int Main2(int argc, wchar_t** argv)
{
    PrintLogo();

    if(!g_CommandLineParameters.Parse(argc, argv))
    {
        wprintf(L"ERROR: Invalid command line syntax.\n");
        PrintHelp();
        return (int)ExitCode::CommandLineError;
    }

    if(g_CommandLineParameters.m_Help)
    {
        PrintHelp();
        return (int)ExitCode::Help;
    }

    VulkanUsage vulkanUsage;
    vulkanUsage.Init();

    if(g_CommandLineParameters.m_List)
    {
        vulkanUsage.PrintPhysicalDeviceList();
        return (int)ExitCode::GPUList;
    }

    g_hPhysicalDevice = vulkanUsage.SelectPhysicalDevice(g_CommandLineParameters.m_GPUSelection);
    TEST(g_hPhysicalDevice);

    return MainWindow();
}

int wmain(int argc, wchar_t** argv)
{
    int result = 0;
    try
    {
        result = Main2(argc, argv);
        TEST(g_CpuAllocCount.load() == 0);
    }
    CATCH_PRINT_ERROR(return (int)ExitCode::RuntimeError;)
    return result;
}

#else // #ifdef _WIN32

#include "VmaUsage.h"

int main()
{
}

#endif // #ifdef _WIN32
