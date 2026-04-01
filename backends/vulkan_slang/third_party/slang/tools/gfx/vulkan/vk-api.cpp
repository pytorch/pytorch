// vk-api.cpp
#include "vk-api.h"

#include "core/slang-list.h"

namespace gfx
{
using namespace Slang;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VulkanApi !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#define VK_API_CHECK_FUNCTION(x) &&hasFunction(#x, (void*)x)
#define VK_API_CHECK_FUNCTIONS(FUNCTION_LIST) true FUNCTION_LIST(VK_API_CHECK_FUNCTION)

static bool hasFunction(const char* name, void* ptr)
{
    if (ptr)
        return true;
#if 0
    fprintf(stderr, "Missing required Vulkan function: %s\n", name);
#endif
    return false;
}

bool VulkanApi::areDefined(ProcType type) const
{
    switch (type)
    {
    case ProcType::Global:
        return VK_API_CHECK_FUNCTIONS(VK_API_ALL_GLOBAL_PROCS);
    case ProcType::Instance:
        return VK_API_CHECK_FUNCTIONS(VK_API_ALL_INSTANCE_PROCS) &&
               VK_API_CHECK_FUNCTIONS(VK_API_INSTANCE_KHR_PROCS);
    case ProcType::Device:
        return VK_API_CHECK_FUNCTIONS(VK_API_DEVICE_PROCS);
    default:
        {
            assert(!"Unhandled type");
            return false;
        }
    }
}

Slang::Result VulkanApi::initGlobalProcs(const VulkanModule& module)
{
#define VK_API_GET_GLOBAL_PROC(x) x = (PFN_##x)module.getFunction(#x);

    // Initialize all the global functions
    VK_API_ALL_GLOBAL_PROCS(VK_API_GET_GLOBAL_PROC)

    if (!areDefined(ProcType::Global))
    {
        return SLANG_FAIL;
    }
    m_module = &module;
    return SLANG_OK;
}

Slang::Result VulkanApi::initInstanceProcs(VkInstance instance)
{
    assert(instance && vkGetInstanceProcAddr != nullptr);

#define VK_API_GET_INSTANCE_PROC(x) x = (PFN_##x)vkGetInstanceProcAddr(instance, #x);

    VK_API_ALL_INSTANCE_PROCS(VK_API_GET_INSTANCE_PROC)

    // Get optional
    VK_API_INSTANCE_PROCS_OPT(VK_API_GET_INSTANCE_PROC)

    if (!areDefined(ProcType::Instance))
    {
        return SLANG_FAIL;
    }


    m_instance = instance;
    return SLANG_OK;
}

Slang::Result VulkanApi::initPhysicalDevice(VkPhysicalDevice physicalDevice)
{
    assert(m_physicalDevice == VK_NULL_HANDLE);
    m_physicalDevice = physicalDevice;

    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &m_deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_deviceMemoryProperties);

    return SLANG_OK;
}

Slang::Result VulkanApi::initDeviceProcs(VkDevice device)
{
    assert(m_instance && device && vkGetDeviceProcAddr != nullptr);

#define VK_API_GET_DEVICE_PROC(x) x = (PFN_##x)vkGetDeviceProcAddr(device, #x);

    VK_API_ALL_DEVICE_PROCS(VK_API_GET_DEVICE_PROC)

    if (!areDefined(ProcType::Device))
    {
        return SLANG_FAIL;
    }

    if (!vkGetBufferDeviceAddressKHR && vkGetBufferDeviceAddressEXT)
        vkGetBufferDeviceAddressKHR = vkGetBufferDeviceAddressEXT;
    if (!vkGetBufferDeviceAddress && vkGetBufferDeviceAddressKHR)
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
    if (!vkGetSemaphoreCounterValue && vkGetSemaphoreCounterValueKHR)
        vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
    if (!vkSignalSemaphore && vkSignalSemaphoreKHR)
        vkSignalSemaphore = vkSignalSemaphoreKHR;
    m_device = device;
    return SLANG_OK;
}

int VulkanApi::findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties) const
{
    assert(typeBits);

    const int numMemoryTypes = int(m_deviceMemoryProperties.memoryTypeCount);

    // bit holds current test bit against typeBits. Ie bit == 1 << typeBits

    uint32_t bit = 1;
    for (int i = 0; i < numMemoryTypes; ++i, bit += bit)
    {
        auto const& memoryType = m_deviceMemoryProperties.memoryTypes[i];
        if ((typeBits & bit) && (memoryType.propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    // assert(!"failed to find a usable memory type");
    return -1;
}

int VulkanApi::findQueue(VkQueueFlags reqFlags) const
{
    assert(m_physicalDevice != VK_NULL_HANDLE);

    uint32_t numQueueFamilies = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &numQueueFamilies, nullptr);

    Slang::List<VkQueueFamilyProperties> queueFamilies;
    queueFamilies.setCount(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(
        m_physicalDevice,
        &numQueueFamilies,
        queueFamilies.getBuffer());

    // Find a queue that can service our needs
    // VkQueueFlags reqQueueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;

    int queueFamilyIndex = -1;
    for (int i = 0; i < int(numQueueFamilies); ++i)
    {
        if ((queueFamilies[i].queueFlags & reqFlags) == reqFlags)
        {
            return i;
        }
    }

    return -1;
}

} // namespace gfx
