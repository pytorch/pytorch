// vk-texture.cpp
#include "vk-texture.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

TextureResourceImpl::TextureResourceImpl(const Desc& desc, DeviceImpl* device)
    : Parent(desc), m_device(device)
{
}

TextureResourceImpl::~TextureResourceImpl()
{
    auto& vkAPI = m_device->m_api;
    if (!m_isWeakImageReference)
    {
        vkAPI.vkFreeMemory(vkAPI.m_device, m_imageMemory, nullptr);
        vkAPI.vkDestroyImage(vkAPI.m_device, m_image, nullptr);
    }
    if (sharedHandle.handleValue != 0)
    {
#if SLANG_WINDOWS_FAMILY
        CloseHandle((HANDLE)sharedHandle.handleValue);
#endif
    }
}

Result TextureResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = (uint64_t)m_image;
    outHandle->api = InteropHandleAPI::Vulkan;
    return SLANG_OK;
}

Result TextureResourceImpl::getSharedHandle(InteropHandle* outHandle)
{
    // Check if a shared handle already exists for this resource.
    if (sharedHandle.handleValue != 0)
    {
        *outHandle = sharedHandle;
        return SLANG_OK;
    }

    // If a shared handle doesn't exist, create one and store it.
#if SLANG_WINDOWS_FAMILY
    VkMemoryGetWin32HandleInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    info.pNext = nullptr;
    info.memory = m_imageMemory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    auto& api = m_device->m_api;
    PFN_vkGetMemoryWin32HandleKHR vkCreateSharedHandle;
    vkCreateSharedHandle = api.vkGetMemoryWin32HandleKHR;
    if (!vkCreateSharedHandle)
    {
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(
        vkCreateSharedHandle(m_device->m_device, &info, (HANDLE*)&outHandle->handleValue) !=
        VK_SUCCESS);
#else
    VkMemoryGetFdInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    info.pNext = nullptr;
    info.memory = m_imageMemory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto& api = m_device->m_api;
    PFN_vkGetMemoryFdKHR vkCreateSharedHandle;
    vkCreateSharedHandle = api.vkGetMemoryFdKHR;
    if (!vkCreateSharedHandle)
    {
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(
        vkCreateSharedHandle(m_device->m_device, &info, (int*)&outHandle->handleValue) !=
        VK_SUCCESS);
#endif
    outHandle->api = InteropHandleAPI::Vulkan;
    return SLANG_OK;
}
Result TextureResourceImpl::setDebugName(const char* name)
{
    Parent::setDebugName(name);
    auto& api = m_device->m_api;
    if (api.vkDebugMarkerSetObjectNameEXT)
    {
        VkDebugMarkerObjectNameInfoEXT nameDesc = {};
        nameDesc.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
        nameDesc.object = (uint64_t)m_image;
        nameDesc.objectType = VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT;
        nameDesc.pObjectName = name;
        api.vkDebugMarkerSetObjectNameEXT(api.m_device, &nameDesc);
    }
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
