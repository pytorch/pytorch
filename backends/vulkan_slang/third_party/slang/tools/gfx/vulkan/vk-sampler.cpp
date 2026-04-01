// vk-sampler.cpp
#include "vk-sampler.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

SamplerStateImpl::SamplerStateImpl(DeviceImpl* device)
    : m_device(device)
{
}

SamplerStateImpl::~SamplerStateImpl()
{
    m_device->m_api.vkDestroySampler(m_device->m_api.m_device, m_sampler, nullptr);
}

Result SamplerStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = (uint64_t)(m_sampler);
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
