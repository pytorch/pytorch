// vk-sampler.h
#pragma once

#include "vk-base.h"
#include "vk-device.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class SamplerStateImpl : public SamplerStateBase
{
public:
    VkSampler m_sampler;
    RefPtr<DeviceImpl> m_device;
    SamplerStateImpl(DeviceImpl* device);
    ~SamplerStateImpl();
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace vk
} // namespace gfx
