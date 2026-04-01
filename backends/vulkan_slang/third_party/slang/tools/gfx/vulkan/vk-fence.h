// vk-fence.h
#pragma once

#include "vk-base.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class FenceImpl : public FenceBase
{
public:
    VkSemaphore m_semaphore = VK_NULL_HANDLE;
    RefPtr<DeviceImpl> m_device;

    FenceImpl(DeviceImpl* device);

    ~FenceImpl();

    Result init(const IFence::Desc& desc);

    virtual SLANG_NO_THROW Result SLANG_MCALL getCurrentValue(uint64_t* outValue) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setCurrentValue(uint64_t value) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;
};

} // namespace vk
} // namespace gfx
