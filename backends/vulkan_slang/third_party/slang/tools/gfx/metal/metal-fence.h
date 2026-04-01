// metal-fence.h
#pragma once

#include "metal-base.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class FenceImpl : public FenceBase
{
public:
    RefPtr<DeviceImpl> m_device;
    NS::SharedPtr<MTL::SharedEvent> m_event;
    NS::SharedPtr<MTL::SharedEventListener> m_eventListener;

    ~FenceImpl();

    Result init(DeviceImpl* device, const IFence::Desc& desc);

    virtual SLANG_NO_THROW Result SLANG_MCALL getCurrentValue(uint64_t* outValue) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setCurrentValue(uint64_t value) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;

    bool waitForFence(uint64_t value, uint64_t timeout);
};

} // namespace metal
} // namespace gfx
