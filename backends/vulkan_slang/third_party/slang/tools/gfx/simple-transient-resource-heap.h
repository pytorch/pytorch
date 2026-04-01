// simple-render-pass-layout.h
#pragma once

// Provide a simple no-op implementation for `ITransientResourceHeap` for targets that
// already support version management.

#include "renderer-shared.h"
#include "slang-gfx.h"

namespace gfx
{
template<typename TDevice, typename TCommandBuffer>
class SimpleTransientResourceHeap : public TransientResourceHeapBase
{
public:
    Slang::RefPtr<TDevice> m_device;
    Slang::ComPtr<IBufferResource> m_constantBuffer;

public:
    Result init(TDevice* device, const ITransientResourceHeap::Desc& desc)
    {
        m_device = device;
        IBufferResource::Desc bufferDesc = {};
        bufferDesc.type = IResource::Type::Buffer;
        bufferDesc.allowedStates =
            ResourceStateSet(ResourceState::ConstantBuffer, ResourceState::CopyDestination);
        bufferDesc.defaultState = ResourceState::ConstantBuffer;
        bufferDesc.sizeInBytes = desc.constantBufferSize;
        bufferDesc.memoryType = MemoryType::Upload;
        SLANG_RETURN_ON_FAIL(
            device->createBufferResource(bufferDesc, nullptr, m_constantBuffer.writeRef()));
        return SLANG_OK;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandBuffer(ICommandBuffer** outCommandBuffer) override
    {
        Slang::RefPtr<TCommandBuffer> newCmdBuffer = new TCommandBuffer();
        newCmdBuffer->init(m_device, this);
        returnComPtr(outCommandBuffer, newCmdBuffer);
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL synchronizeAndReset() override
    {
        ++getVersionCounter();
        return SLANG_OK;
    }
};
} // namespace gfx
