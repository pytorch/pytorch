// vk-transient-heap.h
#pragma once

#include "vk-base.h"
#include "vk-buffer.h"
#include "vk-command-buffer.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class TransientResourceHeapImpl
    : public TransientResourceHeapBaseImpl<DeviceImpl, BufferResourceImpl>
{
private:
    typedef TransientResourceHeapBaseImpl<DeviceImpl, BufferResourceImpl> Super;

public:
    VkCommandPool m_commandPool;
    DescriptorSetAllocator m_descSetAllocator;
    List<VkFence> m_fences;
    Index m_fenceIndex = -1;
    List<RefPtr<CommandBufferImpl>> m_commandBufferPool;
    uint32_t m_commandBufferAllocId = 0;
    VkFence getCurrentFence() { return m_fences[m_fenceIndex]; }
    void advanceFence();

    Result init(const ITransientResourceHeap::Desc& desc, DeviceImpl* device);
    ~TransientResourceHeapImpl();

public:
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandBuffer(ICommandBuffer** outCommandBuffer) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL synchronizeAndReset() override;
};

} // namespace vk
} // namespace gfx
