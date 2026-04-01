// metal-transient-heap.cpp
#include "metal-transient-heap.h"

#include "metal-device.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

Result TransientResourceHeapImpl::init(const ITransientResourceHeap::Desc& desc, DeviceImpl* device)
{
    Super::init(
        desc,
        256, // TODO
        device);

    return SLANG_OK;
}

TransientResourceHeapImpl::~TransientResourceHeapImpl() {}

Result TransientResourceHeapImpl::createCommandBuffer(ICommandBuffer** outCmdBuffer)
{
    RefPtr<CommandBufferImpl> commandBuffer = new CommandBufferImpl();
    SLANG_RETURN_ON_FAIL(commandBuffer->init(m_device, this));
    returnComPtr(outCmdBuffer, commandBuffer);
    return SLANG_OK;
}

Result TransientResourceHeapImpl::synchronizeAndReset()
{
    Super::reset();
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
