// cuda-command-buffer.cpp
#include "cuda-command-buffer.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

ICommandBuffer* CommandBufferImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandBuffer)
        return static_cast<ICommandBuffer*>(this);
    return nullptr;
}

void CommandBufferImpl::init(DeviceImpl* device, TransientResourceHeapBase* transientHeap)
{
    m_device = device;
    m_transientHeap = transientHeap;
}

SLANG_NO_THROW void SLANG_MCALL CommandBufferImpl::encodeRenderCommands(
    IRenderPassLayout* renderPass,
    IFramebuffer* framebuffer,
    IRenderCommandEncoder** outEncoder)
{
    SLANG_UNUSED(renderPass);
    SLANG_UNUSED(framebuffer);
    *outEncoder = nullptr;
}

SLANG_NO_THROW void SLANG_MCALL
CommandBufferImpl::encodeResourceCommands(IResourceCommandEncoder** outEncoder)
{
    m_resourceCommandEncoder.init(this);
    *outEncoder = &m_resourceCommandEncoder;
}

SLANG_NO_THROW void SLANG_MCALL
CommandBufferImpl::encodeComputeCommands(IComputeCommandEncoder** outEncoder)
{
    m_computeCommandEncoder.init(this);
    *outEncoder = &m_computeCommandEncoder;
}

SLANG_NO_THROW void SLANG_MCALL
CommandBufferImpl::encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder)
{
    *outEncoder = nullptr;
}

SLANG_NO_THROW Result SLANG_MCALL CommandBufferImpl::getNativeHandle(InteropHandle* outHandle)
{
    return SLANG_FAIL;
}

} // namespace cuda
#endif
} // namespace gfx
