// metal-command-buffer.cpp
#include "metal-command-buffer.h"

#include "metal-command-encoder.h"
#include "metal-command-queue.h"
#include "metal-device.h"
#include "metal-shader-object.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

ICommandBuffer* CommandBufferImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandBuffer)
        return static_cast<ICommandBuffer*>(this);
    return nullptr;
}

Result CommandBufferImpl::init(DeviceImpl* device, TransientResourceHeapImpl* transientHeap)
{
    m_device = device;
    m_commandBuffer = NS::RetainPtr(m_device->m_commandQueue->commandBuffer());
    return SLANG_OK;
}

void CommandBufferImpl::encodeRenderCommands(
    IRenderPassLayout* renderPass,
    IFramebuffer* framebuffer,
    IRenderCommandEncoder** outEncoder)
{
    if (!m_renderCommandEncoder)
    {
        m_renderCommandEncoder = new RenderCommandEncoder;
        m_renderCommandEncoder->init(this);
    }
    m_renderCommandEncoder->beginPass(renderPass, framebuffer);
    *outEncoder = m_renderCommandEncoder;
}

void CommandBufferImpl::encodeComputeCommands(IComputeCommandEncoder** outEncoder)
{
    if (!m_computeCommandEncoder)
    {
        m_computeCommandEncoder = new ComputeCommandEncoder;
        m_computeCommandEncoder->init(this);
    }
    *outEncoder = m_computeCommandEncoder;
}

void CommandBufferImpl::encodeResourceCommands(IResourceCommandEncoder** outEncoder)
{
    if (!m_resourceCommandEncoder)
    {
        m_resourceCommandEncoder = new ResourceCommandEncoder;
        m_resourceCommandEncoder->init(this);
    }
    *outEncoder = m_resourceCommandEncoder;
}

void CommandBufferImpl::encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder)
{
    if (!m_rayTracingCommandEncoder)
    {
        m_rayTracingCommandEncoder = new RayTracingCommandEncoder;
        m_rayTracingCommandEncoder->init(this);
    }
    *outEncoder = m_rayTracingCommandEncoder;
}

void CommandBufferImpl::close()
{
    // m_commandBuffer->commit();
}

Result CommandBufferImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<intptr_t>(m_commandBuffer.get());
    return SLANG_OK;
}

MTL::RenderCommandEncoder* CommandBufferImpl::getMetalRenderCommandEncoder(
    MTL::RenderPassDescriptor* renderPassDesc)
{
    if (!m_metalRenderCommandEncoder)
    {
        endMetalCommandEncoder();
        m_metalRenderCommandEncoder =
            NS::RetainPtr(m_commandBuffer->renderCommandEncoder(renderPassDesc));
    }
    return m_metalRenderCommandEncoder.get();
}

MTL::ComputeCommandEncoder* CommandBufferImpl::getMetalComputeCommandEncoder()
{
    if (!m_metalComputeCommandEncoder)
    {
        endMetalCommandEncoder();
        m_metalComputeCommandEncoder = NS::RetainPtr(m_commandBuffer->computeCommandEncoder());
    }
    return m_metalComputeCommandEncoder.get();
}

MTL::BlitCommandEncoder* CommandBufferImpl::getMetalBlitCommandEncoder()
{
    if (!m_metalBlitCommandEncoder)
    {
        endMetalCommandEncoder();
        m_metalBlitCommandEncoder = NS::RetainPtr(m_commandBuffer->blitCommandEncoder());
    }
    return m_metalBlitCommandEncoder.get();
}

void CommandBufferImpl::endMetalCommandEncoder()
{
    if (m_metalRenderCommandEncoder)
    {
        m_metalRenderCommandEncoder->endEncoding();
        m_metalRenderCommandEncoder.reset();
    }
    if (m_metalComputeCommandEncoder)
    {
        m_metalComputeCommandEncoder->endEncoding();
        m_metalComputeCommandEncoder.reset();
    }
    if (m_metalBlitCommandEncoder)
    {
        m_metalBlitCommandEncoder->endEncoding();
        m_metalBlitCommandEncoder.reset();
    }
}


} // namespace metal
} // namespace gfx
