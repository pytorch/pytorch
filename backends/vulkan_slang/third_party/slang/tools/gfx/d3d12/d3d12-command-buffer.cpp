// d3d12-command-buffer.cpp
#include "d3d12-command-buffer.h"

#include "d3d12-transient-heap.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

// There are a pair of cyclic references between a `TransientResourceHeap` and
// a `CommandBuffer` created from the heap. We need to break the cycle upon
// the public reference count of a command buffer dropping to 0.

ICommandBufferD3D12* CommandBufferImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandBuffer ||
        guid == GfxGUID::IID_ICommandBufferD3D12)
        return static_cast<ICommandBufferD3D12*>(this);
    return nullptr;
}

Result CommandBufferImpl::getNativeHandle(InteropHandle* handle)
{
    handle->api = InteropHandleAPI::D3D12;
    handle->handleValue = (uint64_t)m_cmdList.get();
    return SLANG_OK;
}

void CommandBufferImpl::bindDescriptorHeaps()
{
    if (!m_descriptorHeapsBound)
    {
        ID3D12DescriptorHeap* heaps[] = {
            m_transientHeap->getCurrentViewHeap().getHeap(),
            m_transientHeap->getCurrentSamplerHeap().getHeap(),
        };
        m_cmdList->SetDescriptorHeaps(SLANG_COUNT_OF(heaps), heaps);
        m_descriptorHeapsBound = true;
    }
}

void CommandBufferImpl::reinit()
{
    invalidateDescriptorHeapBinding();
    m_rootShaderObject.init(m_renderer);
}

void CommandBufferImpl::init(
    DeviceImpl* renderer,
    ID3D12GraphicsCommandList* d3dCommandList,
    TransientResourceHeapImpl* transientHeap)
{
    m_transientHeap = transientHeap;
    m_renderer = renderer;
    m_cmdList = d3dCommandList;

    reinit();

    m_cmdList->QueryInterface<ID3D12GraphicsCommandList6>(m_cmdList6.writeRef());
    if (m_cmdList6)
    {
        m_cmdList4 = m_cmdList6;
        m_cmdList1 = m_cmdList6;
        return;
    }
#if SLANG_GFX_HAS_DXR_SUPPORT
    m_cmdList->QueryInterface<ID3D12GraphicsCommandList4>(m_cmdList4.writeRef());
    if (m_cmdList4)
    {
        m_cmdList1 = m_cmdList4;
        return;
    }
#endif
    m_cmdList->QueryInterface<ID3D12GraphicsCommandList1>(m_cmdList1.writeRef());
}

void CommandBufferImpl::encodeResourceCommands(IResourceCommandEncoder** outEncoder)
{
    m_resourceCommandEncoder.init(this);
    *outEncoder = &m_resourceCommandEncoder;
}

void CommandBufferImpl::encodeRenderCommands(
    IRenderPassLayout* renderPass,
    IFramebuffer* framebuffer,
    IRenderCommandEncoder** outEncoder)
{
    m_renderCommandEncoder.init(
        m_renderer,
        m_transientHeap,
        this,
        static_cast<RenderPassLayoutImpl*>(renderPass),
        static_cast<FramebufferImpl*>(framebuffer));
    *outEncoder = &m_renderCommandEncoder;
}

void CommandBufferImpl::encodeComputeCommands(IComputeCommandEncoder** outEncoder)
{
    m_computeCommandEncoder.init(m_renderer, m_transientHeap, this);
    *outEncoder = &m_computeCommandEncoder;
}

void CommandBufferImpl::encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder)
{
#if SLANG_GFX_HAS_DXR_SUPPORT
    m_rayTracingCommandEncoder.init(this);
    *outEncoder = &m_rayTracingCommandEncoder;
#else
    *outEncoder = nullptr;
#endif
}

void CommandBufferImpl::close()
{
    m_cmdList->Close();
}

} // namespace d3d12
} // namespace gfx
