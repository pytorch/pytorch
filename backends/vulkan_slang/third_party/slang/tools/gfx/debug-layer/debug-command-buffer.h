// debug-command-buffer.h
#pragma once
#include "debug-base.h"
#include "debug-command-encoder.h"
#include "debug-shader-object.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugCommandBuffer : public DebugObject<ICommandBuffer>, ICommandBufferD3D12
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    DebugTransientResourceHeap* m_transientHeap;

private:
    DebugRenderCommandEncoder m_renderCommandEncoder;
    DebugComputeCommandEncoder m_computeCommandEncoder;
    DebugResourceCommandEncoder m_resourceCommandEncoder;
    DebugRayTracingCommandEncoder m_rayTracingCommandEncoder;

public:
    DebugCommandBuffer();
    ICommandBuffer* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW void SLANG_MCALL encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer,
        IRenderCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeComputeCommands(IComputeCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeResourceCommands(IResourceCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL close() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
    virtual SLANG_NO_THROW void SLANG_MCALL invalidateDescriptorHeapBinding() override;
    virtual SLANG_NO_THROW void SLANG_MCALL ensureInternalDescriptorHeapsBound() override;

private:
    void checkEncodersClosedBeforeNewEncoder();
    void checkCommandBufferOpenWhenCreatingEncoder();

public:
    DebugRootShaderObject rootObject;
    bool isOpen = true;
};

} // namespace debug
} // namespace gfx
