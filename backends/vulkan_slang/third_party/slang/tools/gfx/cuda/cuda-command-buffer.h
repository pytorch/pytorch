// cuda-command-buffer.h
#pragma once
#include "cuda-base.h"
#include "cuda-command-encoder.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class CommandBufferImpl : public ICommandBuffer, public CommandWriter, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandBuffer* getInterface(const Guid& guid);

public:
    DeviceImpl* m_device;
    TransientResourceHeapBase* m_transientHeap;
    ResourceCommandEncoderImpl m_resourceCommandEncoder;
    ComputeCommandEncoderImpl m_computeCommandEncoder;

    void init(DeviceImpl* device, TransientResourceHeapBase* transientHeap);
    virtual SLANG_NO_THROW void SLANG_MCALL encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer,
        IRenderCommandEncoder** outEncoder) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeResourceCommands(IResourceCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeComputeCommands(IComputeCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder) override;

    virtual SLANG_NO_THROW void SLANG_MCALL close() override {}

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace cuda
#endif
} // namespace gfx
