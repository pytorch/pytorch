// debug-command-queue.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugCommandQueue : public DebugObject<ICommandQueue>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    ICommandQueue* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override;
    virtual SLANG_NO_THROW void SLANG_MCALL executeCommandBuffers(
        GfxCount count,
        ICommandBuffer* const* commandBuffers,
        IFence* fence,
        uint64_t valueToSignal) override;
    virtual SLANG_NO_THROW void SLANG_MCALL waitOnHost() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    waitForFenceValuesOnDevice(GfxCount fenceCount, IFence** fences, uint64_t* waitValues) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace debug
} // namespace gfx
