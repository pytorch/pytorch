// cuda-buffer.h
#pragma once
#include "cuda-base.h"
#include "cuda-context.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class BufferResourceImpl : public BufferResource
{
public:
    BufferResourceImpl(const Desc& _desc)
        : BufferResource(_desc)
    {
    }

    ~BufferResourceImpl();

    uint64_t getBindlessHandle();

    void* m_cudaExternalMemory = nullptr;
    void* m_cudaMemory = nullptr;

    RefPtr<CUDAContext> m_cudaContext;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeResourceHandle(InteropHandle* outHandle) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    map(MemoryRange* rangeToRead, void** outPointer) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) override;
};

} // namespace cuda
#endif
} // namespace gfx
