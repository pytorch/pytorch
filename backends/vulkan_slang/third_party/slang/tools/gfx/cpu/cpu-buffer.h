// cpu-buffer.h
#pragma once
#include "cpu-base.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class BufferResourceImpl : public BufferResource
{
public:
    BufferResourceImpl(const Desc& _desc)
        : BufferResource(_desc)
    {
    }

    ~BufferResourceImpl();

    Result init();

    Result setData(size_t offset, size_t size, void const* data);

    void* m_data = nullptr;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    map(MemoryRange* rangeToRead, void** outPointer) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) override;
};

} // namespace cpu
} // namespace gfx
