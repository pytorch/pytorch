// d3d11-buffer.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class BufferResourceImpl : public BufferResource
{
public:
    typedef BufferResource Parent;

    BufferResourceImpl(const IBufferResource::Desc& desc)
        : Parent(desc)
    {
    }

    MapFlavor m_mapFlavor;
    D3D11_USAGE m_d3dUsage;
    ComPtr<ID3D11Buffer> m_buffer;
    ComPtr<ID3D11Buffer> m_staging;
    List<uint8_t> m_uploadStagingBuffer;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    map(MemoryRange* rangeToRead, void** outPointer) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) override;
};

} // namespace d3d11
} // namespace gfx
