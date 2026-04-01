// d3d12-buffer.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class BufferResourceImpl : public gfx::BufferResource
{
public:
    typedef BufferResource Parent;

    BufferResourceImpl(const Desc& desc);

    ~BufferResourceImpl();

    D3D12Resource m_resource; ///< The resource in gpu memory, allocated on the correct heap
                              ///< relative to the cpu access flag

    D3D12_RESOURCE_STATES m_defaultState;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeResourceHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    map(MemoryRange* rangeToRead, void** outPointer) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setDebugName(const char* name) override;
};

} // namespace d3d12
} // namespace gfx
