// d3d12-shader-table.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class ShaderTableImpl : public ShaderTableBase
{
public:
    uint32_t m_rayGenTableOffset;
    uint32_t m_missTableOffset;
    uint32_t m_hitGroupTableOffset;
    uint32_t m_callableTableOffset;

    DeviceImpl* m_device;

    virtual RefPtr<BufferResource> createDeviceBuffer(
        PipelineStateBase* pipeline,
        TransientResourceHeapBase* transientHeap,
        IResourceCommandEncoder* encoder) override;
};

} // namespace d3d12
} // namespace gfx
