// vk-shader-table.h
#pragma once

#include "vk-base.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class ShaderTableImpl : public ShaderTableBase
{
public:
    uint32_t m_raygenTableSize;
    uint32_t m_missTableSize;
    uint32_t m_hitTableSize;
    uint32_t m_callableTableSize;

    DeviceImpl* m_device;

    virtual RefPtr<BufferResource> createDeviceBuffer(
        PipelineStateBase* pipeline,
        TransientResourceHeapBase* transientHeap,
        IResourceCommandEncoder* encoder) override;
};

} // namespace vk
} // namespace gfx
