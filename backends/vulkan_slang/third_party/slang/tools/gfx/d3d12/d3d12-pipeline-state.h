// d3d12-ray-tracing.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class PipelineStateImpl : public PipelineStateBase
{
public:
    PipelineStateImpl(DeviceImpl* device)
        : m_device(device)
    {
    }
    DeviceImpl* m_device;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    void init(const GraphicsPipelineStateDesc& inDesc);
    void init(const ComputePipelineStateDesc& inDesc);
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
    virtual Result ensureAPIPipelineStateCreated() override;
};

#if SLANG_GFX_HAS_DXR_SUPPORT
class RayTracingPipelineStateImpl : public PipelineStateBase
{
public:
    ComPtr<ID3D12StateObject> m_stateObject;
    DeviceImpl* m_device;
    RayTracingPipelineStateImpl(DeviceImpl* device);
    void init(const RayTracingPipelineStateDesc& inDesc);
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
    virtual Result ensureAPIPipelineStateCreated() override;
};
#endif

} // namespace d3d12
} // namespace gfx
