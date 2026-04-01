// metal-pipeline-state.h
#pragma once

#include "metal-base.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class PipelineStateImpl : public PipelineStateBase
{
public:
    DeviceImpl* m_device;
    NS::SharedPtr<MTL::RenderPipelineState> m_renderPipelineState;
    NS::SharedPtr<MTL::DepthStencilState> m_depthStencilState;
    NS::SharedPtr<MTL::ComputePipelineState> m_computePipelineState;
    MTL::Size m_threadGroupSize;
    NS::UInteger m_vertexBufferOffset;

    PipelineStateImpl(DeviceImpl* device);
    ~PipelineStateImpl();

    void init(const GraphicsPipelineStateDesc& desc);
    void init(const ComputePipelineStateDesc& desc);
    void init(const RayTracingPipelineStateDesc& desc);

    Result createMetalComputePipelineState();
    Result createMetalRenderPipelineState();

    virtual Result ensureAPIPipelineStateCreated() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class RayTracingPipelineStateImpl : public PipelineStateImpl
{
public:
    Dictionary<String, Index> shaderGroupNameToIndex;
    Int shaderGroupCount;

    RayTracingPipelineStateImpl(DeviceImpl* device);

    virtual Result ensureAPIPipelineStateCreated() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace metal
} // namespace gfx
