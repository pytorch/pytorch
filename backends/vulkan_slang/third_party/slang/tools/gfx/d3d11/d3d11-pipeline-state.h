// d3d11-pipeline-state.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class PipelineStateImpl : public PipelineStateBase
{
public:
};

class GraphicsPipelineStateImpl : public PipelineStateImpl
{
public:
    UINT m_rtvCount;

    RefPtr<InputLayoutImpl> m_inputLayout;
    ComPtr<ID3D11DepthStencilState> m_depthStencilState;
    ComPtr<ID3D11RasterizerState> m_rasterizerState;
    ComPtr<ID3D11BlendState> m_blendState;

    float m_blendColor[4];
    UINT m_sampleMask;

    void init(const GraphicsPipelineStateDesc& inDesc);
};

class ComputePipelineStateImpl : public PipelineStateImpl
{
public:
    void init(const ComputePipelineStateDesc& inDesc);
};

} // namespace d3d11
} // namespace gfx
