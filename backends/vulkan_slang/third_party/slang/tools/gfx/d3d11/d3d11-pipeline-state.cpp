// d3d11-pipeline-state.cpp
#include "d3d11-pipeline-state.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

void GraphicsPipelineStateImpl::init(const GraphicsPipelineStateDesc& inDesc)
{
    PipelineStateBase::PipelineStateDesc pipelineDesc;
    pipelineDesc.graphics = inDesc;
    pipelineDesc.type = PipelineType::Graphics;
    initializeBase(pipelineDesc);
}

void ComputePipelineStateImpl::init(const ComputePipelineStateDesc& inDesc)
{
    PipelineStateBase::PipelineStateDesc pipelineDesc;
    pipelineDesc.compute = inDesc;
    pipelineDesc.type = PipelineType::Compute;
    initializeBase(pipelineDesc);
}

} // namespace d3d11
} // namespace gfx
