// cpu-pipeline-state.cpp
#include "cpu-pipeline-state.h"

#include "cpu-shader-program.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

ShaderProgramImpl* PipelineStateImpl::getProgram()
{
    return static_cast<ShaderProgramImpl*>(m_program.Ptr());
}

void PipelineStateImpl::init(const ComputePipelineStateDesc& inDesc)
{
    PipelineStateDesc pipelineDesc;
    pipelineDesc.type = PipelineType::Compute;
    pipelineDesc.compute = inDesc;
    initializeBase(pipelineDesc);
}

} // namespace cpu
} // namespace gfx
