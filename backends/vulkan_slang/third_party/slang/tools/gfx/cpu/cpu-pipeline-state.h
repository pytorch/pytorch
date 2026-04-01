// cpu-pipeline-state.h
#pragma once
#include "cpu-base.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class PipelineStateImpl : public PipelineStateBase
{
public:
    ShaderProgramImpl* getProgram();

    void init(const ComputePipelineStateDesc& inDesc);
};

} // namespace cpu
} // namespace gfx
