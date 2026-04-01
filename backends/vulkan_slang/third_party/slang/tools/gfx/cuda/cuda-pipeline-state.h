// cuda-pipeline-state.h
#pragma once
#include "cuda-base.h"
#include "cuda-shader-program.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class PipelineStateImpl : public PipelineStateBase
{
public:
};

class ComputePipelineStateImpl : public PipelineStateImpl
{
public:
    RefPtr<ShaderProgramImpl> shaderProgram;
    void init(const ComputePipelineStateDesc& inDesc);
};

} // namespace cuda
#endif
} // namespace gfx
