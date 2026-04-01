// debug-pipeline-state.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugPipelineState : public DebugObject<IPipelineState>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IPipelineState* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace debug
} // namespace gfx
