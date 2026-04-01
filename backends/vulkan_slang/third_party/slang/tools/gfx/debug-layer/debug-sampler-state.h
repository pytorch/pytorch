// debug-sampler-state.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugSamplerState : public DebugObject<ISamplerState>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    ISamplerState* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;
};

} // namespace debug
} // namespace gfx
