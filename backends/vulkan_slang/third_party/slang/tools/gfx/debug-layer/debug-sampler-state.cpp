// debug-sampler-state.cpp
#include "debug-sampler-state.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

Result DebugSamplerState::getNativeHandle(InteropHandle* outNativeHandle)
{
    SLANG_GFX_API_FUNC;

    return baseObject->getNativeHandle(outNativeHandle);
}

} // namespace debug
} // namespace gfx
