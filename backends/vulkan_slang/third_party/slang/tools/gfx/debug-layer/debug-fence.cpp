// debug-fence.cpp
#include "debug-fence.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

Result DebugFence::getSharedHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getSharedHandle(outHandle);
}

Result DebugFence::getNativeHandle(InteropHandle* outNativeHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getNativeHandle(outNativeHandle);
}

Result DebugFence::getCurrentValue(uint64_t* outValue)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getCurrentValue(outValue);
}

Result DebugFence::setCurrentValue(uint64_t value)
{
    SLANG_GFX_API_FUNC;
    if (value < maxValueToSignal)
    {
        GFX_DIAGNOSE_ERROR_FORMAT(
            "Cannot set fence value (%d) to lower than pending signal value (%d) on the fence.",
            value,
            maxValueToSignal);
    }
    return baseObject->setCurrentValue(value);
}

} // namespace debug
} // namespace gfx
