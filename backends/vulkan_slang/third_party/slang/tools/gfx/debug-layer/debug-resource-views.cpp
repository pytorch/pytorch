// debug-resource-views.cpp
#include "debug-resource-views.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

IResourceView::Desc* DebugResourceView::getViewDesc()
{
    SLANG_GFX_API_FUNC;

    return baseObject->getViewDesc();
}

Result DebugResourceView::getNativeHandle(InteropHandle* outNativeHandle)
{
    SLANG_GFX_API_FUNC;

    return baseObject->getNativeHandle(outNativeHandle);
}

DeviceAddress DebugAccelerationStructure::getDeviceAddress()
{
    SLANG_GFX_API_FUNC;

    return baseObject->getDeviceAddress();
}

Result DebugAccelerationStructure::getNativeHandle(InteropHandle* outNativeHandle)
{
    SLANG_GFX_API_FUNC;

    return baseObject->getNativeHandle(outNativeHandle);
}

IResourceView::Desc* DebugAccelerationStructure::getViewDesc()
{
    SLANG_GFX_API_FUNC;

    return baseObject->getViewDesc();
}

} // namespace debug
} // namespace gfx
