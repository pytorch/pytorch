// debug-buffer.cpp
#include "debug-buffer.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

IResource::Type DebugBufferResource::getType()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getType();
}

IBufferResource::Desc* DebugBufferResource::getDesc()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDesc();
}

DeviceAddress DebugBufferResource::getDeviceAddress()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDeviceAddress();
}

Result DebugBufferResource::getNativeResourceHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getNativeResourceHandle(outHandle);
}

Result DebugBufferResource::getSharedHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getSharedHandle(outHandle);
}

Result DebugBufferResource::setDebugName(const char* name)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setDebugName(name);
}

const char* DebugBufferResource::getDebugName()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDebugName();
}

Result DebugBufferResource::map(MemoryRange* rangeToRead, void** outPointer)
{
    SLANG_GFX_API_FUNC;
    return baseObject->map(rangeToRead, outPointer);
}

Result DebugBufferResource::unmap(MemoryRange* writtenRange)
{
    return baseObject->unmap(writtenRange);
}

} // namespace debug
} // namespace gfx
