// debug-texture.cpp
#include "debug-texture.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

IResource::Type DebugTextureResource::getType()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getType();
}

ITextureResource::Desc* DebugTextureResource::getDesc()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDesc();
}

Result DebugTextureResource::getNativeResourceHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getNativeResourceHandle(outHandle);
}

Result DebugTextureResource::getSharedHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getSharedHandle(outHandle);
}

Result DebugTextureResource::setDebugName(const char* name)
{
    return baseObject->setDebugName(name);
}

const char* DebugTextureResource::getDebugName()
{
    return baseObject->getDebugName();
}

} // namespace debug
} // namespace gfx
