// debug-query.cpp
#include "debug-query.h"

#include "debug-helper-functions.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

Result DebugQueryPool::getResult(GfxIndex index, GfxCount count, uint64_t* data)
{
    SLANG_GFX_API_FUNC;

    if (index < 0 || index + count > desc.count)
        GFX_DIAGNOSE_ERROR("index is out of bounds.");
    return baseObject->getResult(index, count, data);
}

Result DebugQueryPool::reset()
{
    SLANG_GFX_API_FUNC;
    return baseObject->reset();
}

} // namespace debug
} // namespace gfx
