// d3d11-buffer.cpp
#include "d3d11-buffer.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

SLANG_NO_THROW DeviceAddress SLANG_MCALL BufferResourceImpl::getDeviceAddress()
{
    return 0;
}

SLANG_NO_THROW Result SLANG_MCALL
BufferResourceImpl::map(MemoryRange* rangeToRead, void** outPointer)
{
    SLANG_UNUSED(rangeToRead);
    SLANG_UNUSED(outPointer);
    return SLANG_FAIL;
}

SLANG_NO_THROW Result SLANG_MCALL BufferResourceImpl::unmap(MemoryRange* writtenRange)
{
    SLANG_UNUSED(writtenRange);
    return SLANG_FAIL;
}

} // namespace d3d11
} // namespace gfx
