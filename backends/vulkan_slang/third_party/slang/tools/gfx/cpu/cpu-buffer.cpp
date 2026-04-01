// cpu-buffer.cpp
#include "cpu-buffer.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

BufferResourceImpl::~BufferResourceImpl()
{
    if (m_data)
    {
        free(m_data);
    }
}

Result BufferResourceImpl::init()
{
    m_data = malloc(m_desc.sizeInBytes);
    if (!m_data)
        return SLANG_E_OUT_OF_MEMORY;
    return SLANG_OK;
}

Result BufferResourceImpl::setData(size_t offset, size_t size, void const* data)
{
    memcpy((char*)m_data + offset, data, size);
    return SLANG_OK;
}

SLANG_NO_THROW DeviceAddress SLANG_MCALL BufferResourceImpl::getDeviceAddress()
{
    return (DeviceAddress)m_data;
}

SLANG_NO_THROW Result SLANG_MCALL
BufferResourceImpl::map(MemoryRange* rangeToRead, void** outPointer)
{
    SLANG_UNUSED(rangeToRead);
    if (outPointer)
        *outPointer = m_data;
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL BufferResourceImpl::unmap(MemoryRange* writtenRange)
{
    SLANG_UNUSED(writtenRange);
    return SLANG_OK;
}

} // namespace cpu
} // namespace gfx
