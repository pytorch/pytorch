// cuda-buffer.cpp
#include "cuda-buffer.h"

#include "cuda-helper-functions.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

BufferResourceImpl::~BufferResourceImpl()
{
    if (m_cudaMemory)
    {
        SLANG_CUDA_ASSERT_ON_FAIL(cuMemFree((CUdeviceptr)m_cudaMemory));
    }
}

uint64_t BufferResourceImpl::getBindlessHandle()
{
    return (uint64_t)m_cudaMemory;
}

DeviceAddress BufferResourceImpl::getDeviceAddress()
{
    return (DeviceAddress)m_cudaMemory;
}

Result BufferResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = getBindlessHandle();
    outHandle->api = InteropHandleAPI::CUDA;
    return SLANG_OK;
}

Result BufferResourceImpl::map(MemoryRange* rangeToRead, void** outPointer)
{
    SLANG_UNUSED(rangeToRead);
    SLANG_UNUSED(outPointer);
    return SLANG_FAIL;
}

Result BufferResourceImpl::unmap(MemoryRange* writtenRange)
{
    SLANG_UNUSED(writtenRange);
    return SLANG_FAIL;
}

} // namespace cuda
#endif
} // namespace gfx
