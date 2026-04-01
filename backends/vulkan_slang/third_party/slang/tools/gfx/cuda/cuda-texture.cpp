// cuda-texture.cpp
#include "cuda-texture.h"

#include "cuda-helper-functions.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

TextureResourceImpl::~TextureResourceImpl()
{
    if (m_cudaSurfObj)
    {
        SLANG_CUDA_ASSERT_ON_FAIL(cuSurfObjectDestroy(m_cudaSurfObj));
    }
    if (m_cudaTexObj)
    {
        SLANG_CUDA_ASSERT_ON_FAIL(cuTexObjectDestroy(m_cudaTexObj));
    }
    if (m_cudaArray)
    {
        SLANG_CUDA_ASSERT_ON_FAIL(cuArrayDestroy(m_cudaArray));
    }
    if (m_cudaMipMappedArray)
    {
        SLANG_CUDA_ASSERT_ON_FAIL(cuMipmappedArrayDestroy(m_cudaMipMappedArray));
    }
}

uint64_t TextureResourceImpl::getBindlessHandle()
{
    return (uint64_t)m_cudaTexObj;
}

Result TextureResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = getBindlessHandle();
    outHandle->api = InteropHandleAPI::CUDA;
    return SLANG_OK;
}

} // namespace cuda
#endif
} // namespace gfx
