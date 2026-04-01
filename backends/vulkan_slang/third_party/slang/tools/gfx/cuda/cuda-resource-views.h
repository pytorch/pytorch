// cuda-resource-views.h
#pragma once
#include "cuda-base.h"
#include "cuda-buffer.h"
#include "cuda-texture.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class ResourceViewImpl : public ResourceViewBase
{
public:
    RefPtr<BufferResourceImpl> memoryResource = nullptr;
    RefPtr<TextureResourceImpl> textureResource = nullptr;
    void* proxyBuffer = nullptr;
};

} // namespace cuda
#endif
} // namespace gfx
