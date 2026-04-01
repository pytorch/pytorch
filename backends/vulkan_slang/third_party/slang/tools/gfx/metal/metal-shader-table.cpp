// metal-shader-table.cpp
#include "metal-shader-table.h"

#include "metal-device.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

RefPtr<BufferResource> ShaderTableImpl::createDeviceBuffer(
    PipelineStateBase* pipeline,
    TransientResourceHeapBase* transientHeap,
    IResourceCommandEncoder* encoder)
{
    return RefPtr<BufferResource>(0);
}

} // namespace metal
} // namespace gfx
