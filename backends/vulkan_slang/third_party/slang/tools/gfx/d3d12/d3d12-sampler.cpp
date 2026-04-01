// d3d12-sampler.cpp
#include "d3d12-sampler.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

SamplerStateImpl::~SamplerStateImpl()
{
    m_allocator->free(m_descriptor);
}

Result SamplerStateImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::D3D12CpuDescriptorHandle;
    outHandle->handleValue = m_descriptor.cpuHandle.ptr;
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
