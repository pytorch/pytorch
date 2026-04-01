// d3d11-scopeNVAPI.cpp
#include "d3d11-scopeNVAPI.h"

#include "d3d11-device.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

SlangResult ScopeNVAPI::init(DeviceImpl* device, Index regIndex)
{
    if (!device->m_nvapi)
    {
        // There is nothing to set as nvapi is not set
        return SLANG_OK;
    }

#ifdef GFX_NVAPI
    NvAPI_Status nvapiStatus = NvAPI_D3D11_SetNvShaderExtnSlot(device->m_device, NvU32(regIndex));
    if (nvapiStatus != NVAPI_OK)
    {
        return SLANG_FAIL;
    }
#endif

    // Record the renderer so it can be freed
    m_renderer = device;
    return SLANG_OK;
}

ScopeNVAPI::~ScopeNVAPI()
{
    // If the m_renderer is not set, it must not have been set up
    if (m_renderer)
    {
#ifdef GFX_NVAPI
        // Disable the slot used
        NvAPI_Status nvapiStatus = NvAPI_D3D11_SetNvShaderExtnSlot(m_renderer->m_device, ~0);
        SLANG_ASSERT(nvapiStatus == NVAPI_OK);
#endif
    }
}

} // namespace d3d11
} // namespace gfx
