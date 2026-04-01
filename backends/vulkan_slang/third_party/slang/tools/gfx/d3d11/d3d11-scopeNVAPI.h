// d3d11-scopeNVAPI.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class ScopeNVAPI
{
public:
    ScopeNVAPI()
        : m_renderer(nullptr)
    {
    }
    SlangResult init(DeviceImpl* renderer, Index regIndex);
    ~ScopeNVAPI();

protected:
    DeviceImpl* m_renderer;
};

} // namespace d3d11
} // namespace gfx
