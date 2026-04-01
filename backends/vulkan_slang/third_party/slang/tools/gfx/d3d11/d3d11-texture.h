// d3d11-texture.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class TextureResourceImpl : public TextureResource
{
public:
    typedef TextureResource Parent;

    TextureResourceImpl(const Desc& desc)
        : Parent(desc)
    {
    }
    ComPtr<ID3D11Resource> m_resource;
};

} // namespace d3d11
} // namespace gfx
