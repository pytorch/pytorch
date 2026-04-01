// d3d11-sampler.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class SamplerStateImpl : public SamplerStateBase
{
public:
    ComPtr<ID3D11SamplerState> m_sampler;
};

} // namespace d3d11
} // namespace gfx
