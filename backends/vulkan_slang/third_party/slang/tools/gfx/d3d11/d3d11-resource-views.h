// d3d11-resource-views.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class ResourceViewImpl : public ResourceViewBase
{
public:
    enum class Type
    {
        SRV,
        UAV,
        DSV,
        RTV,
    };
    Type m_type;
};

class ShaderResourceViewImpl : public ResourceViewImpl
{
public:
    ComPtr<ID3D11ShaderResourceView> m_srv;
};

class UnorderedAccessViewImpl : public ResourceViewImpl
{
public:
    ComPtr<ID3D11UnorderedAccessView> m_uav;
};

class DepthStencilViewImpl : public ResourceViewImpl
{
public:
    ComPtr<ID3D11DepthStencilView> m_dsv;
    DepthStencilClearValue m_clearValue;
};

class RenderTargetViewImpl : public ResourceViewImpl
{
public:
    ComPtr<ID3D11RenderTargetView> m_rtv;
    float m_clearValue[4];
};

} // namespace d3d11
} // namespace gfx
