// d3d11-shader-program.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class ShaderProgramImpl : public ShaderProgramBase
{
public:
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11ComputeShader> m_computeShader;
};

} // namespace d3d11
} // namespace gfx
