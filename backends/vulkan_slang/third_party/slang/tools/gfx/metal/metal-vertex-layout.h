// metal-vertex-layout.h
#pragma once

#include "metal-base.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class InputLayoutImpl : public InputLayoutBase
{
public:
    List<InputElementDesc> m_inputElements;
    List<VertexStreamDesc> m_vertexStreams;

    Result init(const IInputLayout::Desc& desc);
    NS::SharedPtr<MTL::VertexDescriptor> createVertexDescriptor(
        NS::UInteger vertexBufferIndexOffset);
};

} // namespace metal
} // namespace gfx
